import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer 

import datetime
from utils.eval_acc import compute_acc_for_df, compute_acc_for_df_eval
from dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)
import time
from tqdm import tqdm

from engine.trainer import TrainerDF

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'IVLP_VL_Adapter_Prompt',
                      "vision_depth": cfg.TRAINER.IVLP.PROMPT_DEPTH_VISION,
                      "language_depth": cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT, "vision_ctx": cfg.TRAINER.IVLP.N_CTX_VISION,
                      "language_ctx": cfg.TRAINER.IVLP.N_CTX_TEXT,
                      "add_linear": cfg.ADD_LINEAR,
                      "use_classtoken": cfg.USE_CLASSTOKEN,
                      "use_cross_attention": cfg.USE_CROSSATTENTION,
                      "independent_cross_attention": cfg.INDEPENDENT_CROSS_ATTENTION,
                      "independent_learnable_vision": cfg.INDEPENDENT_LEARNABLE_VISION,
                      "insert_layer": cfg.INSERT_LAYER_ATTN
                      }
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class FixedEmbeddings():
    def __init__(self, cfg, classnames, clip_model):
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        prompt_prefix = "a photo of a"
        print('Vision Prompting Design')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Vision prompting: {cfg.TRAINER.VPT.N_CTX_VISION}")
        print(f"Using fixed hand crated prompts")

        # FIXME
        if cfg.DATASET.NAME == "OfficeHomeDF":
            root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
            dataset_dir = osp.join(root, "office_home_dg")
            embedding_file = dataset_dir + "/a_photo_of_a_cls.pt"
            

        elif cfg.DATASET.NAME == "DomainNetMiniDF":
            embedding_file = "/home/gotoyuta/lab/Dataset/domainnet/a_photo_of_a_cls.pt"
        elif cfg.DATASET.NAME == "ImageNetDF":
            embedding_file = "/home/gotoyuta/lab/Dataset/IMAGENET/a_photo_of_a_cls.pt"

        if osp.exists(embedding_file):
            print(f"Loading text features from {embedding_file}")
            text_features = torch.load(embedding_file)
        else:
            print(f"Generating text features and saving to {embedding_file}")
            classnames = [name.replace("_", " ") for name in classnames]
            prompts = [prompt_prefix + " " + name + "." for name in classnames]
            tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
            with torch.no_grad():
                text_features = clip_model.encode_text(tokenized_prompts)
            torch.save(text_features, embedding_file)

        self.fixed_embeddings = text_features

    def return_fixed_embeddings(self):
        return self.fixed_embeddings

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class VLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        # Make sure Language depth >= 1
        assert cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT >= 1, "In Independent VL prompting, Language prompt depth should be >=1" \
                                                        "\nPlease use VPT trainer if you want to learn only vision " \
                                                        "branch  "
        n_ctx = cfg.TRAINER.IVLP.N_CTX_TEXT
        ctx_init = cfg.TRAINER.IVLP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and (n_ctx) <= 4:
            # Use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # Random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print(f"Independent V-L design")
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        print(f"Number of context words (tokens) for Vision prompting: {cfg.TRAINER.IVLP.N_CTX_VISION}")
        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        if cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT == 0:
            self.embeddings = FixedEmbeddings(cfg, classnames, clip_model)
        else :
            self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model)
            self.tokenized_prompts = self.prompt_learner.tokenized_prompts
            self.text_encoder = TextEncoder(clip_model)
        # self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model)
        # self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        # self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.use_vision_adapter = cfg.USE_VISION_ADAPTER
        self.use_text_adapter = cfg.USE_TEXT_ADAPTER
        if self.use_vision_adapter:
            self.vision_adapter = Adapter(self.image_encoder.output_dim, clip_model.dtype)
        if self.use_text_adapter:
            self.text_adapter = Adapter(self.image_encoder.output_dim, clip_model.dtype)
        # self.vision_adapter = Adapter(self.image_encoder.output_dim, clip_model.dtype)
        # self.text_adapter = Adapter(self.image_encoder.output_dim, clip_model.dtype)
        if cfg.USE_DOMAIN_CLASIFIER_LOSS:
            if cfg.DOMAIN_CLASS_DIVIDED:
                if cfg.IS_DOMAIN_DIVIDED:
                    if cfg.DATASET.NAME == "Office31DF":
                        self.domain_classifier = nn.Linear(self.image_encoder.output_dim, 3*len(classnames))
                    else:
                        self.domain_classifier = nn.Linear(self.image_encoder.output_dim, 4*len(classnames))
                else :
                    self.domain_classifier = nn.Linear(self.image_encoder.output_dim, 2*len(classnames))
            else :
                if cfg.IS_DOMAIN_DIVIDED:
                    if cfg.DATASET.NAME == "Office31DF":
                        self.domain_classifier = nn.Linear(self.image_encoder.output_dim, 3)
                    else :
                        self.domain_classifier = nn.Linear(self.image_encoder.output_dim, 4)
                else :
                    self.domain_classifier = nn.Linear(self.image_encoder.output_dim, 2)
            self.domain_classifier.to(self.dtype)
        self.use_domain_cls_loss = cfg.USE_DOMAIN_CLASIFIER_LOSS
        self.text_depth = cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT 

    def forward(self, image, label=None):
        # tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        # prompts = self.prompt_learner()
        # text_features = self.text_encoder(prompts, tokenized_prompts)
        if self.text_depth == 0:
            text_features = self.embeddings.return_fixed_embeddings().cuda()
        else :
            tokenized_prompts = self.tokenized_prompts
            prompts = self.prompt_learner()
            text_features = self.text_encoder(prompts, tokenized_prompts)
        image_features = self.image_encoder(image.type(self.dtype))
        
        # image_features = self.vision_adapter(image_features)
        # text_features = self.text_adapter(text_features)
        if self.use_vision_adapter:
            image_features = self.vision_adapter(image_features)
        if self.use_text_adapter:
            text_features = self.text_adapter(text_features)


        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()

        if self.use_domain_cls_loss:
            domain_logit = self.domain_classifier(image_features)
            return logits, image_features, text_features, domain_logit
        # if self.prompt_learner.training:
        #     return F.cross_entropy(logits, label)

        return logits, image_features, text_features

class Adapter(nn.Module):
    def __init__(self, c_in, dtype, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False).to(dtype=dtype),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False).to(dtype=dtype),
            nn.ReLU(inplace=True)
        )
        self.dtype = dtype

    def forward(self, x):
        x = self.fc(x)
        return x.type(self.dtype)

@TRAINER_REGISTRY.register()
class IVLP_VL_Adapter_Prompt(TrainerDF):
        
    def check_cfg(self, cfg):
        assert cfg.TRAINER.IVLP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.IVLP.PREC == "fp32" or cfg.TRAINER.IVLP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                elif "adapter" in name:
                    param.requires_grad_(True)
                elif "domain_classifier" in name:
                    param.requires_grad_(True)
                elif "cross_attn" in name:
                    param.requires_grad_(True)
                elif "added_linear" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("VLPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.IVLP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)