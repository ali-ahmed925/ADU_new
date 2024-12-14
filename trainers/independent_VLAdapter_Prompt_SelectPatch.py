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
    design_details = {"trainer": 'IVLP_VL_Adapter_Prompt_SelectPatch',
                      "vision_depth": cfg.TRAINER.IVLP.PROMPT_DEPTH_VISION,
                      "language_depth": cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT, "vision_ctx": cfg.TRAINER.IVLP.N_CTX_VISION,
                      "language_ctx": cfg.TRAINER.IVLP.N_CTX_TEXT,
                      "add_linear": cfg.ADD_LINEAR,
                      "use_classtoken": cfg.USE_CLASSTOKEN,
                      "topk": cfg.TOPK,
                      "select_method": "block_shuffle"
                      }
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


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
        self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.vision_adapter = Adapter(self.image_encoder.output_dim, clip_model.dtype)
        self.text_adapter = Adapter(self.image_encoder.output_dim, clip_model.dtype)
        if cfg.USE_DOMAIN_CLASIFIER_LOSS:
            if cfg.DOMAIN_CLASS_DIVIDED:
                if cfg.IS_DOMAIN_DIVIDED:
                    self.domain_classifier = nn.Linear(self.image_encoder.output_dim, 4*len(classnames))
                else :
                    self.domain_classifier = nn.Linear(self.image_encoder.output_dim, 2*len(classnames))
            else :
                if cfg.IS_DOMAIN_DIVIDED:
                    self.domain_classifier = nn.Linear(self.image_encoder.output_dim, 4)
                else :
                    self.domain_classifier = nn.Linear(self.image_encoder.output_dim, 2)
            self.domain_classifier.to(self.dtype)
        self.use_domain_cls_loss = cfg.USE_DOMAIN_CLASIFIER_LOSS

    def forward(self, image, block_shuffled_image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts)
        image_features = self.image_encoder(image.type(self.dtype), block_shuffled_image.type(self.dtype))
        
        image_features = self.vision_adapter(image_features)
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

from utils.data_augmentation import *
from utils.loss_fn import *
from utils.eval_acc import *
import pandas as pd
from torchvision.transforms import v2
@TRAINER_REGISTRY.register()
class IVLP_VL_Adapter_Prompt_SelectPatch(TrainerDF):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.grid_num = cfg.GRID
        self.resize = v2.Resize(size=224)
        self.vflip = v2.RandomVerticalFlip(p=1)
        self.blur = v2.GaussianBlur(kernel_size=(5,9), sigma=(10.,30.))
        
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
    
    def forward_backward(self, batch):
        image, label, domain = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output, img_feat, txt_feat = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            # destructed_image = image
            destructed_image = self.blur(image)
            destructed_image = get_jigsaw_tensor(destructed_image, resize=(224,224), grid=self.grid_num)
            destructed_image = self.resize(destructed_image)

            if self.use_domain_classifier_loss :
                output, img_feat, txt_feat, domain_output = self.model(image, destructed_image)
            else :
                output, img_feat, txt_feat = self.model(image, destructed_image)

            if not self.cfg.NO_FORGET:
                entropy = Entropy()
                false_check_tensor = torch.zeros_like(domain, dtype=torch.bool)
                
                ###########################################
                # preservation loss
                # deletion loss
                ############################################
                # for prv_domain in prv_domain_list:
                prv_domain_index = [self.domain_list.index(prv_d) for prv_d in self.prv_domain_list if prv_d in self.domain_list]
                prv_domain_mask = torch.isin(domain, torch.tensor(prv_domain_index).to(self.device))
                # for del_domain in del_domain_list:
                del_domain_index = [self.domain_list.index(del_d) for del_d in self.del_domain_list if del_d in self.domain_list]
                del_domain_mask = torch.isin(domain, torch.tensor(del_domain_index).to(self.device))
                
                if torch.equal(false_check_tensor, prv_domain_mask):
                    loss_prv = 0
                else :
                    loss_prv = F.cross_entropy(output[prv_domain_mask], label[prv_domain_mask])
                if torch.equal(false_check_tensor, del_domain_mask):
                    loss_del = 0
                else :
                    loss_del = entropy(output[del_domain_mask])
                loss = loss_prv - loss_del

                # select target label to calculate domain class label
                if self.domain_class_divided :
                    if self.is_domain_divided:
                        target_label_list = []
                        for lb, dm in zip(label, domain):
                            new_label = int(lb.item()*len(self.domain_list) + dm.item())
                            target_label_list.append(new_label)
                        target_label = torch.tensor(target_label_list).to(self.device)
                    else :
                        label_div = prv_domain_mask.int().long()
                        target_label_list = []
                        for lb, dm in zip(label_div, domain):
                            new_label = int(lb.item()*len(self.domain_list) + dm.item())
                            target_label_list.append(new_label)
                        target_label = torch.tensor(target_label_list).to(self.device)
                else :
                    if self.is_domain_divided:
                        target_label = domain
                    else :
                        target_label = prv_domain_mask.int().long()

                ######################################################################
                # domain loss (domain classifier loss, nearest neighbor loss or both)
                #####################################################################

                if self.use_domain_classifier_loss :
                    domain_cls_loss = F.cross_entropy(domain_output, target_label)
                    loss += domain_cls_loss
                if self.use_nearest_neighbor_loss :
                    domain_nn_loss = self.nnl(img_feat, target_label)
                    loss += domain_nn_loss
            else :
                loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)
        
        if not self.cfg.NO_FORGET:
            loss_summary = {
                "loss": loss.item(),
                "loss_prv": loss_prv.item() if isinstance(loss_prv, torch.Tensor) else loss_prv,
                "loss_del": loss_del.item() if isinstance(loss_del, torch.Tensor) else loss_del,
                "loss_domain_cls": domain_cls_loss.item() if self.use_domain_classifier_loss else 0,
                "loss_domain_nn": domain_nn_loss.item() if self.use_nearest_neighbor_loss else 0,
                # "acc": compute_accuracy(output, label)[0].item(),
            }
            acc = compute_acc_for_df(output, label, prv_domain_mask, del_domain_mask, domain, self.domain_list, device=self.device)
            loss_summary.update(acc)
            # loss_summary.update(acc)
        else :
            loss_summary = {
                "loss": loss.item(),
                "acc": compute_accuracy(output, label)[0].item()
            }
            # acc = compute_accuracy(output, label)[0].item()
        # acc = compute_acc_for_df(output, label, prv_domain_mask, del_domain_mask, domain, self.domain_list, device=self.device)
        
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary
    
    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_tensorboard()
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")
        eval_dict = {}
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label, domain = self.parse_batch_test(batch)
            # destructed_image = input
            destructed_image = self.blur(input)
            destructed_image = get_jigsaw_tensor(destructed_image, resize=(224,224), grid=self.grid_num)
            destructed_image = self.resize(destructed_image)
            if self.use_domain_classifier_loss :
                output, img_feat, txt_feat, domain_logit = self.model_inference(input, destructed_image)
            else :
                output, img_feat, txt_feat = self.model_inference(input, destructed_image)
            self.evaluator.process(output, label)

            # for prv_domain in prv_domain_list:
            prv_domain_index = [self.domain_list.index(prv_d) for prv_d in self.prv_domain_list if prv_d in self.domain_list]
            prv_domain_mask = torch.isin(domain, torch.tensor(prv_domain_index).to(self.device))
            # for del_domain in del_domain_list:
            del_domain_index = [self.domain_list.index(del_d) for del_d in self.del_domain_list if del_d in self.domain_list]
            del_domain_mask = torch.isin(domain, torch.tensor(del_domain_index).to(self.device))
            if self.domain_class_divided :
                if self.is_domain_divided:
                    target_label_list = []
                    for lb, dm in zip(label, domain):
                        new_label = int(lb.item()*len(self.domain_list) + dm.item())
                        target_label_list.append(new_label)
                    target_label = torch.tensor(target_label_list).to(self.device)
                else :
                    label_div = prv_domain_mask.int().long()
                    target_label_list = []
                    for lb, dm in zip(label_div, domain):
                        new_label = int(lb.item()*len(self.domain_list) + dm.item())
                        target_label_list.append(new_label)
                    target_label = torch.tensor(target_label_list).to(self.device)
            else :
                if self.is_domain_divided:
                    target_label = domain
                else :
                    target_label = prv_domain_mask.int().long()
            if self.domain_class_divided:
                if self.use_domain_classifier_loss:
                    eval_dict = compute_acc_for_df_eval(
                        eval_dict,
                        output,
                        label,
                        prv_domain_mask,
                        del_domain_mask,
                        domain,
                        self.domain_list,
                        domain_logit,
                        target_label,
                        self.is_domain_divided,
                        self.use_domain_classifier_loss,
                        self.device,
                        domain_class_divided=self.domain_class_divided,
                        classnames=self.classnames
                    )
            else :
                if self.use_domain_classifier_loss:
                    eval_dict = compute_acc_for_df_eval(
                        eval_dict,
                        output,
                        label,
                        prv_domain_mask,
                        del_domain_mask,
                        domain,
                        self.domain_list,
                        domain_logit,
                        target_label,
                        self.is_domain_divided,
                        self.use_domain_classifier_loss,
                        self.device
                    )
                else :
                    eval_dict = compute_acc_for_df_eval(
                        eval_dict,
                        output,
                        label,
                        prv_domain_mask,
                        del_domain_mask,
                        domain,
                        self.domain_list,
                        None,
                        target_label,
                        self.is_domain_divided,
                        self.use_domain_classifier_loss,
                        self.device
                    )
            if batch_idx == 0:
                label_all = label
                domain_all = domain
                img_feat_all = img_feat
                # input_all = input

            else :
                label_all = torch.cat((label_all, label))
                domain_all = torch.cat((domain_all, domain))
                img_feat_all = torch.cat((img_feat_all, img_feat))
                # input_all = torch.cat((input_all, input))

        for cls_id, clsname in enumerate(self.classnames):
            cls_specific_index = (label_all == cls_id)
            if torch.all(cls_specific_index == False):
                pass
            else:
                domain_metadata = []
                for d in domain_all[cls_specific_index]:
                    domain_metadata.append(self.domain_list[int(d)])
                tag = f"{split}/tsne-plot/{clsname}"
                # self.write_embedding(img_feat[cls_specific_index], domain_metadata, input[cls_specific_index], global_step=batch_idx, tag=tag)
                self.write_embedding(img_feat_all[cls_specific_index], domain_metadata, tag=tag)

        results = self.evaluator.evaluate()
        #############################################################
        #
        # add csv files
        #
        ##############################################################
        csv_col_name = ""
        for idx, dname in enumerate(self.del_domain_list):
            if idx == len(self.domain_list) - 1:
                csv_col_name += f"{dname}"
            else :
                csv_col_name += f"{dname} "

        if csv_col_name not in self.df.columns:
            new_column_values = [
                round((eval_dict["correct_prv"] / eval_dict["total_prv"])*100, 3),
                round((1 - eval_dict["correct_del"] / eval_dict["total_del"])*100, 3), 
                round((eval_dict["correct_del"] / eval_dict["total_del"])*100, 3)
            ]
            specific_acc = "("
            for idx, dname in enumerate(self.del_domain_list):
                sp_acc = (eval_dict[f"correct_{dname}"] / eval_dict[f"total_{dname}"])*100
                if idx == len(self.del_domain_list) - 1:
                    specific_acc += f"{sp_acc:.3f})"
                else:
                    specific_acc += f"{sp_acc:.3f}, "

            new_column_values.append(specific_acc)

            self.df[csv_col_name] = new_column_values
            self.df.to_csv(self.csv_file_path)

        ############################################################
        #
        # print result
        #
        ############################################################

        if not self.cfg.NO_FORGET:
            print("==========peservation or delete acc===============")
            for name in ["prv", "del"]:
                acc = eval_dict[f"correct_{name}"] / eval_dict[f"total_{name}"]
                print(f"{name} : {acc:.5f}")
            print("==============domain specific acc=================")
            for domain_name in self.domain_list:
                acc = eval_dict[f"correct_{domain_name}"] / eval_dict[f"total_{domain_name}"]
                print(f"{domain_name} : {acc:.5f}")
            if self.domain_class_divided:
                if self.use_domain_classifier_loss:
                    print("==================domain DC accuracy==================")
                    if self.is_domain_divided:
                        for idx in range(len(self.domain_list)*len(self.classnames)):
                            cls = self.classnames[int(idx / len(self.domain_list))]
                            dm = self.domain_list[int(idx % len(self.domain_list))]
                            acc = eval_dict[f"correct_{cls}_{dm}_DC"] / eval_dict[f"total_{cls}_{dm}_DC"]
                            print(f"{cls} {dm} : {acc:.5f}")
            else:
                if self.use_domain_classifier_loss:
                    print("==================domain DC accuracy==================")
                    if self.is_domain_divided:
                        for domain_name in self.domain_list:
                            acc = eval_dict[f"correct_{domain_name}_DC"] / eval_dict[f"total_{domain_name}_DC"]
                            print(f"{domain_name} : {acc:.5f}")
                    else:
                        for domain_name in ["prv", "del"]:
                            acc = eval_dict[f"correct_{domain_name}_DC"] / eval_dict[f"total_{domain_name}_DC"]
                            print(f"{domain_name} : {acc:.5f}")
                    acc = eval_dict["correct_domain"] / eval_dict["total_domain"]
                    print("==================domain DC accuracy tot==================")
                    print(f"domain acc : {acc:.5f}")
            print("===================================================")


        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]
    
    def write_embedding(self, mat, meta_data, label_img=None, global_step=None, tag=None):
        if self._writer is None:
            # Do nothing if writer is not initialized
            # Note that writer is only used when training is needed
            pass
        else:
            self._writer.add_embedding(mat, meta_data, label_img, global_step, tag)
    
    def model_inference(self, input, destructed_input):
        return self.model(input, destructed_input)