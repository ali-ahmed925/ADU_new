import os.path as osp
from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
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
import matplotlib.pyplot as plt
import os
import numpy as np
from clip.model import get_similarity_map


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
    design_details = { "trainer": "VPT_Local",
                    "vision_depth": cfg.TRAINER.VPT.PROMPT_DEPTH_VISION,
                      "vision_ctx": cfg.TRAINER.VPT.N_CTX_VISION,
                      "language_depth": 0,
                      "language_ctx": 0}
    assert cfg.TRAINER.VPT.PROMPT_DEPTH_VISION >= 1, "For Vision Prompting, PROMPT_DEPTH_VISION should be >= 1"
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model.float()

def load_clip_to_cpu_expert(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'ZeroshotCLIP_Local',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model.float()


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

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            text_features = clip_model.encode_text(tokenized_prompts)

        self.fixed_embeddings = text_features

    def return_fixed_embeddings(self):
        return self.fixed_embeddings


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.embeddings = FixedEmbeddings(cfg, classnames, clip_model)
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, label=None, training=False):
        logit_scale = self.logit_scale.exp()

        text_features = self.embeddings.return_fixed_embeddings().cuda()
        image_features, local_feat = self.image_encoder(image.type(self.dtype))

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()

        # if training:
        #     return F.cross_entropy(logits, label)

        return logits, image_features, local_feat, text_features

from engine.trainer import TrainerDF_Local, TrainerDF_Local_Distill

@TRAINER_REGISTRY.register()
class VPT_Local_Distill(TrainerDF_Local_Distill):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.VPT.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.VPT.PREC == "fp32" or cfg.TRAINER.VPT.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)
        self.model_expert = load_clip_to_cpu_expert(cfg)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
        
        for _, param in self.model_expert.named_parameters():
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
        self.model_expert.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.VPT.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = 1
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

    # def parse_batch_test(self, batch):
    #     input = batch["img"]
    #     label = batch["label"]
    #     domain = batch["domain"]
    #     impath = batch["impath"]

    #     input = input.to(self.device)
    #     label = label.to(self.device)
    #     domain = domain.to(self.device)

    #     return input, label, domain, impath
    
    # @torch.no_grad()
    # def test(self, split=None):
    #     """A generic testing pipeline."""
    #     self.set_tensorboard()
    #     self.set_model_mode("eval")
    #     self.evaluator.reset()

    #     if split is None:
    #         split = self.cfg.TEST.SPLIT

    #     if split == "val" and self.val_loader is not None:
    #         data_loader = self.val_loader
    #     else:
    #         split = "test"  # in case val_loader is None
    #         data_loader = self.test_loader

    #     print(f"Evaluate on the *{split}* set")
    #     eval_dict = {}
    #     for batch_idx, batch in enumerate(tqdm(data_loader)):
    #         input, label, domain, impath = self.parse_batch_test(batch)
    #         output, img_feat, local_feat, txt_feat = self.model_inference(input)
    #         self.evaluator.process(output, label)

    #         # for prv_domain in prv_domain_list:
    #         prv_domain_index = [self.domain_list.index(prv_d) for prv_d in self.prv_domain_list if prv_d in self.domain_list]
    #         prv_domain_mask = torch.isin(domain, torch.tensor(prv_domain_index).to(self.device))
    #         # for del_domain in del_domain_list:
    #         del_domain_index = [self.domain_list.index(del_d) for del_d in self.del_domain_list if del_d in self.domain_list]
    #         del_domain_mask = torch.isin(domain, torch.tensor(del_domain_index).to(self.device))

    #         eval_dict = compute_acc_for_df_eval(
    #             eval_dict,
    #             output,
    #             label,
    #             prv_domain_mask,
    #             del_domain_mask,
    #             domain,
    #             self.domain_list,
    #             self.device
    #         ) 
    #         if batch_idx == 0:
    #             label_all = label
    #             domain_all = domain
    #             img_feat_all = img_feat
    #             # input_all = input

    #         else :
    #             label_all = torch.cat((label_all, label))
    #             domain_all = torch.cat((domain_all, domain))
    #             img_feat_all = torch.cat((img_feat_all, img_feat))
    #             # input_all = torch.cat((input_all, input))
        
            

    #     # for cls_id, clsname in enumerate(self.classnames):
    #     #     cls_specific_index = (label_all == cls_id)
    #     #     if torch.all(cls_specific_index == False):
    #     #         pass
    #     #     else:
    #     #         domain_metadata = []
    #     #         for d in domain_all[cls_specific_index]:
    #     #             domain_metadata.append(self.domain_list[int(d)])
    #     #         tag = f"{split}/tsne-plot/{clsname}"
    #     #         # self.write_embedding(img_feat[cls_specific_index], domain_metadata, input[cls_specific_index], global_step=batch_idx, tag=tag)
    #     #         self.write_embedding(img_feat_all[cls_specific_index], domain_metadata, tag=tag)

    #     results = self.evaluator.evaluate()
    #     if not self.cfg.NO_FORGET:
    #         print("==========peservation or delete acc===============")
    #         for name in ["prv", "del"]:
    #             acc = eval_dict[f"correct_{name}"] / eval_dict[f"total_{name}"]
    #             print(f"{name} : {acc:.2f}")
    #         print("==============domain specific acc=================")
    #         for domain_name in self.domain_list:
    #             acc = eval_dict[f"correct_{domain_name}"] / eval_dict[f"total_{domain_name}"]
    #             print(f"{domain_name} : {acc:.2f}")
    #         print("===================================================")

    #     # for k, v in results.items():
    #     #     tag = f"{split}/{k}"
    #     #     self.write_scalar(tag, v, self.epoch)

    #     return list(results.values())[0]
    
# def attnmap_save():
#     img_np = input.cpu().numpy()
#     # img_np = np.transpose(img_np, (0, 2, 3, 1))
#     RGB_img = np.transpose(img_np, (0, 2, 3, 1))*255
#     cv2_img = cv2.cvtColor (RGB_img.astype('uint8'), cv2.COLOR_RGB2BGR)
    
#     # root="/nas/data/gotoyuta/Dataset/pacs/images/"
#     features = local_feat @ txt_feat.t()
#     similarity_map = get_similarity_map(features[:,:, :], img_np.shape[2:])
#     save_parent_dir = "/nas/data/gotoyuta/Result_Domain_Forgetting/samples/pacs_df/vpt/DOMAIN1/photo"
#     for b in range(similarity_map.shape[0]):
#         filename=os.path.basename(impath[b])
#         filename_only, _ = os.path.splitext(filename)
#         tmp_save_dir = save_parent_dir + "/" + self.domain_list[domain[b].item()] + "/" + self.classnames[label[b].item()] + "/"
#         os.makedirs(tmp_save_dir, exist_ok=True)
#         predicted = torch.argmax(output[b]).item()
#         if predicted == label[b].item():
#             save_dir = tmp_save_dir + "/True/" + filename_only + "/"
#             os.makedirs(save_dir, exist_ok=True)
#         else :
#             save_dir = tmp_save_dir + "/False/" + filename_only + "/"
#             os.makedirs(save_dir, exist_ok=True)
#         img_cv2 = cv2.imread(impath[b])
#         img_cv2 = resize_and_center_crop(img_cv2, 224)
#         plt.imsave(save_dir + "org.png", cv2.cvtColor(img_cv2.astype('uint8'), cv2.COLOR_BGR2RGB))
#         # for n in range(similarity_map.shape[-1]):
#         n = label[b].item()
#         vis = (similarity_map[b, :, :, n].cpu().numpy() * 255).astype('uint8')
#         vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
#         vis = img_cv2 * 0.7 + vis * 0.3
#         vis = cv2.cvtColor(vis.astype('uint8'), cv2.COLOR_BGR2RGB)
#         # vis = vis.astype('uint8')
#         plt.imsave(save_dir +f"gt-{self.classnames[label[b].item()]}_predicted-{self.classnames[predicted]}.png" , vis)


import cv2
    
def resize_and_center_crop(image, target_size=224):
    # 元の画像の高さと幅を取得
    height, width = image.shape[:2]
    
    # 短い辺をターゲットサイズにリサイズ
    if width < height:
        new_width = target_size
        new_height = int(target_size * (height / width))
    else:
        new_height = target_size
        new_width = int(target_size * (width / height))
    
    # リサイズ処理
    resized_image = cv2.resize(image, (new_width, new_height))

    # 中央クロップのためのオフセットを計算
    top = (new_height - target_size) // 2
    left = (new_width - target_size) // 2
    
    # 中央クロップ処理
    cropped_image = resized_image[top:top + target_size, left:left + target_size]

    return cropped_image