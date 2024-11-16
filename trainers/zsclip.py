import torch
import torch.nn as nn

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.model import convert_weights

from .coop import load_clip_to_cpu
from .imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT
from tqdm import tqdm
from utils.eval_acc import compute_acc_for_df, compute_acc_for_df_eval

import cv2
from PIL import Image
import matplotlib.pyplot as plt

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
    "OfficeHomeDF": "a photo of a {}.",
    "DomainNetDF": "a photo of a {}",
    "PACSDF": "a photo of a {}",
}

from dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)

import os.path as osp
import time
from engine.trainer import TrainerDF
@TRAINER_REGISTRY.register()
class ZeroshotCLIP(TrainerDF):
    def __init__(self, cfg):
        super().__init__(cfg)

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model

    def model_inference(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits, image_features, self.text_features

    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]
        domain = batch["domain"]

        input = input.to(self.device)
        label = label.to(self.device)
        domain = domain.to(self.device)

        return input, label, domain
    
    def set_tensorboard(self):
        if not self.cfg.EVAL_ONLY:
            return
        else : # before_trainのコピペ
            directory = self.cfg.OUTPUT_DIR
            # if self.cfg.RESUME:
            #     directory = self.cfg.RESUME
            self.start_epoch = 0 # self.resume_model_if_exist(directory) #FIXME

            # Initialize summary writer
            writer_dir = osp.join(self.output_dir, "tensorboard")
            mkdir_if_missing(writer_dir)
            self.init_writer(writer_dir)

            # Remember the starting time (for computing the elapsed time)
            self.time_start = time.time()

    
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
            output, img_feat, txt_feat = self.model_inference(input)
            self.evaluator.process(output, label)

            # for prv_domain in prv_domain_list:
            prv_domain_index = [self.domain_list.index(prv_d) for prv_d in self.prv_domain_list if prv_d in self.domain_list]
            prv_domain_mask = torch.isin(domain, torch.tensor(prv_domain_index).to(self.device))
            # for del_domain in del_domain_list:
            del_domain_index = [self.domain_list.index(del_d) for del_d in self.del_domain_list if del_d in self.domain_list]
            del_domain_mask = torch.isin(domain, torch.tensor(del_domain_index).to(self.device))

            eval_dict = compute_acc_for_df_eval(
                eval_dict,
                output,
                label,
                prv_domain_mask,
                del_domain_mask,
                domain,
                self.domain_list,
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
            
            # img_np = input.cpu().numpy()
            # cv2_img = cv2.cvtColor (img_np, cv2.COLOR_RGB2BGR)
            

            # features = img_feat @ txt_feat.t()
            # similarity_map = clip.get_similarity_map(features[:,1:, :], cv2_img.shape[:2])

            # for b in range(similarity_map.shape[0]):
            #     for n in range(similarity_map.shape[-1]):
            #         vis = (similarity_map[b, :, :, n].cpu().numpy() * 255).astype('uint8')
            #         vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
            #         vis = cv2_img * 0.4 + vis * 0.6
            #         vis = cv2.cvtColor(vis.astype('uint8'), cv2.COLOR_BGR2RGB)
            #         print('CLIP:', self.classnames[n])
            #         plt.imsave(f"{self.classnames[n]}.png", vis)


        # meta_data_cls_agno = []
        # meta_data_cls_agno = []
        # tag = f"{split}/cls-agnostic"

        # print(label_all.shape, img_feat_all.shape)
        # for idx, (lbl, dlbl) in enumerate(zip(label_all, domain_all)):  
        #     # if idx == (label_all.size(0) -1 ) :
        #     #     meta_data_cls_agno.append(f"{lbl.item()}\t{dlbl.item()}")
        #     # else:
        #     meta_data_cls_agno.append([f"{lbl.item()}", f"{self.domain_list[dlbl.item()]}"])

        # self.write_embedding(img_feat_all, meta_data_cls_agno, tag=tag, metadata_header=["Object_Class", "Domain_Class"])

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
        if not self.cfg.NO_FORGET:
            print("==========peservation or delete acc===============")
            for name in ["prv", "del"]:
                acc = eval_dict[f"correct_{name}"] / eval_dict[f"total_{name}"]
                print(f"{name} : {acc:.2f}")
            print("==============domain specific acc=================")
            for domain_name in self.domain_list:
                acc = eval_dict[f"correct_{domain_name}"] / eval_dict[f"total_{domain_name}"]
                print(f"{domain_name} : {acc:.2f}")
            print("===================================================")

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]
    
    def write_embedding(self, mat, meta_data, label_img=None, global_step=None, tag=None, metadata_header=None):
        if self._writer is None:
            # Do nothing if writer is not initialized
            # Note that writer is only used when training is needed
            pass
        else:
            self._writer.add_embedding(mat, meta_data, label_img, global_step, tag, metadata_header=metadata_header)