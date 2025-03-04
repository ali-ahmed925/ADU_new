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

import time
import numpy as np
import os.path as osp
import datetime
from collections import OrderedDict
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from dassl.data import DataManager
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)
from dassl.modeling import build_head, build_backbone
from dassl.evaluation import build_evaluator

from dassl.engine import SimpleTrainer

from utils.loss_fn import Entropy, cossine_embedding_loss, get_entropy, get_entropy_local,orthogonality_loss
from dassl.metrics.accuracy import compute_accuracy
from utils.eval_acc import compute_acc_for_df, compute_acc_for_df_eval
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from utils.data_augmentation import get_jigsaw_tensor
import pandas as pd
from torchvision.transforms import v2
from clip import clip

from dassl.metrics.accuracy import compute_accuracy

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
    "DomainNetMiniDF": "a photo of a {}",
    "VLCSDF": "a photo of a {}",
    "Office31DF": "a photo of a {}",
    "ImageNetDF": "a photo of a {}",
}

from dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)

import os.path as osp
import time
from engine.trainer import TrainerDF
import torch
import numpy as np
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm  # 進捗表示用
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
@TRAINER_REGISTRY.register()
class ZeroshotCLIP_CLUSTER(TrainerDF):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.grid_num = 16
        self.resize = v2.Resize(size=224)
        self.vflip = v2.RandomVerticalFlip(p=1)
        self.blur = v2.GaussianBlur(kernel_size=(5,9), sigma=(10.,30.))

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
    
    def write_embedding(self, mat, meta_data, label_img=None, global_step=None, tag=None, metadata_header=None):
        if self._writer is None:
            # Do nothing if writer is not initialized
            # Note that writer is only used when training is needed
            pass
        else:
            self._writer.add_embedding(mat, meta_data, label_img, global_step, tag, metadata_header=metadata_header)
    
    def train_loop(self):
        # def train(self):
        data_loader = self.train_loader_x
        split = "train"
        """A generic testing pipeline."""
        self.set_tensorboard()
        self.set_model_mode("eval")
        self.evaluator.reset()

        # print(f"Evaluate on the *{split}* set")
        eval_dict = {}
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader)):
                input, label, domain = self.parse_batch_test(batch)
                # destructed_image = self.blur(input)
                # destructed_image = get_jigsaw_tensor(destructed_image, resize=(224,224), grid=self.grid_num)
                # destructed_image = self.resize(destructed_image)
                # input=destructed_image
                if self.use_domain_classifier_loss :
                    output, img_feat, txt_feat, domain_logit = self.model_inference(input)
                else :
                    output, img_feat, txt_feat = self.model_inference(input)
                self.evaluator.process(output, label)

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
        features = img_feat_all# .cpu().numpy()
        label_all = label_all# .cpu().numpy()
        domain_all = domain_all# .cpu().numpy()
        soft_label_eu = torch.randn(img_feat_all.shape[0], len(self.domain_list)).to(torch.half).cuda()
        soft_label_cos = torch.randn(img_feat_all.shape[0], len(self.domain_list)).to(torch.half).cuda()
        # for c_ind, c in enumerate(self.classnames):
        #     index_l = torch.nonzero(label_all == c_ind, as_tuple=True)[0]
        #     feat_masked = features[index_l]
        a = []
        # for dl in range(len(self.domain_list)):
        #     index_dl = torch.nonzero(domain_all[index_l] == dl, as_tuple=True)[0]
        #     selected_values = torch.index_select(feat_masked, dim=0, index=index_dl)
        #     a.append(selected_values.mean(dim = 0).unsqueeze(0))
        # num_clusters = 4  # クラスタ数（art, clipart, real, product）
        # centroid = torch.cat(a)
        # dists = torch.cdist(feat_masked, centroid)
        # s_labels = F.softmax(-dists, dim=1)
        # soft_label_eu[index_l] = s_labels
        
        # cos_dists = F.cosine_similarity(feat_masked.unsqueeze(1), centroid.unsqueeze(0), dim=2)
        # s_labels_cos = F.softmax(cos_dists, dim=1)
        # soft_label_cos[index_l] = s_labels_cos

        for dl in range(len(self.domain_list)):
            index_dl = torch.nonzero(domain_all == dl, as_tuple=True)[0]
            selected_values = torch.index_select(features, dim=0, index=index_dl)
            a.append(selected_values.mean(dim = 0).unsqueeze(0))
        num_clusters = 4  # クラスタ数（art, clipart, real, product）
        centroid = torch.cat(a)
        dists = torch.cdist(features, centroid)
        s_labels = F.softmax(-dists, dim=1)
        soft_label_eu = s_labels
        
        cos_dists = F.cosine_similarity(features.unsqueeze(1), centroid.unsqueeze(0), dim=2)
        s_labels_cos = F.softmax(cos_dists, dim=1)
        soft_label_cos = s_labels_cos
        # feat_masked = feat_masked.cpu().numpy()
        # ppppp = domain_all[index_l].cpu().numpy()
        features = features.cpu().numpy()
        ppppp = domain_all.cpu().numpy()
        centroid = centroid.cpu().numpy()

        # gmm = self.new_method(features, num_clusters)

        # 各サンプルがどのクラスタに属するか確率を取得
        # cluster_probs = gmm.predict_proba(features)  # (N_samples, num_clusters)
        # cluster_labels = gmm.predict(features)  # 各サンプルのクラスタID

        # import pandas as pd

        # df = pd.DataFrame(cluster_probs, columns=[f'Cluster_{i}' for i in range(num_clusters)])
        # df['Cluster_Label'] = cluster_labels
        # df["True_Label"] = domain_all.cpu().numpy()
        # df.to_csv("gmm_cluster_results_pp.csv", index=False)

           #  次元削減（2D）
        from matplotlib.colors import ListedColormap
        colors = ['red', 'blue', 'green', 'purple']
        tsne = TSNE(n_components=2, random_state=42)
        cent_and_feat = np.vstack((centroid, features))
        # features_2d = tsne.fit_transform(feat_masked)
        # cent = tsne.fit_transform(centroid)
        cent_and_feat_2d = tsne.fit_transform(cent_and_feat)        # クラスタごとに色を変えてプロット
        features_2d = cent_and_feat_2d[4:,:]
        cent = cent_and_feat_2d[:4,:]
        plt.figure(figsize=(8, 6))
        for i in range(num_clusters):
            plt.scatter(features_2d[ppppp == i, 0], features_2d[ppppp == i, 1], label=f'{self.domain_list[i]}', color=colors[i])
            plt.scatter(cent[i, 0], cent[i, 1], marker="^", color=colors[i])

        plt.legend()
        plt.title(f"centroid TSNE OfficeHome tot")
        plt.savefig(f"./centroid_tsne/office_home/tsne_cluster_centroid_OfficeHome.png")
        import pickle
        save_path = f"soft_label_officehome_tot_datasetseed{self.cfg.DATASET.SEED}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump({"cosine": soft_label_cos, "euclidean": soft_label_eu}, f)
        return 0

    def new_method(self, features, num_clusters):
        gmm = GaussianMixture(n_components=num_clusters, covariance_type='full', random_state=42)
        gmm.fit(features)
        return gmm

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
                if self.use_domain_classifier_loss :
                    output, img_feat, txt_feat, domain_output = self.model(image)
                else :
                    output, img_feat, txt_feat = self.model(image)

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
                    if self.use_orthogonal_loss:
                        domain_orthogonal_loss = orthogonality_loss(img_feat, target_label)
                        loss += domain_orthogonal_loss
                else :
                    loss = F.cross_entropy(output, label)
                self.model_backward_and_update(loss)
            
            # if not self.cfg.NO_FORGET:
            #     # loss_summary = {
            #     #     "loss": loss.item(),
            #     #     "loss_prv": loss_prv.item() if isinstance(loss_prv, torch.Tensor) else loss_prv,
            #     #     "loss_del": loss_del.item() if isinstance(loss_del, torch.Tensor) else loss_del,
            #     #     "loss_domain_cls": domain_cls_loss.item() if self.use_domain_classifier_loss else 0,
            #     #     "loss_domain_nn": domain_nn_loss.item() if self.use_nearest_neighbor_loss else 0,
            #     #     "loss_domain_ortho": domain_orthogonal_loss.item() if self.use_orthogonal_loss else 0,
            #     #     # "acc": compute_accuracy(output, label)[0].item(),
            #     # }
            #     acc = compute_acc_for_df(output, label, prv_domain_mask, del_domain_mask, domain, self.domain_list, device=self.device)
            #     loss_summary.update(acc)
            #     # loss_summary.update(acc)
            # else :
            #     loss_summary = {
            #         "loss": loss.item(),
            #         "acc": compute_accuracy(output, label)[0].item()
            #     }
                # acc = compute_accuracy(output, label)[0].item()
            # acc = compute_acc_for_df(output, label, prv_domain_mask, del_domain_mask, domain, self.domain_list, device=self.device)
            
            # if (self.batch_idx + 1) == self.num_batches:
            #     self.update_lr()
            loss_summary = {
                "loss": 0
            }

            return loss_summary