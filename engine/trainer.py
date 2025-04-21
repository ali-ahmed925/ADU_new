import time
import numpy as np
import os.path as osp
import datetime
from collections import OrderedDict
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# from dassl.data import DataManager
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)
from dassl.modeling import build_head, build_backbone
from dassl.evaluation import build_evaluator

from dassl.engine import SimpleTrainer, TrainerBase, SimpleNet

from utils.loss_fn import Entropy, cossine_embedding_loss, get_entropy, get_entropy_local,orthogonality_loss
from dassl.metrics.accuracy import compute_accuracy
from utils.eval_acc import compute_acc_for_df, compute_acc_for_df_eval
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from utils.data_augmentation import get_jigsaw_tensor
import pandas as pd
from torchvision.transforms import v2
from clip import clip
from .dataset_manager import DataManager
import cv2
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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
    "ImageNetDF": "a photo of a {}"
}

def median_pairwise_distance(x, y):
    """Heuristic: median of pairwise distances for sigma."""
    with torch.no_grad():
        pairwise_distances = torch.cdist(x, y, p=2)
        return pairwise_distances.median()

def median_sigma(x):
    with torch.no_grad():
        pairwise_dists = torch.cdist(x, x, p=2)
        return pairwise_dists.median()

def gaussian_kernel(x, y, sigma=1.0):
    """Compute the Gaussian (RBF) kernel between x and y."""
    x_norm = x.pow(2).sum(1).view(-1, 1)
    y_norm = y.pow(2).sum(1).view(1, -1)
    dist = x_norm + y_norm - 2 * torch.mm(x, y.t())
    return torch.exp(-dist / (2 * sigma ** 2))

def mmd(x, y, sigma=1.0):
    """Compute the Maximum Mean Discrepancy (MMD) between two samples."""
    K_xx = gaussian_kernel(x, x, sigma)
    K_yy = gaussian_kernel(y, y, sigma)
    K_xy = gaussian_kernel(x, y, sigma)
    m = x.size(0)
    n = y.size(0)

    mmd_val = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    return mmd_val

def total_pairwise_mmd(features, domain_labels, sigma=1.0):
    """Compute total pairwise MMD across all domains."""
    unique_domains = domain_labels.unique()
    mmd_sum = 0
    count = 0

    for i in range(len(unique_domains)):
        for j in range(i + 1, len(unique_domains)):
            d1 = unique_domains[i]
            d2 = unique_domains[j]
            z1 = features[domain_labels == d1]
            z2 = features[domain_labels == d2]

            if z1.size(0) > 1 and z2.size(0) > 1:
                sigma = median_pairwise_distance(z1, z2)
                mmd_val = mmd(z1, z2, sigma)
                mmd_sum += mmd_val
                count += 1

    return mmd_sum / count if count > 0 else torch.tensor(0.0, device=features.device)


def linear_kernel(x):
    return x @ x.t()

def rbf_kernel(x, sigma=1.0):
    x_norm = (x ** 2).sum(dim=1).view(-1, 1)
    dist = x_norm + x_norm.t() - 2 * x @ x.t()
    return torch.exp(-dist / (2 * sigma ** 2))

def center_gram(K):
    n = K.size(0)
    dtype = K.dtype  # Match dtype
    device = K.device
    H = torch.eye(n, device=device, dtype=dtype) - torch.ones(n, n, device=device, dtype=dtype) / n
    return H @ K @ H

def hsic(features, domain_labels, kernel_x='rbf', kernel_y='linear', sigma=1.0):
    """
    HSIC between feature tensor and integer domain labels.

    Args:
        features: (N, D) tensor of features
        domain_labels: (N,) tensor of integer domain labels
        kernel_x: 'rbf' or 'linear' for features
        kernel_y: 'rbf' or 'linear' for domain labels
        sigma: bandwidth for RBF kernel
    """
    # Convert integer labels to one-hot encoding
    num_classes = int(domain_labels.max().item()) + 1
    domain_one_hot = F.one_hot(domain_labels, num_classes=num_classes).float()

    # Compute kernels
    K = rbf_kernel(features, sigma) if kernel_x == 'rbf' else linear_kernel(features)
    L = rbf_kernel(domain_one_hot, sigma) if kernel_y == 'rbf' else linear_kernel(domain_one_hot)

    # Center and compute HSIC
    Kc = center_gram(K)
    Lc = center_gram(L)
    hsic_val = (Kc * Lc).sum() # / ((features.size(0) - 1) ** 2)
    return hsic_val



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
    design_details = {"trainer": 'CoOp',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model
class SimpleTrainer_(TrainerBase):
    """A simple trainer class implementing generic functions."""

    def __init__(self, cfg):
        super().__init__()
        self.check_cfg(cfg)

        if torch.cuda.is_available() and cfg.USE_CUDA:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Save as attributes some frequently used variables
        self.start_epoch = self.epoch = 0
        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.output_dir = cfg.OUTPUT_DIR

        self.cfg = cfg
        self.build_data_loader()
        self.build_model()
        self.evaluator = build_evaluator(cfg, lab2cname=self.lab2cname)
        self.best_result = -np.inf

    def check_cfg(self, cfg):
        """Check whether some variables are set correctly for
        the trainer (optional).

        For example, a trainer might require a particular sampler
        for training such as 'RandomDomainSampler', so it is good
        to do the checking:

        assert cfg.DATALOADER.SAMPLER_TRAIN == 'RandomDomainSampler'
        """
        pass

    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        dm = DataManager(self.cfg)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        self.dm = dm

    def build_model(self):
        """Build and register model.

        The default builds a classification model along with its
        optimizer and scheduler.

        Custom trainers can re-implement this method if necessary.
        """
        cfg = self.cfg

        print("Building model")
        self.model = SimpleNet(cfg, cfg.MODEL, self.num_classes)
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)
        print(f"# params: {count_num_param(self.model):,}")
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("model", self.model, self.optim, self.sched)

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Detected {device_count} GPUs (use nn.DataParallel)")
            self.model = nn.DataParallel(self.model)

    def train(self):
        super().train(self.start_epoch, self.max_epoch)

    def before_train(self):
        directory = self.cfg.OUTPUT_DIR
        if self.cfg.RESUME:
            directory = self.cfg.RESUME
        self.start_epoch = self.resume_model_if_exist(directory)

        # Initialize summary writer
        writer_dir = osp.join(self.output_dir, "tensorboard")
        mkdir_if_missing(writer_dir)
        self.init_writer(writer_dir)

        # Remember the starting time (for computing the elapsed time)
        self.time_start = time.time()

    def after_train(self):
        print("Finish training")

        do_test = not self.cfg.TEST.NO_TEST
        if do_test:
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                print("Deploy the model with the best val performance")
                self.load_model(self.output_dir)
            else:
                print("Deploy the last-epoch model")
            self.test()

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")

        # Close writer
        self.close_writer()

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            curr_result = self.test(split="val")
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    val_result=curr_result,
                    model_name="model-best.pth.tar"
                )

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
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

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    def model_inference(self, input):
        return self.model(input)

    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]

        input = input.to(self.device)
        label = label.to(self.device)

        return input, label

    def get_current_lr(self, names=None):
        names = self.get_model_names(names)
        name = names[0]
        return self._optims[name].param_groups[0]["lr"]


    
    

class TrainerDF(SimpleTrainer_):
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.DATASET.NAME == "OfficeHomeDF":
            self.domain_list = ["art", "clipart", "product", "real_world"]
            self.del_domain_list = cfg.DATASET.FORGETDOMAINS

        elif cfg.DATASET.NAME == "DomainNetDF":
            self.del_domain_list = cfg.DATASET.FORGETDOMAINS
            self.domain_list = [
                "clipart", "infograph", "painting", "quickdraw", "real", "sketch"
            ]
        elif cfg.DATASET.NAME == "DomainNetMiniDF":
            self.del_domain_list = cfg.DATASET.FORGETDOMAINS
            self.domain_list = [
                "clipart", "painting", "real", "sketch"
            ]
        elif cfg.DATASET.NAME == "VLCSDF":
            self.del_domain_list = cfg.DATASET.FORGETDOMAINS
            self.domain_list = ["caltech", "labelme", "pascal", "sun"]
            
        elif cfg.DATASET.NAME == "PACSDF":
            self.del_domain_list = cfg.DATASET.FORGETDOMAINS
            self.domain_list = ["art_painting", "cartoon", "photo", "sketch"]
        
        elif cfg.DATASET.NAME == "Office31DF":
            self.del_domain_list = cfg.DATASET.FORGETDOMAINS
            self.domain_list = ["amazon", "webcam", "dslr"]
        elif cfg.DATASET.NAME == "VisDA17DF":
            self.del_domain_list = cfg.DATASET.FORGETDOMAINS
            self.domain_list = ["synthetic", "real"]
        elif cfg.DATASET.NAME == "ImageNetDF":
            self.del_domain_list = cfg.DATASET.FORGETDOMAINS
            self.domain_list = ["real", "sketch"]
            # self.classnames = [
            #     "dog", "elephant", "giraffe", "guitar", "horse", "house", "person"
            # ]
        assert (set(self.domain_list) | set(self.del_domain_list)) == set(self.domain_list)
        self.prv_domain_list = list(set(self.domain_list) - set(self.del_domain_list))
        self.classnames = self.dm.dataset.classnames
        
        self.use_domain_classifier_loss = cfg.USE_DOMAIN_CLASIFIER_LOSS
        self.use_nearest_neighbor_loss = cfg.USER_NEAREST_NEIGHBOR_LOSS
        self.is_domain_divided = cfg.IS_DOMAIN_DIVIDED
        self.domain_class_divided = cfg.DOMAIN_CLASS_DIVIDED
        if self.use_nearest_neighbor_loss:
            self.nnl = SoftNearestNeighborsLoss()
        self.use_orthogonal_loss = cfg.USE_ORTHOGONAL_LOSS
        self.use_softlabel_dloss = cfg.USE_SOFT_DOMAIN_LABEL
        self.ddl_loss_weight = cfg.DDL_LOSS_WEIGHT
        self.soft_label_update_epoch = cfg.SOFT_LABEL_UPDATE_EPOCH

        # if self.use_orthogonal_loss:
        #     sel
        # self.kldiv_for_ddl_pena = cfg.USE_KLDIV_PENALTY
        # self.is_only_prv_for_kldiv = cfg.ONLY_KLDIV_FOR_PRV
        # from geomloss import SamplesLoss
        # self.emd_loss = SamplesLoss("sinkhorn", p=2)
        # if self.kldiv_for_ddl_pena is not None :
        #     self.model_expert = load_clip_to_cpu_expert(cfg)
        #     self.model_expert.cuda()
        #     self.model_expert.eval()

        self.csv_file_path = cfg.CSV_FILE_PATH
        if not osp.exists(self.csv_file_path):
            row_names = ["Prv Acc.", "Del Err.", "Del Acc.", "Specific Acc."]
            self.df = pd.DataFrame(index=row_names)
            self.df.to_csv(self.csv_file_path)
            
        else:
            print(f"File already exists: {self.csv_file_path}")
            self.df = pd.read_csv(self.csv_file_path, index_col=0)

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()


    
    # def train(self, start_epoch, max_epoch):
    #     """Generic training loops."""
    #     self.start_epoch = start_epoch
    #     self.max_epoch = max_epoch

    #     self.before_train()
    #     for self.epoch in range(self.start_epoch, self.max_epoch):
    #         self.before_epoch()
    #         self.run_epoch()
    #         self.after_epoch()
    #     self.after_train()
    
    def train_loop(self):
        super().train()
        return self.metrics_dict
    
    def forward_backward(self, batch):
        if self.use_softlabel_dloss:
            image, label, domain, domain_soft_label = self.parse_batch_train(batch)
        else:
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
                output, img_feat, txt_feat, domain_output, patch_output = self.model(image)
            else :
                output, img_feat, txt_feat, patch_output = self.model(image)
            
            # if self.kldiv_for_ddl_pena:
            #     with torch.no_grad():
            #         expert_feat = self.model_expert.encode_image(image)

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
                    #domain_cls_loss = F.cross_entropy(domain_output, target_label) * self.ddl_loss_weight
                    domain_cls_loss = hsic(domain_output, target_label) * self.ddl_loss_weight* - total_pairwise_mmd(domain_output, target_label) * self.cfg.MMD_WEIGHT
                    print(hsic(domain_output, target_label))
                    print(total_pairwise_mmd(domain_output, target_label))

                    loss += domain_cls_loss
                if self.use_nearest_neighbor_loss :
                    domain_nn_loss = self.nnl(img_feat, target_label)
                    loss += domain_nn_loss
                if self.use_orthogonal_loss:
                    domain_orthogonal_loss = orthogonality_loss(img_feat, target_label)
                    loss += domain_orthogonal_loss 

                # if self.kldiv_for_ddl_pena is not None:
                #     if self.is_only_prv_for_kldiv:
                #         img_feat_ = img_feat * img_feat.norm(dim=-1, keepdim=True)
                #         if self.kldiv_for_ddl_pena == "kldiv":
                #             domain_kl_div_loss = F.kl_div(F.log_softmax(img_feat_[prv_domain_mask], dim=1),F.softmax(expert_feat[prv_domain_mask], dim=1), reduction="batchmean")
                #         elif self.kldiv_for_ddl_pena == "wasserstein":
                #             domain_wasserstein_loss = self.emd_loss(img_feat_[prv_domain_mask].to(torch.float), expert_feat[prv_domain_mask].to(torch.float))
                #             domain_kl_div_loss = domain_wasserstein_loss.to(torch.half)
                #         else :
                #             AssertionError

                #         # 
                #         # print(expert_feat[prv_domain_mask].shape, img_feat[prv_domain_mask].shape, expert_feat.norm(dim=1, keepdim=True)[prv_domain_mask].shape)
                        
                #         # domain_kl_div_loss = F.mse_loss(img_feat[prv_domain_mask],(expert_feat / expert_feat.norm(dim=-1, keepdim=True))[prv_domain_mask], reduction="mean")
                        
                #     else :  
                #         img_feat_ = img_feat * img_feat.norm(dim=-1, keepdim=True)
                #         if self.kldiv_for_ddl_pena == "kldiv":
                #             domain_kl_div_loss = F.kl_div(F.log_softmax(img_feat_, dim=1),F.softmax(expert_feat, dim=1), reduction="batchmean")
                #         elif self.kldiv_for_ddl_pena == "wasserstein":
                #             domain_wasserstein_loss = self.emd_loss(img_feat_.to(torch.float), expert_feat.to(torch.float))
                #             domain_kl_div_loss = domain_wasserstein_loss.to(torch.half)
                #         # domain_kl_div_loss = F.kl_div(F.log_softmax(img_feat, dim=1),F.log_softmax(expert_feat.norm(dim=-1, keepdim=True), dim=1), reduction="batchmean")
                #     loss += domain_kl_div_loss
            else :
                loss = F.cross_entropy(output, label)

            self.model_backward_and_update(loss)

            if (self.epoch + 1) % self.soft_label_update_epoch ==  0:
                if self.use_softlabel_dloss:
                    novel_soft_label = F.softmax(domain_output, dim=1).detach().cpu()
                    self.train_loader_x.dataset.update_softlabel(batch["index"], novel_soft_label)
        
        
        if not self.cfg.NO_FORGET:
            loss_summary = {
                "loss": loss.item(),
                "loss_prv": loss_prv.item() if isinstance(loss_prv, torch.Tensor) else loss_prv,
                "loss_del": loss_del.item() if isinstance(loss_del, torch.Tensor) else loss_del,
                "loss_domain_cls": domain_cls_loss.item() if self.use_domain_classifier_loss else 0,
                "loss_domain_nn": domain_nn_loss.item() if self.use_nearest_neighbor_loss else 0,
                "loss_domain_ortho": domain_orthogonal_loss.item() if self.use_orthogonal_loss else 0,
                # "loss_domain_kl_div_loss": domain_kl_div_loss.item() if self.kldiv_for_ddl_pena is not None else 0
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

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        domain = batch["domain"]
        if self.use_softlabel_dloss:
            soft_domain_label = batch["soft_domain_label"]
        
        input = input.to(dtype=input.dtype, device=self.device)
        label = label.to(dtype=label.dtype, device=self.device)
        domain = domain.to(dtype=domain.dtype, device=self.device)
        if self.use_softlabel_dloss:
            soft_domain_label = batch["soft_domain_label"]
            soft_domain_label = soft_domain_label.to(dtype=soft_domain_label.dtype, device=self.device)
            return input, label, domain, soft_domain_label
        return input, label, domain

    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]
        domain = batch["domain"]

        input = input.to(dtype=input.dtype, device=self.device)
        label = label.to(dtype=label.dtype, device=self.device)
        domain = domain.to(dtype=domain.dtype, device=self.device)

        return input, label, domain
    
    def set_tensorboard(self):
        if not self.cfg.EVAL_ONLY:
            return
        else : # before_trainのコピペ
            directory = self.cfg.OUTPUT_DIR
            if self.cfg.RESUME:
                directory = self.cfg.RESUME
            self.start_epoch = self.resume_model_if_exist(directory)

            # Initialize summary writer
            writer_dir = osp.join(self.output_dir, "tensorboard")
            mkdir_if_missing(writer_dir)
            self.init_writer(writer_dir)

            # Remember the starting time (for computing the elapsed time)
            self.time_start = time.time()

    @torch.no_grad()
    def get_attention_map(self, split=None):
        self.set_model_mode("eval")
        self.evaluator.reset()
        data_loader = self.train_loader_x
        # data_loader = self.test_loader
        # if split is None:
        #     split = self.cfg.TEST.SPLIT

        # if split == "val" and self.val_loader is not None:
        #     data_loader = self.val_loader
        # else:
        #     split = "test"  # in case val_loader is None
        #     data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")
        eval_dict = {}
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label, domain = self.parse_batch_test(batch)
            if self.use_domain_classifier_loss :
                output, img_feat, txt_feat, domain_logit, patch_output = self.model_inference(input)
            else :
                output, img_feat, txt_feat, patch_output = self.model_inference(input)
            # self.evaluator.process(output, label)

            # for prv_domain in prv_domain_list:
            prv_domain_index = [self.domain_list.index(prv_d) for prv_d in self.prv_domain_list if prv_d in self.domain_list]
            prv_domain_mask = torch.isin(domain, torch.tensor(prv_domain_index).to(self.device))
            # for del_domain in del_domain_list:
            del_domain_index = [self.domain_list.index(del_d) for del_d in self.del_domain_list if del_d in self.domain_list]
            del_domain_mask = torch.isin(domain, torch.tensor(del_domain_index).to(self.device))
            save_base_dir = self.output_dir + "/train_samples/"
            mkdir_if_missing(save_base_dir)
            image_number = batch["index"]
            for idx, (impath, attention_map)in enumerate(zip(batch["impath"], patch_output)):
                pass
                precision = torch.argmax(output[idx])
                original_image = cv2.imread(impath)  # 画像を読み込む
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # OpenCV は BGR なので RGB に変換
                original_image = cv2.resize(original_image, (224, 224))  # 224x224 にリサイズ
                original_image = original_image / 255.0  # 正規化 (0~1)
                attention_map = attention_map[:,label[idx]].view(14,14).cpu().numpy()
                attention_map_resized = cv2.resize(attention_map.astype(np.float32), (224, 224))

                heatmap = (attention_map_resized - attention_map_resized.min()) / (attention_map_resized.max() - attention_map_resized.min())
                heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
                superimposed_img = cv2.addWeighted(np.uint8(255 * original_image), 0.6, heatmap, 0.4, 0)
                if self.domain_list[domain[idx]] in self.del_domain_list:
                    if precision == label[idx]:
                        save_dir = save_base_dir + "del_domains/" + "TP/"
                    else :
                        save_dir = save_base_dir + "del_domains/" + "FP/"
                else :
                    if precision == label[idx]:
                        save_dir = save_base_dir + "prv_domains/" + "TP/"
                    else :
                        save_dir = save_base_dir + "prv_domains/" + "FP/"

                mkdir_if_missing(save_dir)
                original_image = (original_image * 255).astype(np.uint8)

                cv2.imwrite(save_dir + f"{image_number[idx]}_{self.classnames[label[idx]]}_{self.domain_list[domain[idx]]}_original.png", cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
                cv2.imwrite(save_dir + f"{image_number[idx]}_{self.classnames[label[idx]]}_{self.domain_list[domain[idx]]}_attention_gt{label[idx]}_pr{precision}.png", cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
    
    @torch.no_grad()
    def get_tsne_plots(self, split=None):
        self.set_model_mode("eval")
        self.evaluator.reset()
        data_loader = self.train_loader_x
        # data_loader = self.test_loader
        # if split is None:
        #     split = self.cfg.TEST.SPLIT

        # if split == "val" and self.val_loader is not None:
        #     data_loader = self.val_loader
        # else:
        #     split = "test"  # in case val_loader is None
        #     data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")
        eval_dict = {}
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label, domain = self.parse_batch_test(batch)
            if self.use_domain_classifier_loss :
                output, img_feat, txt_feat, domain_logit, patch_output = self.model_inference(input)
            else :
                output, img_feat, txt_feat, patch_output = self.model_inference(input)
            # self.evaluator.process(output, label)
            if batch_idx == 0:
                label_all = label
                domain_all = domain
                img_feat_all = img_feat
                # input_all = input

            else :
                label_all = torch.cat((label_all, label))
                domain_all = torch.cat((domain_all, domain))
                img_feat_all = torch.cat((img_feat_all, img_feat))

            # for prv_domain in prv_domain_list:
            # prv_domain_index = [self.domain_list.index(prv_d) for prv_d in self.prv_domain_list if prv_d in self.domain_list]
            # prv_domain_mask = torch.isin(domain, torch.tensor(prv_domain_index).to(self.device))
            # # for del_domain in del_domain_list:
            # del_domain_index = [self.domain_list.index(del_d) for del_d in self.del_domain_list if del_d in self.domain_list]
            # del_domain_mask = torch.isin(domain, torch.tensor(del_domain_index).to(self.device))
        clslist = ["aircraft_carrier", "tiger", "whale"]
        marker_list = ['o', 's', 'D', '^', 'v', 'p', '*', 'x', '+', '<', '>']
        target_cls_index = [i for i, value in enumerate(self.classnames) if value in clslist]
        save_base_dir = self.output_dir + "/test_samples_tsne/"
        # filtered_array = arrayA[np.isin(arrayA, listB)]
        img_feat_all_numpy = img_feat_all.cpu().numpy()
        domain_all_numpy = domain_all.cpu().numpy()
        label_all_numpy = label_all.cpu().numpy()
        # filterd_mask = np.isin(label_all_numpy, target_cls_index)
        # img_feat_all = img_feat_all[filterd_mask]
        # domain_all_numpy = domain_all_numpy[filterd_mask]
        # label_all_numpy = label_all_numpy[filterd_mask]
        mkdir_if_missing(save_base_dir)
        perplexity = min(30, domain_all_numpy.shape[0] - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity,random_state=42)
        # print(f"{self.classnames[i]}",img_feat_all_numpy.shape)
        tsne_results = tsne.fit_transform(img_feat_all_numpy)
        unique_domains = np.unique(domain_all_numpy)
        unique_labels = np.unique(label_all_numpy)

        cmap = plt.cm.get_cmap('tab10', len(unique_domains))
        
        plt.figure(figsize=(6, 4))
        # for i_, domain in enumerate(unique_domains):
        #     for j_, label in enumerate(unique_labels):
        #         idx = np.where((domain_all_numpy == domain) & (label_all_numpy == label))
                
        #         if len(idx[0]) == 0:
        #             continue  # データがない場合はスキップ
                
        #         # 色はドメイン、マーカーはラベルに基づく
        #         color = cmap(i_)
        #         marker = marker_list[j_ % len(marker_list)]  # マーカーリストをループ
                
        #         plt.scatter(tsne_results[idx, 0], tsne_results[idx, 1],
        #             color=color, marker=marker, label=f"{self.domain_list[i_]}-{clslist[j_]}",
        #             s=50, alpha=0.7)

        for i_, domain in enumerate(unique_domains):
            idx = np.where(np.array(domain_all_numpy) == domain)
            plt.scatter(tsne_results[idx, 0], tsne_results[idx, 1],
                        color=cmap(i_), label=self.domain_list[i_], s=10, alpha=0.7)
        plt.axis('off')
        plt.legend()
        # plt.title("t-SNE of Image Features by Domain")
        # plt.xlabel("t-SNE 1")
        # plt.ylabel("t-SNE 2")
        # now_class = self.classnames[i]
        plt.tight_layout()
        plt.savefig(save_base_dir + "atotalclass_tsne_s10.png")
        # for i in range(len(self.classnames)):
        #     img_feat_all_numpy_ = img_feat_all_numpy[label_all_numpy == i]
        #     domain_all_numpy_ = domain_all_numpy[label_all_numpy == i]
        #     perplexity = min(30, domain_all_numpy_.shape[0] - 1)
        #     tsne = TSNE(n_components=2, perplexity=perplexity,random_state=42)
        #     print(f"{self.classnames[i]}",img_feat_all_numpy_.shape)
        #     tsne_results = tsne.fit_transform(img_feat_all_numpy_)

        #     unique_domains = np.unique(domain_all_numpy_)
        #     cmap = plt.cm.get_cmap('tab10', len(unique_domains))
            
        #     plt.figure(figsize=(6, 4))
        #     for i_, domain in enumerate(unique_domains):
        #         idx = np.where(np.array(domain_all_numpy_) == domain)
        #         plt.scatter(tsne_results[idx, 0], tsne_results[idx, 1],
        #                     color=cmap(i_), label=self.domain_list[i_], s=50, alpha=0.7)
        #     plt.axis('off')
        #     plt.legend()
        #     # plt.title("t-SNE of Image Features by Domain")
        #     # plt.xlabel("t-SNE 1")
        #     # plt.ylabel("t-SNE 2")
        #     now_class = self.classnames[i]
        #     plt.tight_layout()
        #     plt.savefig(save_base_dir + f"{now_class}_tsne.png")



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
            if self.use_domain_classifier_loss :
                output, img_feat, txt_feat, domain_logit, output_patch = self.model_inference(input)
            else :
                output, img_feat, txt_feat, output_patch = self.model_inference(input)
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
            metrics_A = eval_dict[f"correct_prv"] / eval_dict[f"total_prv"]
            metrics_F = 1 - eval_dict[f"correct_del"] / eval_dict[f"total_del"]
            metrics_H = 2 * metrics_A * metrics_F / (metrics_A + metrics_F)

            self.metrics_dict = {
                "A" : 100 * metrics_A,
                "F" : 100 * metrics_F,
                "H" : 100 * metrics_H 
            }


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

from utils.loss_fn import entropy_local_topk
class TrainerDF_Local(TrainerDF):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.topk = cfg.TOPK
        self.entropy_mask = cfg.ENTROPY_MASK
        self.masked_dc = cfg.MASKED_DC
        self.masked_nn = cfg.MASKED_NN

        self.num_vision_context = cfg.TRAINER.IVLP.N_CTX_VISION
        self.block_shuffle_selection = cfg.BLOCK_SHUFFLE_SELECTION
        self.grid_num = cfg.GRID
        self.resize = v2.Resize(size=224)
        self.vflip = v2.RandomVerticalFlip(p=1)
        self.blur = v2.GaussianBlur(kernel_size=(5,9), sigma=(10.,30.))

        if self.block_shuffle_selection :
            self.model_expert = load_clip_to_cpu_expert(cfg)
            self.model_expert.cuda()
            self.model_expert.eval()
    
        self.block_shuffle_selection_non_expert=cfg.BLOCK_SHUFFLE_SELECTION_NONEXP
        if self.block_shuffle_selection_non_expert:
            
            pass

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()
    
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
            if self.use_domain_classifier_loss:
                output, local_output, img_feat, txt_feat, local_img_feat, domain_logits = self.model(image)
            else :
                output, local_output, img_feat, txt_feat, local_img_feat = self.model(image)
            if not self.cfg.NO_FORGET:
                entropy = Entropy()
                false_check_tensor = torch.zeros_like(domain, dtype=torch.bool)
                
                # for prv_domain in prv_domain_list:
                prv_domain_index = [self.domain_list.index(prv_d) for prv_d in self.prv_domain_list if prv_d in self.domain_list]
                prv_domain_mask = torch.isin(domain, torch.tensor(prv_domain_index).to(self.device))
                # for del_domain in del_domain_list:
                del_domain_index = [self.domain_list.index(del_d) for del_d in self.del_domain_list if del_d in self.domain_list]
                del_domain_mask = torch.isin(domain, torch.tensor(del_domain_index).to(self.device))
                
                if torch.equal(false_check_tensor, prv_domain_mask):
                    loss_prv = 0
                else :
                    # num_of_local_feature = local_feat.shape[1]
                    # batch_size_prv = local_feat[prv_domain_mask].shape[0]
                    # output_local = local_feat[del_domain_mask].view(batch_size_prv*num_of_local_feature, -1)
                    loss_prv = F.cross_entropy(output[prv_domain_mask], label[prv_domain_mask])
                    # loss_prv_local = entropy_local_topk(output_local, label[prv_domain_mask],num_of_local_feature)
                if torch.equal(false_check_tensor, del_domain_mask):
                    loss_del = 0
                else :
                    # num_of_local_feature = local_feat.shape[1]
                    # batch_size_del = local_feat[del_domain_mask].shape[0]
                    # output_local = local_feat[del_domain_mask].view(batch_size_del*num_of_local_feature, -1)
                    loss_del = entropy(output[del_domain_mask])
                    # loss_del_local = entropy_local_topk(output_local, label[del_domain_mask],num_of_local_feature)
                loss = loss_prv - loss_del

                if self.is_domain_divided:
                    target_label = domain
                else :
                    target_label = prv_domain_mask.int().long()

                ######################################################################
                # domain loss (domain classifier loss, nearest neighbor loss or both)
                #####################################################################
                if self.entropy_mask:
                    # num_of_local_feature = local_output.shape[1]
                    # batch_size = local_output.shape[0]
                    # entropy_masks = get_masksk(local_output, label, num_of_local_feature)
                    
                    local_entropy = get_entropy_local(local_output[:,self.num_vision_context:]) # exclude context vector
                    topk_index = torch.topk(local_entropy, k=self.topk, dim=1)[1]
                    # print(local_output.shape)
                    ###########################################
                    ## mask preparation
                    ###########################################
                    mask = torch.zeros_like(local_entropy)
                    mask.scatter_(1, topk_index, 1)
                    patch_size = int(local_entropy.size(1)**0.5)
                    
                    px_per_patch = int(image.size(2) / patch_size)
                
                    mask = mask.repeat_interleave(px_per_patch, dim=1)
                
                    mask = mask.view(-1, patch_size, image.size(2)).repeat_interleave(px_per_patch, dim=1)
                    
                    ############################################
                    ## masked image generation
                    ############################################
                    masked_image = image * mask.unsqueeze(1)
                    
                    if self.use_domain_classifier_loss:
                        _, _, img_feat_masked, _, _, domain_logits_masked = self.model(masked_image)
                    else :
                        _, _, img_feat_masked, _, _ = self.model(masked_image)
                elif self.block_shuffle_selection :
                    destructed_image = image
                    destructed_image = self.blur(image)
                    destructed_image = get_jigsaw_tensor(destructed_image, resize=(224,224), grid=self.grid_num)
                    destructed_image = self.resize(destructed_image)
                    domain_specific_features = self.model_expert.encode_image(destructed_image)
                    domain_specific_features = domain_specific_features.unsqueeze(1)
                    domain_focus_simmap = torch.matmul(local_img_feat[:, self.num_vision_context:,:], domain_specific_features.transpose(-1, -2)).squeeze(-1)
                    # domain_focus_simmap = local_img_feat[:,self.num_vision_context:] @ domain_specific_features

                    topk_index = torch.topk(domain_focus_simmap, k=self.topk, dim=1)[1]
                    # print(local_output.shape)
                    ###########################################
                    ## mask preparation
                    ###########################################
                    mask = torch.zeros_like(domain_focus_simmap)
                    mask.scatter_(1, topk_index, 1)
                    patch_size = int(domain_focus_simmap.size(1)**0.5)
                    
                    px_per_patch = int(image.size(2) / patch_size)
                
                    mask = mask.repeat_interleave(px_per_patch, dim=1)
                
                    mask = mask.view(-1, patch_size, image.size(2)).repeat_interleave(px_per_patch, dim=1)
                    ############################################
                    ## masked image generation
                    ############################################
                    masked_image = image * mask.unsqueeze(1)
                    
                    if self.use_domain_classifier_loss:
                        _, _, img_feat_masked, _, _, domain_logits_masked = self.model(masked_image)
                    else :
                        _, _, img_feat_masked, _, _ = self.model(masked_image)
                
                elif self.block_shuffle_selection_non_expert:
                    destructed_image = image
                    destructed_image = self.blur(image)
                    destructed_image = get_jigsaw_tensor(destructed_image, resize=(224,224), grid=self.grid_num)
                    destructed_image = self.resize(destructed_image)
                    domain_specific_features, _ = self.model.image_encoder(destructed_image.half())
                    domain_specific_features = domain_specific_features.unsqueeze(1)
                    domain_focus_simmap = torch.matmul(local_img_feat[:, self.num_vision_context:,:], domain_specific_features.transpose(-1, -2)).squeeze(-1)
                    # domain_focus_simmap = local_img_feat[:,self.num_vision_context:] @ domain_specific_features

                    topk_index = torch.topk(domain_focus_simmap, k=self.topk, dim=1)[1]
                    # print(local_output.shape)
                    ###########################################
                    ## mask preparation
                    ###########################################
                    mask = torch.zeros_like(domain_focus_simmap)
                    mask.scatter_(1, topk_index, 1)
                    patch_size = int(domain_focus_simmap.size(1)**0.5)
                    
                    px_per_patch = int(image.size(2) / patch_size)
                
                    mask = mask.repeat_interleave(px_per_patch, dim=1)
                
                    mask = mask.view(-1, patch_size, image.size(2)).repeat_interleave(px_per_patch, dim=1)
                    ############################################
                    ## masked image generation
                    ############################################
                    masked_image = image * mask.unsqueeze(1)
                    
                    if self.use_domain_classifier_loss:
                        _, _, img_feat_masked, _, _, domain_logits_masked = self.model(masked_image)
                    else :
                        _, _, img_feat_masked, _, _ = self.model(masked_image)

                if self.use_domain_classifier_loss :
                    if self.masked_dc:
                        domain_cls_loss = F.cross_entropy(domain_logits_masked, target_label)
                    else:
                        domain_cls_loss = F.cross_entropy(domain_logits, target_label)
                    loss += domain_cls_loss
                if self.use_nearest_neighbor_loss :
                    if self.masked_nn:
                        domain_nn_loss = self.nnl(img_feat_masked, target_label)
                    else :
                        domain_nn_loss = self.nnl(img_feat, target_label)
                    loss += domain_nn_loss

            else :
                # print(type(output))
                # print(type(label))

                # print(output.shape)
                # print(label.shape)
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
    
    def set_tensorboard(self):
        if not self.cfg.EVAL_ONLY:
            return
        else : # before_trainのコピペ
            directory = self.cfg.OUTPUT_DIR
            if self.cfg.RESUME:
                directory = self.cfg.RESUME
            self.start_epoch = self.resume_model_if_exist(directory)

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
            # output, img_feat, local_feat, txt_feat = self.model_inference(input)
            if self.use_domain_classifier_loss:
                output, local_output, img_feat, txt_feat, local_img_feat, domain_logits = self.model_inference(input)
            else :
                output, local_output, img_feat, txt_feat, local_img_feat = self.model_inference(input)
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

        if not self.cfg.NO_FORGET:
            print("==========peservation or delete acc===============")
            for name in ["prv", "del"]:
                acc = eval_dict[f"correct_{name}"] / eval_dict[f"total_{name}"]
                print(f"{name} : {acc:.5f}")
            print("==============domain specific acc=================")
            for domain_name in self.domain_list:
                acc = eval_dict[f"correct_{domain_name}"] / eval_dict[f"total_{domain_name}"]
                print(f"{domain_name} : {acc:.5f}")
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

class TrainerDF_Local_SelectPatch(TrainerDF_Local):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.only_masked = cfg.TRAINER.IVLP_VLADAPTER_LOCAL_SELECTPATCH.ONLY_MASKED
        self.select_method = cfg.TRAINER.IVLP_VLADAPTER_LOCAL_SELECTPATCH.SELECT_METHOD
        self.model_expert = load_clip_to_cpu_expert(cfg)
        self.model_expert.cuda()
        self.model_expert.eval()

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
            if self.select_method == "entropy":
                if self.use_domain_classifier_loss:
                    output, output_masked, local_output, img_feat, img_feat_masked, txt_feat, local_img_feat, domain_logits, domain_logits_masked = self.model(image)
                else :
                    output, output_masked, local_output, img_feat, img_feat_masked, txt_feat, local_img_feat = self.model(image)
            elif self.select_method == "entropy_distill":
                # text_features = self.model_expert.encode_text()
                if self.use_domain_classifier_loss:
                    output, output_masked, local_output, img_feat, img_feat_masked, txt_feat, local_img_feat, domain_logits, domain_logits_masked = self.model(image, selection_feature=self.text_features_expert)
                else :
                    output, output_masked, local_output, img_feat, img_feat_masked, txt_feat, local_img_feat = self.model(image, selection_feature=self.text_features_expert)
            elif self.select_method == "block_shuffle_distill":
                destructed_image = image
                destructed_image = self.blur(image)
                destructed_image = get_jigsaw_tensor(destructed_image, resize=(224,224), grid=self.grid_num)
                destructed_image = self.resize(destructed_image)
                with torch.no_grad():
                    domain_specific_features = self.model_expert.encode_image(destructed_image)
                if self.use_domain_classifier_loss:
                    output, output_masked, local_output, img_feat, img_feat_masked, txt_feat, local_img_feat, domain_logits, domain_logits_masked = self.model(image, selection_feature=domain_specific_features)
                else :
                    output, output_masked, local_output, img_feat, img_feat_masked, txt_feat, local_img_feat = self.model(image, selection_feature=domain_specific_features)
            elif self.select_method == "block_shuffle":
                destructed_image = image
                destructed_image = self.blur(image)
                destructed_image = get_jigsaw_tensor(destructed_image, resize=(224,224), grid=self.grid_num)
                destructed_image = self.resize(destructed_image)
                domain_specific_features = self.model.encode_image()
                if self.use_domain_classifier_loss:
                    output, output_masked, local_output, img_feat, img_feat_masked, txt_feat, local_img_feat, domain_logits, domain_logits_masked = self.model(image, block_shuffled_img=destructed_image)
                else :
                    output, output_masked, local_output, img_feat, img_feat_masked, txt_feat, local_img_feat = self.model(image, selection_feature=domain_specific_features)

            
            if not self.cfg.NO_FORGET:
                entropy = Entropy()
                false_check_tensor = torch.zeros_like(domain, dtype=torch.bool)
                
                # for prv_domain in prv_domain_list:
                prv_domain_index = [self.domain_list.index(prv_d) for prv_d in self.prv_domain_list if prv_d in self.domain_list]
                prv_domain_mask = torch.isin(domain, torch.tensor(prv_domain_index).to(self.device))
                # for del_domain in del_domain_list:
                del_domain_index = [self.domain_list.index(del_d) for del_d in self.del_domain_list if del_d in self.domain_list]
                del_domain_mask = torch.isin(domain, torch.tensor(del_domain_index).to(self.device))
                if self.only_masked :
                    if torch.equal(false_check_tensor, prv_domain_mask):
                        loss_prv = 0
                    else :
                        loss_prv = F.cross_entropy(output_masked[prv_domain_mask], label[prv_domain_mask])
                    if torch.equal(false_check_tensor, del_domain_mask):
                        loss_del = 0
                    else :
                        loss_del = entropy(output_masked[del_domain_mask])
                    loss = loss_prv - loss_del
                else:
                    if torch.equal(false_check_tensor, prv_domain_mask):
                        loss_prv = 0
                    else :
                        loss_prv = F.cross_entropy(output[prv_domain_mask], label[prv_domain_mask])
                    if torch.equal(false_check_tensor, del_domain_mask):
                        loss_del = 0
                    else :
                        loss_del = entropy(output[del_domain_mask])
                    loss = loss_prv - loss_del

                if self.is_domain_divided:
                    target_label = domain
                else :
                    target_label = prv_domain_mask.int().long()

                ######################################################################
                # domain loss (domain classifier loss, nearest neighbor loss or both)
                #####################################################################
                # if self.entropy_mask:
                    
                # elif self.block_shuffle_selection :
                #     pass
                # elif self.block_shuffle_selection_non_expert:
                #     pass

                if self.use_domain_classifier_loss :
                    if self.masked_dc:
                        domain_cls_loss = F.cross_entropy(domain_logits_masked, target_label)
                    else:
                        domain_cls_loss = F.cross_entropy(domain_logits, target_label)
                    loss += domain_cls_loss
                if self.use_nearest_neighbor_loss :
                    if self.masked_nn:
                        domain_nn_loss = self.nnl(img_feat_masked, target_label)
                    else :
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
            # output, img_feat, local_feat, txt_feat = self.model_inference(input)
            if self.use_domain_classifier_loss:
                output, output_masked, local_output, img_feat, img_feat_masked, txt_feat, local_img_feat, domain_logits, domain_logits_masked = self.model_inference(input)
            else :
                output, output_masked, local_output, img_feat, img_feat_masked, txt_feat, local_img_feat = self.model_inference(input)
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

        if not self.cfg.NO_FORGET:
            print("==========peservation or delete acc===============")
            for name in ["prv", "del"]:
                acc = eval_dict[f"correct_{name}"] / eval_dict[f"total_{name}"]
                print(f"{name} : {acc:.5f}")
            print("==============domain specific acc=================")
            for domain_name in self.domain_list:
                acc = eval_dict[f"correct_{domain_name}"] / eval_dict[f"total_{domain_name}"]
                print(f"{domain_name} : {acc:.5f}")
            print("===================================================")

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]


from utils.loss_fn import entropy_local_topk_distilled
class TrainerDF_Local_Distill(TrainerDF_Local):
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.DATASET.NAME == "OfficeHomeDF":
            self.domain_list = ["art", "clipart", "product", "real_world"]
            self.del_domain_list = cfg.DATASET.FORGETDOMAINS

        elif cfg.DATASET.NAME == "DomainNetDF":
            self.del_domain_list = cfg.DATASET.FORGETDOMAINS
            self.domain_list = [
                "clipart", "infograph", "painting", "quickdraw", "real", "sketch"
            ]
    
        elif cfg.DATASET.NAME == "PACSDF":
            self.del_domain_list = cfg.DATASET.FORGETDOMAINS
            self.domain_list = ["art_painting", "cartoon", "photo", "sketch"]
            # self.classnames = [
            #     "dog", "elephant", "giraffe", "guitar", "horse", "house", "person"
            # ]
        assert (set(self.domain_list) | set(self.del_domain_list)) == set(self.domain_list)
        self.prv_domain_list = list(set(self.domain_list) - set(self.del_domain_list))
        self.classnames = self.dm.dataset.classnames
    
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
            output, img_feat, local_feat, txt_feat = self.model(image)
            output_expert, local_feat_expert = self.model_expert.encode_image(image)
            if not self.cfg.NO_FORGET:
                entropy = Entropy()
                false_check_tensor = torch.zeros_like(domain, dtype=torch.bool)
                
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
                    loss_del_local = 0
                else :
                    num_of_local_feature = local_feat.shape[1]
                    batch_size_del = local_feat[del_domain_mask].shape[0]
                    output_local = local_feat[del_domain_mask].view(batch_size_del*num_of_local_feature, -1)
                    output_local_expert = local_feat_expert[del_domain_mask].view(batch_size_del*num_of_local_feature, -1)
                    loss_del = entropy(output[del_domain_mask])
                    loss_del_local = entropy_local_topk_distilled(output_local, output_local_expert, label[del_domain_mask], num_of_local_feature, top_k=self.cfg.TOPK)
                loss = loss_prv - loss_del - loss_del_local
            else :
                # print(type(output))
                # print(type(label))

                # print(output.shape)
                # print(label.shape)
                loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)
        
        if not self.cfg.NO_FORGET:
            loss_summary = {
                "loss": loss.item(),
                "loss_prv": loss_prv.item() if isinstance(loss_prv, torch.Tensor) else loss_prv,
                "loss_del": loss_del.item() if isinstance(loss_del, torch.Tensor) else loss_del,
                "loss_del_local": loss_del_local.item() if isinstance(loss_del_local, torch.Tensor) else loss_del_local,
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
from utils.data_augmentation import get_jigsaw_tensor
class TrainerDF_Local_DC_Divided(TrainerDF_Local):
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.DATASET.NAME == "OfficeHomeDF":
            self.domain_list = ["art", "clipart", "product", "real_world"]
            self.del_domain_list = cfg.DATASET.FORGETDOMAINS

        elif cfg.DATASET.NAME == "DomainNetDF":
            self.del_domain_list = cfg.DATASET.FORGETDOMAINS
            self.domain_list = [
                "clipart", "infograph", "painting", "quickdraw", "real", "sketch"
            ]
    
        elif cfg.DATASET.NAME == "PACSDF":
            self.del_domain_list = cfg.DATASET.FORGETDOMAINS
            self.domain_list = ["art_painting", "cartoon", "photo", "sketch"]
            # self.classnames = [
            #     "dog", "elephant", "giraffe", "guitar", "horse", "house", "person"
            # ]
        assert (set(self.domain_list) | set(self.del_domain_list)) == set(self.domain_list)
        self.prv_domain_list = list(set(self.domain_list) - set(self.del_domain_list))
        self.classnames = self.dm.dataset.classnames

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
            output, img_feat, local_feat, txt_feat, domain_logit = self.model(image)

            if self.cfg.BLOCK_SHUFFLE:
                shuffled_image = get_jigsaw_tensor(image, grid=self.cfg.GRID, device=self.device)
                _, _, _, _, domain_logit = self.model(shuffled_image)
            if not self.cfg.NO_FORGET:
                entropy = Entropy()
                false_check_tensor = torch.zeros_like(domain, dtype=torch.bool)
                
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
                    loss_del_local = 0
                else :
                    num_of_local_feature = local_feat.shape[1]
                    batch_size_del = local_feat[del_domain_mask].shape[0]
                    output_local = local_feat[del_domain_mask].view(batch_size_del*num_of_local_feature, -1)
                    loss_del = entropy(output[del_domain_mask])
                    loss_del_local = entropy_local_topk(output_local, label[del_domain_mask],num_of_local_feature)
                domain_loss = F.cross_entropy(domain_logit, domain)
                loss = loss_prv - loss_del - loss_del_local + domain_loss
            else :
                # print(type(output))
                # print(type(label))

                # print(output.shape)
                # print(label.shape)
                loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)
        
        if not self.cfg.NO_FORGET:
            loss_summary = {
                "loss": loss.item(),
                "loss_prv": loss_prv.item() if isinstance(loss_prv, torch.Tensor) else loss_prv,
                "loss_del": loss_del.item() if isinstance(loss_del, torch.Tensor) else loss_del,
                "loss_del_local": loss_del_local.item() if isinstance(loss_del_local, torch.Tensor) else loss_del_local,
                "domain_loss": domain_loss.item()
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
            output, img_feat, local_feat, txt_feat, _ = self.model_inference(input)
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

class TrainerDF_Local_DC(TrainerDF_Local):
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.DATASET.NAME == "OfficeHomeDF":
            self.domain_list = ["art", "clipart", "product", "real_world"]
            self.del_domain_list = cfg.DATASET.FORGETDOMAINS

        elif cfg.DATASET.NAME == "DomainNetDF":
            self.del_domain_list = cfg.DATASET.FORGETDOMAINS
            self.domain_list = [
                "clipart", "infograph", "painting", "quickdraw", "real", "sketch"
            ]
        elif cfg.DATASET.NAME == "DomainNetMiniDF":
            self.del_domain_list = cfg.DATASET.FORGETDOMAINS
            self.domain_list = [
                "clipart", "painting", "real", "sketch"
            ]
    
        elif cfg.DATASET.NAME == "PACSDF":
            self.del_domain_list = cfg.DATASET.FORGETDOMAINS
            self.domain_list = ["art_painting", "cartoon", "photo", "sketch"]
            # self.classnames = [
            #     "dog", "elephant", "giraffe", "guitar", "horse", "house", "person"
            # ]
        assert (set(self.domain_list) | set(self.del_domain_list)) == set(self.domain_list)
        self.prv_domain_list = list(set(self.domain_list) - set(self.del_domain_list))
        self.classnames = self.dm.dataset.classnames

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
            output, img_feat, local_feat, txt_feat, domain_logit = self.model(image)
            if not self.cfg.NO_FORGET:
                entropy = Entropy()
                false_check_tensor = torch.zeros_like(domain, dtype=torch.bool)
                
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
                    loss_del_local = 0
                else :
                    num_of_local_feature = local_feat.shape[1]
                    batch_size_del = local_feat[del_domain_mask].shape[0]
                    output_local = local_feat[del_domain_mask].view(batch_size_del*num_of_local_feature, -1)
                    loss_del = entropy(output[del_domain_mask])
                    loss_del_local = entropy_local_topk(output_local, label[del_domain_mask],num_of_local_feature)
                target_label = prv_domain_mask.int()
                domain_loss = F.cross_entropy(domain_logit, target_label.long())
                loss = loss_prv - loss_del - loss_del_local + domain_loss
            else :
                # print(type(output))
                # print(type(label))

                # print(output.shape)
                # print(label.shape)
                loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)
        
        if not self.cfg.NO_FORGET:
            loss_summary = {
                "loss": loss.item(),
                "loss_prv": loss_prv.item() if isinstance(loss_prv, torch.Tensor) else loss_prv,
                "loss_del": loss_del.item() if isinstance(loss_del, torch.Tensor) else loss_del,
                "loss_del_local": loss_del_local.item() if isinstance(loss_del_local, torch.Tensor) else loss_del_local,
                "domain_loss": domain_loss.item()
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
            output, img_feat, local_feat, txt_feat, _ = self.model_inference(input)
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
                print(f"{name} : {acc:.5f}")
            print("==============domain specific acc=================")
            for domain_name in self.domain_list:
                acc = eval_dict[f"correct_{domain_name}"] / eval_dict[f"total_{domain_name}"]
                print(f"{domain_name} : {acc:.5f}")
            print("===================================================")

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]
    
from utils.loss_fn import SoftNearestNeighborsLoss

class TrainerDF_NNL_Local(TrainerDF_Local):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.nnl = SoftNearestNeighborsLoss()
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
            output, img_feat, local_feat, txt_feat = self.model(image)
            if not self.cfg.NO_FORGET:
                entropy = Entropy()
                false_check_tensor = torch.zeros_like(domain, dtype=torch.bool)
                
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
                    loss_del_local = 0
                else :
                    num_of_local_feature = local_feat.shape[1]
                    batch_size_del = local_feat[del_domain_mask].shape[0]
                    output_local = local_feat[del_domain_mask].view(batch_size_del*num_of_local_feature, -1)
                    loss_del = entropy(output[del_domain_mask])
                    loss_del_local = entropy_local_topk(output_local, label[del_domain_mask],num_of_local_feature)
                target_label = prv_domain_mask.int()
                domain_loss = self.nnl(img_feat, target_label.long())
                loss = loss_prv - loss_del - loss_del_local + domain_loss
            else :
                # print(type(output))
                # print(type(label))

                # print(output.shape)
                # print(label.shape)
                loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)
        
        if not self.cfg.NO_FORGET:
            loss_summary = {
                "loss": loss.item(),
                "loss_prv": loss_prv.item() if isinstance(loss_prv, torch.Tensor) else loss_prv,
                "loss_del": loss_del.item() if isinstance(loss_del, torch.Tensor) else loss_del,
                "loss_del_local": loss_del_local.item() if isinstance(loss_del_local, torch.Tensor) else loss_del_local,
                "domain_loss": domain_loss.item()
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
    
class TrainerDF_NNL_Local_PromptGenerator(TrainerDF_NNL_Local):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.nnl = SoftNearestNeighborsLoss()
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
            output, img_feat, local_feat, txt_feat, domain_specific_prompt = self.model(image)
            if not self.cfg.NO_FORGET:
                entropy = Entropy()
                false_check_tensor = torch.zeros_like(domain, dtype=torch.bool)
                
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
                    loss_del_local = 0
                else :
                    num_of_local_feature = local_feat.shape[1]
                    batch_size_del = local_feat[del_domain_mask].shape[0]
                    output_local = local_feat[del_domain_mask].view(batch_size_del*num_of_local_feature, -1)
                    loss_del = entropy(output[del_domain_mask])
                    loss_del_local = entropy_local_topk(output_local, label[del_domain_mask],num_of_local_feature)
                target_label = prv_domain_mask.int()
                domain_loss = self.nnl(img_feat, target_label.long())
                domain_prompt_loss = self.nnl(domain_specific_prompt.view(domain_specific_prompt.shape[0]*domain_specific_prompt.shape[1], -1), domain, domain_specific_prompt.shape[1])
                loss = loss_prv - loss_del - loss_del_local + domain_loss + domain_prompt_loss
            else :
                # print(type(output))
                # print(type(label))

                # print(output.shape)
                # print(label.shape)
                loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)
        
        if not self.cfg.NO_FORGET:
            loss_summary = {
                "loss": loss.item(),
                "loss_prv": loss_prv.item() if isinstance(loss_prv, torch.Tensor) else loss_prv,
                "loss_del": loss_del.item() if isinstance(loss_del, torch.Tensor) else loss_del,
                "loss_del_local": loss_del_local.item() if isinstance(loss_del_local, torch.Tensor) else loss_del_local,
                "domain_loss": domain_loss.item(),
                "domain_prompt_loss": domain_prompt_loss.item()
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
            output, img_feat, local_feat, txt_feat,_ = self.model_inference(input)
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
                print(f"{name} : {acc:.5f}")
            print("==============domain specific acc=================")
            for domain_name in self.domain_list:
                acc = eval_dict[f"correct_{domain_name}"] / eval_dict[f"total_{domain_name}"]
                print(f"{domain_name} : {acc:.5f}")
            print("===================================================")

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

class TrainerDF_NNL(SimpleTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.DATASET.NAME == "OfficeHomeDF":
            self.domain_list = ["art", "clipart", "product", "real_world"]
            self.del_domain_list = cfg.DATASET.FORGETDOMAINS

        elif cfg.DATASET.NAME == "DomainNetDF":
            self.del_domain_list = cfg.DATASET.FORGETDOMAINS
            self.domain_list = [
                "clipart", "infograph", "painting", "quickdraw", "real", "sketch"
            ]
        elif cfg.DATASET.NAME == "DomainNetMiniDF":
            self.del_domain_list = cfg.DATASET.FORGETDOMAINS
            self.domain_list = [
                "clipart", "painting", "real", "sketch"
            ]
        elif cfg.DATASET.NAME == "PACSDF":
            self.del_domain_list = cfg.DATASET.FORGETDOMAINS
            self.domain_list = ["art_painting", "cartoon", "photo", "sketch"]
            # self.classnames = [
            #     "dog", "elephant", "giraffe", "guitar", "horse", "house", "person"
            # ]
        assert (set(self.domain_list) | set(self.del_domain_list)) == set(self.domain_list)
        self.prv_domain_list = list(set(self.domain_list) - set(self.del_domain_list))
        self.classnames = self.dm.dataset.classnames
        self.nllloss = SoftNearestNeighborsLoss()
        self.lmd_domain_loss = cfg.LMD_DOMAIN_LOSS

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()
    
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
            output, img_feat, txt_feat = self.model(image)
            if not self.cfg.NO_FORGET:
                entropy = Entropy()
                
                false_check_tensor = torch.zeros_like(domain, dtype=torch.bool)
                
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
                target_label = prv_domain_mask.int()
                # domain_loss = cossine_embedding_loss(img_feat, prv_domain_mask, label)
                domain_loss = self.nllloss(img_feat, target_label.long())

                loss = loss_prv - loss_del + self.lmd_domain_loss * domain_loss
            else :
                # print(type(output))
                # print(type(label))

                # print(output.shape)
                # print(label.shape)
                loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)
        
        if not self.cfg.NO_FORGET:
            loss_summary = {
                "loss": loss.item(),
                "loss_prv": loss_prv.item() if isinstance(loss_prv, torch.Tensor) else loss_prv,
                "loss_del": loss_del.item() if isinstance(loss_del, torch.Tensor) else loss_del,
                "domain_loss": domain_loss.item(),
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

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        domain = batch["domain"]
        
        input = input.to(self.device)
        label = label.to(self.device)
        domain = domain.to(self.device)
        return input, label, domain

    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]
        domain = batch["domain"]

        input = input.to(self.device)
        label = label.to(self.device)
        domain = domain.to(self.device)

        return input, label, domain
    
    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
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
                print(f"{name} : {acc:.5f}")
            print("==============domain specific acc=================")
            for domain_name in self.domain_list:
                acc = eval_dict[f"correct_{domain_name}"] / eval_dict[f"total_{domain_name}"]
                print(f"{domain_name} : {acc:.5f}")
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
class TrainerDF_NNL_Divided(TrainerDF_NNL):
    
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
            output, img_feat, txt_feat = self.model(image)
            if not self.cfg.NO_FORGET:
                entropy = Entropy()
                
                false_check_tensor = torch.zeros_like(domain, dtype=torch.bool)
                
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
                target_label = prv_domain_mask.int()
                # domain_loss = cossine_embedding_loss(img_feat, prv_domain_mask, label)
                domain_loss = self.nllloss(img_feat, domain)

                loss = loss_prv - loss_del + domain_loss
            else :
                # print(type(output))
                # print(type(label))

                # print(output.shape)
                # print(label.shape)
                loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)
        
        if not self.cfg.NO_FORGET:
            loss_summary = {
                "loss": loss.item(),
                "loss_prv": loss_prv.item() if isinstance(loss_prv, torch.Tensor) else loss_prv,
                "loss_del": loss_del.item() if isinstance(loss_del, torch.Tensor) else loss_del,
                "domain_loss": domain_loss.item(),
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

class TrainerDF_CosEmb(SimpleTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.DATASET.NAME == "OfficeHomeDF":
            self.domain_list = ["art", "clipart", "product", "real_world"]
            self.del_domain_list = cfg.DATASET.FORGETDOMAINS

        elif cfg.DATASET.NAME == "DomainNetDF":
            self.del_domain_list = cfg.DATASET.FORGETDOMAINS
            self.domain_list = [
                "clipart", "infograph", "painting", "quickdraw", "real", "sketch"
            ]
        elif cfg.DATASET.NAME == "PACSDF":
            self.del_domain_list = cfg.DATASET.FORGETDOMAINS
            self.domain_list = ["art_painting", "cartoon", "photo", "sketch"]
            # self.classnames = [
            #     "dog", "elephant", "giraffe", "guitar", "horse", "house", "person"
            # ]
        assert (set(self.domain_list) | set(self.del_domain_list)) == set(self.domain_list)
        self.prv_domain_list = list(set(self.domain_list) - set(self.del_domain_list))
        self.classnames = self.dm.dataset.classnames

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()
    
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
            output, img_feat, txt_feat = self.model(image)
            if not self.cfg.NO_FORGET:
                entropy = Entropy()
                
                false_check_tensor = torch.zeros_like(domain, dtype=torch.bool)
                
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
                
                domain_loss = cossine_embedding_loss(img_feat, prv_domain_mask, label)

                loss = loss_prv - loss_del + domain_loss
            else :
                # print(type(output))
                # print(type(label))

                # print(output.shape)
                # print(label.shape)
                loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)
        
        if not self.cfg.NO_FORGET:
            loss_summary = {
                "loss": loss.item(),
                "loss_prv": loss_prv.item() if isinstance(loss_prv, torch.Tensor) else loss_prv,
                "loss_del": loss_del.item() if isinstance(loss_del, torch.Tensor) else loss_del,
                "domain_loss": domain_loss.item(),
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

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        domain = batch["domain"]
        
        input = input.to(self.device)
        label = label.to(self.device)
        domain = domain.to(self.device)
        return input, label, domain

    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]
        domain = batch["domain"]

        input = input.to(self.device)
        label = label.to(self.device)
        domain = domain.to(self.device)

        return input, label, domain
    
    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
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
    
    def write_embedding(self, mat, meta_data, label_img=None, global_step=None, tag=None):
        if self._writer is None:
            # Do nothing if writer is not initialized
            # Note that writer is only used when training is needed
            pass
        else:
            self._writer.add_embedding(mat, meta_data, label_img, global_step, tag)



class TrainerDF_DC(SimpleTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.DATASET.NAME == "OfficeHomeDF":
            self.domain_list = ["art", "clipart", "product", "real_world"]
            self.del_domain_list = cfg.DATASET.FORGETDOMAINS

        elif cfg.DATASET.NAME == "DomainNetDF":
            self.del_domain_list = cfg.DATASET.FORGETDOMAINS
            self.domain_list = [
                "clipart", "infograph", "painting", "quickdraw", "real", "sketch"
            ]
        elif cfg.DATASET.NAME == "DomainNetMiniDF":
            self.del_domain_list = cfg.DATASET.FORGETDOMAINS
            self.domain_list = [
                "clipart", "painting", "real", "sketch"
            ]
        elif cfg.DATASET.NAME == "PACSDF":
            self.del_domain_list = cfg.DATASET.FORGETDOMAINS
            self.domain_list = ["art_painting", "cartoon", "photo", "sketch"]
            # self.classnames = [
            #     "dog", "elephant", "giraffe", "guitar", "horse", "house", "person"
            # ]
        assert (set(self.domain_list) | set(self.del_domain_list)) == set(self.domain_list)
        self.prv_domain_list = list(set(self.domain_list) - set(self.del_domain_list))
        self.classnames = self.dm.dataset.classnames

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()
    
    def forward_backward(self, batch):
        image, label, domain = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output, img_feat, txt_feat, domain_output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output, img_feat, txt_feat, domain_output = self.model(image)
            
            if self.cfg.BLOCK_SHUFFLE:
                shuffled_image = get_jigsaw_tensor(image, grid=self.cfg.GRID, device=self.device)
                _, _, _, domain_output = self.model(shuffled_image)
            if not self.cfg.NO_FORGET:
                entropy = Entropy()
                false_check_tensor = torch.zeros_like(domain, dtype=torch.bool)
                
                # for prv_domain in prv_domain_list:
                prv_domain_index = [self.domain_list.index(prv_d) for prv_d in self.prv_domain_list if prv_d in self.domain_list]
                prv_domain_mask = torch.isin(domain, torch.tensor(prv_domain_index).to(self.device))
                # for del_domain in del_domain_list:
                del_domain_index = [self.domain_list.index(del_d) for del_d in self.del_domain_list if del_d in self.domain_list]
                del_domain_mask = torch.isin(domain, torch.tensor(del_domain_index).to(self.device))
                
                if torch.equal(false_check_tensor, prv_domain_mask):
                    loss_prv = 0
                    # domain_loss_prv = 0
                else :
                    loss_prv = F.cross_entropy(output[prv_domain_mask], label[prv_domain_mask])
                    # domain_loss_prv = entropy(domain_output[prv_domain_mask])
                if torch.equal(false_check_tensor, del_domain_mask):
                    loss_del = 0
                    # domain_loss_del = 0
                else :
                    loss_del = entropy(output[del_domain_mask])
                    # domain_loss_del = F.cross_entropy(domain_output[del_domain_mask], domain[del_domain_mask])
                target_label = prv_domain_mask.int()
                domain_loss = F.cross_entropy(domain_output, target_label.long())
                loss = loss_prv - loss_del + domain_loss
            else :
                # print(type(output))
                # print(type(label))

                # print(output.shape)
                # print(label.shape)
                loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)
        
        if not self.cfg.NO_FORGET:
            loss_summary = {
                "loss": loss.item(),
                "loss_prv": loss_prv.item() if isinstance(loss_prv, torch.Tensor) else loss_prv,
                "loss_del": loss_del.item() if isinstance(loss_del, torch.Tensor) else loss_del,
                "domain_loss": domain_loss.item(),
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

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        domain = batch["domain"]
        
        input = input.to(self.device)
        label = label.to(self.device)
        domain = domain.to(self.device)
        return input, label, domain

    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]
        domain = batch["domain"]

        input = input.to(self.device)
        label = label.to(self.device)
        domain = domain.to(self.device)

        return input, label, domain
    
    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
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
            output, img_feat, txt_feat, _ = self.model_inference(input)
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
                print(f"{name} : {acc:.5f}")
            print("==============domain specific acc=================")
            for domain_name in self.domain_list:
                acc = eval_dict[f"correct_{domain_name}"] / eval_dict[f"total_{domain_name}"]
                print(f"{domain_name} : {acc:.5f}")
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

class TrainerDF_DC_Divided(TrainerDF_DC):
    def forward_backward(self, batch):
        image, label, domain = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output, img_feat, txt_feat, domain_output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output, img_feat, txt_feat, domain_output = self.model(image)
            
            if self.cfg.BLOCK_SHUFFLE:
                shuffled_image = get_jigsaw_tensor(image, grid=self.cfg.GRID, device=self.device)
                _, _, _, domain_output = self.model(shuffled_image)
            if not self.cfg.NO_FORGET:
                entropy = Entropy()
                false_check_tensor = torch.zeros_like(domain, dtype=torch.bool)
                
                # for prv_domain in prv_domain_list:
                prv_domain_index = [self.domain_list.index(prv_d) for prv_d in self.prv_domain_list if prv_d in self.domain_list]
                prv_domain_mask = torch.isin(domain, torch.tensor(prv_domain_index).to(self.device))
                # for del_domain in del_domain_list:
                del_domain_index = [self.domain_list.index(del_d) for del_d in self.del_domain_list if del_d in self.domain_list]
                del_domain_mask = torch.isin(domain, torch.tensor(del_domain_index).to(self.device))
                
                if torch.equal(false_check_tensor, prv_domain_mask):
                    loss_prv = 0
                    # domain_loss_prv = 0
                else :
                    loss_prv = F.cross_entropy(output[prv_domain_mask], label[prv_domain_mask])
                    # domain_loss_prv = entropy(domain_output[prv_domain_mask])
                if torch.equal(false_check_tensor, del_domain_mask):
                    loss_del = 0
                    # domain_loss_del = 0
                else :
                    loss_del = entropy(output[del_domain_mask])
                    # domain_loss_del = F.cross_entropy(domain_output[del_domain_mask], domain[del_domain_mask])
                # target_label = prv_domain_mask.int()
                domain_loss = F.cross_entropy(domain_output, domain)
                loss = loss_prv - loss_del + domain_loss
            else :
                # print(type(output))
                # print(type(label))

                # print(output.shape)
                # print(label.shape)
                loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)
        
        if not self.cfg.NO_FORGET:
            loss_summary = {
                "loss": loss.item(),
                "loss_prv": loss_prv.item() if isinstance(loss_prv, torch.Tensor) else loss_prv,
                "loss_del": loss_del.item() if isinstance(loss_del, torch.Tensor) else loss_del,
                "domain_loss": domain_loss.item(),
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


class TrainerDC(SimpleTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.DATASET.NAME == "OfficeHomeDF":
            self.domain_list = ["art", "clipart", "product", "real_world"]
            self.del_domain_list = cfg.DATASET.FORGETDOMAINS

        elif cfg.DATASET.NAME == "DomainNetDF":
            self.domain_list = [
                "clipart", "infograph", "painting", "quickdraw", "real", "sketch"
            ]
            self.prv_domain_list = [
                "clipart", "infograph", "quickdraw", "real", "sketch"
            ]
            self.del_domain_list = [
                "painting"
            ]
        elif cfg.DATASET.NAME == "PACSDF":
            self.del_domain_list = cfg.DATASET.FORGETDOMAINS
            self.domain_list = ["art_painting", "cartoon", "photo", "sketch"]
            # self.classnames = [
            #     "dog", "elephant", "giraffe", "guitar", "horse", "house", "person"
            # ]
        assert (set(self.domain_list) | set(self.del_domain_list)) == set(self.domain_list)
        self.prv_domain_list = list(set(self.domain_list) - set(self.del_domain_list))
        self.classnames = self.dm.dataset.classnames
        self.prv_domain_index = [self.domain_list.index(d) for d in self.prv_domain_list]
        self.del_domain_index = [self.domain_list.index(d) for d in self.del_domain_list]

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()
    
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
            output, img_feat = self.model(image)
            # preservation domain = 0 , deletion domain = 1
            # binary_label = torch.where(torch.isin(domain, torch.tensor(self.prv_domain_index).to(domain.device)), 0, 1)
                # print(type(output))
                # print(type(label))

                # print(output.shape)
                # print(label.shape)
            loss = F.cross_entropy(output, domain)
            self.model_backward_and_update(loss)
        
        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, domain)[0].item()
        }
            # acc = compute_accuracy(output, label)[0].item()
        # acc = compute_acc_for_df(output, label, prv_domain_mask, del_domain_mask, domain, self.domain_list, device=self.device)
        
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        domain = batch["domain"]
        
        input = input.to(self.device)
        label = label.to(self.device)
        domain = domain.to(self.device)
        return input, label, domain

    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]
        domain = batch["domain"]

        input = input.to(self.device)
        label = label.to(self.device)
        domain = domain.to(self.device)

        return input, label, domain
    
    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
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
            output, img_feat = self.model_inference(input)
            # binary_label = torch.where(torch.isin(domain, torch.tensor(self.prv_domain_index).to(domain.device)), 0, 1)

            self.evaluator.process(output, domain)
            # # for prv_domain in prv_domain_list:
            # prv_domain_index = [self.domain_list.index(prv_d) for prv_d in self.prv_domain_list if prv_d in self.domain_list]
            # prv_domain_mask = torch.isin(domain, torch.tensor(prv_domain_index).to(self.device))
            # # for del_domain in del_domain_list:
            # del_domain_index = [self.domain_list.index(del_d) for del_d in self.del_domain_list if del_d in self.domain_list]
            # del_domain_mask = torch.isin(domain, torch.tensor(del_domain_index).to(self.device))
            
            # eval_dict = compute_acc_for_df_eval(
            #     eval_dict,
            #     output,
            #     label,
            #     prv_domain_mask,
            #     del_domain_mask,
            #     domain,
            #     self.domain_list,
            #     self.device
            # ) 
            if batch_idx == 0:
                # label_all = domain
                domain_all = domain
                img_feat_all = img_feat
                # input_all = input

            else :
                # label_all = torch.cat((label_all, binary_label))
                domain_all = torch.cat((domain_all, domain))
                img_feat_all = torch.cat((img_feat_all, img_feat))
                # input_all = torch.cat((input_all, input))

        domain_metadata = []
        for d in domain_all:
            domain_metadata.append(self.domain_list[int(d)])
        tag = f"{split}/tsne-plot/"
        # self.write_embedding(img_feat[cls_specific_index], domain_metadata, input[cls_specific_index], global_step=batch_idx, tag=tag)
        self.write_embedding(img_feat_all, domain_metadata, tag=tag)

        results = self.evaluator.evaluate()
        # if not self.cfg.NO_FORGET:
        #     print("==========peservation or delete acc===============")
        #     for name in ["prv", "del"]:
        #         acc = eval_dict[f"correct_{name}"] / eval_dict[f"total_{name}"]
        #         print(f"{name} : {acc:.2f}")
        #     print("==============domain specific acc=================")
        #     for domain_name in self.domain_list:
        #         acc = eval_dict[f"correct_{domain_name}"] / eval_dict[f"total_{domain_name}"]
        #         print(f"{domain_name} : {acc:.2f}")
        #     print("===================================================")

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