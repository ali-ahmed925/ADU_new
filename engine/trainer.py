import time
import numpy as np
import os.path as osp
import datetime
from collections import OrderedDict
import torch
import torch.nn as nn
from tqdm import tqdm
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

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

def median_pairwise_distance(x, y):
    """Heuristic: median of pairwise distances for sigma."""
    with torch.no_grad():
        pairwise_distances = torch.cdist(x, y, p=2)
        return pairwise_distances.median()

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
        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
            if getattr(self, 'early_stop', False):
                print(f"Early stopping triggered at epoch {self.epoch}")
                break
        self.after_train()

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
            
            start_time = time.time()

            self.test()

            end_time = time.time()
            print(f"Time taken for inference: {end_time - start_time:.4f} seconds")

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

        # Early stopping logic
        if hasattr(self, 'train_loss'):
            if not hasattr(self, 'best_train_loss'):
                self.best_train_loss = self.train_loss
                self.patience_counter = 0
            else:
                if self.train_loss < self.best_train_loss - 1e-4:
                    self.best_train_loss = self.train_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

            patience = 15 # Increased patience for unlearning
            if self.patience_counter >= patience:
                self.early_stop = True

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
            self.domain_list = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]

        elif cfg.DATASET.NAME in ("DomainNetMiniDF", "DomainNetMiniPaperDF"):
            self.del_domain_list = cfg.DATASET.FORGETDOMAINS
            self.domain_list = ["clipart", "painting", "real", "sketch"]

        elif cfg.DATASET.NAME == "ImageNetDF":
            self.del_domain_list = cfg.DATASET.FORGETDOMAINS
            self.domain_list = ["real", "sketch"]

        assert (set(self.domain_list) | set(self.del_domain_list)) == set(self.domain_list)
        self.prv_domain_list = list(set(self.domain_list) - set(self.del_domain_list))
        self.classnames = self.dm.dataset.classnames
        self.del_class_list = getattr(cfg.DATASET, "FORGETCLASSES", [])
        
        self.use_domain_classifier_loss = getattr(cfg, "USE_DOMAIN_CLASIFIER_LOSS", getattr(cfg, "USE_DOMAIN_CLASSIFIER_LOSS", True))
        self.forget_loss_type = getattr(cfg, "FORGET_LOSS_TYPE", "entropy")
        self.no_retain_loss = getattr(cfg, "NO_RETAIN_LOSS", False)
        self.forget_weight = getattr(cfg, "FORGET_WEIGHT", 1.0)
        # P2 (suppress_flat): relative strength of the flatten (anti-leak) term
        # vs the suppression (NegGrad-strength push-down) term.
        self.flat_weight = getattr(cfg, "FLAT_WEIGHT", 1.0)
        # P2: cap on the per-image suppression CE so the push-down cannot run to
        # -inf (unbounded CE-ascent = NegGrad instability). ln(126)=4.84 is the
        # uniform level; cap 6.0 => target prob ~0.0025, safely below uniform,
        # then the gradient switches off. Prevents the loss explosion.
        self.suppress_cap = getattr(cfg, "SUPPRESS_CAP", 6.0)
        # P2c (suppress_marg): weight on the batch-MARGINAL entropy term (rewards
        # different forget images predicting different classes -> breaks the
        # across-image funnel that pins the leak on one class, e.g. 'cat').
        self.marg_weight = getattr(cfg, "MARG_WEIGHT", 1.0)
        self.exclude_forget_class_from_retain = getattr(cfg, "EXCLUDE_FORGET_CLASS_FROM_RETAIN", False)
        self.kernel = "gaussian"
        self.is_domain_divided = True
        self.domain_class_divided = False
        self.ddl_loss_weight = cfg.DDL_LOSS_WEIGHT

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

        if "loss" in losses.meters:
            self.train_loss = losses.meters["loss"].avg

    def train_loop(self):
        super().train()
        return self.metrics_dict
    
    def _build_forget_batch(self):
        """Load ALL forget (class,domain) images once, for the suppress_marg
        dedicated forward. Stores PIL images + labels + a train transform so we
        re-augment freshly each step (augmentation adds a little extra diversity)."""
        from PIL import Image
        from dassl.data.transforms import build_transform
        self._forget_tfm = build_transform(self.cfg, is_train=True)
        cls_idx = [self.classnames.index(c) for c in self.del_class_list if c in self.classnames]
        dom_idx = [self.domain_list.index(d) for d in self.del_domain_list if d in self.domain_list]
        items = [it for it in self.dm.dataset.train_x
                 if it.label in cls_idx and it.domain in dom_idx]
        self._forget_pil = [Image.open(it.impath).convert("RGB") for it in items]
        self._forget_labels = torch.tensor([it.label for it in items], device=self.device)
        print(f"[MARG] forget-batch built: {len(items)} images "
              f"(classes={cls_idx}, domains={dom_idx})", flush=True)

    def forward_backward(self, batch):
        image, label, domain = self.parse_batch_train(batch)

        model_out = self.model(image)
        if len(model_out) == 5:
            output, img_feat, txt_feat, domain_output, _ = model_out
        else:
            output, img_feat, txt_feat, _ = model_out
            domain_output = None

        entropy = Entropy()
        false_check_tensor = torch.zeros_like(domain, dtype=torch.bool)
                
        ###########################################
        # preservation loss
        # deletion loss
        ############################################
        del_domain_index = [self.domain_list.index(del_d) for del_d in self.del_domain_list if del_d in self.domain_list]
        domain_part = torch.isin(domain, torch.tensor(del_domain_index).to(self.device))

        if self.del_class_list:
            # Forget only specific (class, domain) pairs — e.g. tiger/sketch
            del_class_index = [self.classnames.index(c) for c in self.del_class_list if c in self.classnames]
            class_part = torch.isin(label, torch.tensor(del_class_index).to(self.device))
            del_domain_mask = domain_part & class_part
            if self.exclude_forget_class_from_retain:
                # Lever P1: forget class gets NO retain gradient in ANY domain
                prv_domain_mask = ~class_part
            else:
                # v1 default: retain everything except the forget cell
                # (incl. the forget class in other domains — the anchor stays ON)
                prv_domain_mask = ~del_domain_mask
        else:
            # Original behaviour: forget entire domain
            del_domain_mask = domain_part
            prv_domain_index = [self.domain_list.index(prv_d) for prv_d in self.prv_domain_list if prv_d in self.domain_list]
            prv_domain_mask = torch.isin(domain, torch.tensor(prv_domain_index).to(self.device))

        ### Imbalanced domain labels
        ## need to change prv_domain_mask, del_domain_mask, domain, and domain_output

        if torch.equal(false_check_tensor, prv_domain_mask):
            loss_prv = 0
        else :
            loss_prv = F.cross_entropy(output[prv_domain_mask], label[prv_domain_mask])

        if self.forget_loss_type == "suppress_marg":
            # P2c: BATCH-MARGINAL diversity. The 'cat' concentration is an
            # ACROSS-image funnel (all forgotten tigers land on cat), which no
            # per-image term can fix (variance-min and entropy-max both plateau
            # at cat~29%). Here we forward ALL forget images together every step
            # and add entropy-MAX on the MARGINAL (batch-averaged) non-target
            # prediction: if every image predicts cat the marginal peaks ->
            # penalized; to satisfy it, different images must pick different
            # classes. Plus per-image entropy + the same bounded suppression.
            if not hasattr(self, "_forget_pil"):
                self._build_forget_batch()
            imgs = torch.stack([self._forget_tfm(p) for p in self._forget_pil]).to(self.device)
            Lf = self.model(imgs)[0]                          # (m, C) scaled logits
            tf = self._forget_labels                          # (m,) all = forget class
            ce = F.cross_entropy(Lf, tf, reduction="none")
            loss_suppress = ce.clamp(max=self.suppress_cap).mean()
            keep = torch.ones_like(Lf, dtype=torch.bool)
            keep[torch.arange(Lf.shape[0], device=Lf.device), tf] = False
            Lrest = Lf[keep].view(Lf.shape[0], Lf.shape[1] - 1)   # (m, C-1) non-target
            p = F.softmax(Lrest, dim=1)
            H_cond = -(p * torch.log(p + 1e-12)).sum(dim=1).mean()    # per-image (maximize)
            p_bar = p.mean(dim=0)                                     # marginal over the m images
            H_marg = -(p_bar * torch.log(p_bar + 1e-12)).sum()       # marginal (maximize)
            self._sf_suppress = float(loss_suppress)
            self._sf_flat = float(H_cond)
            self._sf_marg = float(H_marg)
            loss_del = -loss_suppress - self.flat_weight * H_cond - self.marg_weight * H_marg
        elif torch.equal(false_check_tensor, del_domain_mask):
            loss_del = 0
        else :
            if self.forget_loss_type == "neggrad":
                # NegGrad: gradient ascent on the forget set's CE
                loss_del = F.cross_entropy(output[del_domain_mask], label[del_domain_mask])
            elif self.forget_loss_type == "flat":
                # Logit-variance minimization: drive all scaled logits equal
                # (uniform softmax by construction; pulls the target logit down
                #  toward the mean). Scaled by logit_scale so the term is
                #  magnitude-matched to the retain CE (scale^2 fix).
                model_ = self.model.module if hasattr(self.model, "module") else self.model
                s = img_feat[del_domain_mask] @ txt_feat.t()
                loss_del = (model_.logit_scale.exp() * s).var(dim=1).mean()
            elif self.forget_loss_type == "suppress_flat":
                # P2: SUPPRESS the target logit hard (NegGrad-strength push-down
                #     so forgetting actually happens & propagates across domains)
                #     + FLATTEN the remaining non-target logits (so the freed
                #     probability mass spreads uniformly instead of piling onto
                #     the nearest neighbour -> leak-free by construction).
                L = output[del_domain_mask]                 # (n, C) scaled logits
                tgt = label[del_domain_mask]                # all = the forget class
                # BOUNDED suppression: clamp per-image CE at suppress_cap so the
                # push-down saturates (no -inf runaway / NegGrad blow-up).
                ce = F.cross_entropy(L, tgt, reduction="none")     # (n,)
                loss_suppress = ce.clamp(max=self.suppress_cap).mean()
                keep = torch.ones_like(L, dtype=torch.bool)
                keep[torch.arange(L.shape[0], device=L.device), tgt] = False
                L_rest = L[keep].view(L.shape[0], L.shape[1] - 1)   # non-target logits
                loss_flat = L_rest.var(dim=1).mean()        # MINIMIZE (spread the rest)
                # bake in the internal signs: this whole term is ADDED below, so
                # minimizing it = maximize (capped) suppression + minimize residual variance.
                self._sf_suppress = float(loss_suppress)
                self._sf_flat = float(loss_flat)
                loss_del = -loss_suppress + self.flat_weight * loss_flat
            elif self.forget_loss_type == "suppress_entropy":
                # P2b: identical bounded suppression, but the anti-leak term is
                # OUTPUT-level entropy-MAX on the non-target softmax instead of
                # feature-level variance-min. The entropy gradient SATURATES as
                # the distribution flattens, so it spreads the freed mass without
                # the aggressive feature funnelling that variance-min causes
                # (which we observed re-peaking onto 'cat' at high flat_weight).
                L = output[del_domain_mask]
                tgt = label[del_domain_mask]
                ce = F.cross_entropy(L, tgt, reduction="none")
                loss_suppress = ce.clamp(max=self.suppress_cap).mean()
                keep = torch.ones_like(L, dtype=torch.bool)
                keep[torch.arange(L.shape[0], device=L.device), tgt] = False
                L_rest = L[keep].view(L.shape[0], L.shape[1] - 1)   # non-target logits
                logp = F.log_softmax(L_rest, dim=1)
                H = -(logp.exp() * logp).sum(dim=1).mean()  # non-target entropy, MAXIMIZE
                self._sf_suppress = float(loss_suppress)
                self._sf_flat = float(H)                    # store H in the same log slot
                loss_del = -loss_suppress - self.flat_weight * H
            else:
                loss_del = entropy(output[del_domain_mask])

        # ---- combine (sign depends on objective) ----
        # entropy / neggrad are MAXIMIZED (subtracted); flat is MINIMIZED (added)
        base = 0 if self.no_retain_loss else loss_prv
        if isinstance(loss_del, torch.Tensor):
            # 'flat' and 'suppress_flat' have their signs baked in -> ADD;
            # 'entropy'/'neggrad' are maximized -> SUBTRACT.
            if self.forget_loss_type in ("flat", "suppress_flat", "suppress_entropy", "suppress_marg"):
                loss = base + self.forget_weight * loss_del
            else:
                loss = base - loss_del
        else:
            loss = base

        # magnitude sanity log: first 8 batches that CONTAIN a forget image
        # (~94% of batches have none, so chronological logging would mislead)
        if isinstance(loss_del, torch.Tensor) and getattr(self, "_forgetlog_n", 0) < 8:
            self._forgetlog_n = getattr(self, "_forgetlog_n", 0) + 1
            extra = ""
            if self.forget_loss_type in ("suppress_flat", "suppress_entropy"):
                term = "flat(var)" if self.forget_loss_type == "suppress_flat" else "ent(H)"
                extra = (f" [suppress(CE)={self._sf_suppress:.4f} "
                         f"{term}={self._sf_flat:.4f} flat_w={self.flat_weight}]")
            elif self.forget_loss_type == "suppress_marg":
                extra = (f" [suppress(CE)={self._sf_suppress:.4f} "
                         f"H_cond={self._sf_flat:.4f} H_marg={self._sf_marg:.4f} "
                         f"flat_w={self.flat_weight} marg_w={self.marg_weight}]")
            print(f"[FORGET-BATCH {self._forgetlog_n}/8] n_forget={int(del_domain_mask.sum())} "
                  f"loss_prv={float(loss_prv):.4f} "
                  f"loss_forget[{self.forget_loss_type}]={float(loss_del):.4f} "
                  f"(weight={self.forget_weight}){extra}", flush=True)

        ######################################################################
        # domain loss (domain classifier loss, nearest neighbor loss or both)
        #####################################################################
        if domain_output is not None:
            target_label = domain
            mmd_loss = total_pairwise_mmd(domain_output.float(), target_label.float()) * self.cfg.MMD_WEIGHT
            ddl = F.cross_entropy(domain_output, target_label) * self.ddl_loss_weight
            domain_cls_loss = ddl - mmd_loss
            loss += domain_cls_loss
        else:
            domain_cls_loss = torch.tensor(0.0)

        if isinstance(loss, torch.Tensor):
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item() if isinstance(loss, torch.Tensor) else float(loss),
            "loss_prv": loss_prv.item() if isinstance(loss_prv, torch.Tensor) else loss_prv,
            "loss_del": loss_del.item() if isinstance(loss_del, torch.Tensor) else loss_del,
            "loss_domain_cls": domain_cls_loss.item() ,
        }
        acc = compute_acc_for_df(output, label, prv_domain_mask, del_domain_mask, domain, self.domain_list, device=self.device)
        loss_summary.update(acc)
        
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        domain = batch["domain"]
        
        input = input.to(dtype=input.dtype, device=self.device)
        label = label.to(dtype=label.dtype, device=self.device)
        domain = domain.to(dtype=domain.dtype, device=self.device)

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
            
            model_out = self.model_inference(input)
            if len(model_out) == 5:
                output, img_feat, _, domain_logit, _ = model_out
            else:
                output, img_feat, _, _ = model_out
                domain_logit = None
            
            self.evaluator.process(output, label)
            
            prv_domain_index = [self.domain_list.index(prv_d) for prv_d in self.prv_domain_list if prv_d in self.domain_list]
            prv_domain_mask = torch.isin(domain, torch.tensor(prv_domain_index).to(self.device))
            
            del_domain_index = [self.domain_list.index(del_d) for del_d in self.del_domain_list if del_d in self.domain_list]
            del_domain_mask = torch.isin(domain, torch.tensor(del_domain_index).to(self.device))

            target_label_list = []
            for lb, dm in zip(label, domain):
                new_label = int(lb.item()*len(self.domain_list) + dm.item())
                target_label_list.append(new_label)
            target_label = torch.tensor(target_label_list).to(self.device)

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

            if batch_idx == 0:
                label_all = label
                domain_all = domain
                img_feat_all = img_feat

            else :
                label_all = torch.cat((label_all, label))
                domain_all = torch.cat((domain_all, domain))
                img_feat_all = torch.cat((img_feat_all, img_feat))

        for cls_id, clsname in enumerate(self.classnames):
            cls_specific_index = (label_all == cls_id)

            if torch.all(cls_specific_index == False):
                pass
            else:
                domain_metadata = []
                for d in domain_all[cls_specific_index]:
                    domain_metadata.append(self.domain_list[int(d)])
                tag = f"{split}/tsne-plot/{clsname}"
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
                round((eval_dict.get("correct_prv", 0) / eval_dict.get("total_prv", 1))*100, 3),
                round((1 - eval_dict.get("correct_del", 0) / eval_dict.get("total_del", 1))*100, 3), 
                round((eval_dict.get("correct_del", 0) / eval_dict.get("total_del", 1))*100, 3)
            ]
            if len(self.del_domain_list) == 0:
                specific_acc = "()"
            else:
                specific_acc = "("
                for idx, dname in enumerate(self.del_domain_list):
                    sp_acc = (eval_dict.get(f"correct_{dname}", 0) / eval_dict.get(f"total_{dname}", 1))*100
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

        print("==========peservation or delete acc===============")
        for name in ["prv", "del"]:
            acc = eval_dict.get(f"correct_{name}", 0) / eval_dict.get(f"total_{name}", 1)
            print(f"{name} : {acc:.5f}")
        print("==============domain specific acc=================")
        for domain_name in self.domain_list:
            acc = eval_dict.get(f"correct_{domain_name}", 0) / eval_dict.get(f"total_{domain_name}", 1)
            print(f"{domain_name} : {acc:.5f}")

        print("==============per class per domain acc============")
        print(f"{'Class':<15} ", end="")
        for domain_name in self.domain_list:
            print(f"{domain_name:<10} ", end="")
        print("")
        for cls in self.classnames:
            print(f"{cls:<15} ", end="")
            for domain_name in self.domain_list:
                correct = eval_dict.get(f"correct_clsdom_{cls}_{domain_name}", 0)
                total = eval_dict.get(f"total_clsdom_{cls}_{domain_name}", 0)
                acc = (correct / total) * 100 if total > 0 else 0
                print(f"{acc:<9.2f}% ", end="")
            print("")

        if self.domain_class_divided:
            if self.use_domain_classifier_loss:
                print("==================domain DC accuracy==================")
                if self.is_domain_divided:
                    for idx in range(len(self.domain_list)*len(self.classnames)):
                        cls = self.classnames[int(idx / len(self.domain_list))]
                        dm = self.domain_list[int(idx % len(self.domain_list))]
                        acc = eval_dict.get(f"correct_{cls}_{dm}_DC", 0) / eval_dict.get(f"total_{cls}_{dm}_DC", 1)
                        print(f"{cls} {dm} : {acc:.5f}")
        else:
            if self.use_domain_classifier_loss:
                print("==================domain DC accuracy==================")
                if self.is_domain_divided:
                    for domain_name in self.domain_list:
                        acc = eval_dict.get(f"correct_{domain_name}_DC", 0) / eval_dict.get(f"total_{domain_name}_DC", 1)
                        print(f"{domain_name} : {acc:.5f}")
                else:
                    for domain_name in ["prv", "del"]:
                        acc = eval_dict.get(f"correct_{domain_name}_DC", 0) / eval_dict.get(f"total_{domain_name}_DC", 1)
                        print(f"{domain_name} : {acc:.5f}")
                acc = eval_dict.get("correct_domain", 0) / eval_dict.get("total_domain", 1)
                print("==================domain DC accuracy tot==================")
                print(f"domain acc : {acc:.5f}")
        print("===================================================")
        metrics_A = eval_dict.get("correct_prv", 0) / eval_dict.get("total_prv", 1)
        metrics_F = 1 - eval_dict.get("correct_del", 0) / eval_dict.get("total_del", 1)
        metrics_H = 2 * metrics_A * metrics_F / (metrics_A + metrics_F) if (metrics_A + metrics_F) > 0 else 0

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