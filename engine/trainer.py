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

from utils.loss_fn import Entropy
from utils.eval_acc import compute_acc_for_df, compute_acc_for_df_eval
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

class TrainerDF(SimpleTrainer):
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
        assert (set(self.domain_list) | set(self.del_domain_list)) == set(self.domain_list)
        self.prv_domain_list = list(set(self.domain_list) - set(self.del_domain_list))


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
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            is_DF = True #FIXME
            if is_DF :
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
                loss = loss_prv - loss_del
            else :
                loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)
        loss_summary = {
            "loss": loss.item(),
            "loss_prv": loss_prv.item() if isinstance(loss_prv, torch.Tensor) else loss_prv,
            "loss_del": loss_del.item() if isinstance(loss_del, torch.Tensor) else loss_del,
            # "acc": compute_accuracy(output, label)[0].item(),
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
            output = self.model_inference(input)
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

        results = self.evaluator.evaluate()
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
    
def isin(elements, test_elements):
    # Check if each element in `elements` is present in `test_elements`
    return torch.stack([torch.any(elements == x, dim=0) for x in test_elements])