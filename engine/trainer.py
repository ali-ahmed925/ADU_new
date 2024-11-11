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

from utils.loss_fn import Entropy, cossine_embedding_loss
from dassl.metrics.accuracy import compute_accuracy
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
                loss = loss_prv - loss_del
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
from utils.loss_fn import entropy_local_topk
class TrainerDF_Local(SimpleTrainer):
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
            output, img_feat, local_feat, txt_feat = self.model_inference(input)
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
    
from utils.loss_fn import SoftNearestNeighborsLoss
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