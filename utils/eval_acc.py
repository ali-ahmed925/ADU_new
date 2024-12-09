import torch
from dassl.metrics.accuracy import compute_accuracy

import numpy as np
import os.path as osp
from collections import OrderedDict, defaultdict
import torch
from sklearn.metrics import f1_score, confusion_matrix
from dassl.evaluation.build import EVALUATOR_REGISTRY
from dassl.evaluation.evaluator import Classification

# @EVALUATOR_REGISTRY.register()
# class ClassificationDetail(Classification):
#     def __init__(self, cfg, lab2cname=None, **kwargs):
#         super().__init__(cfg, lab2cname, **kwargs)

def compute_acc_for_df(output, target, prv_mask, del_mask, domain_label, domain_list, device):
    acc_dict = {}
    acc_dict["acc"] = compute_accuracy(output, target)[0].item()
    false_check_tensor = torch.zeros_like(prv_mask, dtype=torch.bool)
    
    if torch.equal(false_check_tensor, prv_mask):
        acc_dict["acc_prv"] = -1
    else :
        acc_dict["acc_prv"] = compute_accuracy(output[prv_mask], target[prv_mask])[0].item()
    if torch.equal(false_check_tensor, del_mask):
        acc_dict["acc_del"] = -1
    else :
        acc_dict["acc_del"] = compute_accuracy(output[del_mask], target[del_mask])[0].item()

    for domain in domain_list:
        mask = torch.isin(domain_label, torch.tensor([domain_list.index(domain)]).to(device)) # (domain_label != domain_list.index(domain))
        if mask_false_check(mask):
            acc_dict[f"{domain}"] = -1
        else :
            acc_dict[f"{domain}"] = compute_accuracy(output[mask], target[mask])[0].item()

    return acc_dict

def compute_acc_for_df_eval(acc_dict, 
                            output, 
                            target, 
                            prv_mask, 
                            del_mask, 
                            domain_label, 
                            domain_list,
                            domain_logit,
                            target_label,
                            is_divided,
                            use_domain_classifier,
                            device,
                            domain_class_divided=False,
                            classnames=[],
                            ):
    # acc_dict = {}
    # acc_dict["acc"] = compute_accuracy(output, target)[0].item()
    false_check_tensor = torch.zeros_like(prv_mask, dtype=torch.bool)
    
    if torch.equal(false_check_tensor, prv_mask):
        # acc_dict["acc_prv"] = -1
        pass
    else :
        correct, total = process(output[prv_mask], target[prv_mask])
        if "correct_prv" in acc_dict:
            acc_dict["correct_prv"] += correct
            acc_dict["total_prv"] += total
        else :
            acc_dict["correct_prv"] = correct
            acc_dict["total_prv"] = total
    if torch.equal(false_check_tensor, del_mask):
        pass
    else :
        correct, total = process(output[del_mask], target[del_mask])
        if "correct_del" in acc_dict:
            acc_dict["correct_del"] += correct
            acc_dict["total_del"] += total
        else :
            acc_dict["correct_del"] = correct
            acc_dict["total_del"] = total

    for domain in domain_list:
        mask = torch.isin(domain_label, torch.tensor([domain_list.index(domain)]).to(device)) # (domain_label != domain_list.index(domain))
        if mask_false_check(mask):
            pass
        else :
            correct, total = process(output[mask], target[mask])
            if f"correct_{domain}" in acc_dict:
                acc_dict[f"correct_{domain}"] += correct
                acc_dict[f"total_{domain}"] += total
            else :
                acc_dict[f"correct_{domain}"] = correct
                acc_dict[f"total_{domain}"] = total
    if domain_class_divided:
        if use_domain_classifier:
            correct, total = process(domain_logit, target_label)
            if "correct_domain" in acc_dict :
                acc_dict["correct_domain"] += correct
                acc_dict["total_domain"] += total
            else :
                acc_dict["correct_domain"] = correct
                acc_dict["total_domain"] = total
            if is_divided :
                for idx in range(len(domain_list)*len(classnames)):
                    mask = torch.isin(target_label, torch.tensor([domain_list.index(domain)]).to(device))
                    correct, total = process(domain_logit[mask], target_label[mask])
                    cls = classnames[int(idx / len(domain_list))]
                    dm = domain_list[int(idx % len(domain_list))]
                    if f"correct_{cls}_{dm}_DC" in acc_dict:
                        acc_dict[f"correct_{cls}_{dm}_DC"] += correct
                        acc_dict[f"total_{cls}_{dm}_DC"] += total
                    else :
                        acc_dict[f"correct_{cls}_{dm}_DC"] = correct
                        acc_dict[f"total_{cls}_{dm}_DC"] = total
                    pass
    else :
        if use_domain_classifier:
            correct, total = process(domain_logit, target_label)
            if "correct_domain" in acc_dict :
                acc_dict["correct_domain"] += correct
                acc_dict["total_domain"] += total
            else :
                acc_dict["correct_domain"] = correct
                acc_dict["total_domain"] = total
            if is_divided :
                for domain in domain_list:
                    mask = torch.isin(target_label, torch.tensor([domain_list.index(domain)]).to(device)) # (domain_label != domain_list.index(domain))
                    correct, total = process(domain_logit[mask], target_label[mask])
                    if f"correct_{domain}_DC" in acc_dict:
                        acc_dict[f"correct_{domain}_DC"] += correct
                        acc_dict[f"total_{domain}_DC"] += total
                    else :
                        acc_dict[f"correct_{domain}_DC"] = correct
                        acc_dict[f"total_{domain}_DC"] = total
            else :
                for domain in ["del", "prv"] :
                    mask = torch.isin(target_label, torch.tensor([["prv", "del"].index(domain)]).to(device)) # (domain_label != domain_list.index(domain))
                    correct, total = process(domain_logit[mask], target_label[mask])
                    if f"correct_{domain}_DC" in acc_dict:
                        acc_dict[f"correct_{domain}_DC"] += correct
                        acc_dict[f"total_{domain}_DC"] += total
                    else :
                        acc_dict[f"correct_{domain}_DC"] = correct
                        acc_dict[f"total_{domain}_DC"] = total
                    # acc_dict[f"{domain}"] = compute_accuracy(output[mask], target[mask])[0].item()

    return acc_dict

def mask_false_check(mask):
    false_check_tensor = torch.zeros_like(mask, dtype=torch.bool)
    return torch.equal(false_check_tensor, mask)
# def compute_domain_specific_accuracy(output, target, domain_list):

def process(mo, gt):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        pred = mo.max(1)[1]
        matches = pred.eq(gt).float()
        correct = int(matches.sum().item())
        total = gt.shape[0]

        # self._y_true.extend(gt.data.cpu().numpy().tolist())
        # self._y_pred.extend(pred.data.cpu().numpy().tolist())

        # if self._per_class_res is not None:
        #     for i, label in enumerate(gt):
        #         label = label.item()
        #         matches_i = int(matches[i].item())
        #         self._per_class_res[label].append(matches_i)
        return correct, total