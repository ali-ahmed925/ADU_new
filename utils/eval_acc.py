import torch
from dassl.metrics.accuracy import compute_accuracy
# def compute_prv_del_accuracy(output, target, prv_mask, del_mask, topk=(1, )):
#     """Computes the accuracy over the k top predictions for
#     the specified values of k.

#     Args:
#         output (torch.Tensor): prediction matrix with shape (batch_size, num_classes).
#         target (torch.LongTensor): ground truth labels with shape (batch_size).
#         topk (tuple, optional): accuracy at top-k will be computed. For example,
#             topk=(1, 5) means accuracy at top-1 and top-5 will be computed.

#     Returns:
#         list: accuracy at top-k.
#     """
#     maxk = max(topk)
#     batch_size = target.size(0)

#     if isinstance(output, (tuple, list)):
#         output = output[0]

#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     false_check_tensor = torch.zeros_like(prv_mask, dtype=torch.bool)
#     if torch.equal(false_check_tensor, prv_mask):
#         correct_prv = pred[prv_mask].eq(target[prv_mask].view(1, -1).expand_as(pred[prv_mask]))
#     else :
#         correct_prv = -1
#     if torch.equal(false_check_tensor, del_mask):
#         correct_del = pred[del_mask].eq(target[del_mask].view(1, -1).expand_as(pred[del_mask]))
#     else :
#         correct_del = -1
#     res_prv = []
#     res_del = []
#     for k in topk:
#         if correct_prv != -1:
#             correct_k = correct_prv[:k].view(-1).float().sum(0, keepdim=True)
#             acc = correct_k.mul_(100.0 / batch_size)
#             res_prv.append(acc)
#         else :
#             res_prv.append(correct_prv)
#         if correct_del != -1:
#             correct_k = correct_del[:k].view(-1).float().sum(0, keepdim=True)
#             acc = correct_k.mul_(100.0 / batch_size)
#             res_del.append(acc)
#         else :
#             res_del.append(correct_del)

#     return res_prv, res_del
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

def compute_acc_for_df_eval(acc_dict, output, target, prv_mask, del_mask, domain_label, domain_list, device):
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