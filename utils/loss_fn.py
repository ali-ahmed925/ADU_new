import torch
from torch import nn

# for arc face 
# from __future__ import print_function
# from __future__ import division
# import torch
# import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

# def get_entropy(output):
#   ### softmaxせずに渡す (batch_size, # of classes)
#   prob = F.softmax(output,dim=1)
#   entropy = torch.sum(prob * torch.log(prob + 1e-5), dim=1)
  
#   return -torch.mean(entropy) 
def entropy_local_topk(p, label, num_of_local_feature, top_k=3):
    """
    Extract non-Top-K regions and calculate entropy.
    """
    label_repeat = label.repeat_interleave(num_of_local_feature)
    p = F.softmax(p, dim=-1)
    pred_topk = torch.topk(p, k=top_k, dim=1)[1]
    contains_label = pred_topk.eq(torch.tensor(label_repeat).unsqueeze(1)).any(dim=1)
    selected_p = p[contains_label]
    if selected_p.shape[0] == 0:
        return torch.tensor([0]).cuda()
    return -torch.mean(torch.sum(selected_p * torch.log(selected_p + 1e-5), 1))

    # return -torch.mean(torch.sum(selected_p * torch.log(selected_p+1e-5), 1))
def entropy_local_topk_distilled(local_out, local_out_expert, label, num_of_local_feature, top_k=3):
    """
    Extract non-Top-K regions and calculate entropy.
    """
    label_repeat = label.repeat_interleave(num_of_local_feature)

    local_out_expert = F.softmax(local_out_expert, dim=-1)
    local_out = F.softmax(local_out, dim=-1)

    pred_topk = torch.topk(local_out_expert, k=top_k, dim=1)[1]
    contains_label = pred_topk.eq(torch.tensor(label_repeat).unsqueeze(1)).any(dim=1)

    selected_p = local_out[contains_label]
    if selected_p.shape[0] == 0:
        return torch.tensor([0]).cuda()
    return -torch.mean(torch.sum(selected_p * torch.log(selected_p + 1e-5), 1))


def cossine_embedding_loss(input, domain_label, label):
    cosemb = torch.nn.CosineEmbeddingLoss()
    loss = 0
    target_label = domain_label.int()*2 - 1
    # for cls_id, cls in enumerate(classnames):
    #     cls_specific_index = (label == cls_id)
        # if torch.all(cls_specific_index == False):
        #     pass
        # else :
        #     if input[cls_specific_index].dim() == 1:
        #         target = domain_label[cls_specific_index] * domain_label[cls_specific_index]
        #         loss += cosemb(input[cls_specific_index].unsqueeze(0), input[cls_specific_index].unsqueeze(0), target)
        #     else :
    for idx in range(input.size(0)):
        # cls_bool = (label == label[idx])
        domain_bool = (target_label == target_label[idx])
        # y = (cls_bool & domain_bool).int()*2 - 1
        label_bool = torch.isin(label, label[idx])
        y = (label_bool & domain_bool).int()*2 - 1
        # print(input.shape)
        # print("hello",y[label_bool].shape, input[idx].shape, input[label_bool].shape)
        # print(label_bool)
        loss += cosemb(input[idx].unsqueeze(0), input[label_bool], y[label_bool])

    return loss / input.size(0)

class Entropy(nn.Module):
    def __init__(self, is_activation:str="softmax2d", eps:float = 1e-7): #FIXME
        super().__init__()
        self.is_activation = is_activation
        # self.class_num = class_num
        if self.is_activation == "softmax2d":
            self.activation = torch.nn.Softmax(dim=1)
        self.eps = eps
    def forward(self, x)->torch.tensor:
        if self.is_activation:
            x = self.activation(x)                
        # t = 1 / self.class_num
        entropy = torch.sum(x * torch.log(x + 1e-5), dim=1)
  
        return -torch.mean(entropy)
        # for_loss = torch.sum((t * torch.log( x + self.eps)), 1)
        # for_loss = - (torch.sum(for_loss) /len(for_loss))
        # return for_loss
class SoftNearestNeighborsLoss(nn.Module):
    def __init__(self, temperature=0.1, distance_type='L2', mahalanobis_cov=None):
        """
        Calculate the distance between each pair of candidates. 
        Pairs with the same label are considered positive,while pairs with different labels are negative.

        Arguements:
            candidates (torch.Tensor): A tensor representing the candidates to evaluate for contrastive loss.
                                       Each candidate is expected to have associated positives and negatives
                                       from the other candidates. The tensor shape is (B, C), where B is the
                                       batch size and C represents candidate features.
            labels (torch.Tensor): A tensor of (domain) labels for each candidate, with shape (B), where B is the batch size.
            temperature (float): 温度パラメータ。距離の鋭敏さを調整。
            distance_type (str): 距離タイプ。'L2', 'cosine', 'mahalanobis', 'L1'から選択。
            mahalanobis_cov (torch.Tensor or None): マハラノビス距離の共分散行列。指定しない場合は単位行列。
        Return:
            loss (torch.Tensor)
        """
        super().__init__()
        self.temperature = temperature
        self.distance_type = distance_type
        self.mahalanobis_cov = mahalanobis_cov
    
    def forward(self, candidates, labels):
        if len(candidates) != len(labels):
            raise ValueError(f"There are {len(candidates)} candidates, but only {(len(labels))} labels")
        
        device = candidates.device
        b, embed_dim = candidates.shape
        scale = embed_dim**-0.5 
        
        mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).to(device).float()
        mask.fill_diagonal_(0)

        # 距離の計算を距離タイプごとに分岐
        if self.distance_type == 'L2':
            # ユークリッド距離（L2距離）の二乗
            distance_matrix = torch.cdist(candidates, candidates, p=2) ** 2
        
        elif self.distance_type == 'cosine':
            # コサイン距離（1 - コサイン類似度）
            normalized_candidates = nn.functional.normalize(candidates, p=2, dim=1)
            cosine_similarity = torch.mm(normalized_candidates, normalized_candidates.T)
            distance_matrix = 1 - cosine_similarity  # 類似度から距離へ変換
        
        elif self.distance_type == 'mahalanobis':
            # マハラノビス距離
            if self.mahalanobis_cov is None:
                cov_inv = torch.eye(embed_dim, device=device)  # 単位行列を使う
            else:
                cov_inv = torch.inverse(self.mahalanobis_cov).to(device)
            # 各ペアのマハラノビス距離を計算
            distance_matrix = torch.cdist(candidates @ cov_inv, candidates, p=2) ** 2

        elif self.distance_type == 'L1':
            # マンハッタン距離（L1距離）
            distance_matrix = torch.cdist(candidates, candidates, p=1)
        
        else:
            raise ValueError(f"Unsupported distance type: {self.distance_type}")

        # 距離を温度とスケールに応じてスコアへ変換
        exp_distance_matrix = torch.exp(-distance_matrix * scale / self.temperature)
        
        numerators = (exp_distance_matrix * mask).sum(dim=1)
        denominators = exp_distance_matrix.sum(dim=1)

        indices = numerators.nonzero()
        numerators = numerators[indices]
        denominators = denominators[indices]

        r = torch.log(numerators / denominators)
        loss = -r.mean()

        return loss
# class SoftNearestNeighborsLoss(nn.Module):
#     def __init__(self, temperature=0.1):
#         super().__init__()

#         self.temperature = temperature
    
#     def forward(self, candidates, labels):
#         """
#         Calculate the distance between each pair of candidates. 
#         Pairs with the same label are considered positive,while pairs with different labels are negative.

#         Arguements:
#             candidates (torch.Tensor): A tensor representing the candidates to evaluate for contrastive loss.
#                                        Each candidate is expected to have associated positives and negatives
#                                        from the other candidates. The tensor shape is (B, C), where B is the
#                                        batch size and C represents candidate features.
#             labels (torch.Tensor): A tensor of (domain) labels for each candidate, with shape (B), where B is the batch size.
#         Return:
#             loss (torch.Tensor)
#         """
#         if len(candidates) != len(labels):
#             raise ValueError(f"There are {len(candidates)} candidates, but only {(len(labels))} labels")
#         device = candidates.device
#         b, embed_dim = candidates.shape

#         scale = embed_dim**-0.5 
        
#         mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).to(device).float()
#         mask.fill_diagonal_(0)

#         distance_matrix = torch.cdist(candidates, candidates, p=2) ** 2 
#         exp_distance_matrix = torch.exp(-distance_matrix * scale / self.temperature) 
        
#         numerators = (exp_distance_matrix * mask).sum(dim=1)
#         denominators = exp_distance_matrix.sum(dim=1) 

#         # Remove the candidates that has no positive
#         indices = numerators.nonzero()
#         numerators = numerators[indices]
#         denominators = denominators[indices]

#         r = torch.log(numerators / denominators)
#         loss = -r.mean()

#         return loss

# class ArcMarginProduct(nn.Module):
#     r"""Implement of large margin arc distance: :
#         Args:
#             in_features: size of each input sample
#             out_features: size of each output sample
#             s: norm of input feature
#             m: margin

#             cos(theta + m)
#         """
#     def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
#         super(ArcMarginProduct, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.s = s
#         self.m = m
#         self.weight = Parameter(torch.FloatTensor(out_features, in_features))
#         nn.init.xavier_uniform_(self.weight)

#         self.easy_margin = easy_margin
#         self.cos_m = math.cos(m)
#         self.sin_m = math.sin(m)
#         self.th = math.cos(math.pi - m)
#         self.mm = math.sin(math.pi - m) * m

#     def forward(self, input, label):
#         # --------------------------- cos(theta) & phi(theta) ---------------------------
#         cosine = F.linear(F.normalize(input), F.normalize(self.weight))
#         sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
#         phi = cosine * self.cos_m - sine * self.sin_m
#         if self.easy_margin:
#             phi = torch.where(cosine > 0, phi, cosine)
#         else:
#             phi = torch.where(cosine > self.th, phi, cosine - self.mm)
#         # --------------------------- convert label to one-hot ---------------------------
#         # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
#         one_hot = torch.zeros(cosine.size(), device='cuda')
#         one_hot.scatter_(1, label.view(-1, 1).long(), 1)
#         # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
#         output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
#         output *= self.s
#         # print(output)

#         return output