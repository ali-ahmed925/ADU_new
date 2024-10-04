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