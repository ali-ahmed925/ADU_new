import torch
from torch import nn

class EntropyMaximizationLoss(nn.Module):
    def __init__(self, is_activation:str="softmax2d", class_num:int = 65,eps:float = 1e-7): #FIXME
        super().__init__()
        self.is_activation = is_activation
        self.class_num = class_num
        if self.is_activation == "softmax2d":
            self.activation = torch.nn.Softmax(dim=1)
        self.eps = eps
    def forward(self, x)->torch.tensor:
        if self.is_activation:
            x = self.activation(x)                
        t = 1 / self.class_num
        for_loss = torch.sum((t * torch.log( x + self.eps)), 1)
        for_loss = - (torch.sum(for_loss) /len(for_loss))
        return for_loss