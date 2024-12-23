from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from typing import Type, Tuple, Union, Optional
NoneType = Type[None]

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]

class AttentionPool2d_Local(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None, return_local_features: bool = True) -> NoneType:
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        self.return_local_features = return_local_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_local = x.reshape(b, c, h, w).permute(0, 2, 3, 1)

        if self.return_local_features:
            x_local = F.linear(x_local, self.v_proj.weight, self.v_proj.bias)
            x_local = F.linear(x_local, self.c_proj.weight, self.c_proj.bias)

        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )  # The final number of token is given by the query

        x_local = x_local.reshape(b, h * w, -1)
        return x.squeeze(0), x_local


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x

class ModifiedResNet_Local(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs antialiasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers: list, output_dim: int, heads: int, input_resolution: int = 224, width: int = 64, return_local_features: bool = True) -> NoneType:
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d_Local(input_resolution // 32, embed_dim, heads, output_dim, return_local_features)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Module:
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def stem(x: torch.Tensor) -> torch.Tensor:
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x_global, x_local = self.attnpool(x)
        return x_global, x_local

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class ResidualAttentionBlock_Local(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None) -> NoneType:
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def attention_weight(self, x: torch.Tensor) -> torch.Tensor:  # ADDED
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask)[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ln_1(x)
        y = y.permute(1, 0, 2)
        y = F.linear(y, self.attn.in_proj_weight, self.attn.in_proj_bias)
        # The in_proj_weight performs the q_proj, k_proj, v_proj projections
        N, L, C = y.shape
        y = y.view(N, L, 3, C // 3).permute(2, 0, 1, 3).reshape(3 * N, L, C // 3)
        y = F.linear(y, self.attn.out_proj.weight, self.attn.out_proj.bias)
        q, k, v = y.tensor_split(3, dim=0)
        v = v.permute(1, 0, 2)
        q = q.permute(1, 0, 2)
        k = k.permute(1, 0, 2)
        v += x
        v = v + self.mlp(self.ln_2(v))

        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x, q, k, v
    

class ResidualAttentionBlock_IVLP_Local(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, add_prompt=False, text_layer=False,i=0, design_details=None) -> NoneType:
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)

        self.text_layer = text_layer
        self.attn_mask = attn_mask

        if i != 0:
            self.add_prompt = add_prompt
            if self.add_prompt:
                if self.text_layer:
                    self.n_ctx_text = design_details["language_ctx"]  # hyperparameter
                    ctx_vectors = torch.empty(self.n_ctx_text, d_model)
                else:
                    self.n_ctx_visual = design_details["vision_ctx"]  # hyperparameter
                    ctx_vectors = torch.empty(self.n_ctx_visual, d_model)
                # Code snippet for per layer visual prompts
                nn.init.normal_(ctx_vectors, std=0.02)
                self.VPT_shallow = nn.Parameter(ctx_vectors)
        else:
            self.add_prompt = False


    def attention(self, x: torch.Tensor) -> torch.Tensor:
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def attention_weight(self, x: torch.Tensor) -> torch.Tensor:  # ADDED
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask)[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.add_prompt:
            # Also see if this is textual transformer layer or not
            if not self.text_layer:
                # Remove the outputs produced by learnable tokens of previous layer
                prefix = x[0:x.shape[0] - self.n_ctx_visual, :, :]
                # Create/configure learnable tokens of this layer
                visual_context = self.VPT_shallow.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                # Add the learnable tokens of this layer with the input, by replacing the previous
                # layer learnable tokens
                x = torch.cat([prefix, visual_context], dim=0)
            else:
                # Appending the learnable tokens in different way
                # x -> [77, NCLS, DIM]
                # First remove the learnable tokens from previous layer
                prefix = x[:1, :, :]
                suffix = x[1 + self.n_ctx_text:, :, :]
                # Create/configure learnable tokens of this layer
                textual_context = self.VPT_shallow.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                # Add the learnable tokens of this layer with the input, replaced by previous
                # layer learnable tokens
                x = torch.cat([prefix, textual_context, suffix], dim=0)   
        y = self.ln_1(x)
        y = y.permute(1, 0, 2)
        y = F.linear(y, self.attn.in_proj_weight, self.attn.in_proj_bias)
        # The in_proj_weight performs the q_proj, k_proj, v_proj projections
        N, L, C = y.shape
        y = y.view(N, L, 3, C // 3).permute(2, 0, 1, 3).reshape(3 * N, L, C // 3)
        y = F.linear(y, self.attn.out_proj.weight, self.attn.out_proj.bias)
        q, k, v = y.tensor_split(3, dim=0)
        v = v.permute(1, 0, 2)
        q = q.permute(1, 0, 2)
        k = k.permute(1, 0, 2)
        v += x
        v = v + self.mlp(self.ln_2(v))

        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x, q, k, v

# class ResidualAttentionBlock_IVLP_Prompt(nn.Module):
#     def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, add_prompt=False, text_layer=False,i=0, design_details=None) -> NoneType:
#         super().__init__()

#         self.attn = nn.MultiheadAttention(d_model, n_head)
#         self.ln_1 = LayerNorm(d_model)
#         self.mlp = nn.Sequential(OrderedDict([
#             ("c_fc", nn.Linear(d_model, d_model * 4)),
#             ("gelu", QuickGELU()),
#             ("c_proj", nn.Linear(d_model * 4, d_model))
#         ]))
#         self.ln_2 = LayerNorm(d_model)

#         self.text_layer = text_layer
#         self.attn_mask = attn_mask

#         if i != 0:
#             self.add_prompt = add_prompt
#             if self.add_prompt:
#                 if self.text_layer:
#                     self.n_ctx_text = design_details["language_ctx"]  # hyperparameter
#                     ctx_vectors = torch.empty(self.n_ctx_text, d_model)
#                 else:
#                     self.n_ctx_visual = design_details["vision_ctx"]  # hyperparameter
#                     ctx_vectors = torch.empty(self.n_ctx_visual, d_model)
#                 # Code snippet for per layer visual prompts
#                 nn.init.normal_(ctx_vectors, std=0.02)
#                 self.VPT_shallow = nn.Parameter(ctx_vectors)
#         else:
#             self.add_prompt = False
#         self.insert_layer = design_details["insert_layer"] - 1
#         if i == self.insert_layer:
#             self.cross_attn = nn.MultiheadAttention(d_model, n_head)
    
#     def cross_atention(self, q: torch.Tensor, k:torch.Tensor, v:torch.Tensor):
#         self.attn_mask = self.attn_mask.to(dtype=q.dtype, device=k.device) if self.attn_mask is not None else None
#         return self.cross_attn(query=q, key=k, value=v)[0]

#     def attention(self, x: torch.Tensor) -> torch.Tensor:
#         self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
#         return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

#     def attention_weight(self, x: torch.Tensor) -> torch.Tensor:  # ADDED
#         self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
#         return self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask)[1]

#     def forward(self, x: torch.Tensor, idx: int) -> torch.Tensor:
#         if self.add_prompt:
#             # Also see if this is textual transformer layer or not
#             if not self.text_layer:
#                 # Remove the outputs produced by learnable tokens of previous layer
#                 prefix = x[0:x.shape[0] - self.n_ctx_visual, :, :]
#                 # Create/configure learnable tokens of this layer
#                 visual_context = self.VPT_shallow.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
#                 # Add the learnable tokens of this layer with the input, by replacing the previous
#                 # layer learnable tokens
#                 if idx == self.insert_layer :
#                     visual_context = self.cross_atention(visual_context, prefix, prefix)
#                 x = torch.cat([prefix, visual_context], dim=0)
#             else:
#                 # Appending the learnable tokens in different way
#                 # x -> [77, NCLS, DIM]
#                 # First remove the learnable tokens from previous layer
#                 prefix = x[:1, :, :]
#                 suffix = x[1 + self.n_ctx_text:, :, :]
#                 # Create/configure learnable tokens of this layer
#                 textual_context = self.VPT_shallow.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
#                 # Add the learnable tokens of this layer with the input, replaced by previous
#                 # layer learnable tokens
#                 x = torch.cat([prefix, textual_context, suffix], dim=0)   
#         y = self.ln_1(x)
#         y = y.permute(1, 0, 2)
#         y = F.linear(y, self.attn.in_proj_weight, self.attn.in_proj_bias)
#         # The in_proj_weight performs the q_proj, k_proj, v_proj projections
#         N, L, C = y.shape
#         y = y.view(N, L, 3, C // 3).permute(2, 0, 1, 3).reshape(3 * N, L, C // 3)
#         y = F.linear(y, self.attn.out_proj.weight, self.attn.out_proj.bias)
#         q, k, v = y.tensor_split(3, dim=0)
#         v = v.permute(1, 0, 2)
#         q = q.permute(1, 0, 2)
#         k = k.permute(1, 0, 2)
#         v += x
#         v = v + self.mlp(self.ln_2(v))

#         x = x + self.attention(self.ln_1(x))
#         x = x + self.mlp(self.ln_2(x))

#         return x, q, k, v

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            QuickGELU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x.permute(1, 0, 2)
        return self.net(x).permute(1, 0, 2)

class CrossAttention(nn.Module):
    def __init__(
            self, 
            embed_dim: int,
            num_cross_attention_heads:int,
            attention_dropout: float=0,
            ffn_dropout: float = 0    
                 ):
        super().__init__()
        self.pre_norm1 = nn.LayerNorm(embed_dim)
        self.pre_norm2 = nn.LayerNorm(embed_dim)
        self.cross_attention = torch.nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_cross_attention_heads
        )
        self.post_norm = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim)
    
    def forward(self, x, visual_feature):
        x = self.pre_norm1(x)
        visual_feature = self.pre_norm2(visual_feature)

        x = self.cross_attention(x, visual_feature, visual_feature)[0] + x
        x = self.post_norm(x)
        x = self.ffn(x) + x
        return x

    # def cross_attention_promptgen(self, q: torch.Tensor, kv: torch.Tensor):
    #     # self.attn_mask = self.attn_mask.to(dtype=q.dtype, device=q.device) if self.attn_mask is not None else None
    #     self.cross_attn = self.cross_attn.to(dtype=q.dtype)
    #     return self.cross_attn(q, kv)

class ResidualAttentionBlock_IVLP_Prompt(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, add_prompt=False,
                 text_layer=False, i=0, design_details=None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        # Only add learnable tokens if flag is set True
        # For the first iteration i, we should not add the learnable parameters
        # as it is already been taken care of in the very start, for both text
        # and the visual branch
        self.independent_learnable_vision = design_details["independent_learnable_vision"]
        self.text_layer = text_layer
        self.attn_mask = attn_mask
        if i != 0:
            self.add_prompt = add_prompt
            if self.add_prompt:
                if self.text_layer:
                    self.n_ctx_text = design_details["language_ctx"]  # hyperparameter
                    ctx_vectors = torch.empty(self.n_ctx_text, d_model)
                    nn.init.normal_(ctx_vectors, std=0.02)
                    self.VPT_shallow = nn.Parameter(ctx_vectors)
                else:
                    self.n_ctx_visual = design_details["vision_ctx"]  # hyperparameter
                    if self.independent_learnable_vision:
                        ctx_vectors = torch.empty(self.n_ctx_visual, d_model)
                        nn.init.normal_(ctx_vectors, std=0.02)
                        self.VPT_shallow = nn.Parameter(ctx_vectors)
                    else :
                        pass
                # Code snippet for per layer visual prompts
        else:
            self.add_prompt = False
        self.insert_layer = design_details["vision_depth"] - 1
        self.use_classtoken = design_details["use_classtoken"]
        # self.use_cross_attention = design_details["use_cross_attention"]
        if i == self.insert_layer:
            if not self.text_layer:
                self.cross_attn = CrossAttention(d_model, n_head)

    def cross_atention(self, q: torch.Tensor, k:torch.Tensor, v:torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=q.dtype, device=k.device) if self.attn_mask is not None else None
        return self.cross_attn(query=q, key=k, value=v)[0]
    
    def cross_attention_promptgen(self, q: torch.Tensor, kv: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=q.dtype, device=q.device) if self.attn_mask is not None else None
        self.cross_attn = self.cross_attn.to(dtype=q.dtype)
        return self.cross_attn(q, kv)
        # pass

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, idx: int, vpt_share: Union[torch.Tensor, NoneType] = None):
        # Will need to append the learnable tokens for this layer here
        # Check if flag was set for this layer or not
        if self.add_prompt:
            # Also see if this is textual transformer layer or not
            if not self.text_layer:
                # Remove the outputs produced by learnable tokens of previous layer
                prefix = x[0:x.shape[0] - self.n_ctx_visual, :, :]
                # Create/configure learnable tokens of this layer
                if self.independent_learnable_vision:
                    visual_context = self.VPT_shallow.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                else :
                    visual_context = vpt_share.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                # Add the learnable tokens of this layer with the input, by replacing the previous
                # layer learnable tokens
                if idx == self.insert_layer :
                    if self.use_classtoken:
                        kv = prefix[0,:,:].unsqueeze(0)
                    else:   
                        kv = prefix[1:,:,:]
                    visual_context = self.cross_attention_promptgen(visual_context, kv)
                x = torch.cat([prefix, visual_context], dim=0)
            else:
                # Appending the learnable tokens in different way
                # x -> [77, NCLS, DIM]
                # First remove the learnable tokens from previous layer
                prefix = x[:1, :, :]
                suffix = x[1 + self.n_ctx_text:, :, :]
                # Create/configure learnable tokens of this layer
                textual_context = self.VPT_shallow.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                # Add the learnable tokens of this layer with the input, replaced by previous
                # layer learnable tokens
                x = torch.cat([prefix, textual_context, suffix], dim=0)                
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    

class ResidualAttentionBlock_IVLP_Prompt_SelectPatch(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, add_prompt=False,
                 text_layer=False, i=0, design_details=None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        # Only add learnable tokens if flag is set True
        # For the first iteration i, we should not add the learnable parameters
        # as it is already been taken care of in the very start, for both text
        # and the visual branch
        self.text_layer = text_layer
        self.attn_mask = attn_mask
        if i != 0:
            self.add_prompt = add_prompt
            if self.add_prompt:
                if self.text_layer:
                    self.n_ctx_text = design_details["language_ctx"]  # hyperparameter
                    ctx_vectors = torch.empty(self.n_ctx_text, d_model)
                else:
                    self.n_ctx_visual = design_details["vision_ctx"]  # hyperparameter
                    ctx_vectors = torch.empty(self.n_ctx_visual, d_model)
                # Code snippet for per layer visual prompts
                nn.init.normal_(ctx_vectors, std=0.02)
                self.VPT_shallow = nn.Parameter(ctx_vectors)
        else:
            self.add_prompt = False
        self.insert_layer = design_details["vision_depth"] - 1
        self.use_classtoken = design_details["use_classtoken"]
        self.topk = design_details["topk"]
        self.select_method = design_details["select_method"]
        if i == self.insert_layer:
            if not self.text_layer:
                # if self.use_cross_attention:
                self.cross_attn = CrossAttention(d_model, n_head)
                # else:
                #     self.cross_attn = nn.MultiheadAttention(d_model, n_head)
                # if self.add_linear :
                #     self.added_linear = nn.Sequential(
                #         nn.Linear(d_model, d_model // 4, bias=False), # .to(dtype=dtype),
                #         nn.ReLU(inplace=True),
                #         nn.Linear(d_model // 4, d_model, bias=False), # .to(dtype=dtype),
                #         nn.ReLU(inplace=True)
                #     )

    def cross_atention(self, q: torch.Tensor, k:torch.Tensor, v:torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=q.dtype, device=k.device) if self.attn_mask is not None else None
        return self.cross_attn(query=q, key=k, value=v)[0]
    
    def cross_attention_promptgen(self, q: torch.Tensor, kv: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=q.dtype, device=q.device) if self.attn_mask is not None else None
        self.cross_attn = self.cross_attn.to(dtype=q.dtype)
        return self.cross_attn(q, kv)
        # pass

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, idx: int, selective_feature:Union[torch.Tensor, NoneType] = None):
        # Will need to append the learnable tokens for this layer here
        # Check if flag was set for this layer or not
        if self.add_prompt:
            # Also see if this is textual transformer layer or not
            if not self.text_layer:
                # Remove the outputs produced by learnable tokens of previous layer
                prefix = x[0:x.shape[0] - self.n_ctx_visual, :, :]
                # Create/configure learnable tokens of this layer
                visual_context = self.VPT_shallow.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                # Add the learnable tokens of this layer with the input, by replacing the previous
                # layer learnable tokens
                if idx == self.insert_layer :   
                    kv = prefix[1:,:,:]
                    if self.select_method == "block_shuffle":
                        selective_feature = selective_feature.unsqueeze(0)
                        # domain_focus_simmap = torch.matmul(kv, selective_feature).unsqueeze(-1)
                        domain_focus_simmap = torch.sum(selective_feature * kv, dim=-1)
                        topk_index = torch.topk(domain_focus_simmap, k=self.topk, dim=0)[1]
                        mask = torch.zeros_like(domain_focus_simmap)
                        mask.scatter_(0, topk_index, 1)
                        kv = mask.unsqueeze(-1) * kv


                    visual_context = self.cross_attention_promptgen(visual_context, kv)
                    
                x = torch.cat([prefix, visual_context], dim=0)
            else:
                # Appending the learnable tokens in different way
                # x -> [77, NCLS, DIM]
                # First remove the learnable tokens from previous layer
                prefix = x[:1, :, :]
                suffix = x[1 + self.n_ctx_text:, :, :]
                # Create/configure learnable tokens of this layer
                textual_context = self.VPT_shallow.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                # Add the learnable tokens of this layer with the input, replaced by previous
                # layer learnable tokens
                x = torch.cat([prefix, textual_context, suffix], dim=0)                
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ResidualAttentionBlock_IVLP(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, add_prompt=False,
                 text_layer=False, i=0, design_details=None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        # Only add learnable tokens if flag is set True
        # For the first iteration i, we should not add the learnable parameters
        # as it is already been taken care of in the very start, for both text
        # and the visual branch
        self.text_layer = text_layer
        self.attn_mask = attn_mask
        if i != 0:
            self.add_prompt = add_prompt
            if self.add_prompt:
                if self.text_layer:
                    self.n_ctx_text = design_details["language_ctx"]  # hyperparameter
                    ctx_vectors = torch.empty(self.n_ctx_text, d_model)
                else:
                    self.n_ctx_visual = design_details["vision_ctx"]  # hyperparameter
                    ctx_vectors = torch.empty(self.n_ctx_visual, d_model)
                # Code snippet for per layer visual prompts
                nn.init.normal_(ctx_vectors, std=0.02)
                self.VPT_shallow = nn.Parameter(ctx_vectors)
        else:
            self.add_prompt = False

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        # Will need to append the learnable tokens for this layer here
        # Check if flag was set for this layer or not
        if self.add_prompt:
            # Also see if this is textual transformer layer or not
            if not self.text_layer:
                # Remove the outputs produced by learnable tokens of previous layer
                prefix = x[0:x.shape[0] - self.n_ctx_visual, :, :]
                # Create/configure learnable tokens of this layer
                visual_context = self.VPT_shallow.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                # Add the learnable tokens of this layer with the input, by replacing the previous
                # layer learnable tokens
                x = torch.cat([prefix, visual_context], dim=0)
            else:
                # Appending the learnable tokens in different way
                # x -> [77, NCLS, DIM]
                # First remove the learnable tokens from previous layer
                prefix = x[:1, :, :]
                suffix = x[1 + self.n_ctx_text:, :, :]
                # Create/configure learnable tokens of this layer
                textual_context = self.VPT_shallow.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                # Add the learnable tokens of this layer with the input, replaced by previous
                # layer learnable tokens
                x = torch.cat([prefix, textual_context, suffix], dim=0)                
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ResidualAttentionBlock_MaPLe(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, design_details=None,
                 text_layer=False, i=0):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        # For the first iteration i, we do not need to add the learnable parameters here
        # as it will be added in the beginning, for both text and the vision branch
        self.text_layer = text_layer
        self.attn_mask = attn_mask
        # This must be consistent with the config file prompt
        self.compound_prompt_nctx = design_details['maple_length']
        if i == 0:
            self.first_layer = True
        else:
            self.first_layer = False

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, inputs):
        # For the first layer, we do not need to add any duplicate, as it is already added
        # as the shallow version
        x = inputs[0]
        compound_prompts_deeper = inputs[1]
        counter = inputs[2]
        if not self.first_layer:
            if len(compound_prompts_deeper) > 0:
                # This means that deeper compound prompts are turned on
                # Here it behaves differently for text and visual side
                # Forward function is same for both

                if not self.text_layer:
                    # First check if the ith layer needs compound prompts or not
                    if not (counter > len(compound_prompts_deeper) - 1):
                        # Remove the outputs produced by learnable tokens of previous layer
                        prefix = x[0:x.shape[0] - self.compound_prompt_nctx, :, :]
                        # Create/configure learnable tokens of this layer
                        visual_context = compound_prompts_deeper[counter]  # extract the correct index
                        visual_context = visual_context.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                        # Add the learnable tokens of this layer with the input, by replacing previous
                        # layer learnable tokens
                        x = torch.cat([prefix, visual_context], dim=0)

                        # Once done, update the counter, so that the next time, it does not use same learnable tokens
                        counter += 1
                else:
                    # First check if the ith layer needs compound prompts or not
                    if not (counter > len(compound_prompts_deeper) - 1):
                        # Appending the learnable tokens in different way
                        # x -> [77, NCLS, DIM]
                        # First remove the learnable tokens from previous layer
                        prefix = x[:1, :, :]
                        suffix = x[1 + self.compound_prompt_nctx:, :, :]
                        # Create/configure learnable tokens of this layer
                        textual_context = compound_prompts_deeper[counter]
                        textual_context = textual_context.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                        # Add the learnable tokens of this layer with the input, replaced by previous
                        # layer learnable tokens
                        x = torch.cat([prefix, textual_context, suffix], dim=0)
                        # Once done, update the counter, so that the next time, it does not use same learnable tokens
                        counter += 1
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return [x, compound_prompts_deeper, counter]  # return again as a list, so that nn.seq can work
    
class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, prompts_needed=0,
                 text_layer=False, design_details=None):
        super().__init__()
        self.width = width
        self.layers = layers
        # Implements respective encoder blocks for a given design choice
        current_trainer = design_details['trainer']
        if current_trainer == 'IVLP' or current_trainer == 'VPT':
            self.resblocks = nn.Sequential(*[ResidualAttentionBlock_IVLP(width, heads, attn_mask, True,
                                                                         text_layer, i,
                                                                         design_details) if prompts_needed > i
                                             else ResidualAttentionBlock_IVLP(width, heads, attn_mask, False,
                                                                              text_layer, i, design_details)
                                             for i in range(layers)])
        elif current_trainer == "IVLP_Local":
            self.resblocks = nn.Sequential(*[ResidualAttentionBlock_IVLP_Local(width, heads, attn_mask, True,
                                                                         text_layer, i,
                                                                         design_details) if prompts_needed > i
                                             else ResidualAttentionBlock_IVLP_Local(width, heads, attn_mask, False,
                                                                              text_layer, i, design_details)
                                             for i in range(layers)])
        elif current_trainer in 'IVLP':
            if current_trainer in 'Local':
                self.resblocks = nn.Sequential(*[ResidualAttentionBlock_IVLP_Local(width, heads, attn_mask, True,
                                                                         text_layer, i,
                                                                         design_details) if prompts_needed > i
                                             else ResidualAttentionBlock_IVLP_Local(width, heads, attn_mask, False,
                                                                              text_layer, i, design_details)
                                             for i in range(layers)])
            else :
                self.resblocks = nn.Sequential(*[ResidualAttentionBlock_IVLP(width, heads, attn_mask, True,
                                                                            text_layer, i,
                                                                            design_details) if prompts_needed > i
                                                else ResidualAttentionBlock_IVLP(width, heads, attn_mask, False,
                                                                                text_layer, i, design_details)
                                             for i in range(layers)])
        elif current_trainer == 'MaPLe':
            self.resblocks = nn.Sequential(
                *[ResidualAttentionBlock_MaPLe(width, heads, attn_mask, design_details, text_layer, i)
                  for i in range(layers)])
        elif "Local" in current_trainer :
            if "VPT_Local" in current_trainer:
                self.resblocks = nn.Sequential(*[ResidualAttentionBlock_IVLP_Local(width, heads, attn_mask, True,
                                                                         text_layer, i,
                                                                         design_details) if prompts_needed > i
                                             else ResidualAttentionBlock_IVLP_Local(width, heads, attn_mask, False,
                                                                              text_layer, i, design_details)
                                             for i in range(layers)])
            elif "VPT_w_NNL_Local" == current_trainer:
                self.resblocks = nn.Sequential(*[ResidualAttentionBlock_IVLP_Local(width, heads, attn_mask, True,
                                                                         text_layer, i,
                                                                         design_details) if prompts_needed > i
                                             else ResidualAttentionBlock_IVLP_Local(width, heads, attn_mask, False,
                                                                              text_layer, i, design_details)
                                             for i in range(layers)])
            elif "VPT_w_NNL_Local_PromptGenerator" == current_trainer:
                self.resblocks = nn.Sequential(*[ResidualAttentionBlock_IVLP_Local(width, heads, attn_mask, True,
                                                                         text_layer, i,
                                                                         design_details) if prompts_needed > i
                                             else ResidualAttentionBlock_IVLP_Local(width, heads, attn_mask, False,
                                                                              text_layer, i, design_details)
                                             for i in range(layers)])
            else: 
                self.resblocks = nn.Sequential(*[ResidualAttentionBlock_Local(width, heads, attn_mask) for _ in range(layers)])
        else:
            # Corresponds to default CoOp or CoCoOp
            assert current_trainer == 'CoOp' or current_trainer == 'CoCoOp'
            self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
        self.current_trainer = current_trainer
        self.patch_selection = False

    def forward(self, x: torch.Tensor):
        if "Local" in self.current_trainer:
            for i in range(self.layers):
                x, q, k, v = self.resblocks[i](x)
            return x, q, k, v
        else:
            return self.resblocks(x)
        
from utils.loss_fn import *   
class Transformer_Prompt(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, prompts_needed=0,
                 text_layer=False, design_details=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.text_layer = text_layer
        # Implements respective encoder blocks for a given design choice
        current_trainer = design_details['trainer']
        if current_trainer == "IVLP_VL_Adapter_Prompt":
            self.resblocks = nn.Sequential(*[ResidualAttentionBlock_IVLP_Prompt(width, heads, attn_mask, True,
                                                                         text_layer, i,
                                                                         design_details) if prompts_needed > i
                                             else ResidualAttentionBlock_IVLP_Prompt(width, heads, attn_mask, False,
                                                                              text_layer, i, design_details)
                                             for i in range(layers)])
        self.independ_learnable_vision = design_details["independent_learnable_vision"]
        if not self.independ_learnable_vision :
            self.n_ctx_visual = design_details["vision_ctx"]  # hyperparameter
            ctx_vectors = torch.empty(self.n_ctx_visual, width)
            nn.init.normal_(ctx_vectors, std=0.02)
            self.VPT_shallow_share = nn.Parameter(ctx_vectors)
        
        self.current_trainer = current_trainer
        self.n_ctx_visual = design_details["vision_ctx"]
        # self.select_layer = design_details["insert_layer"]

    def forward(self, x: torch.Tensor):
        if "Local" in self.current_trainer:
            for i in range(self.layers):
                x, q, k, v = self.resblocks[i](x, i)
            return x, q, k, v
        else:
            for i in range(self.layers):
                if self.independ_learnable_vision:
                    x = self.resblocks[i](x, i)
                else :
                    x = self.resblocks[i](x, i, self.VPT_shallow_share)
            return x

class _UNUSED_Transformer_Prompt_Multiple(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, prompts_needed=0,
                 text_layer=False, design_details=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.text_layer = text_layer
        # Implements respective encoder blocks for a given design choice
        current_trainer = design_details['trainer']
        
        self.independent_cross_attention = design_details["independent_cross_attention"]
        self.independent_learnable_vision = design_details["independent_learnable_vision"]
        # self.n_ctx_visual = design_details["vision_ctx"]  # hyperparameter
        self.current_trainer = current_trainer
        self.n_ctx_visual = design_details["vision_ctx"]
        self.insert_layer = design_details["vision_depth"] - 1

        if not self.text_layer:
            if self.independent_cross_attention:
                for i in range(design_details["vision_depth"]):
                    if i == 0:
                        self.CROSS_ATTN = nn.ParameterList([CrossAttention(self.width, heads)])
                    else :
                        self.CROSS_ATTN.append(CrossAttention(self.width, heads))

            else:
                self.CROSS_ATTN = nn.ParameterList([CrossAttention(self.width, heads)])

            if self.independent_learnable_vision:
                # for i in range(layers):
                ctx_vectors = torch.empty(design_details["vision_depth"], self.n_ctx_visual, width)
                nn.init.normal_(ctx_vectors, std=0.02)
                self.VPT_Deep = nn.Parameter(ctx_vectors)
                    # if i == 0:
                    #     self.VPT_Deep = nn.ParameterList([nn.Parameter(ctx_vectors)])
                    # else :
                    #     self.VPT_Deep.append(nn.Parameter(ctx_vectors))

            else :
                # for i in range(self.insert_layer + 1):
                ctx_vectors = torch.empty(self.n_ctx_visual, width)
                nn.init.normal_(ctx_vectors, std=0.02)
                #     if i == 0:
                #         self.VPT_Deep = nn.ParameterList([nn.Parameter(ctx_vectors)])
                #     else :
                #         self.VPT_Deep.append(nn.Parameter(ctx_vectors))
                self.VPT_Deep = nn.Parameter(ctx_vectors)
        
        if current_trainer == "IVLP_VL_Adapter_Prompt_Multiple":
            # self.resblocks = nn.Sequential(*[ResidualAttentionBlock_IVLP_Prompt_Multiple(width, heads, attn_mask, True,
            #                                                              text_layer, i,
            #                                                              design_details) if prompts_needed > i
            #                                  else ResidualAttentionBlock_IVLP_Prompt_Multiple(width, heads, attn_mask, False,
            #                                                                   text_layer, i, design_details)
            #                                  for i in range(layers)])
            if self.text_layer :
                self.resblocks = nn.Sequential(*[ResidualAttentionBlock_IVLP_Prompt(width, heads, attn_mask, True,
                                                                         text_layer, i,
                                                                         design_details) if prompts_needed > i
                                             else ResidualAttentionBlock_IVLP_Prompt(width, heads, attn_mask, False,
                                                                              text_layer, i, design_details)
                                             for i in range(layers)])
            else :
                self.resblocks = nn.Sequential(
                    *[ResidualAttentionBlock_IVLP_Prompt_Multiple(width, heads, attn_mask, True,
                                                                            text_layer, i,
                                                                            design_details)
                                                                            for i in range(layers)
                                                                            ]
                )
        self.layers = layers
        # self.select_layer = design_details["insert_layer"]
        
    
    # def freeze_VPT_Deep(self):
    #     # for i in range(len(self.CROSS_ATTN)):
    #     #     if i < self.insert_layer:
    #     #         for param in self.CROSS_ATTN[i].parameters():
    #     #             param.requires_grad = False
    #             # for param in self.VPT_Deep[i].parameters():
    #             #     param.requires_grad = False
    #     for i in range(len(self.VPT_Deep)):
    #         if i < self.insert_layer:
    #             self.VPT_Deep[i].requires_grad = False



    def forward(self, x: torch.Tensor):
        if "Local" in self.current_trainer:
            for i in range(self.layers):
                x, q, k, v = self.resblocks[i](x, i)
            return x, q, k, v
        else:
            if self.text_layer:
                for i in range(self.layers):
                    x = self.resblocks[i](x, i)
                return x
            else :
                for i in range(self.layers):
                    x = self.resblocks[i](x, i, self.VPT_Deep, self.CROSS_ATTN)
                return x

from utils.loss_fn import *   
class Transformer_Prompt_Multiple(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, prompts_needed=0,
                 text_layer=False, design_details=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.text_layer = text_layer
        # Implements respective encoder blocks for a given design choice
        current_trainer = design_details['trainer']
        if current_trainer == "IVLP_VL_Adapter_Prompt_Multiple":
            if not self.text_layer:
                self.resblocks = nn.Sequential(*[ResidualAttentionBlock_IVLP_Prompt_Multiple(width, heads, attn_mask, True,
                                                                            text_layer, i,
                                                                            design_details) if prompts_needed > i
                                                else ResidualAttentionBlock_IVLP_Prompt_Multiple(width, heads, attn_mask, False,
                                                                                text_layer, i, design_details)
                                                for i in range(layers)])
            else :
                self.resblocks = nn.Sequential(*[ResidualAttentionBlock_IVLP_Prompt(width, heads, attn_mask, True,
                                                                            text_layer, i,
                                                                            design_details) if prompts_needed > i
                                                else ResidualAttentionBlock_IVLP_Prompt(width, heads, attn_mask, False,
                                                                                text_layer, i, design_details)
                                                for i in range(layers)])
        
        self.current_trainer = current_trainer
        self.n_ctx_visual = design_details["vision_ctx"]
        # self.select_layer = design_details["insert_layer"]

    def forward(self, x: torch.Tensor):
        if "Local" in self.current_trainer:
            for i in range(self.layers):
                x, q, k, v = self.resblocks[i](x, i)
            return x, q, k, v
        else:
            for i in range(self.layers):
                x = self.resblocks[i](x, i)
            return x
    
class ResidualAttentionBlock_IVLP_Prompt_Multiple(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, add_prompt=False,
                 text_layer=False, i=0, design_details=None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        # Only add learnable tokens if flag is set True
        # For the first iteration i, we should not add the learnable parameters
        # as it is already been taken care of in the very start, for both text
        # and the visual branch
        self.text_layer = text_layer
        self.attn_mask = attn_mask
        if i != 0:
            self.add_prompt = add_prompt
            if self.add_prompt:
                if self.text_layer:
                    self.n_ctx_text = design_details["language_ctx"]  # hyperparameter
                    ctx_vectors = torch.empty(self.n_ctx_text, d_model)
                else:
                    self.n_ctx_visual = design_details["vision_ctx"]  # hyperparameter
                    ctx_vectors = torch.empty(self.n_ctx_visual, d_model)
                # Code snippet for per layer visual prompts
                nn.init.normal_(ctx_vectors, std=0.02)
                self.VPT_shallow = nn.Parameter(ctx_vectors)
        else:
            self.add_prompt = False
        self.insert_layer = design_details["vision_depth"] - 1
        self.use_classtoken = design_details["use_classtoken"]
        # self.use_cross_attention = design_details["use_cross_attention"]
        if i <= self.insert_layer:
            if not self.text_layer:
                self.cross_attn = CrossAttention(d_model, n_head)

    def cross_atention(self, q: torch.Tensor, k:torch.Tensor, v:torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=q.dtype, device=k.device) if self.attn_mask is not None else None
        return self.cross_attn(query=q, key=k, value=v)[0]
    
    def cross_attention_promptgen(self, q: torch.Tensor, kv: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=q.dtype, device=q.device) if self.attn_mask is not None else None
        self.cross_attn = self.cross_attn.to(dtype=q.dtype)
        return self.cross_attn(q, kv)
        # pass

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, idx: int):
        # Will need to append the learnable tokens for this layer here
        # Check if flag was set for this layer or not
        if self.add_prompt:
            # Also see if this is textual transformer layer or not
            if not self.text_layer:
                # Remove the outputs produced by learnable tokens of previous layer
                prefix = x[0:x.shape[0] - self.n_ctx_visual, :, :]
                # Create/configure learnable tokens of this layer
                visual_context = self.VPT_shallow.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                # Add the learnable tokens of this layer with the input, by replacing the previous
                # layer learnable tokens
                if idx <= self.insert_layer :
                    if self.use_classtoken:
                        kv = prefix[0,:,:].unsqueeze(0)
                    else:   
                        kv = prefix[1:,:,:]
                    visual_context = self.cross_attention_promptgen(visual_context, kv)
                x = torch.cat([prefix, visual_context], dim=0)
            else:
                # Appending the learnable tokens in different way
                # x -> [77, NCLS, DIM]
                # First remove the learnable tokens from previous layer
                prefix = x[:1, :, :]
                suffix = x[1 + self.n_ctx_text:, :, :]
                # Create/configure learnable tokens of this layer
                textual_context = self.VPT_shallow.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                # Add the learnable tokens of this layer with the input, replaced by previous
                # layer learnable tokens
                x = torch.cat([prefix, textual_context, suffix], dim=0)                
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
                

class _UNUSED_ResidualAttentionBlock_IVLP_Prompt_Multiple(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, add_prompt=False,
                 text_layer=False, i=0, design_details=None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        # Only add learnable tokens if flag is set True
        # For the first iteration i, we should not add the learnable parameters
        # as it is already been taken care of in the very start, for both text
        # and the visual branch
        self.text_layer = text_layer
        self.attn_mask = attn_mask
        self.n_ctx_visual = design_details["vision_ctx"]
        # if i != 0:
        #     self.add_prompt = add_prompt
        #     if self.add_prompt:
        #         if self.text_layer:
        #             self.n_ctx_text = design_details["language_ctx"]  # hyperparameter
        #             ctx_vectors = torch.empty(self.n_ctx_text, d_model)
        #         else:
        #             self.n_ctx_visual = design_details["vision_ctx"]  # hyperparameter
        #             ctx_vectors = torch.empty(self.n_ctx_visual, d_model)
        #         # Code snippet for per layer visual prompts
        #         nn.init.normal_(ctx_vectors, std=0.02)
        #         self.VPT_shallow = nn.Parameter(ctx_vectors)
        # else:
        #     self.add_prompt = False
        self.independent_cross_attention = design_details["independent_cross_attention"]
        self.independent_learnable_vision = design_details["independent_learnable_vision"]
        self.insert_layer = design_details["vision_depth"] - 1
        self.use_classtoken = design_details["use_classtoken"]
        # self.use_cross_attention = design_details["use_cross_attention"]
        # if i == self.insert_layer:
        #     if not self.text_layer:
        #         self.cross_attn = CrossAttention(d_model, n_head)

    # def cross_atention(self, q: torch.Tensor, k:torch.Tensor, v:torch.Tensor):
    #     self.attn_mask = self.attn_mask.to(dtype=q.dtype, device=k.device) if self.attn_mask is not None else None
    #     return self.cross_attn(query=q, key=k, value=v)[0]
    
    # def cross_attention_promptgen(self, q: torch.Tensor, kv: torch.Tensor):
    #     self.attn_mask = self.attn_mask.to(dtype=q.dtype, device=q.device) if self.attn_mask is not None else None
    #     self.cross_attn = self.cross_attn.to(dtype=q.dtype)
    #     return self.cross_attn(q, kv)
    #     # pass

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, idx: int, vpt_deep, cross_attn):
        # Will need to append the learnable tokens for this layer here
        # Check if flag was set for this layer or not
        # Also see if this is textual transformer layer or not
        if not self.text_layer:
            if idx <= self.insert_layer:
                if self.independent_learnable_vision:
                    visual_context = vpt_deep[idx].expand(x.shape[1], -1, -1).permute(1, 0, 2).to(dtype=x.dtype)
                else:
                    visual_context = vpt_deep.expand(x.shape[1], -1, -1).permute(1, 0, 2).to(dtype=x.dtype)
                prefix = x[0:x.shape[0] - self.n_ctx_visual, :, :]
                # elif idx >= self.insert_layer:
                if self.use_classtoken:
                    kv = prefix[0,:,:].unsqueeze(0)
                else:   
                    kv = prefix[1:,:,:]
                if self. independent_cross_attention:
                    cross_attn = cross_attn.to(dtype=kv.dtype)
                    visual_context = cross_attn[idx](visual_context, kv)
                else :
                    cross_attn = cross_attn.to(dtype=kv.dtype)
                    visual_context = cross_attn[0](visual_context, kv)
                
                x = torch.cat([prefix, visual_context], dim=0)
            # else:
            else :
                pass
            
            #     # Appending the learnable tokens in different way
            #     # x -> [77, NCLS, DIM]
            #     # First remove the learnable tokens from previous layer
            #     prefix = x[:1, :, :]
            #     suffix = x[1 + self.n_ctx_text:, :, :]
            #     # Create/configure learnable tokens of this layer
            #     textual_context = self.VPT_shallow.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
            #     # Add the learnable tokens of this layer with the input, replaced by previous
            #     # layer learnable tokens
            #     x = torch.cat([prefix, textual_context, suffix], dim=0)                
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class _UNUSED_Transformer_Prompt_SelectPatch(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, prompts_needed=0,
                 text_layer=False, design_details=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.text_layer = text_layer
        # Implements respective encoder blocks for a given design choice
        current_trainer = design_details['trainer']
        if current_trainer == "IVLP_VL_Adapter_Prompt_SelectPatch":
            if self.text_layer:
                self.resblocks = nn.Sequential(*[ResidualAttentionBlock_IVLP_Prompt(width, heads, attn_mask, True,
                                                                         text_layer, i,
                                                                         design_details) if prompts_needed > i
                                             else ResidualAttentionBlock_IVLP_Prompt(width, heads, attn_mask, False,
                                                                              text_layer, i, design_details)
                                             for i in range(layers)])
            else :
                self.resblocks = nn.Sequential(*[ResidualAttentionBlock_IVLP_Prompt_SelectPatch(width, heads, attn_mask, True,
                                                                            text_layer, i,
                                                                            design_details) if prompts_needed > i
                                                else ResidualAttentionBlock_IVLP_Prompt_SelectPatch(width, heads, attn_mask, False,
                                                                                text_layer, i, design_details)
                                                for i in range(layers)])
        
        self.current_trainer = current_trainer
        self.n_ctx_visual = design_details["vision_ctx"]
        self.insert_layer = design_details["vision_depth"] - 1
        # self.select_layer = design_details["insert_layer"]

    def forward(self, x: torch.Tensor, x_block_shuffled: Union[torch.Tensor, NoneType] = None):
        
        if self.text_layer:
            for i in range(self.layers):
                x = self.resblocks[i](x, i)
            return x
        else :
            for i in range(self.layers):
                if i < self.insert_layer :
                    x = self.resblocks[i](x, i)
                    x_block_shuffled = self.resblocks[i](x_block_shuffled, i)
                elif i == self.insert_layer :
                    x = self.resblocks[i](x, i, x_block_shuffled[0])
                elif i > self.insert_layer :
                    x = self.resblocks[i](x, i)
            # for i in range(self.layers):
            #     x_block_shuffled = self.resblocks[i](x_block_shuffled, i)
            # for i in range(self.layers):
            #     if i < self.insert_layer :
            #         x = self.resblocks[i](x, i)
            #     elif i == self.insert_layer:
            #         x = self.resblocks[i](x, i, x_block_shuffled)
            #     elif i > self.insert_layer :
            #         x = self.resblocks[i](x, i)
            return x
            
                



# from utils.loss_fn import *   
class _UNUSED_Transformer_SelectPatch(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, prompts_needed=0,
                 text_layer=False, design_details=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.text_layer = text_layer
        # Implements respective encoder blocks for a given design choice
        current_trainer = design_details['trainer']
        # if current_trainer == 'IVLP' or current_trainer == 'VPT':
        #     self.resblocks = nn.Sequential(*[ResidualAttentionBlock_IVLP(width, heads, attn_mask, True,
        #                                                                  text_layer, i,
        #                                                                  design_details) if prompts_needed > i
        #                                      else ResidualAttentionBlock_IVLP(width, heads, attn_mask, False,
        #                                                                       text_layer, i, design_details)
        #                                      for i in range(layers)])
        if current_trainer == "IVLP_VL_Adapter_Local_SelectPatch":
            if self.text_layer:
                self.resblocks = nn.Sequential(*[ResidualAttentionBlock_IVLP_Prompt_SelectPatch(width, heads, attn_mask, True,
                                                                         text_layer, i,
                                                                         design_details) if prompts_needed > i
                                             else ResidualAttentionBlock_IVLP_Prompt_SelectPatch(width, heads, attn_mask, False,
                                                                              text_layer, i, design_details)
                                             for i in range(layers)])
            else :
                self.resblocks = nn.Sequential(*[ResidualAttentionBlock_IVLP_Prompt_SelectPatch(width, heads, attn_mask, True,
                                                                            text_layer, i,
                                                                            design_details) if prompts_needed > i
                                                else ResidualAttentionBlock_IVLP_Prompt_SelectPatch(width, heads, attn_mask, False,
                                                                                text_layer, i, design_details)
                                                for i in range(layers)])
        
        self.current_trainer = current_trainer
        self.n_ctx_visual = design_details["vision_ctx"]
        self.topk = design_details["topk"]
        self.proj = torch.load("/home/gotoyuta/lab/Domain-Forgetting/proj-vit-b16.pt", weights_only=True) # FIXME
        self.proj.requires_grad = False
        self.patch_selection = True
        self.select_method = design_details["select_method"]
        self.select_layer = design_details["select_layer"]

    def forward(self, x: torch.Tensor, selective_feature: Union[torch.Tensor, NoneType] = None):
        if not self.training:
            if self.text_layer :
                if "Local" in self.current_trainer:
                    for i in range(self.layers):
                        x, q, k, v = self.resblocks[i](x)
                    return x, q, k, v
                else:
                    return self.resblocks(x)
            else :
                if "Local" in self.current_trainer:
                    for i in range(self.layers):
                        x, q, k, v = self.resblocks[i](x)
                    return x, q, k, v, None, None
                else:
                    return self.resblocks(x)
        else:
            if self.text_layer :
                if "Local" in self.current_trainer:
                    for i in range(self.layers):
                        x, q, k, v = self.resblocks[i](x)
                    return x, q, k, v
                else:
                    return self.resblocks(x)
            else:
                if "Local" in self.current_trainer:
                    if self.patch_selection:
                        for i in range(self.layers):
                            if i < self.select_layer - 1:
                                x, q, k, v = self.resblocks[i](x)
                            elif i == self.select_layer - 1:
                                x, q, k, v = self.resblocks[i](x)

                                ##########value featrureの準備########
                                v_proj = v @ self.proj
                                normalized_v = v_proj / v_proj.norm(dim=-1, keepdim=True)
                                normalized_v = normalized_v.permute(1, 0, 2) # LND -> NLD

                                if self.select_method == "entropy":
                                    local_logits = normalized_v @ selective_feature.t()
                                    local_entropy = get_entropy_local(local_logits[:, 1 + self.n_ctx_visual:])
                                    topk_index = torch.topk(local_entropy, k=self.topk, dim=1)[1]
                                    mask = torch.zeros_like(local_entropy)
                                elif self.select_method == "entropy_distill":
                                    local_logits = normalized_v @ selective_feature.t()
                                    local_entropy = get_entropy_local(local_logits[:, 1 + self.n_ctx_visual:])
                                    topk_index = torch.topk(local_entropy, k=self.topk, dim=1)[1]
                                    mask = torch.zeros_like(local_entropy)
                                elif self.select_method == "block_shuffle_distill":
                                    domain_specific_features = selective_feature.unsqueeze(1)
                                    domain_focus_simmap = torch.matmul(normalized_v[:, 1 + self.n_ctx_visual:,:], domain_specific_features.transpose(-1, -2)).squeeze(-1)
                                    topk_index = torch.topk(domain_focus_simmap, k=self.topk, dim=1)[1]
                                    mask = torch.zeros_like(domain_focus_simmap)
                                # if self.input_type == "image":
                                mask.scatter_(1, topk_index, 1)
                                margine_context_mask = torch.ones((mask.shape[0], 1 + self.n_ctx_visual)).cuda()
                                mask = torch.cat((margine_context_mask, mask), dim = 1).half()
                                x_masked = x.permute(1, 0, 2) * mask.unsqueeze(-1)
                                x_masked = x_masked.permute(1, 0, 2)

                                # return x, q, k, v, x_masked, v_masked
                                # return x, q, k, v, x_masked.permute(1, 0, 2)
                                #################################
                                # patch select function
                                ##################################
                            else :
                                x, q, k, v = self.resblocks[i](x)
                                x_masked, _, _, v_masked = self.resblocks[i](x_masked)
                        return x, q, k, v, x_masked, v_masked
                    else :
                        for i in range(self.layers):
                            x, q, k, v = self.resblocks[i](x)
                        return x, q, k, v
                else:
                    return self.resblocks(x)


class Transformer_SelectPatch_FullMask(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, prompts_needed=0,
                 text_layer=False, design_details=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.text_layer = text_layer
        # Implements respective encoder blocks for a given design choice
        current_trainer = design_details['trainer']
        # if current_trainer == 'IVLP' or current_trainer == 'VPT':
        #     self.resblocks = nn.Sequential(*[ResidualAttentionBlock_IVLP(width, heads, attn_mask, True,
        #                                                                  text_layer, i,
        #                                                                  design_details) if prompts_needed > i
        #                                      else ResidualAttentionBlock_IVLP(width, heads, attn_mask, False,
        #                                                                       text_layer, i, design_details)
        #                                      for i in range(layers)])
        if current_trainer == "IVLP_VL_Adapter_Local_SelectPatch":
            self.resblocks = nn.Sequential(*[ResidualAttentionBlock_IVLP_Local(width, heads, attn_mask, True,
                                                                         text_layer, i,
                                                                         design_details) if prompts_needed > i
                                             else ResidualAttentionBlock_IVLP_Local(width, heads, attn_mask, False,
                                                                              text_layer, i, design_details)
                                             for i in range(layers)])
        elif current_trainer == "IVLP_VL_Adapter_Local_SelectPatch_FullMask":
            self.resblocks = nn.Sequential(*[ResidualAttentionBlock_IVLP_Local(width, heads, attn_mask, True,
                                                                         text_layer, i,
                                                                         design_details) if prompts_needed > i
                                             else ResidualAttentionBlock_IVLP_Local(width, heads, attn_mask, False,
                                                                              text_layer, i, design_details)
                                             for i in range(layers)])

        
        self.current_trainer = current_trainer
        self.n_ctx_visual = design_details["vision_ctx"]
        self.topk = design_details["topk"]
        self.proj = torch.load("/home/gotoyuta/lab/Domain-Forgetting/proj-vit-b16.pt", weights_only=True) # FIXME
        self.proj.requires_grad = False
        self.patch_selection = True
        self.select_method = design_details["select_method"]
        self.select_layer = design_details["select_layer"]

    def forward(self, x: torch.Tensor, selective_feature: Union[torch.Tensor, NoneType] = None):
        if not self.training:
            if self.text_layer :
                if "Local" in self.current_trainer:
                    for i in range(self.layers):
                        x, q, k, v = self.resblocks[i](x)
                    return x, q, k, v
                else:
                    return self.resblocks(x)
            else :
                if "Local" in self.current_trainer:
                    for i in range(self.layers):
                        x, q, k, v = self.resblocks[i](x)
                    return x, q, k, v, None, None
                else:
                    return self.resblocks(x)
        else:
            if self.text_layer :
                if "Local" in self.current_trainer:
                    for i in range(self.layers):
                        x, q, k, v = self.resblocks[i](x)
                    return x, q, k, v
                else:
                    return self.resblocks(x)
            else:
                if "Local" in self.current_trainer:
                    if self.patch_selection:
                        for i in range(self.layers):
                            if i < self.select_layer - 1:
                                x, q, k, v = self.resblocks[i](x)
                            elif i == self.select_layer - 1:
                                x, q, k, v = self.resblocks[i](x)

                                ##########value featrureの準備########
                                v_proj = v @ self.proj
                                normalized_v = v_proj / v_proj.norm(dim=-1, keepdim=True)
                                normalized_v = normalized_v.permute(1, 0, 2) # LND -> NLD

                                if self.select_method == "entropy":
                                    local_logits = normalized_v @ selective_feature.t()
                                    local_entropy = get_entropy_local(local_logits[:, 1 + self.n_ctx_visual:])
                                    topk_index = torch.topk(local_entropy, k=self.topk, dim=1)[1]
                                    mask = torch.zeros_like(local_entropy)
                                elif self.select_method == "entropy_distill":
                                    local_logits = normalized_v @ selective_feature.t()
                                    local_entropy = get_entropy_local(local_logits[:, 1 + self.n_ctx_visual:])
                                    topk_index = torch.topk(local_entropy, k=self.topk, dim=1)[1]
                                    mask = torch.zeros_like(local_entropy)
                                elif self.select_method == "block_shuffle_distill":
                                    domain_specific_features = selective_feature.unsqueeze(1)
                                    domain_focus_simmap = torch.matmul(normalized_v[:, 1 + self.n_ctx_visual:,:], domain_specific_features.transpose(-1, -2)).squeeze(-1)
                                    topk_index = torch.topk(domain_focus_simmap, k=self.topk, dim=1)[1]
                                    mask = torch.zeros_like(domain_focus_simmap)
                                # if self.input_type == "image":
                                mask.scatter_(1, topk_index, 1)
                                margine_context_mask = torch.ones((mask.shape[0], 1 + self.n_ctx_visual)).cuda()
                                mask = torch.cat((margine_context_mask, mask), dim = 1).half()
                                x_masked = x.permute(1, 0, 2) * mask.unsqueeze(-1)
                                x_masked = x_masked.permute(1, 0, 2)

                                # return x, q, k, v, x_masked, v_masked
                                # return x, q, k, v, x_masked.permute(1, 0, 2)
                                #################################
                                # patch select function
                                ##################################
                            else :
                                x_masked = x.permute(1, 0, 2) * mask.unsqueeze(-1)
                                # print(x_masked[0,self.n_ctx_visual+1:, 0])
                                x_masked = x_masked.permute(1, 0, 2)
                                x, q, k, v = self.resblocks[i](x)
                                x_masked, _, _, v_masked = self.resblocks[i](x_masked)
                        return x, q, k, v, x_masked, v_masked
                    else :
                        for i in range(self.layers):
                            x, q, k, v = self.resblocks[i](x)
                        return x, q, k, v
                else:
                    return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int,
                 output_dim: int, design_details):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        if design_details["vision_depth"] == 0:
            self.VPT_shallow = False
        else:
            self.VPT_shallow = True
        if self.VPT_shallow:
            # Add visual prompt tokens here
            n_ctx = design_details["vision_ctx"]  # hyperparameter
            ctx_vectors = torch.empty(n_ctx, width)
            nn.init.normal_(ctx_vectors, std=0.02)
            self.VPT = nn.Parameter(ctx_vectors)
            # self.VPT.half()
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        # hyper-parameter if need to add prompt embeddings inside to the input
        # of transformer block or not:
        self.prompt_till_layer_visual = design_details["vision_depth"]
        self.transformer = Transformer(width, layers, heads, prompts_needed=self.prompt_till_layer_visual,
                                       design_details=design_details)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                                                            device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        # After positional embeddings, we will attach prompts with the model, remember only those
        # are trainable parameters here in whole image encoder.
        if self.VPT_shallow:
            visual_ctx = self.VPT.expand(x.shape[0], -1, -1).half()
            x = torch.cat([x, visual_ctx], dim=1)
        else:
            assert self.prompt_till_layer_visual == 0

        # Normal code as before
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class VisionTransformer_Prompt(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int,
                 output_dim: int, design_details):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        if design_details["vision_depth"] == 0:
            self.VPT_shallow = False
        else:
            self.VPT_shallow = True
        if self.VPT_shallow:
            # Add visual prompt tokens here
            n_ctx = design_details["vision_ctx"]  # hyperparameter
            ctx_vectors = torch.empty(n_ctx, width)
            nn.init.normal_(ctx_vectors, std=0.02)
            self.VPT = nn.Parameter(ctx_vectors)
            # self.VPT.half()
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        # hyper-parameter if need to add prompt embeddings inside to the input
        # of transformer block or not:
        self.prompt_till_layer_visual = design_details["vision_depth"]
        self.transformer = Transformer_Prompt(width, layers, heads, prompts_needed=self.prompt_till_layer_visual,
                                       design_details=design_details)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                                                            device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        # After positional embeddings, we will attach prompts with the model, remember only those
        # are trainable parameters here in whole image encoder.
        if self.VPT_shallow:
            visual_ctx = self.VPT.expand(x.shape[0], -1, -1).half()
            x = torch.cat([x, visual_ctx], dim=1)
        else:
            assert self.prompt_till_layer_visual == 0

        # Normal code as before
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x

class VisionTransformer_Prompt_Multiple(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int,
                 output_dim: int, design_details):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        if design_details["vision_depth"] == 0:
            self.VPT_shallow = False
        else:
            self.VPT_shallow = True
        if self.VPT_shallow:
            # Add visual prompt tokens here
            n_ctx = design_details["vision_ctx"]  # hyperparameter
            ctx_vectors = torch.empty(n_ctx, width)
            nn.init.normal_(ctx_vectors, std=0.02)
            self.VPT = nn.Parameter(ctx_vectors)
            # self.VPT.half()
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        # hyper-parameter if need to add prompt embeddings inside to the input
        # of transformer block or not:
        self.prompt_till_layer_visual = design_details["vision_depth"]
        self.transformer = Transformer_Prompt_Multiple(width, layers, heads, prompts_needed=self.prompt_till_layer_visual,
                                       design_details=design_details)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                                                            device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        # After positional embeddings, we will attach prompts with the model, remember only those
        # are trainable parameters here in whole image encoder.
        if self.VPT_shallow:
            visual_ctx = self.VPT.expand(x.shape[0], -1, -1).half()
            x = torch.cat([x, visual_ctx], dim=1)
        else:
            assert self.prompt_till_layer_visual == 0

        # Normal code as before
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x

class VisionTransformer_Prompt_SelectPatch(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int,
                 output_dim: int, design_details):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        if design_details["vision_depth"] == 0:
            self.VPT_shallow = False
        else:
            self.VPT_shallow = True
        if self.VPT_shallow:
            # Add visual prompt tokens here
            n_ctx = design_details["vision_ctx"]  # hyperparameter
            ctx_vectors = torch.empty(n_ctx, width)
            nn.init.normal_(ctx_vectors, std=0.02)
            self.VPT = nn.Parameter(ctx_vectors)
            # self.VPT.half()
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        # hyper-parameter if need to add prompt embeddings inside to the input
        # of transformer block or not:
        self.prompt_till_layer_visual = design_details["vision_depth"]
        self.transformer = Transformer_Prompt_SelectPatch(width, layers, heads, prompts_needed=self.prompt_till_layer_visual,
                                       design_details=design_details)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, x_block_shuffle: torch.Tensor):
        
        x = self.preprocess(x)
        x_block_shuffle = self.preprocess(x_block_shuffle)
        # Normal code as before
        x = self.ln_pre(x)
        x_block_shuffle = self.ln_pre(x_block_shuffle)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x_block_shuffle = x_block_shuffle.permute(1, 0, 2) # NLD -> LND

        x = self.transformer(x, x_block_shuffle)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x

    def preprocess(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                                                            device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        # After positional embeddings, we will attach prompts with the model, remember only those
        # are trainable parameters here in whole image encoder.
        if self.VPT_shallow:
            visual_ctx = self.VPT.expand(x.shape[0], -1, -1).half()
            x = torch.cat([x, visual_ctx], dim=1)
        else:
            assert self.prompt_till_layer_visual == 0
        return x


class VisionTransformer_MaPLe(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 design_details):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        self.VPT_shallow = True
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        # hyper-parameter if need to add prompt embeddings inside to the input
        # of transformer block or not:
        self.prompt_till_layer_visual = 0
        self.transformer = Transformer(width, layers, heads, design_details=design_details)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, shared_ctx, compound_deeper_prompts):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        # After positional embeddings, we will attach prompts with the model, remember only those
        # are trainable parameters here in whole image encoder.
        if self.VPT_shallow:
            visual_ctx = shared_ctx.expand(x.shape[0], -1, -1).half()
            x = torch.cat([x, visual_ctx], dim=1)
        else:
            assert self.prompt_till_layer_visual == 0

        # Normal code as before
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        # Again combine the inputs, so nn.sequential can work
        outputs = self.transformer([x, compound_deeper_prompts, 0])  # third argument is counter
        x = outputs[0]
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x

class VisionTransformer_Local(nn.Module):
    def __init__(
        self,
        input_resolution: int,
        patch_size: int, width: int,
        layers: int,
        heads: int,
        output_dim: int,
        return_local_features: bool = True,
        design_details: bool = False
    ) -> NoneType:
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        if design_details["vision_depth"] == 0:
            self.VPT_shallow = False
        else:
            self.VPT_shallow = True
        if self.VPT_shallow:
            # Add visual prompt tokens here
            n_ctx = design_details["vision_ctx"]  # hyperparameter
            ctx_vectors = torch.empty(n_ctx, width)
            nn.init.normal_(ctx_vectors, std=0.02)
            self.VPT = nn.Parameter(ctx_vectors)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        # self.transformer = Transformer(width, layers, heads, design_details=design_details)
        self.prompt_till_layer_visual = design_details["vision_depth"]
        self.transformer = Transformer(width, layers, heads, prompts_needed=self.prompt_till_layer_visual,
                                       design_details=design_details)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        self.return_local_features = return_local_features

    @staticmethod
    def get_pixel_from_patch_idx(patch_idx: int, patch_size: int, image_size: int) -> Tuple[int, int]:
        """this function takes a patch idx and returns the list of coordinates of pixels that are in the patch as a list"""
        coordinates = []
        for i in range(patch_size):
            for j in range(patch_size):
                x = (patch_idx // (image_size // patch_size)) * patch_size + i
                y = (patch_idx % (image_size // patch_size)) * patch_size + j
                coordinates.append((x, y))
        return coordinates

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        if self.VPT_shallow:
            visual_ctx = self.VPT.expand(x.shape[0], -1, -1).half()
            x = torch.cat([x, visual_ctx], dim=1)
        else:
            assert self.prompt_till_layer_visual == 0

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x, q, k, v = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        B, _, C = x[:, 1:].shape
        x = self.ln_post(x[:, 0, :])

        if self.return_local_features:
            v = v.permute(1, 0, 2)
            v = self.ln_post(v)
            v = v[:, 1:]
            v = v.reshape(B, -1, C).contiguous()
        if self.proj is not None:
            x = x @ self.proj
            v = v @ self.proj

        return x, v

class VisionTransformer_Local_SelectPatch(nn.Module):
    def __init__(
        self,
        input_resolution: int,
        patch_size: int, width: int,
        layers: int,
        heads: int,
        output_dim: int,
        return_local_features: bool = True,
        design_details: bool = False
    ) -> NoneType:
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        if design_details["vision_depth"] == 0:
            self.VPT_shallow = False
        else:
            self.VPT_shallow = True
        if self.VPT_shallow:
            # Add visual prompt tokens here
            n_ctx = design_details["vision_ctx"]  # hyperparameter
            ctx_vectors = torch.empty(n_ctx, width)
            nn.init.normal_(ctx_vectors, std=0.02)
            self.VPT = nn.Parameter(ctx_vectors)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        # self.transformer = Transformer(width, layers, heads, design_details=design_details)
        self.prompt_till_layer_visual = design_details["vision_depth"]
        self.transformer = Transformer_SelectPatch(width, layers, heads, prompts_needed=self.prompt_till_layer_visual,
                                       design_details=design_details)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        self.return_local_features = return_local_features

    @staticmethod
    def get_pixel_from_patch_idx(patch_idx: int, patch_size: int, image_size: int) -> Tuple[int, int]:
        """this function takes a patch idx and returns the list of coordinates of pixels that are in the patch as a list"""
        coordinates = []
        for i in range(patch_size):
            for j in range(patch_size):
                x = (patch_idx // (image_size // patch_size)) * patch_size + i
                y = (patch_idx % (image_size // patch_size)) * patch_size + j
                coordinates.append((x, y))
        return coordinates

    def forward(self, x: torch.Tensor, select_features: Union[torch.Tensor, NoneType]=None) -> torch.Tensor:
        torch.save(self.ln_post, "./ln_post_weight.pt")
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        if self.VPT_shallow:
            visual_ctx = self.VPT.expand(x.shape[0], -1, -1).half()
            x = torch.cat([x, visual_ctx], dim=1)
        else:
            assert self.prompt_till_layer_visual == 0

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x, q, k, v, x_masked, v_masked = self.transformer(x, select_features)
        x = x.permute(1, 0, 2)  # LND -> NLD
        B, _, C = x[:, 1:].shape
        x = self.ln_post(x[:, 0, :])
        if self.training:
            x_masked = x_masked.permute(1, 0, 2)  # LND -> NLD
            x_masked = self.ln_post(x_masked[:, 0, :])

        if self.return_local_features:
            v = v.permute(1, 0, 2)
            v = self.ln_post(v)
            v = v[:, 1:]
            v = v.reshape(B, -1, C).contiguous()
            if self.training :
                v_masked = v_masked.permute(1, 0, 2)
                v_masked = self.ln_post(v_masked)
                v_masked = v_masked[:, 1:]
                v_masked = v_masked.reshape(B, -1, C).contiguous()
        if self.proj is not None:
            x = x @ self.proj
            v = v @ self.proj
            if self.training :
                x_masked = x_masked @ self.proj
                v_masked = v_masked @ self.proj
        if self.training:
            return x, v, x_masked, v_masked
        else :
            return x, v, None, None

class VisionTransformer_Local_SelectPatch_FullMask(nn.Module):
    def __init__(
        self,
        input_resolution: int,
        patch_size: int, width: int,
        layers: int,
        heads: int,
        output_dim: int,
        return_local_features: bool = True,
        design_details: bool = False
    ) -> NoneType:
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        if design_details["vision_depth"] == 0:
            self.VPT_shallow = False
        else:
            self.VPT_shallow = True
        if self.VPT_shallow:
            # Add visual prompt tokens here
            n_ctx = design_details["vision_ctx"]  # hyperparameter
            ctx_vectors = torch.empty(n_ctx, width)
            nn.init.normal_(ctx_vectors, std=0.02)
            self.VPT = nn.Parameter(ctx_vectors)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        # self.transformer = Transformer(width, layers, heads, design_details=design_details)
        self.prompt_till_layer_visual = design_details["vision_depth"]
        self.transformer = Transformer_SelectPatch_FullMask(width, layers, heads, prompts_needed=self.prompt_till_layer_visual,
                                       design_details=design_details)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        self.return_local_features = return_local_features

    @staticmethod
    def get_pixel_from_patch_idx(patch_idx: int, patch_size: int, image_size: int) -> Tuple[int, int]:
        """this function takes a patch idx and returns the list of coordinates of pixels that are in the patch as a list"""
        coordinates = []
        for i in range(patch_size):
            for j in range(patch_size):
                x = (patch_idx // (image_size // patch_size)) * patch_size + i
                y = (patch_idx % (image_size // patch_size)) * patch_size + j
                coordinates.append((x, y))
        return coordinates

    def forward(self, x: torch.Tensor, select_features: Union[torch.Tensor, NoneType]=None) -> torch.Tensor:
        torch.save(self.ln_post, "./ln_post_weight.pt")
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        if self.VPT_shallow:
            visual_ctx = self.VPT.expand(x.shape[0], -1, -1).half()
            x = torch.cat([x, visual_ctx], dim=1)
        else:
            assert self.prompt_till_layer_visual == 0

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x, q, k, v, x_masked, v_masked = self.transformer(x, select_features)
        x = x.permute(1, 0, 2)  # LND -> NLD
        B, _, C = x[:, 1:].shape
        x = self.ln_post(x[:, 0, :])
        if self.training:
            x_masked = x_masked.permute(1, 0, 2)  # LND -> NLD
            x_masked = self.ln_post(x_masked[:, 0, :])

        if self.return_local_features:
            v = v.permute(1, 0, 2)
            v = self.ln_post(v)
            v = v[:, 1:]
            v = v.reshape(B, -1, C).contiguous()
            if self.training :
                v_masked = v_masked.permute(1, 0, 2)
                v_masked = self.ln_post(v_masked)
                v_masked = v_masked[:, 1:]
                v_masked = v_masked.reshape(B, -1, C).contiguous()
        if self.proj is not None:
            x = x @ self.proj
            v = v @ self.proj
            if self.training :
                x_masked = x_masked @ self.proj
                v_masked = v_masked @ self.proj
        if self.training:
            return x, v, x_masked, v_masked
        else :
            return x, v, None, None

class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 design_details
                 ):
        super().__init__()

        self.context_length = context_length
        trainer = design_details['trainer']
        self.trainer = trainer
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            if trainer == "MaPLe":
                self.visual = VisionTransformer_MaPLe(
                    input_resolution=image_resolution,
                    patch_size=vision_patch_size,
                    width=vision_width,
                    layers=vision_layers,
                    heads=vision_heads,
                    output_dim=embed_dim,
                    design_details=design_details
                )
            elif trainer == "IVLP_VL_Adapter_Local_SelectPatch":
                self.visual = VisionTransformer_Local_SelectPatch(
                    input_resolution=image_resolution,
                    patch_size=vision_patch_size,
                    width=vision_width,
                    layers=vision_layers,
                    heads=vision_heads,
                    output_dim=embed_dim,
                    design_details=design_details
                )
            elif trainer == "IVLP_VL_Adapter_Local_SelectPatch_FullMask":
                self.visual = VisionTransformer_Local_SelectPatch_FullMask(
                    input_resolution=image_resolution,
                    patch_size=vision_patch_size,
                    width=vision_width,
                    layers=vision_layers,
                    heads=vision_heads,
                    output_dim=embed_dim,
                    design_details=design_details
                )
            elif "Local" in trainer:
                self.visual = VisionTransformer_Local(
                    input_resolution=image_resolution,
                    patch_size=vision_patch_size,
                    width=vision_width,
                    layers=vision_layers,
                    heads=vision_heads,
                    output_dim=embed_dim,
                    design_details=design_details
                )
            elif trainer == "IVLP_VL_Adapter_Prompt":
                self.visual= VisionTransformer_Prompt(
                    input_resolution=image_resolution,
                    patch_size=vision_patch_size,
                    width=vision_width,
                    layers=vision_layers,
                    heads=vision_heads,
                    output_dim=embed_dim,
                    design_details=design_details
                ) 
            elif trainer == "IVLP_VL_Adapter_Prompt_SelectPatch":
                self.visual= VisionTransformer_Prompt_SelectPatch(
                    input_resolution=image_resolution,
                    patch_size=vision_patch_size,
                    width=vision_width,
                    layers=vision_layers,
                    heads=vision_heads,
                    output_dim=embed_dim,
                    design_details=design_details
                ) 
            elif trainer == "IVLP_VL_Adapter_Prompt_Multiple":
                self.visual= VisionTransformer_Prompt_Multiple(
                    input_resolution=image_resolution,
                    patch_size=vision_patch_size,
                    width=vision_width,
                    layers=vision_layers,
                    heads=vision_heads,
                    output_dim=embed_dim,
                    design_details=design_details
                ) 
            else:
                self.visual = VisionTransformer(
                    input_resolution=image_resolution,
                    patch_size=vision_patch_size,
                    width=vision_width,
                    layers=vision_layers,
                    heads=vision_heads,
                    output_dim=embed_dim,
                    design_details=design_details
                )
        # hyper-parameter if need to add prompt embeddings inside to the input
        # of transformer block or not:
        prompt_till_layer_text = design_details['language_depth']
        if trainer == "IVLP_VL_Adapter_Local_SelectPatch":
            self.transformer = Transformer_SelectPatch(
                width=transformer_width,
                layers=transformer_layers,
                heads=transformer_heads,
                attn_mask=self.build_attention_mask(),
                prompts_needed=prompt_till_layer_text,
                text_layer=True,
                design_details=design_details
            )
        elif trainer == "IVLP_VL_Adapter_Local_SelectPatch_FullMask":
            self.transformer = Transformer_SelectPatch_FullMask(
                width=transformer_width,
                layers=transformer_layers,
                heads=transformer_heads,
                attn_mask=self.build_attention_mask(),
                prompts_needed=prompt_till_layer_text,
                text_layer=True,
                design_details=design_details
            )
        elif trainer == "IVLP_VL_Adapter_Prompt":
            self.transformer = Transformer_Prompt(
                width=transformer_width,
                layers=transformer_layers,
                heads=transformer_heads,
                attn_mask=self.build_attention_mask(),
                prompts_needed=prompt_till_layer_text,
                text_layer=True,
                design_details=design_details
            )
        elif trainer == "IVLP_VL_Adapter_Prompt_SelectPatch":
            self.transformer = Transformer_Prompt_SelectPatch(
                width=transformer_width,
                layers=transformer_layers,
                heads=transformer_heads,
                attn_mask=self.build_attention_mask(),
                prompts_needed=prompt_till_layer_text,
                text_layer=True,
                design_details=design_details
            )
        elif trainer == "IVLP_VL_Adapter_Prompt_Multiple":
            self.transformer = Transformer_Prompt_Multiple(
                width=transformer_width,
                layers=transformer_layers,
                heads=transformer_heads,
                attn_mask=self.build_attention_mask(),
                prompts_needed=prompt_till_layer_text,
                text_layer=True,
                design_details=design_details
            )
        else :
            self.transformer = Transformer(
                width=transformer_width,
                layers=transformer_layers,
                heads=transformer_heads,
                attn_mask=self.build_attention_mask(),
                prompts_needed=prompt_till_layer_text,
                text_layer=True,
                design_details=design_details
            )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        if "Local" in self.trainer:
            x, q, k, v = self.transformer(x)
        else:
            x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        if "Local" in self.trainer:
            image_features, local_feat = self.encode_image(image)
        else :
            image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, design_details):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in
                        [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, design_details
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    try:
        model.load_state_dict(state_dict)
    except:
        missing_keys, _ = model.load_state_dict(state_dict, strict=False)
        print('Weights not found for some missing keys: ', missing_keys)
    return model.eval()

def get_similarity_map(sm, shape):

    # min-max norm
    sm = (sm - sm.min(1, keepdim=True)[0]) / (sm.max(1, keepdim=True)[0] - sm.min(1, keepdim=True)[0])

    # reshape
    side = int(sm.shape[1] ** 0.5) # square output
    sm = sm.reshape(sm.shape[0], side, side, -1).permute(0, 3, 1, 2)

    # interpolate
    sm = torch.nn.functional.interpolate(sm, shape, mode='bilinear')
    sm = sm.permute(0, 2, 3, 1)
    
    return sm