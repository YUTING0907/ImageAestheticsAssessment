"""
@DESCRIPTION: Train GIAA model
@AUTHOR: yzc-ippl
"""
#from models.attr import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
from torch import nn
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from PIL import Image
from scipy.stats import spearmanr
from torchvision import transforms
from torchvision import models
import torch.optim as optim
from torch.autograd import Variable
import copy
from torch.nn.parameter import Parameter
from tqdm import tqdm
import warnings
from skimage import transform
from torch.utils.data.dataloader import default_collate
import scipy.sparse as sp
import torch.nn.functional as F
import math

warnings.filterwarnings("ignore")
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

print(torch.cuda.device_count())
use_gpu = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 单GPU或者CPU
############### swin + TANet ######### start ##############
# --------------------------------------------------------
# Swin Transformer V2
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)

    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)

    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=[0, 0]):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))

        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h,
                            relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01).to(device))).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, ' \
               f'pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pretrained_window_size (int): Window size in pre-training.
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size))

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask).to(device)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.reduction(x)
        x = self.norm(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        flops += H * W * self.dim // 2
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        pretrained_window_size (int): Local window size in pre-training.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 pretrained_window_size=0):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pretrained_window_size=pretrained_window_size)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwinTransformerV2(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        pretrained_window_sizes (tuple(int)): Pretrained window sizes of each layer.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0], **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               pretrained_window_size=pretrained_window_sizes[i_layer])
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"cpb_mlp", "logit_scale", 'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x).to(device)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops


def swin_transformer_v2(pretrained=True):
    
    model = SwinTransformerV2(img_size=256,window_size=8)

    new_state_dict = {}
    if pretrained:
        print("read swintransformer weights")
        path_to_model = '/home/ps/temp/model/aesthetic2/MTCL_main/code/GIAA/swinv2_tiny_patch4_window8_256.pth'
        state_dict = torch.load(path_to_model, map_location=lambda storage, loc: storage)

        # 如果权重文件有顶层 "model" key，提取其中内容
        if "model" in state_dict:
            state_dict = state_dict["model"]

        '''
        for key, value in state_dict.items():
            if "relative_position_index" in key:
                new_key = key.replace("relative_position_index", "relative_pos_idx")
            elif "attn_mask" in key:
                continue
            else:
                new_key = key
             
            if "patch_embed" in key:
                new_key = key.replace("patch_embed", "patch_embedding")
            elif "proj" in key:
                new_key = key.replace("proj", "linear_embedding")
            elif "layers" in key:
                new_key = key.replace("layers", "stages")
            elif "norm1" in key:
                new_key = key.replace("norm1","normalization_1")
            elif "norm2" in key:
                new_key = key.replace("norm2","normalization_2")
            else:
                new_key = key
            
            new_state_dict[new_key] = value
        '''
        model.load_state_dict(state_dict)
        #model = nn.Sequential(*list(model.children())[:-1])
    #model.head = nn.Sequential(nn.Linear(768,1000),nn.LogSoftmax(dim=1))
    return model.to(device)

class swin_t(nn.Module):
    def __init__(self):
        super(swin_t,self).__init__()
        self.model = swin_transformer_v2(pretrained=True) 
        self.classifier = nn.Linear(
            in_features=1000,
            out_features=2048,
            bias=False
        )

    def forward(self, x):
        x_base = self.model(x).squeeze()
        x  = self.classifier(x_base)
        return x

########################################################################
class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, t))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # self.features.append(nn.AvgPool2d(input_size // 32))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        #avgpool
        self.avgpool = nn.AvgPool2d(input_size // 32)
        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.last_channel, n_class)
            #,DAModule(d_model=16,kernel_size=3,H=7,W=7) # add by yt
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(-1, self.last_channel)
        x = self.classifier(x)
        
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def mobile_net_v2(pretrained=True):
    model = MobileNetV2()

    if pretrained:
        print("read mobilenet weights")
        path_to_model = '/home/ps/temp/model/aesthetic2/MTCL_main/code/GIAA/mobilenetv2.pth.tar'
        state_dict = torch.load(path_to_model, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)
    #model.fc = nn.Sequential(nn.Linear(model.fc.in_features,256),nn.LogSoftmax(dim=1))
    return model


class score(nn.Module):
    def __init__(self):
        super(score,self).__init__()
        model = mobile_net_v2(pretrained = True)
        self.model = model
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1000,256 ),
        )

    def forward(self, x):
        x_base = self.model(x)
        x  = self.classifier(x_base)
        return x

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    rowsum[rowsum==0]=0.0000001
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.00000001
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        # torch.Size([1, 7, 256]) torch.Size([256, 256]) torch.Size([1, 7, 256]) torch.Size([7, 7]) torch.Size([1, 7, 256])
        # print(input.shape,self.weight.shape,support.shape,adj.shape,output.shape)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid*8)
        self.gc2_2 = GraphConvolution(nhid*8, nhid * 16)
        self.dropout = dropout
        self.fc1 = nn.Linear(nhid *16*7, 1)

    def forward(self, x, x2, adj):
        print("=======GCN===torch.Size([1, 7, 256])== torch.Size([1, 7, 256])==torch.Size([7, 7])=")
        x = F.relu(self.gc1(x, adj)) # torch.Size([1, 7, 256])
        x = F.dropout(x, self.dropout, training=self.training)
        x[:, 6, :] = x2[:, 6, :] # Change center node
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2_2(x, adj))
        x = x.view(x.shape[0], 1, -1)
        if x.ndim == 2:
            x = x.view(1, -1)
        x = F.relu(self.fc1(x))
        x = x.reshape((x.shape[0], -1))

        return x


class convNet_GCN(nn.Module):
    def __init__(self, attr_net, scene_net, score_net, gcn_net):
        super(convNet_GCN, self).__init__()
        self.AttrNet = attr_net
        self.ScenceNet = scene_net
        self.score_net = score_net
        self.GCNNet = gcn_net

    def forward(self, x_img):
        print("======convNet_GCN===start===")  # x_img.shape = torch.Size([1, 3, 214, 214])
        Scence_f = self.ScenceNet(x_img)  # x0 = torch.Size([1, 6])  Scence_f = torch.Size([1, 256])
        print(Scence_f.shape)
        score_f = self.score_net(x_img) # torch.Size([1, 256])
        print("-----------------------")
        print(score_f.shape)
        Content_f, Object_f, VividColor_f, DoF_f, ColorHarmony_f, Light_f = self.AttrNet(x_img) # torch.Size([1, 256])
        print(Object_f.shape)
        temp1 = torch.zeros(Scence_f.shape[0], 7, Scence_f.shape[1])  # torch.Size([1, 7, 256])
        temp2 = torch.zeros(Scence_f.shape[0], 7, Scence_f.shape[1])  # torch.Size([1, 7, 256])

        for num in range(Scence_f.shape[0]):
            temp1[num::] = torch.stack((Content_f[num,:],Object_f[num,:],VividColor_f[num,:],DoF_f[num,:],ColorHarmony_f[num,:],Light_f[num,:],Scence_f[num,:]), 0)
            temp2[num::] = torch.stack((Content_f[num,:],Object_f[num,:],VividColor_f[num,:],DoF_f[num,:],ColorHarmony_f[num,:],Light_f[num,:],score_f[num, :]), 0)
        edges_unordered = np.genfromtxt("cora.cites", dtype=np.int32)  # 读入边的信息
        adj = np.zeros((7, 7))
        for [q, p] in edges_unordered:
            adj[q - 1, p - 1] = 1
        adj = torch.from_numpy(adj)
        adj = normalize(adj)
        adj = torch.from_numpy(adj)
        adj = adj.clone().float()
        adj = adj.to(device)
        temp1 = temp1.to(device)
        temp2 = temp2.to(device)
        #  torch.Size([1, 1]) = torch.Size([1, 7, 256]) torch.Size([1, 7, 256]) torch.Size([7, 7])
        out_a = self.GCNNet(temp1, temp2, adj)
        print("======convNet_GCN===end===")
        return out_a

def mos(pretrained=True):
    path_dir = os.path.join(r'/home/ps/temp/model/aesthetic2/MTCL_main/code/GIAA/')
    path_to_model = os.path.join(path_dir, 'AADB_epoch_8_srcc_0.7148_lcc_0.7092_loss_0.6213_.pt')
    
    if pretrained:
        print("read AADB scence model weights")
        model = torch.load(path_to_model, map_location=lambda storage, loc: storage)

        model = torch.nn.Sequential( *( list(model.children())[:-6] ) )
        model_dict = model.state_dict()
        model.load_state_dict(model_dict)
    return model


class MO(nn.Module):
    def __init__(self):
        super(MO,self).__init__()
        model = mos(pretrained = True)
        self.model = model
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(3681,2048 ),
        )

    def forward(self, x):
        x_base = self.model(x)
        x_base = torch.cat(x_base, dim=1)
        x  = self.classifier(x_base)
        return x


def sce_resnet101(pretrained=True):
    path_dir = os.path.join(r'/home/ps/temp/model/EVA/deep-aesthetics-pytorch/')
    path_to_model = os.path.join(path_dir, 'ResNet34_GPU.pth')

    if pretrained:
        print("read eva scence model weights")
        model = torch.load(path_to_model, map_location=lambda storage, loc: storage)

        model = torch.nn.Sequential( *( list(model.children())[:-1] ) )
        
        model_dict = model.state_dict()
        model.load_state_dict(model_dict)
    return model

class sce(nn.Module):
    def __init__(self):
        super(sce,self).__init__()
        model = sce_resnet101(pretrained = True)
        self.model = model
        self.adaptive_pool=nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(
            in_features=512,
            out_features=256,
            bias=False
        )

    def forward(self, x):
        x_base = self.model(x).squeeze()
        print('resnet34:-2:output',x_base.shape)
        #x_base = self.adaptive_pool(x_base)
        #print('avgpool_shape:',x_base.shape)
        x  = self.classifier(x_base)
        print("sce_shape:",x.shape)
        print(x.shape)
        return x
    
################# GCN  ##### end ###############
def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

def Attention(x):
    # x = torch.Size([40, 3, 12, 12])
    batch_size, in_channels, h, w = x.size()
    # quary = torch.Size([40, 3, 144])
    quary = x.view(batch_size, in_channels, -1)
    key = quary
    # quary = torch.Size([40, 144, 3])
    quary = quary.permute(0, 2, 1)

    # sim_map = torch.Size([40, 144, 144])
    sim_map = torch.matmul(quary, key)
    ql2 = torch.norm(quary, dim=2, keepdim=True)
    kl2 = torch.norm(key, dim=1, keepdim=True)
    #  sim_map 进行归一化，具体来说，是将 sim_map 中的每个元素除以相应位置上 ql2 和 kl2 的乘积
    sim_map = torch.div(sim_map, torch.matmul(ql2, kl2).clamp(min=1e-8))

    return sim_map

def resnet365_backbone():
    arch = 'resnet18'
    # load the pre-trained weights
    model_file = '/home/ps/temp/model/TAD66K/resnet18_places365.pth.tar'
    last_model = models.__dict__[arch](num_classes=365)

    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    last_model.load_state_dict(state_dict)
    backbone = nn.Sequential(*list(last_model.children())[:-2])
    # Transformation layer: Convolution + Global Average Pooling
    transform_layer = nn.Sequential(
        nn.Conv2d(512, 4096, kernel_size=1),   # Increase channels to 2048
        nn.AdaptiveAvgPool2d(1)                # Global average pooling to [batch_size, 2048, 1, 1]
    )
    
    # Combine both in a single model
    model = nn.Sequential(
        backbone,
        transform_layer,
        nn.Flatten()  # Final output: [batch_size, 2048]
    )

    #return model

    return model

#hypernet
class L1(nn.Module):

    def __init__(self):
        super(L1, self).__init__()

        self.last_out_w = nn.Linear(4096, 1280)
        self.last_out_b = nn.Linear(4096, 1)

        # initialize
        for i, m_name in enumerate(self._modules):
            if i > 2:
                nn.init.kaiming_normal_(self._modules[m_name].weight.data)

    def forward(self, x):
        res_last_out_w = self.last_out_w(x)
        res_last_out_b = self.last_out_b(x)
        param_out = {}
        param_out['res_last_out_w'] = res_last_out_w
        param_out['res_last_out_b'] = res_last_out_b
        return param_out

class TargetFC(nn.Module):
    def __init__(self, weight, bias):
        super(TargetFC, self).__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, input_):
        out = F.linear(input_, self.weight, self.bias)
        return out

# L3
class TargetNet(nn.Module):

    def __init__(self):
        super(TargetNet, self).__init__()
        # L2
        self.fc1 = nn.Linear(4096, 1280)
        for i, m_name in enumerate(self._modules):
            if i > 2:
                nn.init.kaiming_normal_(self._modules[m_name].weight.data)
        #self.bn1 = nn.BatchNorm1d(100).cuda()
        self.bn1 = nn.BatchNorm1d(1280).to(device)
        self.relu1 = nn.PReLU()
        self.drop1 = nn.Dropout(1 - 0.5)

        self.relu7 = nn.PReLU()
        #self.relu7.cuda()
        self.relu7.to(device)
        self.sig = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, paras):

        q = self.fc1(x)
        q = self.bn1(q)
        q = self.relu1(q)
        q = self.drop1(q)

        self.lin = nn.Sequential(TargetFC(paras['res_last_out_w'], paras['res_last_out_b']))
        q = self.lin(q)
        q = self.softmax(q)
        return q


class L5(nn.Module):
    def __init__(self):
        super(L5, self).__init__()
        back_model = swin_t()
        self.base_model = back_model
        #self.PSPModule = PSPModule(1280, 100) # add by yt
        self.head = nn.Sequential(

            nn.Dropout(p=0.75),
            nn.ReLU(inplace=True), # ReLU在前面
            nn.Linear(1280, 10),
            #nn.Linear(100, 10),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.base_model(x)
        #x = self.PSPModule(x)  # add by yt
        x = x.view(x.size(0), -1)
        #x = self.head(x)
        return x

def MV2():
    model = mobile_net_v2()
    model = nn.Sequential(*list(model.children())[:-1])
    return model


class TANet(nn.Module):
    def __init__(self):
        super(TANet, self).__init__()
        self.res365_last = resnet365_backbone()
        #self.tf_efficientnet_b2 = efficientnet_b2_backbone()
        #self.swin_t = swin_t()
        self.hypernet = L1()

        self.tygertnet = TargetNet()
        self.avg = nn.AdaptiveAvgPool2d((2048, 1))
        self.avg_RGB = nn.AdaptiveAvgPool2d((12, 12))
        self.mobileNet = L5()
        self.softmax = nn.Softmax(dim=1)
        #self.eca = EfficientChannelAttention(12) # add by yt

        # L4
        self.head_rgb = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=0.75),
            #nn.Linear(20736, 10),
            nn.Linear(20736, 2048),
            nn.Softmax(dim=1)
        )

        # L6
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=0.75),
            #nn.Linear(3840, 2048),
            nn.Linear(30, 1),  # modify correlation
            nn.Sigmoid()
        )


    def forward(self, x):

        # L4 attention 第三条线(最底下)
        # 输入x(3,244,244)--self.avg_RGB(x)-->(3,12,12)--Attention(x_temp)-->(144,144)--view(x_temp.size(0), -1)-->(20736)--self.head_rgb(x_temp)-->(10)
        #x_temp = self.eca(x)
        x_temp = self.avg_RGB(x)

        # self.avg_RGB(x) = nn.AdaptiveAvgPool2d((12, 12))
        x_temp = Attention(x_temp)

        x_temp = x_temp.view(x_temp.size(0), -1)

        x_temp = self.head_rgb(x_temp)

        # 第一条w\b(场景)
        # 输入x(3,244,244)--self.res365_last(x)-->(365)---self.hypernet(res365_last_out)-->w(100)b(1)--self.tygertnet(res365_last_out, res365_last_out_weights)-->(40)--self.avg(x2)-->(10)
        # 输入x(3,244,244) --> (365) ->  (40)  -> (10)
        res365_last_out = self.res365_last(x)
        #print("resnet18:",res365_last_out.shape)
        res365_last_out_weights = self.hypernet(res365_last_out)

        res365_last_out_weights_mul_out = self.tygertnet(res365_last_out, res365_last_out_weights)

        x2 = res365_last_out_weights_mul_out.unsqueeze(dim=2)
        x2 = self.avg(x2)
        x2 = x2.squeeze(dim=2)

        # 第二条(美学)
        # 输入x(3,244,244)--self.mobileNet(x)-->(10)--concat()-->(30)--self.head(x)--> (1)
        x1 = self.mobileNet(x)

        x0 = x1.view(x1.size(0), x1.size(1), 1)

        # 转置张量 (40, 100, 1) 到 (40, 1, 100)
        transposed_tensor = torch.transpose(x0, 1, 2)
        # 相乘，得到大小为 (40, 1, 100) 的结果
        result_tensor = torch.matmul(x0, transposed_tensor)
        # 取每个结果的最大值
        x0 = torch.max(result_tensor, dim=2, keepdim=True)[0]

        x0 = x0.view(x0.size(0), -1)
        # 打印结果的大小
        #print("x1: x2: x_temp:",x1.shape,x2.shape,x_temp.shape)
        x = torch.cat([x1, x2, x_temp], 1)
        # L6
        #x = self.head(x)
        return x


class GIAA_model(nn.Module):
    def __init__(self, backbone, out_dim=6144):
        super(GIAA_model, self).__init__()

        #  create backbone
        self.backbone = backbone
        self.backbone.fc = nn.Sequential()

        # create regression
        self.regression = nn.Sequential(nn.Linear(out_dim, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.backbone(x)
        y = self.regression(x)
        return y

'''
class GIAA_model(nn.Module):
    def __init__(self, backbone, out_dim=2048):
        super(GIAA_model, self).__init__()

        #  create backbone
        self.backbone = backbone
        self.backbone.fc = nn.Sequential()

        # create regression
        self.regression = nn.Sequential(nn.Linear(out_dim, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.backbone(x)
        y = self.regression(x)
        return y
'''


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image = transform.resize(image, (new_h, new_w))

        return {'image': image, 'rating': rating}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w]

        return {'image': image, 'rating': rating}


class RandomHorizontalFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']
        if random.random() < self.p:
            image = np.flip(image, 1)
            # image = io.imread(img_name+'.jpg', mode='RGB').convert('RGB')
        return {'image': image, 'rating': rating}


class Normalize(object):
    def __init__(self):
        self.means = np.array([0.485, 0.456, 0.406])
        self.stds = np.array([0.229, 0.224, 0.225])

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']
        im = image / 1.0  # / 255
        im[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
        im[:, :, 1] = (image[:, :, 1] - self.means[1]) / self.stds[1]
        im[:, :, 2] = (image[:, :, 2] - self.means[2]) / self.stds[2]
        image = im
        return {'image': image, 'rating': rating}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).double(),
                'rating': torch.from_numpy(np.float64([rating])).double()}


def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


class FlickrDataset_GIAA(Dataset):
    """Images dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.images_frame = pd.read_csv(csv_file, sep=',')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.images_frame)

    def __getitem__(self, idx):
        # try:
        img_name = str(os.path.join(self.root_dir, str(self.images_frame.iloc[idx, 0])))
        im = Image.open(img_name).convert('RGB')
        if im.mode == 'P':
            im = im.convert('RGB')
        image = np.asarray(im)
        # image = io.imread(img_name+'.jpg', mode='RGB').convert('RGB')
        rating = self.images_frame.iloc[idx, 1:]
        sample = {'image': image, 'rating': rating}

        if self.transform:
            sample = self.transform(sample)
        return sample


def load_data():
    data_dir = os.path.join(r'/home/ps/temp/model/FLICKR-AES/label')
    data_image_dir = os.path.join(r'/home/ps/temp/model/FLICKR-AES/image')
    data_image_train_dir = os.path.join(data_dir, 'FLICKR-AES_GIAA_train.csv')
    data_image_test_dir = os.path.join(data_dir, 'FLICKR-AES_GIAA_test.csv')

    transformed_dataset_train = FlickrDataset_GIAA(
        csv_file=data_image_train_dir,
        root_dir=data_image_dir,
        transform=transforms.Compose(
            [Rescale(output_size=(256, 256)),
             RandomHorizontalFlip(0.5),
             #RandomCrop(
             #    output_size=(256, 256)),
             Normalize(),
             ToTensor(),
             ])
    )
    transformed_dataset_valid = FlickrDataset_GIAA(
        csv_file=data_image_test_dir,
        root_dir=data_image_dir,
        transform=transforms.Compose(
            [Rescale(output_size=(256, 256)),
             Normalize(),
             ToTensor(),
             ])
    )
    data_train = DataLoader(transformed_dataset_train, batch_size=64,
                            shuffle=True, num_workers=8, collate_fn=my_collate, drop_last=False)
    data_valid = DataLoader(transformed_dataset_valid, batch_size=64,
                            shuffle=True, num_workers=8, collate_fn=my_collate, drop_last=False)

    return data_train, data_valid


def exp_lr_scheduler(optimizer, epoch, lr_decay_epoch=1):
    """Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs."""

    decay_rate = 0.5 ** (epoch // lr_decay_epoch)
    if epoch % lr_decay_epoch == 0:
        print('decay_rate is set to {}'.format(decay_rate))

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

    return optimizer


def train_GIAA():
    # parameter setting
    num_epoch = 10
    BestSRCC = -10

    # data
    data_train, data_valid = load_data()

    # model
    #backbone = models.resnet50(pretrained=True)
    # backbone = models.resnext101_32x8d(pretrained=True)
    #model = GIAA_model()
    #backbone = swin_t()
    backbone = TANet()
    model = GIAA_model(backbone)
    #model = nn.DataParallel(model, device_ids=[0, 1])
    model.cuda()
    '''
    #####################
    model_ft = MO()

    num_ftrs = 2048
    Attr1_Model = attr1_Model(0.5, num_ftrs)
    Attr2_Model = attr1_Model(0.5, num_ftrs)
    Attr3_Model = attr1_Model(0.5, num_ftrs)
    Attr4_Model = attr1_Model(0.5, num_ftrs)
    Attr5_Model = attr1_Model(0.5, num_ftrs)
    Attr6_Model = attr1_Model(0.5, num_ftrs)
    Attr_model = convNet(backbone=model_ft, Attr1_Model=Attr1_Model, Attr2_Model=Attr2_Model, Attr3_Model=Attr3_Model, Attr4_Model=Attr4_Model, Attr5_Model=Attr5_Model, Attr6_Model=Attr6_Model )
    scence_model = sce()

    TANet_model = score()
    model_GCN = GCN(nfeat=256, nhid=256, dropout=0.5).to(device)
    model = convNet_GCN(attr_net=Attr_model, scene_net=scence_model, score_net=TANet_model, gcn_net=model_GCN).to(device)
   #####################
    '''
    # optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=5E-2)

    for epoch in range(num_epoch):
        for phase in ['train', 'valid']:
            if phase == 'train':
                print('***********************train***********************')
                model.train()
                optimizer = exp_lr_scheduler(optimizer, epoch)
                loop = tqdm(enumerate(data_train), total=len(data_train), leave=True)
                for batch_idx, data in loop:
                    inputs = data['image']
                    batch_size = inputs.size()[0]
                    labels = data['rating'].view(batch_size, -1)
                    if use_gpu:
                        inputs, labels = Variable(inputs.float().cuda()), Variable(labels.float().cuda())
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    criterion = nn.MSELoss()
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    loop.set_description(f'Epoch [{epoch}/{num_epoch}]--train')
                    loop.set_postfix(loss=loss.item())
            if phase == 'valid':
                print('***********************valid***********************')
                model.eval()
                predicts_score = []
                ratings_score = []
                for batch_idx, data in enumerate(data_valid):
                    inputs = data['image']
                    batch_size = inputs.size()[0]
                    labels = data['rating'].view(batch_size, -1)
                    if use_gpu:
                        inputs, labels = Variable(inputs.float().cuda()), Variable(labels.float().cuda())
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)

                    with torch.no_grad():
                        outputs = model(inputs)
                    outputs = outputs.data.cpu().numpy()
                    labels = labels.data.cpu().numpy()
                    predicts_score += outputs.tolist()
                    ratings_score += labels.tolist()
                srcc = spearmanr(predicts_score, ratings_score)[0]
                print('Valid Regression SRCC:%4f' % srcc)
                if srcc > BestSRCC:
                    BestSRCC = srcc
                    best_model = copy.deepcopy(model)
                    torch.save(best_model.cuda(), '../model/swin-transV2/swin-transV2-FlickrAes-TANet-GIAA.pt')


if __name__ == '__main__':
    train_GIAA()
