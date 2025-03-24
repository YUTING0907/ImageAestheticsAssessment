"""
@DESCRIPTION: Train GIAA model
@AUTHOR: yzc-ippl
"""
from models.attr import *
import torch
from torch import nn
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
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

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

use_gpu = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 单GPU或者CPU
############### GCN + TANet ######### start ##############
class SwinTransformerBlock(nn.Module):
    def __init__(self,
                 dim,
                 n_heads,
                 window_size=7,
                 shift_size=0,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 proj_dropout=0.,
                 attn_dropout=0.,
                 dropout=0.,
                 norm_layer=nn.LayerNorm):
        super(SwinTransformerBlock, self).__init__()

        self.dim = dim
        self.n_heads = n_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size, n_heads,
                                    qkv_bias, attn_dropout, proj_dropout)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = MLP(in_features=dim,
                       hid_features=dim*mlp_ratio,
                       dropout=proj_dropout)

    def forward(self, x, attn_mask):
        h, w = self.h, self.w
        b, _, c = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, hp, wp, _ = x.shape

        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None

        # (n_windows*b,m,m,c)
        x_windows = window_partition(shifted_x, self.window_size)
        # (n_windows*b,m*m,c)
        x_windows = x_windows.view(-1, self.window_size*self.window_size, c)

        attn_windows = self.attn(x_windows, attn_mask)

        attn_windows = attn_windows.view(-1,
                                         self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, hp, wp)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(
                self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :].contiguous()

        x = x.view(b, h*w, c)

        x = shortcut + self.dropout(x)
        x = x + self.dropout(self.mlp(self.norm2(x)))

        return x
class WindowAttention(nn.Module):
    def __init__(self,
                 dim,
                 window_size,
                 n_heads,
                 qkv_bias=True,
                 attn_dropout=0.,
                 proj_dropout=0.):
        super(WindowAttention, self).__init__()

        self.dim = dim
        self.window_size = window_size
        self.n_heads = n_heads
        self.scale = (dim // n_heads) ** -.5

        # ((2m-1)*(2m-1),n_heads)
        # 相对位置参数表长为(2m-1)*(2m-1)
        # 行索引和列索引各有2m-1种可能，故其排列组合有(2m-1)*(2m-1)种可能
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2*window_size - 1) * (2*window_size - 1),
                        n_heads))

        # 构建窗口的绝对位置索引
        # 以window_size=2为例
        # coord_h = coord_w = [0,1]
        # meshgrid([0,1], [0,1])
        # -> [[0,0], [[0,1]
        #     [1,1]], [0,1]]
        # -> [[0,0,1,1],
        #     [0,1,0,1]]
        # (m,)
        coord_h = torch.arange(self.window_size)
        coord_w = torch.arange(self.window_size)
        # (m,)*2 -> (m,m)*2 -> (2,m,m)
        coords = torch.stack(torch.meshgrid([coord_h, coord_w]))
        # (2,m*m)
        coord_flatten = torch.flatten(coords, 1)

        # 构建窗口的相对位置索引
        # (2,m*m,1) - (2,1,m*m) -> (2,m*m,m*m)
        # 以coord_flatten为
        # [[0,0,1,1]
        #  [0,1,0,1]]为例
        # 对于第一个元素[0,0,1,1]
        # [[0],[0],[1],[1]] - [[0,0,1,1]]
        # -> [[0,0,0,0] - [[0,0,1,1] = [[0,0,-1,-1]
        #     [0,0,0,0]    [0,0,1,1]    [0,0,-1,-1]
        #     [1,1,1,1]    [0,0,1,1]    [1,1, 0, 0]
        #     [1,1,1,1]]   [0,0,1,1]]   [1,1, 0, 0]]
        # 相当于每个元素的h减去每个元素的h
        # 例如，第一行[0,0,0,0] - [0,0,1,1] -> [0,0,-1,-1]
        # 即为元素(0,0)相对(0,0)(0,1)(1,0)(1,1)为列(h)方向的差
        # 第二个元素即为每个元素的w减去每个元素的w
        # 于是得到窗口内每个元素相对每个元素高和宽的差
        # 例如relative_coords[0,1,2]
        # 即为窗口的第1个像素(0,1)和第2个像素(1,0)在列(h)方向的差
        relative_coords = coord_flatten[:, :, None] - coord_flatten[:, None, :]
        # (m*m,m*m,2)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()

        # 论文中提到的，将二维相对位置索引转为一维的过程
        # 1. 行列都加上m-1
        # 2. 行乘以2m-1
        # 3. 行列相加
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        # (m*m,m*m,2) -> (m*m,m*m)
        relative_pos_idx = relative_coords.sum(-1)
        self.register_buffer('relative_pos_idx', relative_pos_idx)

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(proj_dropout)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask):
        b, n, c = x.shape
        # (b*n_windows,m*m,total_embed_dim)
        # -> (b*n_windows,m*m,3*total_embed_dim)
        # -> (b*n_windows,m*m,3,n_heads,embed_dim_per_head)
        # -> (3,b*n_windows,n_heads,m*m,embed_dim_per_head)
        qkv = (self.qkv(x)
               .reshape(b, n, 3, self.n_heads, c//self.n_heads)
               .permute(2, 0, 3, 1, 4))
        # (b*n_windows,n_heads,m*m,embed_dim_per_head)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        # (b*n_windows,n_heads,m*m,m*m)
        attn = (q @ k.transpose(-2, -1))

        # (m*m*m*m,n_heads)
        # -> (m*m,m*m,n_heads)
        # -> (n_heads,m*m,m*m)
        # -> (b*n_windows,n_heads,m*m,m*m) + (1,n_heads,m*m,m*m)
        # -> (b*n_windows,n_heads,m*m,m*m)
        relative_pos_bias = (self.relative_position_bias_table[self.relative_pos_idx.view(-1)]
                             .view(self.window_size*self.window_size, self.window_size*self.window_size, -1))
        relative_pos_bias = relative_pos_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_pos_bias.unsqueeze(0)

        if mask is not None:
            # mask: (n_windows,m*m,m*m)
            nw = mask.shape[0]
            # (b*n_windows,n_heads,m*m,m*m)
            # -> (b,n_windows,n_heads,m*m,m*m)
            # + (1,n_windows,1,m*m,m*m)
            # -> (b,n_windows,n_heads,m*m,m*m)
            attn = (attn.view(b//nw, nw, self.n_heads, n, n)
                    + mask.unsqueeze(1).unsqueeze(0))
            # (b,n_windows,n_heads,m*m,m*m)
            # -> (b*n_windows,n_heads,m*m,m*m)
            attn = attn.view(-1, self.n_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_dropout(attn)

        # (b*n_windows,n_heads,m*m,embed_dim_per_head)
        # -> (b*n_windows,m*m,n_heads,embed_dim_per_head)
        # -> (b*n_windows,m*m,total_embed_dim)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_dropout(x)

        return x

class MLP(nn.Module):
    def __init__(self,
                 in_features,
                 hid_features=None,
                 out_features=None,
                 dropout=0.):
        super(MLP, self).__init__()

        out_features = out_features or in_features
        hid_features = hid_features or in_features

        self.fc1 = nn.Linear(in_features, hid_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hid_features, out_features)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.drop1(self.act(self.fc1(x)))
        x = self.drop2(self.fc2(x))
        return x

class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super(PatchMerging, self).__init__()

        self.dim = dim
        self.reduction = nn.Linear(4*dim, 2*dim, bias=False)
        self.norm = norm_layer(4*dim)

    def forward(self, x, h, w):
        # (b,hw,c)
        b, l, c = x.shape
        # (b,hw,c) -> (b,h,w,c)
        x = x.view(b, h, w, c)

        # 如果h,w不是2的整数倍，需要padding
        if (h % 2 == 1) or (w % 2 == 1):
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))

        # (b,h/2,w/2,c)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        # (b,h/2,w/2,c)*4 -> (b,h/2,w/2,4c)
        x = torch.cat([x0, x1, x2, x3], -1)
        # (b,hw/4,4c)
        x = x.view(b, -1, 4*c)

        x = self.norm(x)
        # (b,hw/4,4c) -> (b,hw/4,2c)
        x = self.reduction(x)

        return x

class BasicLayer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 n_heads,
                 window_size,
                 mlp_ratio=4,
                 qkv_bias=True,
                 proj_dropout=0.,
                 attn_dropout=0.,
                 dropout=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None):
        super(BasicLayer, self).__init__()

        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        # 窗口向右和向下的移动数为窗口宽度除以2向下取整
        self.shift_size = window_size // 2

        # 按照每个Stage的深度堆叠若干Block
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim, n_heads, window_size,
                                 0 if (i % 2 == 0) else self.shift_size,
                                 mlp_ratio, qkv_bias, proj_dropout, attn_dropout,
                                 dropout[i] if isinstance(
                                     dropout, list) else dropout,
                                 norm_layer)
            for i in range(depth)])

        self.downsample = downsample(dim=dim, norm_layer=norm_layer) if downsample else None

    def forward(self, x, h, w):
        attn_mask = self.create_mask(x, h, w)
        for blk in self.blocks:
            blk.h, blk.w = h, w
            x = blk(x, attn_mask)

        if self.downsample is not None:
            x = self.downsample(x, h, w)
            # 如果是奇数，相当于做padding后除以2
            # 如果是偶数，相当于直接除以2
            h, w = (h+1) // 2, (w+1) // 2

        return x, h, w

    def create_mask(self, x, h, w):
        # 保证hp,wp是window_size的整数倍
        hp = int(np.ceil(h/self.window_size)) * self.window_size
        wp = int(np.ceil(w/self.window_size)) * self.window_size

        # (1,hp,wp,1)
        img_mask = torch.zeros((1, hp, wp, 1), device=x.device)

        # 将feature map分割成9个区域
        # 例如，对于9x9图片
        # 若window_size=3, shift_size=3//2=1
        # 得到slices为([0,-3],[-3,-1],[-1,])
        # 于是h从0至-4(不到-3)，w从0至-4
        # 即(0,0)(-4,-4)围成的矩形为第1个区域
        # h从0至-4，w从-3至-2
        # 即(0,-3)(-4,-2)围成的矩形为第2个区域...
        # h\w 0 1 2 3 4 5 6 7 8
        # --+--------------------
        # 0 | 0 0 0 0 0 0 1 1 2
        # 1 | 0 0 0 0 0 0 1 1 2
        # 2 | 0 0 0 0 0 0 1 1 2
        # 3 | 0 0 0 0 0 0 1 1 2
        # 4 | 0 0 0 0 0 0 1 1 2
        # 5 | 0 0 0 0 0 0 1 1 2
        # 6 | 3 3 3 3 3 3 4 4 5
        # 7 | 3 3 3 3 3 3 4 4 5
        # 8 | 6 6 6 6 6 6 7 7 8
        # 这样在每个窗口内，相同数字的区域都是连续的
        slices = (slice(0, -self.window_size),
                  slice(-self.window_size, -self.shift_size),
                  slice(-self.shift_size, None))
        cnt = 0
        for h in slices:
            for w in slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        # (1,hp,wp,1) -> (n_windows,m,m,1) m表示window_size
        mask_windows = window_partition(img_mask, self.window_size)
        # (n_windows,m,m,1) -> (n_windows,m*m)
        mask_windows = mask_windows.view(-1,
                                         self.window_size * self.window_size)

        # (n_windows,1,m*m) - (n_windows,m*m,1)
        # -> (n_windows,m*m,m*m)
        # 以window
        # [[4 4 5]
        #  [4 4 5]
        #  [7 7 8]]
        # 为例
        # 展平后为 [4,4,5,4,4,5,7,7,8]
        # [[4,4,5,4,4,5,7,7,8]] - [[4]
        #                          [4]
        #                          [5]
        #                          [4]
        #                          [4]
        #                          [5]
        #                          [7]
        #                          [7]
        #                          [8]]
        # -> [[0,0,-,0,0,-,-,-,-]
        #     [0,0,-,0,0,-,-,-,-]
        #     [...]]
        # 于是有同样数字的区域为0，不同数字的区域为非0
        # attn_mask[1,3]即为窗口的第3个元素(1,0)和第1个元素(0,1)是否相同
        # 若相同，则值为0，否则为非0
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        # 将非0的元素设为-100
        attn_mask = (attn_mask
                     .masked_fill(attn_mask != 0, float(-100.))
                     .masked_fill(attn_mask == 0, float(0.)))

        return attn_mask


class PatchEmbed(nn.Module):
    def __init__(self,
                 patch_size=4,
                 in_c=3,
                 embed_dim=96,
                 norm_layer=None):
        super(PatchEmbed, self).__init__()

        self.patch_size = patch_size
        self.in_c = in_c
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(
            in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # 如果图片的H,W不是patch_size的整数倍，需要padding
        _, _, h, w = x.shape
        if (h % self.patch_size != 0) or (w % self.patch_size != 0):
            x = F.pad(x, (0, self.patch_size - w % self.patch_size,
                          0, self.patch_size - h % self.patch_size,
                          0, 0))

        x = self.proj(x)
        _, _, h, w = x.shape

        # (b,c,h,w) -> (b,c,hw) -> (b,hw,c)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, h, w


class SwinTransformer(nn.Module):
    def __init__(self,
                 patch_size=4,
                 in_c=3,
                 n_classes=1000,
                 embed_dim=96,
                 depths=(2, 2, 6, 2),
                 n_heads=(3, 6, 12, 24),
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 proj_dropout=0.,
                 attn_dropout=0.,
                 dropout=0.,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True):
        super(SwinTransformer, self).__init__()

        self.n_classes = n_classes
        self.n_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        # Stage4输出的channels，即embed_dim*8
        self.n_features = int(embed_dim * 2**(self.n_layers-1))
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(patch_size, in_c, embed_dim,
                                      norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(proj_dropout)

        # 根据深度递增dropout
        dpr = [x.item() for x in torch.linspace(0, dropout, sum(depths))]

        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            layers = BasicLayer(int(embed_dim*2**i), depths[i], n_heads[i],
                                window_size, mlp_ratio, qkv_bias,
                                proj_dropout, attn_dropout, 
                                dpr[sum(depths[:i]):sum(depths[:i+1])],
                                norm_layer, PatchMerging if i < self.n_layers-1 else None)
            self.layers.append(layers)

        self.norm = norm_layer(self.n_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(
            self.n_features, n_classes) if n_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def forward(self, x):
        x, h, w = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x, h, w = layer(x, h, w)

        # (b,l,c)
        x = self.norm(x)
        # (b,l,c) -> (b,c,l) -> (b,c,1)
        x = self.avgpool(x.transpose(1, 2))
        # (b,c,1) -> (b,c)
        x = torch.flatten(x, 1)
        # (b,c) -> (b,n_classes)
        x = self.head(x)

        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.)



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


class attr1_Model(nn.Module):
    def __init__(self, keep_probability, inputsize):
        super(attr1_Model, self).__init__()
        self.drop_prob = (1 - keep_probability)
        self.fc1_1 = nn.Linear(inputsize, 256)
        self.bn1_1 = nn.BatchNorm1d(256)
        self.relu1_1 = nn.PReLU()
        self.drop1_1 = nn.Dropout(self.drop_prob)
        self.fc2_1 = nn.Linear(256, 64)
        self.relu2_1 = nn.PReLU()
        self.drop2_1 = nn.Dropout(p=self.drop_prob)
        self.fc3_1 = nn.Linear(64, 1)
        self.relu3_1 = nn.PReLU()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):

        out_a = self.fc1_1(x)
        
        f_out_a = out_a.clone() # return 256
        out_a = self.bn1_1(out_a)
        out_a = self.relu1_1(out_a)
        out_a = self.drop1_1(out_a)
        out_a = self.fc2_1(out_a)
        out_a = self.relu2_1(out_a)
        out_a = self.drop2_1(out_a)
        out_a = self.fc3_1(out_a)
        out_a = self.relu3_1(out_a)

        return out_a, f_out_a

class convNet(nn.Module):
    def __init__(self, backbone, Attr1_Model, Attr2_Model, Attr3_Model, Attr4_Model, Attr5_Model, Attr6_Model ):
        super(convNet, self).__init__()
        self.backbone=backbone
        self.Attr1_Model=Attr1_Model
        self.Attr2_Model=Attr2_Model
        self.Attr3_Model=Attr3_Model
        self.Attr4_Model=Attr4_Model
        self.Attr5_Model=Attr5_Model
        self.Attr6_Model = Attr6_Model

    def forward(self, x_img):
        
        x=self.backbone(x_img)
        
        x1, f1 = self.Attr1_Model(x) # Interesting Content    f1=(1,256)
        x2, f2 = self.Attr2_Model(x) # Object Emphasis
        x3, f3 = self.Attr3_Model(x) # Vivid Color
        x4, f4 = self.Attr4_Model(x) # Depth of Field
        x5, f5 = self.Attr5_Model(x) # Color Harmony
        x6, f6 = self.Attr6_Model(x) # Good Lighting
        return f1, f2, f3, f4, f5, f6


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


class GIAA_model(nn.Module):
    def __init__(self):
    # ------------Visual Attribute Analysis Network------------- #
        super(GIAA_model,self).__init__() 

        model_ft = MO()
                
        num_ftrs = 2048
        Attr1_Model = attr1_Model(0.5, num_ftrs)
        Attr2_Model = attr1_Model(0.5, num_ftrs)
        Attr3_Model = attr1_Model(0.5, num_ftrs)
        Attr4_Model = attr1_Model(0.5, num_ftrs)
        Attr5_Model = attr1_Model(0.5, num_ftrs)
        Attr6_Model = attr1_Model(0.5, num_ftrs)
        Attr_model = convNet(backbone=model_ft, Attr1_Model=Attr1_Model, Attr2_Model=Attr2_Model, Attr3_Model=Attr3_Model, Attr4_Model=Attr4_Model, Attr5_Model=Attr5_Model, Attr6_Model=Attr6_Model )

        
        self.scence_model = sce()
        
        self.TANet_model = score() 
        self.model_GCN = GCN(nfeat=256, nhid=256, dropout=0.5).to(device)
        self.convNet_GCN = convNet_GCN(attr_net=Attr_model, scene_net=self.scence_model, score_net=self.TANet_model, gcn_net=self.model_GCN).to(device)
    def forward(self,x):
        x = self.convNet_GCN(x)
        return x
################# GCN + TANet ##### end ###############
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
             RandomCrop(
                 output_size=(224, 224)),
             Normalize(),
             ToTensor(),
             ])
    )
    transformed_dataset_valid = FlickrDataset_GIAA(
        csv_file=data_image_test_dir,
        root_dir=data_image_dir,
        transform=transforms.Compose(
            [Rescale(output_size=(224, 224)),
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
    model = GIAA_model()
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
                    torch.save(best_model.cuda(), '../model/ResNet-50/ResNet50-FlickrAes-sixsixGIAA.pt')


if __name__ == '__main__':
    train_GIAA()
