import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.distributions import Normal, Independent, kl
import warnings
warnings.filterwarnings('ignore')
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

import math
warnings.filterwarnings('ignore')


# MLP module with depthwise conv and dropout
class Mlp(nn.Module):
    # Initialize MLP with optional hidden/out features and activation
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or int(in_features * 2)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    # Initialize weights for layers
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    # Forward pass through MLP with dwconv and dropout
    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# Attention module with optional spatial reduction
class Attention(nn.Module):
    # Initialize attention with given dimensions and heads
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = max(1, num_heads // 2)
        head_dim = dim // self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    # Initialize weights for attention internals
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    # Forward pass computing self-attention with optional spatial reduction
    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


# Transformer block combining attention and MLP with residual connections
class Block(nn.Module):
    # Initialize transformer block with normalization, attention and MLP
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio // 2)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    # Initialize weights for block internals
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    # Forward pass for transformer block
    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


# Patch embedding that projects image to patch tokens
class OverlapPatchEmbed(nn.Module):
    # Initialize overlapping patch embedding with conv projection and layer norm
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    # Initialize weights for embedding internals
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    # Forward pass converting image to patch tokens
    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


# Pyramid Vision Transformer improved variant producing multi-scale features
class PyramidVisionTransformerImpr(nn.Module):
    # Initialize PVT-improved with multiple stages and transformer blocks
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # transformer encoder - reduced number of layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        self.apply(self._init_weights)

    # Initialize weights for model internals
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    # Initialize weights (pretrained loading removed)
    def init_weights(self, pretrained=None):
        pass

    # Reset drop path probabilities based on new rate
    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    # Freeze patch embedding parameters
    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    # Specify parameters that should have no weight decay
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}

    # Return classifier head
    def get_classifier(self):
        return self.head

    # Reset classifier head for new number of classes
    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    # Compute multi-scale features from input
    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    # Forward returns multi-scale features
    def forward(self, x):
        x = self.forward_features(x)
        return x


# Depthwise convolution used inside MLP
class DWConv(nn.Module):
    # Initialize depthwise conv
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    # Forward reshape to 2D, apply conv, and restore shape
    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


# Convert patch embedding weights if needed (conv filter conversion)
def _conv_filter(state_dict, patch_size=16):
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


# Small PVT variants with reduced sizes
class pvt_v2_b0(PyramidVisionTransformerImpr):
    # Initialize pvt_v2_b0 variant with smaller embedding dims and depths
    def __init__(self, **kwargs):
        super(pvt_v2_b0, self).__init__(
            patch_size=4, embed_dims=[24, 48, 96, 192],
            num_heads=[1, 2, 3, 4],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[1, 1, 1, 1],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class pvt_v2_b1(PyramidVisionTransformerImpr):
    # Initialize pvt_v2_b1 variant with adjusted sizes
    def __init__(self,** kwargs):
        super(pvt_v2_b1, self).__init__(
            patch_size=4, embed_dims=[32, 64, 128, 256],
            num_heads=[1, 2, 3, 4],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[1, 2, 2, 1],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class pvt_v2_b2(PyramidVisionTransformerImpr):
    # Initialize pvt_v2_b2 variant with adjusted sizes
    def __init__(self, **kwargs):
        super(pvt_v2_b2, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256],
            num_heads=[1, 2, 4, 4],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[2, 2, 3, 2],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class pvt_v2_b3(PyramidVisionTransformerImpr):
    # Initialize pvt_v2_b3 variant with adjusted sizes
    def __init__(self,** kwargs):
        super(pvt_v2_b3, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256],
            num_heads=[1, 2, 4, 4],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[2, 3, 6, 2],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class pvt_v2_b4(PyramidVisionTransformerImpr):
    # Initialize pvt_v2_b4 variant with adjusted sizes
    def __init__(self, **kwargs):
        super(pvt_v2_b4, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256],
            num_heads=[1, 2, 4, 4],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[2, 4, 8, 2],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class pvt_v2_b5(PyramidVisionTransformerImpr):
    # Initialize pvt_v2_b5 variant with adjusted sizes
    def __init__(self,** kwargs):
        super(pvt_v2_b5, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256],
            num_heads=[1, 2, 4, 4],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[2, 4, 12, 2],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


# Prior module for aleatoric uncertainty estimation using convolutional encoder to latent prior
class Aleatoric_Uncertainty_Estimation_Module_Prior(nn.Module):
    # Initialize prior estimation module with encoder convs and FC to latent
    def __init__(self, input_channels, channels, latent_size):
        super(Aleatoric_Uncertainty_Estimation_Module_Prior, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.layer2 = nn.Conv2d(channels, int(1.5 * channels), kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(int(1.5 * channels))
        self.layer3 = nn.Conv2d(int(1.5 * channels), 2 * channels, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(2 * channels)
        self.layer4 = nn.Conv2d(2 * channels, 4 * channels, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(4 * channels)
        self.layer5 = nn.Conv2d(4 * channels, 4 * channels, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(4 * channels)
        self.channel = channels

        self.latent_size = latent_size
        self.fc1 = nn.Linear(4 * channels, latent_size)
        self.fc2 = nn.Linear(4 * channels, latent_size)

        self.leakyrelu = nn.LeakyReLU()

    # Forward encoder and produce mu, logvar and an independent normal distribution
    def forward(self, input_feature):
        output = self.leakyrelu(self.bn1(self.layer1(input_feature)))
        output = self.leakyrelu(self.bn2(self.layer2(output)))
        output = self.leakyrelu(self.bn3(self.layer3(output)))
        output = self.leakyrelu(self.bn4(self.layer4(output)))
        output = self.leakyrelu(self.bn5(self.layer5(output)))
        
        batch_size = output.size(0)
        flatten_size = output.view(batch_size, -1).size(1)
        
        # Dynamically adjust fully connected layers if flattened size changed
        if self.fc1.weight.size(1) != flatten_size:
            self.fc1 = nn.Linear(flatten_size, self.latent_size).to(output.device)
            self.fc2 = nn.Linear(flatten_size, self.latent_size).to(output.device)
        
        output = output.view(batch_size, -1)
        mu = self.fc1(output)
        logvar = self.fc2(output)
        dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)

        return mu, logvar, dist


class Aleatoric_Uncertainty_Estimation_Module_Post(nn.Module):
    """Aleatoric uncertainty estimation module for posterior branch"""
    def __init__(self, input_channels, channels, latent_size):
        super(Aleatoric_Uncertainty_Estimation_Module_Post, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.layer2 = nn.Conv2d(channels, int(1.5 * channels), kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(int(1.5 * channels))
        self.layer3 = nn.Conv2d(int(1.5 * channels), 2 * channels, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(2 * channels)
        self.layer4 = nn.Conv2d(2 * channels, 4 * channels, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(4 * channels)
        self.layer5 = nn.Conv2d(4 * channels, 4 * channels, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(4 * channels)
        self.channel = channels

        self.latent_size = latent_size
        self.fc1 = nn.Linear(4 * channels, latent_size)
        self.fc2 = nn.Linear(4 * channels, latent_size)

        self.leakyrelu = nn.LeakyReLU()

    def forward(self, input_feature):
        output = self.leakyrelu(self.bn1(self.layer1(input_feature)))
        output = self.leakyrelu(self.bn2(self.layer2(output)))
        output = self.leakyrelu(self.bn3(self.layer3(output)))
        output = self.leakyrelu(self.bn4(self.layer4(output)))
        output = self.leakyrelu(self.bn5(self.layer5(output)))
        
        batch_size = output.size(0)
        flatten_size = output.view(batch_size, -1).size(1)
        
        if self.fc1.weight.size(1) != flatten_size:
            self.fc1 = nn.Linear(flatten_size, self.latent_size).to(output.device)
            self.fc2 = nn.Linear(flatten_size, self.latent_size).to(output.device)
        
        output = output.view(batch_size, -1)
        mu = self.fc1(output)
        logvar = self.fc2(output)
        dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)

        return mu, logvar, dist


class Epistemic_Uncertainty_Estimation(nn.Module):
    """Epistemic uncertainty estimation module"""
    def __init__(self, ndf):
        super(Epistemic_Uncertainty_Estimation, self).__init__()
        self.conv1 = nn.Conv2d(5, ndf//2, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf//2, ndf//2, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(ndf//2, ndf//2, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf//2, ndf//2, kernel_size=3, stride=1, padding=1)
        self.classifier = nn.Conv2d(ndf//2, 1, kernel_size=3, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.bn1 = nn.BatchNorm2d(ndf//2)
        self.bn2 = nn.BatchNorm2d(ndf//2)
        self.bn3 = nn.BatchNorm2d(ndf//2)
        self.bn4 = nn.BatchNorm2d(ndf//2)

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.leaky_relu(self.bn3(self.conv3(x)))
        x = self.leaky_relu(self.bn4(self.conv4(x)))
        x = self.classifier(x)
        return x


class DynamicFeatureAdapter(nn.Module):
    """Dynamic feature adapter to fuse inputs with different channels into main network"""
    def __init__(self, in_channels, target_channels, reduction=8):
        super().__init__()
        self.target_channels = target_channels
        
        self.channel_mapper = nn.Sequential(
            nn.Conv2d(in_channels, target_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(target_channels),
            nn.ReLU(inplace=True)
        )
        
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(target_channels, target_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(target_channels // reduction, target_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, target_feat):
        x = self.channel_mapper(x)
        
        if x.shape[2:] != target_feat.shape[2:]:
            x = F.interpolate(x, size=target_feat.shape[2:], mode='bilinear', align_corners=True)
            
        attn = self.attention(x)
        return target_feat + x * attn


class UABCDWithDynamic(nn.Module):
    """UABCD model with dynamic feature input"""
    def __init__(self, fixed_in_channels, latent_dim=128, num_classes=2, dynamic_in_channels=None):
        super(UABCDWithDynamic, self).__init__()
        channel = 64

        self.backbone = pvt_v2_b0()
        self.backbone.patch_embed1.proj = nn.Conv2d(
            fixed_in_channels, 
            self.backbone.patch_embed1.proj.out_channels,
            kernel_size=self.backbone.patch_embed1.proj.kernel_size,
            stride=self.backbone.patch_embed1.proj.stride,
            padding=self.backbone.patch_embed1.proj.padding
        )
        
        self.conv1 = BasicConv2d(2*24, 24, 1)
        self.conv2 = BasicConv2d(2*48, 48, 1)
        self.conv3 = BasicConv2d(2*96, 96, 1)
        self.conv4 = BasicConv2d(2*192, 192, 1)

        self.conv_4 = BasicConv2d(192, channel, 3, 1, 1)
        self.conv_3 = BasicConv2d(96, channel, 3, 1, 1)
        self.conv_2 = BasicConv2d(48, channel, 3, 1, 1)
        self.conv_1 = BasicConv2d(24, channel, 3, 1, 1)

        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.coarse_out = nn.Sequential(
            nn.Conv2d(4 * channel, channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(channel, 1, kernel_size=1, stride=1, bias=True)
        )
        self.sigmoid = nn.Sigmoid()

        self.AUEM_prior = Aleatoric_Uncertainty_Estimation_Module_Prior(
            2 * fixed_in_channels, int(channel / 8), latent_dim)
        self.AUEM_post = Aleatoric_Uncertainty_Estimation_Module_Post(
            2 * fixed_in_channels + 1, int(channel / 8), latent_dim)

        self.decoder_prior = Refined_Change_Map_Generation(latent_dim, num_classes)
        self.decoder_post = Refined_Change_Map_Generation(latent_dim, num_classes)
        
        self.dynamic_branch_count = len(dynamic_in_channels) if dynamic_in_channels else 0
        self.dynamic_adapters = nn.ModuleList()
        if dynamic_in_channels:
            target_channels = [24, 48, 96, 192]
            for in_ch in dynamic_in_channels:
                level_adapters = nn.ModuleList([
                    DynamicFeatureAdapter(in_ch, target_channels[0]),
                    DynamicFeatureAdapter(in_ch, target_channels[1]),
                    DynamicFeatureAdapter(in_ch, target_channels[2]),
                    DynamicFeatureAdapter(in_ch, target_channels[3])
                ])
                self.dynamic_adapters.append(level_adapters)
        
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def get_device(self):
        """Safely get model device to avoid StopIteration errors"""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return device

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(std, device=mu.device)
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def kl_divergence(self, posterior_latent_space, prior_latent_space):
        kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)
        return kl_div
    
    def Coarse_Change_Map_Generation(self, A, B, dynamic_inputs=None):
        layer_1_A, layer_2_A, layer_3_A, layer_4_A = self.backbone(A)
        layer_1_B, layer_2_B, layer_3_B, layer_4_B = self.backbone(B)
        
        if dynamic_inputs is not None and self.dynamic_adapters:
            assert len(dynamic_inputs) == len(self.dynamic_adapters), \
                f"Dynamic input count {len(dynamic_inputs)} does not match adapters {len(self.dynamic_adapters)}"
            for adapter, dyn_inp in zip(self.dynamic_adapters, dynamic_inputs):
                dyn_inp = dyn_inp.to(self.get_device())
                if dyn_inp.shape[2:] != A.shape[2:]:
                    dyn_inp = F.interpolate(dyn_inp, size=A.shape[2:], mode='bilinear', align_corners=True)
                layer_1_A = adapter[0](dyn_inp, layer_1_A)
                layer_2_A = adapter[1](dyn_inp, layer_2_A)
                layer_3_A = adapter[2](dyn_inp, layer_3_A)
                layer_4_A = adapter[3](dyn_inp, layer_4_A)
                layer_1_B = adapter[0](dyn_inp, layer_1_B)
                layer_2_B = adapter[1](dyn_inp, layer_2_B)
                layer_3_B = adapter[2](dyn_inp, layer_3_B)
                layer_4_B = adapter[3](dyn_inp, layer_4_B)

        layer_1 = self.conv_1(self.conv1(torch.cat((layer_1_A, layer_1_B), dim=1)))
        layer_2 = self.conv_2(self.conv2(torch.cat((layer_2_A, layer_2_B), dim=1)))
        layer_3 = self.conv_3(self.conv3(torch.cat((layer_3_A, layer_3_B), dim=1)))
        layer_4 = self.conv_4(self.conv4(torch.cat((layer_4_A, layer_4_B), dim=1)))

        Fusion = torch.cat((self.upsample8(layer_4), self.upsample4(layer_3), self.upsample2(layer_2), layer_1), 1)
        Guidance_out = self.coarse_out(Fusion)
        Coarse_out = self.upsample4(Guidance_out)

        return Guidance_out, Coarse_out, layer_1, layer_2, layer_3, layer_4


    def forward(self, A, B, dynamic_inputs=None, y=None):
        model_device = self.get_device()
        A = A.to(model_device)
        B = B.to(model_device)
        
        if dynamic_inputs is not None:
            dynamic_inputs = [d.to(model_device) for d in dynamic_inputs]
        
        Guidance_out, Coarse_out, layer1, layer2, layer3, layer4 = self.Coarse_Change_Map_Generation(A, B, dynamic_inputs)
        Changed_Guidance = self.sigmoid(Guidance_out)
        Non_Changed_Guidance = 1 - self.sigmoid(Guidance_out)

        if y is None:
            mu_prior, logvar_prior, _ = self.AUEM_prior(torch.cat((A, B), 1))
            z_prior = self.reparametrize(mu_prior, logvar_prior)
            Refined_out_prior = self.decoder_prior(Changed_Guidance, Non_Changed_Guidance, layer4, layer3, layer2, layer1, z_prior)
            return Refined_out_prior
        else:
            y = y.to(model_device)
            mu_prior, logvar_prior, dist_prior = self.AUEM_prior(torch.cat((A, B), 1))
            z_prior = self.reparametrize(mu_prior, logvar_prior)
            mu_post, logvar_post, dist_post = self.AUEM_post(torch.cat((A, B, y), 1))
            z_post = self.reparametrize(mu_post, logvar_post)
            kld = torch.mean(self.kl_divergence(dist_post, dist_prior))
            Refined_out_prior = self.decoder_prior(Changed_Guidance, Non_Changed_Guidance, layer4, layer3, layer2, layer1, z_prior)
            Refined_out_post = self.decoder_post(Changed_Guidance, Non_Changed_Guidance, layer4, layer3, layer2, layer1, z_post)
            return Refined_out_post


class Refined_Change_Map_Generation(nn.Module):
    """Refined change map generation module"""
    def __init__(self, latent_dim, num_classes):
        super(Refined_Change_Map_Generation, self).__init__()
        channel = 64

        self.down8 = nn.Upsample(scale_factor=0.125, mode='bilinear', align_corners=True)
        self.down4 = nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True)
        self.down2 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)

        self.noise_conv = nn.Conv2d(channel + latent_dim, channel, kernel_size=1, padding=0)
        self.spatial_axes = [2, 3]

        self.KGFEM4 = Knowledge_Guided_Feature_Enhancement_Module()
        self.KGFEM3 = Knowledge_Guided_Feature_Enhancement_Module()
        self.KGFEM2 = Knowledge_Guided_Feature_Enhancement_Module()
        self.KGFEM1 = Knowledge_Guided_Feature_Enhancement_Module()

        self.Fusion4 = AggUnit(channel)
        self.Fusion3 = AggUnit(channel)
        self.Fusion2 = AggUnit(channel)
        self.Fusion1 = AggUnit(channel)

        self.out_conv = nn.Sequential(
            nn.Conv2d(channel, 32, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, num_classes, kernel_size=1, stride=1, padding=0),
        )

    def tile(self, a, dim, n_tile):
        """Simulate tf.tile behavior ensuring tensor is on the same device"""
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(
            np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])
        ).to(a.device)
        return torch.index_select(a, dim, order_index)

    def forward(self, Changed_Guidance, Non_Changed_Guidance, layer4, layer3, layer2, layer1, z):
        layer4_height, layer4_width = layer4.shape[2], layer4.shape[3]
        batch_size = layer4.shape[0]
        latent_dim = z.shape[1]

        z_noise = z.view(batch_size, latent_dim, 1, 1)
        z_noise = z_noise.repeat(1, 1, layer4_height, layer4_width)
        if z_noise.shape[2:] != (layer4_height, layer4_width):
            z_noise = F.interpolate(z_noise, size=(layer4_height, layer4_width), mode='bilinear', align_corners=True)

        layer4 = torch.cat((layer4, z_noise), 1)
        layer4 = self.noise_conv(layer4)

        layer4 = self.KGFEM4(layer4, self.down8(Changed_Guidance), self.down8(Non_Changed_Guidance))
        layer3 = self.KGFEM3(layer3, self.down4(Changed_Guidance), self.down4(Non_Changed_Guidance))
        layer2 = self.KGFEM2(layer2, self.down2(Changed_Guidance), self.down2(Non_Changed_Guidance))
        layer1 = self.KGFEM1(layer1, Changed_Guidance, Non_Changed_Guidance)

        Fusion = self.Fusion4(layer4)
        Fusion = self.Fusion3(Fusion, layer3)
        Fusion = self.Fusion2(Fusion, layer2)
        Fusion = self.Fusion1(Fusion, layer1)
        Refined_out = self.out_conv(Fusion)

        return Refined_out


class Knowledge_Guided_Feature_Enhancement_Module(nn.Module):
    """Knowledge guided feature enhancement module"""
    def __init__(self,):
        super(Knowledge_Guided_Feature_Enhancement_Module, self).__init__()
        self.conv_p = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.conv_n = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, w_p, w_n):
        x_p = x * w_p
        x_n = x * w_n

        max_out_p, _ = torch.max(x_p, dim=1, keepdim=True)
        avg_out_p = torch.mean(x_p, dim=1, keepdim=True)
        spatial_out_p = self.sigmoid(self.conv_p(torch.cat([max_out_p, avg_out_p], dim=1)))
        x_p = spatial_out_p * x_p

        max_out_n, _ = torch.max(x_n, dim=1, keepdim=True)
        avg_out_n = torch.mean(x_n, dim=1, keepdim=True)
        spatial_out_n = self.sigmoid(self.conv_n(torch.cat([max_out_n, avg_out_n], dim=1)))
        x_n = spatial_out_n * x_n

        return x + x_p + x_n


class BasicConv2d(nn.Module):
    """Basic convolution + BN + ReLU"""
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class ResidualConvUnit(nn.Module):
    """Residual convolutional unit"""
    def __init__(self, features):
        super(ResidualConvUnit, self).__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        return out + x


class AggUnit(nn.Module):
    """Aggregation unit"""
    def __init__(self, features):
        super(AggUnit, self).__init__()
        self.resConfUnit1 = ResidualConvUnit(features)

    def forward(self, *xs):
        output = xs[0]
        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])
        output = F.interpolate(output, scale_factor=2, mode="bilinear", align_corners=True)
        return output


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import time
    start_time = time.time()

    # Test 1: Standard input
    print("Test 1: Standard input")
    x_before = torch.randn(2, 12, 256, 256)
    x_after = torch.randn(2, 12, 256, 256)
    dynamic_inputs = [torch.randn(2, 4, 256, 256), torch.randn(2, 3, 256, 256)]
    model = UABCDWithDynamic(fixed_in_channels=12, latent_dim=128, dynamic_in_channels=[4, 3])
    output = model(x_before, x_after, dynamic_inputs)
    params = count_parameters(model)
    print(f"Output shape: {output.shape}, Expected: (2, 2, 256, 256)")
    print(f"Model parameters: {params:,}\n")

    # Test 2: Different bands and sizes
    print("Test 2: Different bands and sizes")
    x_before = torch.randn(3, 8, 512, 512)
    x_after = torch.randn(3, 8, 512, 512)
    dynamic_inputs = [torch.randn(3, 5, 512, 512)]
    model = UABCDWithDynamic(fixed_in_channels=8, latent_dim=128, dynamic_in_channels=[5])
    output = model(x_before, x_after, dynamic_inputs)
    params = count_parameters(model)
    print(f"Output shape: {output.shape}, Expected: (3, 2, 512, 512)")
    print(f"Model parameters: {params:,}\n")

    # Test 3: No dynamic branch
    print("Test 3: No dynamic branch")
    x_before = torch.randn(1, 10, 128, 128)
    x_after = torch.randn(1, 10, 128, 128)
    model = UABCDWithDynamic(fixed_in_channels=10, latent_dim=128)
    output = model(x_before, x_after)
    params = count_parameters(model)
    print(f"Output shape: {output.shape}, Expected: (1, 2, 128, 128)")
    print(f"Model parameters: {params:,}\n")

    # Test 4: Multiple dynamic branches
    print("Test 4: Multiple dynamic branches")
    x_before = torch.randn(2, 12, 256, 256)
    x_after = torch.randn(2, 12, 256, 256)
    dynamic_inputs = [torch.randn(2, 4, 256, 256), torch.randn(2, 3, 256, 256), torch.randn(2, 2, 256, 256)]
    model = UABCDWithDynamic(fixed_in_channels=12, latent_dim=128, dynamic_in_channels=[4, 3, 2])
    output = model(x_before, x_after, dynamic_inputs)
    params = count_parameters(model)
    print(f"Output shape: {output.shape}, Expected: (2, 2, 256, 256)")
    print(f"Model parameters: {params:,}\n")

    end_time = time.time()
    print(f"Total test duration: {end_time - start_time:.2f} seconds")


