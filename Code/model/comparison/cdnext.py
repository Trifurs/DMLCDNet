import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from timm.layers import trunc_normal_, DropPath

class LayerNorm(nn.Module):
    """LayerNorm supporting channels_last and channels_first formats"""
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    """Basic ConvNeXt block"""
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                 requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    """Lightweight ConvNeXt encoder"""
    def __init__(self, in_chans=3, depths=[2, 2, 2, 2], dims=[32, 64, 128, 256], drop_path_rate=0.):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) 
                  for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            features.append(x)
        return features


class SqueezeDoubleConv(nn.Module):
    """Squeeze double convolution module"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.GELU())
        self.double_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
        )
        self.acfun = nn.GELU()

    def forward(self, x):
        x = self.squeeze(x)
        block_x = self.double_conv(x)
        return self.acfun(x + block_x)


class SpatiotemporalAttention(nn.Module):
    """Spatiotemporal attention module"""
    def __init__(self, in_channels, inter_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels if inter_channels else in_channels // 2

        self.g = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=self.inter_channels,
                      kernel_size=1, stride=1, padding=0)
        )
                         
        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=in_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels)
        )

        self.theta = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=self.inter_channels,
                      kernel_size=1, stride=1, padding=0),
        )
        self.phi = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=self.inter_channels,
                      kernel_size=1, stride=1, padding=0),
        )
        self.softmax = nn.Softmax(dim=-2)
        
    def forward(self, x1, x2):
        batch_size = x1.size(0)
        g_x1 = self.g(x1).reshape(batch_size, self.inter_channels, -1)
        g_x2 = self.g(x2).reshape(batch_size, self.inter_channels, -1)

        theta_x1 = self.theta(x1).reshape(batch_size, self.inter_channels, -1)
        theta_x2 = theta_x1.permute(0, 2, 1)
        
        phi_x2 = self.phi(x2).reshape(batch_size, self.inter_channels, -1)

        energy_space = torch.matmul(theta_x2, phi_x2)
        attention = self.softmax(energy_space)

        y1 = torch.matmul(g_x1, attention).contiguous()
        y2 = torch.matmul(g_x2, attention.permute(0, 2, 1)).contiguous()
        
        y1 = y1.reshape(batch_size, self.inter_channels, *x2.size()[2:])
        y2 = y2.reshape(batch_size, self.inter_channels, *x1.size()[2:])
        
        return x1 + self.W(y1), x2 + self.W(y2)


class DynamicFeatureAdapter(nn.Module):
    """Dynamic feature adapter for fusing dynamic branch inputs"""
    def __init__(self, in_channels, target_channels, reduction=4):
        super().__init__()
        self.target_channels = target_channels
        
        self.channel_mapper = nn.Sequential(
            nn.Conv2d(in_channels, target_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(target_channels),
            nn.GELU()
        )
        
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(target_channels, target_channels // reduction, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(target_channels // reduction, target_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, target_feat):
        x = self.channel_mapper(x)
        if x.shape[2:] != target_feat.shape[2:]:
            x = F.interpolate(
                x, 
                size=target_feat.shape[2:], 
                mode='bilinear', 
                align_corners=True
            )
        attn = self.attention(x)
        return target_feat + x * attn


class CDNeXtWithDynamic(nn.Module):
    """Change detection model with dynamic branch input support"""
    def __init__(self, fixed_in_channels=3, n_classes=2, img_size=64,
                 dynamic_in_channels: Optional[List[int]] = None):
        super().__init__()
        self.fixed_in_channels = fixed_in_channels
        self.n_classes = n_classes
        self.img_size = img_size
        
        self.depths = [2, 2, 2, 2]
        self.dims = [32, 64, 128, 256]
        self.encoder = ConvNeXt(in_chans=fixed_in_channels, depths=self.depths, dims=self.dims)
        
        self.temporal_attentions = nn.ModuleList([
            SpatiotemporalAttention(self.dims[0]),
            SpatiotemporalAttention(self.dims[1]),
            SpatiotemporalAttention(self.dims[2]),
            SpatiotemporalAttention(self.dims[3])
        ])
        
        self.change_conv = nn.ModuleList([
            SqueezeDoubleConv(self.dims[0] * 2 + self.dims[1], self.dims[0]),
            SqueezeDoubleConv(self.dims[1] * 2 + self.dims[2], self.dims[1]),
            SqueezeDoubleConv(self.dims[2] * 2 + self.dims[3], self.dims[2]),
            SqueezeDoubleConv(self.dims[3] * 2, self.dims[3])
        ])
        
        self.final_fusion = SqueezeDoubleConv(sum(self.dims), self.dims[-1])
        self.final_upsample = nn.Sequential(
            nn.Conv2d(self.dims[-1], self.dims[-1] // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.dims[-1] // 2),
            nn.GELU(),
            nn.Conv2d(self.dims[-1] // 2, n_classes, kernel_size=1)
        )
        
        self.dynamic_adapters = nn.ModuleList()
        if dynamic_in_channels:
            for in_ch in dynamic_in_channels:
                level_adapters = nn.ModuleList([
                    DynamicFeatureAdapter(in_ch, self.dims[0]),
                    DynamicFeatureAdapter(in_ch, self.dims[1]),
                    DynamicFeatureAdapter(in_ch, self.dims[2]),
                    DynamicFeatureAdapter(in_ch, self.dims[3])
                ])
                self.dynamic_adapters.append(level_adapters)
        
        self.device_ids = None

    def forward(self, x1, x2, dynamic_inputs=None):
        input_size = x1.shape[2:]
        feats1 = self.encoder(x1)
        feats2 = self.encoder(x2)
        
        if dynamic_inputs is not None and self.dynamic_adapters:
            assert len(dynamic_inputs) == len(self.dynamic_adapters), \
                f"Dynamic inputs {len(dynamic_inputs)} and adapters {len(self.dynamic_adapters)} mismatch"
            for adapter, dyn_inp in zip(self.dynamic_adapters, dynamic_inputs):
                if dyn_inp.shape[2:] != x1.shape[2:]:
                    dyn_inp = F.interpolate(
                        dyn_inp, 
                        size=x1.shape[2:], 
                        mode='bilinear', 
                        align_corners=True
                    )
                for i in range(4):
                    feats1[i] = adapter[i](dyn_inp, feats1[i])
                    feats2[i] = adapter[i](dyn_inp, feats2[i])
        
        fusion_features = []
        change = None
        for i in reversed(range(4)):
            feat1, feat2 = self.temporal_attentions[i](feats1[i], feats2[i])
            if i == 3:
                change = torch.cat([feat1, feat2], dim=1)
            else:
                change = torch.cat([feat1, feat2, change], dim=1)
            change = self.change_conv[i](change)
            fusion_features.append(change)
            if i > 0:
                change = F.interpolate(change, scale_factor=2, mode='bilinear', align_corners=True)
        
        for i in range(4):
            fusion_features[i] = F.interpolate(
                fusion_features[i], 
                size=input_size, 
                mode='bilinear', 
                align_corners=True
            )
        
        final = torch.cat(fusion_features, dim=1)
        final = self.final_fusion(final)
        if final.shape[2:] != input_size:
            final = F.interpolate(final, size=input_size, mode='bilinear', align_corners=True)
        
        output = self.final_upsample(final)
        return output

    def train(self, mode=True):
        """Override train mode to support multi-GPU"""
        super().train(mode)
        return self

    def cuda(self, device=None):
        """Override cuda method to support multi-GPU"""
        if device is None and torch.cuda.device_count() > 1 and not self.device_ids:
            self.device_ids = list(range(torch.cuda.device_count()))
            return nn.DataParallel(self, device_ids=self.device_ids).cuda()
        return super().cuda(device)


def count_parameters(model):
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import time
    start_time = time.time()

    print("Test1: Standard input")
    x_before = torch.randn(2, 12, 256, 256)
    x_after = torch.randn(2, 12, 256, 256)
    dynamic_inputs = [torch.randn(2, 4, 256, 256), torch.randn(2, 3, 256, 256)]
    model = CDNeXtWithDynamic(fixed_in_channels=12, dynamic_in_channels=[4, 3], img_size=256)
    output = model(x_before, x_after, dynamic_inputs)
    params = count_parameters(model)
    print(f"Output shape: {output.shape} Expected: (2, 2, 256, 256)")
    print(f"Model parameters: {params:,}\n")

    print("Test2: Different channels and sizes")
    x_before = torch.randn(3, 8, 512, 512)
    x_after = torch.randn(3, 8, 512, 512)
    dynamic_inputs = [torch.randn(3, 5, 512, 512)]
    model = CDNeXtWithDynamic(fixed_in_channels=8, dynamic_in_channels=[5], img_size=512)
    output = model(x_before, x_after, dynamic_inputs)
    params = count_parameters(model)
    print(f"Output shape: {output.shape} Expected: (3, 2, 512, 512)")
    print(f"Model parameters: {params:,}\n")

    print("Test3: No dynamic branch")
    x_before = torch.randn(1, 10, 128, 128)
    x_after = torch.randn(1, 10, 128, 128)
    model = CDNeXtWithDynamic(fixed_in_channels=10, img_size=128)
    output = model(x_before, x_after)
    params = count_parameters(model)
    print(f"Output shape: {output.shape} Expected: (1, 2, 128, 128)")
    print(f"Model parameters: {params:,}\n")

    print("Test4: Multiple dynamic branches")
    x_before = torch.randn(2, 12, 256, 256)
    x_after = torch.randn(2, 12, 256, 256)
    dynamic_inputs = [torch.randn(2, 1, 256, 256) for _ in range(7)]
    model = CDNeXtWithDynamic(fixed_in_channels=12, dynamic_in_channels=[1]*7, img_size=256)
    output = model(x_before, x_after, dynamic_inputs)
    params = count_parameters(model)
    print(f"Output shape: {output.shape} Expected: (2, 2, 256, 256)")
    print(f"Model parameters: {params:,}\n")

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")
