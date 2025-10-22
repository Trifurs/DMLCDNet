import torch
import torch.nn as nn
import numpy as np
import math
from typing import List, Optional

class SSFC(torch.nn.Module):
    """Self-Similarity Feature Calibration Module"""
    def __init__(self, in_ch):
        super(SSFC, self).__init__()

    def forward(self, x):
        _, _, h, w = x.size()

        q = x.mean(dim=[2, 3], keepdim=True)
        k = x
        square = (k - q).pow(2)
        sigma = square.sum(dim=[2, 3], keepdim=True) / (h * w)
        att_score = square / (2 * sigma + np.finfo(np.float32).eps) + 0.5
        att_weight = nn.Sigmoid()(att_score)

        return x * att_weight


class CMConv(nn.Module):
    """Centralized Multi-Dilation Convolution"""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=3, groups=1, dilation_set=4,
                 bias=False):
        super(CMConv, self).__init__()
        self.prim = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding=dilation, dilation=dilation,
                              groups=groups * dilation_set, bias=bias)
        self.prim_shift = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding=2 * dilation, dilation=2 * dilation,
                                    groups=groups * dilation_set, bias=bias)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=bias)

        def backward_hook(grad):
            out = grad.clone()
            out[self.mask] = 0
            return out

        self.mask = torch.zeros(self.conv.weight.shape, dtype=torch.bool, device=self.conv.weight.device)
        _in_channels = in_ch // (groups * dilation_set)
        _out_channels = out_ch // (groups * dilation_set)
        for i in range(dilation_set):
            for j in range(groups):
                self.mask[(i + j * groups) * _out_channels: (i + j * groups + 1) * _out_channels,
                i * _in_channels: (i + 1) * _in_channels, :, :] = True
                self.mask[((i + dilation_set // 2) % dilation_set + j * groups) *
                          _out_channels: ((i + dilation_set // 2) % dilation_set + j * groups + 1) * _out_channels,
                i * _in_channels: (i + 1) * _in_channels, :, :] = True
        self.conv.weight.data[self.mask] = 0
        self.conv.weight.register_hook(backward_hook)
        self.groups = groups

    def forward(self, x):
        x_split = (z.chunk(2, dim=1) for z in x.chunk(self.groups, dim=1))
        x_merge = torch.cat(tuple(torch.cat((x2, x1), dim=1) for (x1, x2) in x_split), dim=1)
        x_shift = self.prim_shift(x_merge)
        return self.prim(x) + self.conv(x) + x_shift


class MSDConv_SSFC(nn.Module):
    """Multi-Scale Dilated Convolution with SSFC"""
    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1, padding=0, ratio=2, aux_k=3, dilation=3):
        super(MSDConv_SSFC, self).__init__()
        self.out_ch = out_ch
        native_ch = math.ceil(out_ch / ratio)
        aux_ch = native_ch * (ratio - 1)

        # native feature maps
        self.native = nn.Sequential(
            nn.Conv2d(in_ch, native_ch, kernel_size, stride, padding=padding, dilation=1, bias=False),
            nn.BatchNorm2d(native_ch),
            nn.ReLU(inplace=True),
        )

        # auxiliary feature maps
        self.aux = nn.Sequential(
            CMConv(native_ch, aux_ch, aux_k, 1, padding=1, groups=max(1, int(native_ch / 4)), dilation=dilation,
                   bias=False),
            nn.BatchNorm2d(aux_ch),
            nn.ReLU(inplace=True),
        )

        self.att = SSFC(aux_ch)

    def forward(self, x):
        x1 = self.native(x)
        x2 = self.att(self.aux(x1))
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_ch, :, :]


class First_DoubleConv(nn.Module):
    """First Double Convolution Block"""
    def __init__(self, in_ch, out_ch):
        super(First_DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class DoubleConv(nn.Module):
    """Standard Double Convolution Block"""
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.Conv = nn.Sequential(
            MSDConv_SSFC(in_ch, out_ch, dilation=3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            MSDConv_SSFC(out_ch, out_ch, dilation=3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.Conv(input)


class DynamicFeatureAdapter(nn.Module):
    """Dynamic Feature Adapter for fusing different channel inputs into main network"""
    def __init__(self, in_channels, target_channels, reduction=4):
        super().__init__()
        self.target_channels = target_channels
        
        # Channel mapping
        self.channel_mapper = nn.Sequential(
            nn.Conv2d(in_channels, target_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(target_channels),
            nn.ReLU(inplace=True)
        )
        
        # Attention gating
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(target_channels, max(1, target_channels // reduction), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, target_channels // reduction), target_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, target_feat):
        x = self.channel_mapper(x)
        if x.shape[2:] != target_feat.shape[2:]:
            x = torch.nn.functional.interpolate(
                x, size=target_feat.shape[2:], mode='bilinear', align_corners=True
            )
        attn = self.attention(x)
        return target_feat + x * attn


class USSFCNetWithDynamic(nn.Module):
    """USSFCNet with optional dynamic input branches"""
    def __init__(self, fixed_in_channels, out_ch=2, ratio=0.5, dynamic_in_channels: Optional[List[int]] = None):
        super(USSFCNetWithDynamic, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        base_ch = int(32 * ratio)

        self.Conv1_1 = First_DoubleConv(fixed_in_channels, base_ch)
        self.Conv1_2 = First_DoubleConv(fixed_in_channels, base_ch)
        self.Conv2_1 = DoubleConv(base_ch, base_ch * 2)
        self.Conv2_2 = DoubleConv(base_ch, base_ch * 2)
        self.Conv3_1 = DoubleConv(base_ch * 2, base_ch * 4)
        self.Conv3_2 = DoubleConv(base_ch * 2, base_ch * 4)
        self.Conv4_1 = DoubleConv(base_ch * 4, base_ch * 8)
        self.Conv4_2 = DoubleConv(base_ch * 4, base_ch * 8)
        self.Conv5_1 = DoubleConv(base_ch * 8, base_ch * 16)
        self.Conv5_2 = DoubleConv(base_ch * 8, base_ch * 16)

        self.Up5 = nn.ConvTranspose2d(base_ch * 16, base_ch * 8, 2, stride=2)
        self.Up_conv5 = DoubleConv(base_ch * 16, base_ch * 8)

        self.Up4 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.Up_conv4 = DoubleConv(base_ch * 8, base_ch * 4)

        self.Up3 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.Up_conv3 = DoubleConv(base_ch * 4, base_ch * 2)

        self.Up2 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.Up_conv2 = DoubleConv(base_ch * 2, base_ch)

        self.Conv_1x1 = nn.Conv2d(base_ch, out_ch, kernel_size=1, stride=1, padding=0)
        
        # Dynamic branch adapters
        self.dynamic_branch_count = len(dynamic_in_channels) if dynamic_in_channels else 0
        self.dynamic_adapters = nn.ModuleList()
        if dynamic_in_channels:
            target_channels = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16]
            for in_ch in dynamic_in_channels:
                level_adapters = nn.ModuleList([
                    DynamicFeatureAdapter(in_ch, target_channels[0]),
                    DynamicFeatureAdapter(in_ch, target_channels[1]),
                    DynamicFeatureAdapter(in_ch, target_channels[2]),
                    DynamicFeatureAdapter(in_ch, target_channels[3]),
                    DynamicFeatureAdapter(in_ch, target_channels[4])
                ])
                self.dynamic_adapters.append(level_adapters)

    def forward(self, x1, x2, dynamic_inputs=None):
        c1_1 = self.Conv1_1(x1)
        c2_1 = self.Maxpool(c1_1); c2_1 = self.Conv2_1(c2_1)
        c3_1 = self.Maxpool(c2_1); c3_1 = self.Conv3_1(c3_1)
        c4_1 = self.Maxpool(c3_1); c4_1 = self.Conv4_1(c4_1)
        c5_1 = self.Maxpool(c4_1); c5_1 = self.Conv5_1(c5_1)

        c1_2 = self.Conv1_2(x2)
        c2_2 = self.Maxpool(c1_2); c2_2 = self.Conv2_2(c2_2)
        c3_2 = self.Maxpool(c2_2); c3_2 = self.Conv3_2(c3_2)
        c4_2 = self.Maxpool(c3_2); c4_2 = self.Conv4_2(c4_2)
        c5_2 = self.Maxpool(c4_2); c5_2 = self.Conv5_2(c5_2)

        if dynamic_inputs is not None and self.dynamic_adapters:
            assert len(dynamic_inputs) == len(self.dynamic_adapters)
            for adapter, dyn_inp in zip(self.dynamic_adapters, dynamic_inputs):
                if dyn_inp.shape[2:] != x1.shape[2:]:
                    dyn_inp = torch.nn.functional.interpolate(
                        dyn_inp, size=x1.shape[2:], mode='bilinear', align_corners=True
                    )
                c1_1 = adapter[0](dyn_inp, c1_1); c2_1 = adapter[1](dyn_inp, c2_1)
                c3_1 = adapter[2](dyn_inp, c3_1); c4_1 = adapter[3](dyn_inp, c4_1); c5_1 = adapter[4](dyn_inp, c5_1)
                c1_2 = adapter[0](dyn_inp, c1_2); c2_2 = adapter[1](dyn_inp, c2_2)
                c3_2 = adapter[2](dyn_inp, c3_2); c4_2 = adapter[3](dyn_inp, c4_2); c5_2 = adapter[4](dyn_inp, c5_2)

        x1 = torch.abs(torch.sub(c1_1, c1_2))
        x2 = torch.abs(torch.sub(c2_1, c2_2))
        x3 = torch.abs(torch.sub(c3_1, c3_2))
        x4 = torch.abs(torch.sub(c4_1, c4_2))
        x5 = torch.abs(torch.sub(c5_1, c5_2))

        d5 = self.Up5(x5); d5 = torch.cat((x4, d5), dim=1); d5 = self.Up_conv5(d5)
        d4 = self.Up4(d5); d4 = torch.cat((x3, d4), dim=1); d4 = self.Up_conv4(d4)
        d3 = self.Up3(d4); d3 = torch.cat((x2, d3), dim=1); d3 = self.Up_conv3(d3)
        d2 = self.Up2(d3); d2 = torch.cat((x1, d2), dim=1); d2 = self.Up_conv2(d2)

        return self.Conv_1x1(d2)


def count_parameters(model):
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import time
    start_time = time.time()

    print("Test 1: Standard input")
    x_before = torch.randn(2, 12, 256, 256)
    x_after = torch.randn(2, 12, 256, 256)
    dynamic_inputs = [torch.randn(2, 4, 256, 256), torch.randn(2, 3, 256, 256)]
    model = USSFCNetWithDynamic(fixed_in_channels=12, dynamic_in_channels=[4, 3])
    output = model(x_before, x_after, dynamic_inputs)
    params = count_parameters(model)
    print(f"Output shape: {output.shape}, Expected: (2, 2, 256, 256)")
    print(f"Model parameters: {params:,}\n")

    print("Test 2: Different channels and sizes")
    x_before = torch.randn(3, 8, 512, 512)
    x_after = torch.randn(3, 8, 512, 512)
    dynamic_inputs = [torch.randn(3, 5, 512, 512)]
    model = USSFCNetWithDynamic(fixed_in_channels=8, dynamic_in_channels=[5])
    output = model(x_before, x_after, dynamic_inputs)
    params = count_parameters(model)
    print(f"Output shape: {output.shape}, Expected: (3, 2, 512, 512)")
    print(f"Model parameters: {params:,}\n")

    print("Test 3: Without dynamic branch")
    x_before = torch.randn(1, 10, 128, 128)
    x_after = torch.randn(1, 10, 128, 128)
    model = USSFCNetWithDynamic(fixed_in_channels=10)
    output = model(x_before, x_after)
    params = count_parameters(model)
    print(f"Output shape: {output.shape}, Expected: (1, 2, 128, 128)")
    print(f"Model parameters: {params:,}\n")

    print("Test 4: Multiple dynamic branches")
    x_before = torch.randn(2, 12, 256, 256)
    x_after = torch.randn(2, 12, 256, 256)
    dynamic_inputs = [torch.randn(2, 1, 256, 256) for _ in range(7)]
    model = USSFCNetWithDynamic(fixed_in_channels=12, dynamic_in_channels=[1]*7)
    output = model(x_before, x_after, dynamic_inputs)
    params = count_parameters(model)
    print(f"Output shape: {output.shape}, Expected: (2, 2, 256, 256)")
    print(f"Model parameters: {params:,}\n")

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")

