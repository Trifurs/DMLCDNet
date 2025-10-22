import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

def conv3x3(in_planes, out_planes, stride=1, padding=1):
    """3x3 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, bias=False)

class BasicConv2d(nn.Module):
    """Basic 2D convolution with BatchNorm and ReLU"""
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

class ChannelAttention(nn.Module):
    """Channel attention module"""
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        mid = max(1, in_planes // ratio)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_planes, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(self.avg_pool(x))

class SpatialAttention(nn.Module):
    """Spatial attention module"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sig(self.conv(max_out))

class LightweightGCM(nn.Module):
    """Simplified GCM with two branches instead of four"""
    def __init__(self, in_channel, out_channel):
        super(LightweightGCM, self).__init__()
        self.branch0 = BasicConv2d(in_channel, out_channel, kernel_size=1, padding=0)
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, kernel_size=1, padding=0),
            BasicConv2d(out_channel, out_channel, kernel_size=3, padding=2, dilation=2)
        )
        self.conv_cat = BasicConv2d(2*out_channel, out_channel, kernel_size=3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x_cat = self.conv_cat(torch.cat((x0, x1), dim=1))
        out = self.relu(x_cat + self.conv_res(x))
        return out

class TinyAggregationInit(nn.Module):
    """Lightweight attention generator, outputs 1-channel attention map (1/4 resolution)"""
    def __init__(self, channel):
        super(TinyAggregationInit, self).__init__()
        self.conv1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv2 = BasicConv2d(channel*2, channel, 3, padding=1)
        self.conv3 = nn.Conv2d(channel, 1, 1)

    def forward(self, x1, x2, x3):
        up2 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        fuse2 = self.conv1(up2) * x2
        up1 = F.interpolate(fuse2, scale_factor=2, mode='bilinear', align_corners=True)
        fuse1 = self.conv1(up1) * x3
        down1 = F.interpolate(fuse1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        cat = torch.cat([x1, down1], dim=1)
        cat = self.conv2(cat)
        att = self.conv3(cat)
        return att

class TinyAggregationFinal(nn.Module):
    """Lightweight final aggregation, returns fused features"""
    def __init__(self, channel):
        super(TinyAggregationFinal, self).__init__()
        self.conv = BasicConv2d(channel*3, channel, 3, padding=1)

    def forward(self, x1, x2, x3):
        u2 = F.interpolate(x2, size=x1.shape[2:], mode='bilinear', align_corners=True)
        u3 = F.interpolate(x3, size=x1.shape[2:], mode='bilinear', align_corners=True)
        cat = torch.cat([x1, u2, u3], dim=1)
        return self.conv(cat)

class Refine(nn.Module):
    """Simple refinement with up/down sampling"""
    def __init__(self):
        super(Refine, self).__init__()

    def forward(self, attention, x1, x2, x3):
        att_up4 = F.interpolate(attention, size=x1.shape[2:], mode='bilinear', align_corners=True)
        att_up2 = F.interpolate(attention, size=x2.shape[2:], mode='bilinear', align_corners=True)
        att_up1 = F.interpolate(attention, size=x3.shape[2:], mode='bilinear', align_corners=True)
        x1 = x1 + x1 * att_up4
        x2 = x2 + x2 * att_up2
        x3 = x3 + x3 * att_up1
        return x1, x2, x3

class DynamicFeatureAdapter(nn.Module):
    """Lightweight dynamic adapter: 1x1 projection + simple attention"""
    def __init__(self, in_channels, target_channels, reduction=4):
        super().__init__()
        self.channel_mapper = nn.Sequential(
            nn.Conv2d(in_channels, target_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(target_channels),
            nn.ReLU(inplace=True)
        )
        mid = max(1, target_channels // reduction)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(target_channels, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, target_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x, target_feat):
        x = self.channel_mapper(x)
        if x.shape[2:] != target_feat.shape[2:]:
            x = F.interpolate(x, size=target_feat.shape[2:], mode='bilinear', align_corners=True)
        att = self.attention(x)
        return target_feat + x * att

class SimpleEncoder(nn.Module):
    """Lightweight encoder: three layers (original, 1/2, 1/4)"""
    def __init__(self, in_channels, base_channels=32):
        super().__init__()
        self.conv0 = BasicConv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.conv0b = BasicConv2d(base_channels, base_channels, kernel_size=3, padding=1)
        self.down1 = nn.Sequential(
            BasicConv2d(base_channels, base_channels*2, kernel_size=3, stride=2, padding=1),
            BasicConv2d(base_channels*2, base_channels*2, kernel_size=3, padding=1)
        )
        self.down2 = nn.Sequential(
            BasicConv2d(base_channels*2, base_channels*4, kernel_size=3, stride=2, padding=1),
            BasicConv2d(base_channels*4, base_channels*4, kernel_size=3, padding=1)
        )

    def forward(self, x):
        l1 = self.conv0(x)
        l1 = self.conv0b(l1)
        l2 = self.down1(l1)
        l3 = self.down2(l2)
        return l1, l2, l3

class C2FNetWithDynamic(nn.Module):
    """Compact model with dynamic adaptation and attention"""
    def __init__(self, fixed_in_channels, output_nc=2, dynamic_in_channels: Optional[List[int]] = None, base_ch=32):
        super().__init__()
        self.base_ch = base_ch
        self.dynamic_branch_count = len(dynamic_in_channels) if dynamic_in_channels else 0
        self.encoder = SimpleEncoder(fixed_in_channels, base_channels=base_ch)
        self.rfb1 = LightweightGCM(base_ch, base_ch//4)
        self.rfb2 = LightweightGCM(base_ch*2, base_ch//4)
        self.rfb3 = LightweightGCM(base_ch*4, base_ch//4)
        small_channel = base_ch//4
        self.agg1 = TinyAggregationInit(small_channel)
        self.HA = Refine()
        self.rfb1_2 = LightweightGCM(small_channel, small_channel)
        self.rfb2_2 = LightweightGCM(small_channel, small_channel)
        self.rfb3_2 = LightweightGCM(small_channel, small_channel)
        self.agg2 = TinyAggregationFinal(small_channel)
        self.atten_A_channel_1 = ChannelAttention(base_ch)
        self.atten_A_channel_2 = ChannelAttention(base_ch*2)
        self.atten_A_channel_3 = ChannelAttention(base_ch*4)
        self.atten_A_spatial_1 = SpatialAttention()
        self.atten_A_spatial_2 = SpatialAttention()
        self.atten_A_spatial_3 = SpatialAttention()
        self.atten_B_channel_1 = ChannelAttention(base_ch)
        self.atten_B_channel_2 = ChannelAttention(base_ch*2)
        self.atten_B_channel_3 = ChannelAttention(base_ch*4)
        self.atten_B_spatial_1 = SpatialAttention()
        self.atten_B_spatial_2 = SpatialAttention()
        self.atten_B_spatial_3 = SpatialAttention()
        self.dynamic_adapters = nn.ModuleList()
        if dynamic_in_channels:
            for in_ch in dynamic_in_channels:
                level_adapters = nn.ModuleList([
                    DynamicFeatureAdapter(in_ch, base_ch),
                    DynamicFeatureAdapter(in_ch, base_ch*2),
                    DynamicFeatureAdapter(in_ch, base_ch*4)
                ])
                self.dynamic_adapters.append(level_adapters)
        self.head = nn.Sequential(
            nn.Conv2d(small_channel, max(8, small_channel), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(max(8, small_channel)),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(8, small_channel), output_nc, kernel_size=1)
        )

    def forward(self, A, B, dynamic_inputs=None):
        l1_A, l2_A, l3_A = self.encoder(A)
        l1_B, l2_B, l3_B = self.encoder(B)
        if dynamic_inputs is not None and self.dynamic_adapters:
            assert len(dynamic_inputs) == len(self.dynamic_adapters), \
                f"Number of dynamic inputs {len(dynamic_inputs)} does not match adapters {len(self.dynamic_adapters)}"
            for adapter, dyn_inp in zip(self.dynamic_adapters, dynamic_inputs):
                if dyn_inp.shape[2:] != A.shape[2:]:
                    dyn_inp = F.interpolate(dyn_inp, size=A.shape[2:], mode='bilinear', align_corners=True)
                l1_A = adapter[0](dyn_inp, l1_A)
                l2_A = adapter[1](dyn_inp, l2_A)
                l3_A = adapter[2](dyn_inp, l3_A)
                l1_B = adapter[0](dyn_inp, l1_B)
                l2_B = adapter[1](dyn_inp, l2_B)
                l3_B = adapter[2](dyn_inp, l3_B)
        l1_A = l1_A * self.atten_A_channel_1(l1_A)
        l1_A = l1_A * self.atten_A_spatial_1(l1_A)
        l1_B = l1_B * self.atten_B_channel_1(l1_B)
        l1_B = l1_B * self.atten_B_spatial_1(l1_B)
        layer1 = l1_A + l1_B
        l2_A = l2_A * self.atten_A_channel_2(l2_A)
        l2_A = l2_A * self.atten_A_spatial_2(l2_A)
        l2_B = l2_B * self.atten_B_channel_2(l2_B)
        l2_B = l2_B * self.atten_B_spatial_2(l2_B)
        layer2 = l2_A + l2_B
        l3_A = l3_A * self.atten_A_channel_3(l3_A)
        l3_A = l3_A * self.atten_A_spatial_3(l3_A)
        l3_B = l3_B * self.atten_B_channel_3(l3_B)
        l3_B = l3_B * self.atten_B_spatial_3(l3_B)
        layer3 = l3_A + l3_B
        layer1_gcm = self.rfb1(layer1)
        layer2_gcm = self.rfb2(layer2)
        layer3_gcm = self.rfb3(layer3)
        attention_map = self.agg1(layer3_gcm, layer2_gcm, layer1_gcm)
        layer1_ref, layer2_ref, layer3_ref = self.HA(torch.sigmoid(attention_map), layer1_gcm, layer2_gcm, layer3_gcm)
        f1 = self.rfb1_2(layer1_ref)
        f2 = self.rfb2_2(layer2_ref)
        f3 = self.rfb3_2(layer3_ref)
        y = self.agg2(f3, f2, f1)
        y = self.head(y)
        y = F.interpolate(y, size=A.shape[2:], mode='bilinear', align_corners=True)
        return y

    def prepare_for_distributed(self, device):
        """Placeholder for distributed preparation"""
        self.to(device)
        return self

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ----------------- Test Entry -----------------
if __name__ == "__main__":
    import time
    start_time = time.time()

    # Test1: standard input
    print("Test1: Standard input")
    x_before = torch.randn(2, 12, 256, 256)
    x_after = torch.randn(2, 12, 256, 256)
    dynamic_inputs = [
        torch.randn(2, 4, 256, 256),
        torch.randn(2, 3, 256, 256)
    ]
    model = C2FNetWithDynamic(fixed_in_channels=12, dynamic_in_channels=[4, 3])
    output = model(x_before, x_after, dynamic_inputs)
    params = count_parameters(model)
    print(f"Output shape: {output.shape}, Expected: (2, 2, 256, 256)")
    print(f"Model parameters: {params:,}\n")

    # Test2: different bands and sizes
    print("Test2: Different bands and sizes")
    x_before = torch.randn(3, 8, 512, 512)
    x_after = torch.randn(3, 8, 512, 512)
    dynamic_inputs = [torch.randn(3, 5, 512, 512)]
    model = C2FNetWithDynamic(fixed_in_channels=8, dynamic_in_channels=[5])
    output = model(x_before, x_after, dynamic_inputs)
    params = count_parameters(model)
    print(f"Output shape: {output.shape}, Expected: (3, 2, 512, 512)")
    print(f"Model parameters: {params:,}\n")

    # Test3: no dynamic branch
    print("Test3: No dynamic branch")
    x_before = torch.randn(1, 10, 128, 128)
    x_after = torch.randn(1, 10, 128, 128)
    model = C2FNetWithDynamic(fixed_in_channels=10)
    output = model(x_before, x_after)
    params = count_parameters(model)
    print(f"Output shape: {output.shape}, Expected: (1, 2, 128, 128)")
    print(f"Model parameters: {params:,}\n")

    # Test4: multiple dynamic branches (7 single channels)
    print("Test4: Multiple dynamic branches")
    x_before = torch.randn(2, 12, 256, 256)
    x_after = torch.randn(2, 12, 256, 256)
    dynamic_inputs = [torch.randn(2, 1, 256, 256) for _ in range(7)]
    model = C2FNetWithDynamic(fixed_in_channels=12, dynamic_in_channels=[1]*7)
    output = model(x_before, x_after, dynamic_inputs)
    params = count_parameters(model)
    print(f"Output shape: {output.shape}, Expected: (2, 2, 256, 256)")
    print(f"Model parameters: {params:,}\n")

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f}s")
