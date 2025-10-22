import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from typing import List, Optional, Tuple

__all__ = ['ResNet', 'resnet18', 'SEIFNetWithDynamic']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """Basic residual block"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            dilation = 1
        
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """Simplified ResNet with reduced feature levels"""
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, strides=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.strides = strides
        if self.strides is None:
            self.strides = [2, 2, 2]

        self.inplanes = 32
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False]
        if len(replace_stride_with_dilation) != 2:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 2-element tuple")
        self.groups = groups
        self.base_width = width_per_group
        
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=5, stride=self.strides[0], padding=2,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=self.strides[1], padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=self.strides[2],
                                       dilate=replace_stride_with_dilation[0])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)

        return x1, x2

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained=False, progress=True, **kwargs):
    model = ResNet(block, layers,** kwargs)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    """Simplified ResNet-18 with reduced network depth"""
    return _resnet('resnet18', BasicBlock, [1, 1], pretrained, progress,** kwargs)


class ChannelAttentionModule(nn.Module):
    """Simplified channel attention module"""
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    """Simplified spatial attention module"""
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=5, stride=1, padding=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    """Simplified CBAM attention module"""
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


class CoDEM2(nn.Module):
    """Simplified difference enhancement module"""
    def __init__(self, channel_dim):
        super(CoDEM2, self).__init__()
        self.channel_dim = channel_dim
        self.Conv3 = nn.Conv2d(in_channels=2*self.channel_dim, 
                              out_channels=self.channel_dim,
                              kernel_size=3, stride=1, padding=1)
        self.BN1 = nn.BatchNorm2d(self.channel_dim)
        self.ReLU = nn.ReLU(inplace=True)
        self.coAtt_1 = CoordAtt(inp=channel_dim, oup=channel_dim, reduction=32)

    def forward(self, x1, x2):
        f_d = torch.abs(x1 - x2)
        f_c = torch.cat((x1, x2), dim=1)
        z_c = self.ReLU(self.BN1(self.Conv3(f_c)))
        d_aw, d_ah = self.coAtt_1(f_d)
        z_d = f_d * d_aw * d_ah
        return z_d + z_c


class ACFF2(nn.Module):
    """Simplified adaptive cross-feature fusion module"""
    def __init__(self, channel_L, channel_H):
        super(ACFF2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channel_H, 
                              out_channels=channel_L,
                              kernel_size=1, stride=1, padding=0)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.ca = ChannelAttention(in_channels=channel_L, ratio=16)

    def forward(self, f_low, f_high):
        f_high = self.up(self.conv1(f_high))
        adaptive_w = self.ca(f_low + f_high)
        return f_low * adaptive_w + f_high * (1 - adaptive_w)


class ChannelAttention(nn.Module):
    """Simplified channel attention"""
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels//ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels//ratio, in_channels, 1, bias=False)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmod(avg_out + max_out)


class SupervisedAttentionModule(nn.Module):
    """Simplified supervised attention module"""
    def __init__(self, mid_d):
        super(SupervisedAttentionModule, self).__init__()
        self.mid_d = mid_d
        self.cbam = CBAM(channel=self.mid_d)
        self.conv2 = nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(self.mid_d)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        context = self.cbam(x)
        return self.relu(self.bn(self.conv2(context)))


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    """Simplified coordinate attention"""
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(4, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        a_h = a_h.expand(-1, -1, h, w)
        a_w = a_w.expand(-1, -1, h, w)

        return a_w, a_h


class DynamicFeatureAdapter(nn.Module):
    """Simplified dynamic feature adapter"""
    def __init__(self, in_channels, target_dims=[64, 128]):
        super().__init__()
        self.target_dims = target_dims
        self.scale_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, target_dims[0], kernel_size=3, padding=1),
                nn.BatchNorm2d(target_dims[0]),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(in_channels, target_dims[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(target_dims[1]),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
        ])

    def forward(self, x):
        features = []
        for adapter in self.scale_adapters:
            features.append(adapter(x))
        return features


class DynamicFeatureFusion(nn.Module):
    """Simplified dynamic feature fusion module"""
    def __init__(self, channel_dims=[64, 128]):
        super().__init__()
        self.fusion_gates = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2 * dim, dim, kernel_size=1),
                nn.Sigmoid()
            ) for dim in channel_dims
        ])

    def forward(self, diff_features, dynamic_features):
        fused_features = []
        for i, (diff_feat, dyn_feat, gate) in enumerate(zip(
                diff_features, dynamic_features, self.fusion_gates)):
            if dyn_feat.shape[2:] != diff_feat.shape[2:]:
                dyn_feat = F.interpolate(
                    dyn_feat, 
                    size=diff_feat.shape[2:], 
                    mode='bilinear', 
                    align_corners=True
                )
            combined = torch.cat([diff_feat, dyn_feat], dim=1)
            weight = gate(combined)
            fused = weight * diff_feat + (1 - weight) * dyn_feat
            fused_features.append(fused)
        return fused_features


class SEIFNetWithDynamic(nn.Module):
    """Simplified SEIFNet model supporting multi-GPU training"""
    def __init__(self, fixed_in_channels, output_nc=2, dynamic_in_channels: Optional[List[int]] = None):
        super(SEIFNetWithDynamic, self).__init__()
        self.stage_dims = [64, 128]
        self.output_nc = output_nc
        self.dynamic_branch_count = len(dynamic_in_channels) if dynamic_in_channels else 0

        self.encoder = resnet18(pretrained=False)
        self.encoder.conv1 = nn.Conv2d(
            fixed_in_channels, 32, kernel_size=5, stride=2, padding=2, bias=False
        )

        self.diff1 = CoDEM2(self.stage_dims[0])
        self.diff2 = CoDEM2(self.stage_dims[1])

        self.ACFF2 = ACFF2(channel_L=self.stage_dims[0], channel_H=self.stage_dims[1])

        self.sam_p2 = SupervisedAttentionModule(self.stage_dims[1])
        self.sam_p1 = SupervisedAttentionModule(self.stage_dims[0])

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.conv2 = nn.Conv2d(128, 64, kernel_size=1)
        self.conv_final1 = nn.Conv2d(64, output_nc, kernel_size=1)

        self.dynamic_adapters = nn.ModuleList()
        if dynamic_in_channels:
            for in_ch in dynamic_in_channels:
                self.dynamic_adapters.append(DynamicFeatureAdapter(in_ch))
            self.dynamic_fusion = DynamicFeatureFusion(self.stage_dims)

    def forward(self, x1, x2, dynamic_inputs: Optional[List[torch.Tensor]] = None):
        f1 = self.encoder(x1)
        f2 = self.encoder(x2)
        x1_0, x1_1 = f1
        x2_0, x2_1 = f2

        d1 = self.diff1(x1_0, x2_0)
        d2 = self.diff2(x1_1, x2_1)
        diff_features = [d1, d2]

        if dynamic_inputs and self.dynamic_adapters:
            assert len(dynamic_inputs) == len(self.dynamic_adapters), \
                f"Number of dynamic inputs {len(dynamic_inputs)} does not match number of adapters {len(self.dynamic_adapters)}"
            
            all_dynamic_features = []
            for adapter, inp in zip(self.dynamic_adapters, dynamic_inputs):
                if inp.shape[2:] != x1.shape[2:]:
                    inp = F.interpolate(
                        inp, 
                        size=x1.shape[2:], 
                        mode='bilinear', 
                        align_corners=True
                    )
                dyn_feats = adapter(inp)
                all_dynamic_features.append(dyn_feats)
            
            aggregated_dynamic = []
            for i in range(2):
                scale_features = [dyn_feats[i] for dyn_feats in all_dynamic_features]
                aggregated = torch.mean(torch.stack(scale_features), dim=0)
                aggregated_dynamic.append(aggregated)
            
            fused_features = self.dynamic_fusion(diff_features, aggregated_dynamic)
            d1, d2 = fused_features

        p2 = self.sam_p2(d2)
        ACFF_21 = self.ACFF2(d1, p2)
        p1 = self.sam_p1(ACFF_21)

        p2_up = self.upsample2(p2)
        p2_up = self.conv2(p2_up)

        p = p1 + p2_up
        p_up = self.upsample4(p)
        output = self.conv_final1(p_up)

        return output


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import time
    start_time = time.time()

    print("Test 1: Standard input")
    x_before = torch.randn(2, 12, 256, 256)
    x_after = torch.randn(2, 12, 256, 256)
    dynamic_inputs = [
        torch.randn(2, 4, 256, 256),
        torch.randn(2, 3, 256, 256)
    ]
    model = SEIFNetWithDynamic(fixed_in_channels=12, dynamic_in_channels=[4, 3])
    output = model(x_before, x_after, dynamic_inputs)
    params = count_parameters(model)
    print(f"Output shape: {output.shape} Expected: (2, 2, 256, 256)")
    print(f"Model parameters: {params:,}\n")

    print("Test 2: Different bands and sizes")
    x_before = torch.randn(3, 8, 512, 512)
    x_after = torch.randn(3, 8, 512, 512)
    dynamic_inputs = [torch.randn(3, 5, 512, 512)]
    model = SEIFNetWithDynamic(fixed_in_channels=8, dynamic_in_channels=[5])
    output = model(x_before, x_after, dynamic_inputs)
    params = count_parameters(model)
    print(f"Output shape: {output.shape} Expected: (3, 2, 512, 512)")
    print(f"Model parameters: {params:,}\n")

    print("Test 3: No dynamic branches")
    x_before = torch.randn(1, 10, 128, 128)
    x_after = torch.randn(1, 10, 128, 128)
    model = SEIFNetWithDynamic(fixed_in_channels=10)
    output = model(x_before, x_after)
    params = count_parameters(model)
    print(f"Output shape: {output.shape} Expected: (1, 2, 128, 128)")
    print(f"Model parameters: {params:,}")

    print("Test 4: Multiple dynamic branches")
    x_before = torch.randn(2, 12, 256, 256)
    x_after = torch.randn(2, 12, 256, 256)
    dynamic_inputs = [
        torch.randn(2, 1, 256, 256),
        torch.randn(2, 1, 256, 256),
        torch.randn(2, 1, 256, 256),
        torch.randn(2, 1, 256, 256),
        torch.randn(2, 1, 256, 256),
        torch.randn(2, 1, 256, 256),
        torch.randn(2, 1, 256, 256)
    ]
    model = SEIFNetWithDynamic(fixed_in_channels=12, dynamic_in_channels=[1, 1, 1, 1, 1, 1, 1])
    output = model(x_before, x_after, dynamic_inputs)
    params = count_parameters(model)
    print(f"Output shape: {output.shape} Expected: (2, 2, 256, 256)")
    print(f"Model parameters: {params:,}\n")

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")