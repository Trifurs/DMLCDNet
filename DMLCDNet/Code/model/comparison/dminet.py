import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np
import cv2
from typing import List, Optional

def init_weights(m):
    """Initialize layer weights"""
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        
class Conv(nn.Module):
    """Convolutional layer with optional batch normalization and ReLU activation"""
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, f"Input channel mismatch: {x.size()[1]} vs {self.inp_dim}"
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class decode(nn.Module):
    """Decoding module for feature fusion"""
    def __init__(self, in_channel_left, in_channel_down, out_channel, norm_layer=nn.BatchNorm2d):
        super(decode, self).__init__()
        self.conv_d1 = nn.Conv2d(in_channel_down, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv_l = nn.Conv2d(in_channel_left, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(out_channel*2, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn3 = norm_layer(out_channel)

    def forward(self, left, down):
        down_mask = self.conv_d1(down)
        left_mask = self.conv_l(left)
        
        if down.size()[2:] != left.size()[2:]:
            down_ = F.interpolate(down, size=left.size()[2:], mode='bilinear', align_corners=True)
            z1 = F.relu(left_mask * down_, inplace=True)
        else:
            z1 = F.relu(left_mask * down, inplace=True)

        if down_mask.size()[2:] != left.size()[2:]:
            down_mask = F.interpolate(down_mask, size=left.size()[2:], mode='bilinear', align_corners=True)

        z2 = F.relu(down_mask * left, inplace=True)

        out = torch.cat((z1, z2), dim=1)
        return F.relu(self.bn3(self.conv3(out)), inplace=True)

class BasicConv2d(nn.Module):
    """Basic convolutional block with batch normalization and ReLU"""
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class CrossAtt(nn.Module):
    """Cross attention module for feature interaction"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        
        self.query = nn.Conv2d(in_channels, in_channels // 16, kernel_size=1, stride=1)
        self.key = nn.Conv2d(in_channels, in_channels // 16, kernel_size=1, stride=1)
        self.value = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1)

        self.gamma = nn.Parameter(torch.zeros(1)) 
        self.softmax = nn.Softmax(dim=-1)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, input1, input2):
        batch_size, channels, height, width = input1.shape
        
        q1 = self.query(input1).view(batch_size, -1, height * width).permute(0, 2, 1)
        k2 = self.key(input2).view(batch_size, -1, height * width)
        v2 = self.value(input2).view(batch_size, -1, height * width)
        
        q2 = self.query(input2).view(batch_size, -1, height * width).permute(0, 2, 1)
        k1 = self.key(input1).view(batch_size, -1, height * width)
        v1 = self.value(input1).view(batch_size, -1, height * width)

        attn_matrix1 = torch.bmm(q1, k2) / (k2.size(1) ** 0.5)
        attn_matrix1 = self.softmax(attn_matrix1)
        out1 = torch.bmm(v1, attn_matrix1.permute(0, 2, 1))
        out1 = out1.view(batch_size, channels // 2, height, width)
        out1 = torch.cat([out1, input1[:, :channels//2]], dim=1)
        out1 = self.gamma * out1 + input1

        attn_matrix2 = torch.bmm(q2, k1) / (k1.size(1) ** 0.5)
        attn_matrix2 = self.softmax(attn_matrix2)
        out2 = torch.bmm(v2, attn_matrix2.permute(0, 2, 1))
        out2 = out2.view(batch_size, channels // 2, height, width)
        out2 = torch.cat([out2, input2[:, :channels//2]], dim=1)
        out2 = self.gamma * out2 + input2

        feat_sum = self.conv_cat(out1 + out2)
        return feat_sum, out1, out2
    

class DynamicFeatureAdapter(nn.Module):
    """Simplified dynamic feature adapter"""
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
            x = F.interpolate(
                x, 
                size=target_feat.shape[2:], 
                mode='bilinear', 
                align_corners=True
            )
            
        attn = self.attention(x)
        return target_feat + x * attn

class BasicBlock(nn.Module):
    """Basic residual block for ResNet"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet18(nn.Module):
    """Simplified ResNet18 model"""
    def __init__(self, in_channels=3):
        super(ResNet18, self).__init__()
        self.inplanes = 32
        
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(BasicBlock, 32, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 1, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 128, 1, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 256, 1, stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x1, x2, x3, x4

class DMINetWithDynamic(nn.Module):
    """DMINet model with dynamic feature adaptation"""
    def __init__(self, fixed_in_channels, num_classes=2, drop_rate=0.2, 
                 normal_init=True, pretrained=False, show_Feature_Maps=False,
                 dynamic_in_channels: Optional[List[int]] = None):
        super(DMINetWithDynamic, self).__init__()

        self.show_Feature_Maps = show_Feature_Maps
        self.dynamic_branch_count = len(dynamic_in_channels) if dynamic_in_channels else 0
        
        self.resnet = ResNet18(in_channels=fixed_in_channels)
        
        self.cross2 = CrossAtt(128, 128)
        self.cross3 = CrossAtt(64, 64)
        self.cross4 = CrossAtt(32, 32)

        self.Translayer2_1 = BasicConv2d(128, 64, 1)
        self.fam32_1 = decode(64, 64, 64)
        self.Translayer3_1 = BasicConv2d(64, 32, 1)
        self.fam43_1 = decode(32, 32, 32)

        self.Translayer2_2 = BasicConv2d(128, 64, 1)
        self.fam32_2 = decode(64, 64, 64)
        self.Translayer3_2 = BasicConv2d(64, 32, 1)
        self.fam43_2 = decode(32, 32, 32)

        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.final = nn.Sequential(
            Conv(32, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
        )
        self.final2 = nn.Sequential(
            Conv(32, 16, 3, bn=True, relu=True),
            Conv(16, num_classes, 3, bn=False, relu=False)
        )

        self.dynamic_adapters = nn.ModuleList()
        if dynamic_in_channels:
            for in_ch in dynamic_in_channels:
                level_adapters = nn.ModuleList([
                    DynamicFeatureAdapter(in_ch, 32),
                    DynamicFeatureAdapter(in_ch, 64),
                    DynamicFeatureAdapter(in_ch, 128)
                ])
                self.dynamic_adapters.append(level_adapters)

        if normal_init:
            self.init_weights()

    def forward(self, imgs1, imgs2, dynamic_inputs=None):
        c1, c2, c3, _ = self.resnet(imgs1)
        
        c1_img2, c2_img2, c3_img2, _ = self.resnet(imgs2)

        if dynamic_inputs is not None and self.dynamic_adapters:
            assert len(dynamic_inputs) == len(self.dynamic_adapters), \
                f"Number of dynamic inputs {len(dynamic_inputs)} does not match number of adapters {len(self.dynamic_adapters)}"
            
            for adapter, dyn_inp in zip(self.dynamic_adapters, dynamic_inputs):
                if dyn_inp.shape[2:] != imgs1.shape[2:]:
                    dyn_inp = F.interpolate(
                        dyn_inp, 
                        size=imgs1.shape[2:], 
                        mode='bilinear', 
                        align_corners=True
                    )
                
                c1 = adapter[0](dyn_inp, c1)
                c2 = adapter[1](dyn_inp, c2)
                c3 = adapter[2](dyn_inp, c3)
                
                c1_img2 = adapter[0](dyn_inp, c1_img2)
                c2_img2 = adapter[1](dyn_inp, c2_img2)
                c3_img2 = adapter[2](dyn_inp, c3_img2)

        cross_result2, cur1_2, cur2_2 = self.cross2(c3, c3_img2)
        cross_result3, cur1_3, cur2_3 = self.cross3(c2, c2_img2)
        cross_result4, cur1_4, cur2_4 = self.cross4(c1, c1_img2)

        out3 = self.fam32_1(cross_result3, self.Translayer2_1(cross_result2))
        out4 = self.fam43_1(cross_result4, self.Translayer3_1(out3))

        out3_2 = self.fam32_2(torch.abs(cur1_3 - cur2_3), self.Translayer2_2(torch.abs(cur1_2 - cur2_2)))
        out4_2 = self.fam43_2(torch.abs(cur1_4 - cur2_4), self.Translayer3_2(out3_2))

        out4_up = self.upsamplex4(out4)
        out4_2_up = self.upsamplex4(out4_2)
        out_1 = self.final(out4_up)
        out_2 = self.final2(out4_2_up)
        
        final_output = (out_1 + out_2) / 2
        return F.interpolate(final_output, size=imgs1.shape[2:], mode='bilinear', align_corners=True)

    def init_weights(self):
        """Initialize model weights"""
        self.cross2.apply(init_weights)
        self.cross3.apply(init_weights)        
        self.cross4.apply(init_weights)

        self.fam32_1.apply(init_weights)
        self.Translayer2_1.apply(init_weights)
        self.fam43_1.apply(init_weights)
        self.Translayer3_1.apply(init_weights)

        self.fam32_2.apply(init_weights)
        self.Translayer2_2.apply(init_weights)
        self.fam43_2.apply(init_weights)
        self.Translayer3_2.apply(init_weights)

        self.final.apply(init_weights)
        self.final2.apply(init_weights)

        if self.dynamic_adapters:
            for adapter in self.dynamic_adapters:
                adapter.apply(init_weights)


def count_parameters(model):
    """Calculate total number of trainable parameters in the model"""
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
    model = DMINetWithDynamic(fixed_in_channels=12, dynamic_in_channels=[4, 3])
    output = model(x_before, x_after, dynamic_inputs)
    params = count_parameters(model)
    print(f"Output shape: {output.shape} Expected: (2, 2, 256, 256)")
    print(f"Model parameters: {params:,}\n")

    print("Test 2: Different bands and sizes")
    x_before = torch.randn(3, 8, 512, 512)
    x_after = torch.randn(3, 8, 512, 512)
    dynamic_inputs = [torch.randn(3, 5, 512, 512)]
    model = DMINetWithDynamic(fixed_in_channels=8, dynamic_in_channels=[5])
    output = model(x_before, x_after, dynamic_inputs)
    params = count_parameters(model)
    print(f"Output shape: {output.shape} Expected: (3, 2, 512, 512)")
    print(f"Model parameters: {params:,}\n")

    print("Test 3: No dynamic branches")
    x_before = torch.randn(1, 10, 128, 128)
    x_after = torch.randn(1, 10, 128, 128)
    model = DMINetWithDynamic(fixed_in_channels=10)
    output = model(x_before, x_after)
    params = count_parameters(model)
    print(f"Output shape: {output.shape} Expected: (1, 2, 128, 128)")
    print(f"Model parameters: {params:,}\n")

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
    model = DMINetWithDynamic(fixed_in_channels=12, dynamic_in_channels=[1, 1, 1, 1, 1, 1, 1])
    output = model(x_before, x_after, dynamic_inputs)
    params = count_parameters(model)
    print(f"Output shape: {output.shape} Expected: (2, 2, 256, 256)")
    print(f"Model parameters: {params:,}\n")

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")