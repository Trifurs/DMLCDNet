import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import warnings
from torch.utils.checkpoint import checkpoint

warnings.filterwarnings("ignore", message="torch.utils.checkpoint: please pass in use_reentrant")

# Replaces depthwise separable convolution with standard convolution
class StandardConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, 
            padding=padding, bias=bias
        )
        
    def forward(self, x):
        return self.conv(x)

# Double convolution block: two consecutive standard convolutions with BatchNorm and LeakyReLU
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, reduce_ratio=0.5):
        super().__init__()
        mid_channels = max(int(out_channels * reduce_ratio), 4)
        self.double_conv = nn.Sequential(
            StandardConv(in_channels, mid_channels),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            
            StandardConv(mid_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True, negative_slope=0.1)
        )

    def forward(self, x):
        return self.double_conv(x)

# Single convolution block: 1x1 convolution with BatchNorm and LeakyReLU
class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
        )

    def forward(self, x):
        return self.single_conv(x)

# Parallel Convolution Module: extracts features using multiple parallel convolution branches
class PCM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PCM, self).__init__()
        in_channels_4 = max(int(in_channels / 4), 2)
        self.Conv1x1_1 = nn.Conv2d(in_channels, in_channels_4, kernel_size=1, bias=False)

        self.Conv1x1_3 = nn.Conv2d(in_channels, in_channels_4, kernel_size=1, bias=False)
        self.Conv3x3 = StandardConv(in_channels_4, in_channels_4)

        self.Conv1x1_5 = nn.Conv2d(in_channels, in_channels_4, kernel_size=1, bias=False)
        self.Conv5x5 = StandardConv(in_channels_4, in_channels_4, kernel_size=5, padding=2)

        self.Conv1x1_a = nn.Conv2d(in_channels, in_channels_4, kernel_size=1, bias=False)

        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        branch1 = self.Conv1x1_1(x)

        branch2_1 = self.Conv1x1_3(x)
        branch2_2 = self.Conv3x3(branch2_1)

        branch3_1 = self.Conv1x1_5(x)
        branch3_2 = self.Conv5x5(branch3_1)

        branch4_1 = self.Conv1x1_a(x)
        branch4_2 = F.avg_pool2d(branch4_1, kernel_size=3, stride=1, padding=1)

        outputs = [branch1, branch2_2, branch3_2, branch4_2]
        x = torch.cat(outputs, 1)
        x = self.bn(x)
        return F.leaky_relu(x, inplace=True, negative_slope=0.1)

# Output convolution: 1x1 convolution to produce final output
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# Spatial Pooling Block: captures spatial attention along height and width dimensions
class SP_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SP_Block, self).__init__()
        self.conv_1x1 = nn.Conv2d(
            in_channels=channel, out_channels=max(channel // reduction, 2), 
            kernel_size=1, stride=1, bias=False
        )
        self.relu = nn.LeakyReLU(negative_slope=0.1)
        self.bn = nn.BatchNorm2d(max(channel // reduction, 2))
        self.F_h = nn.Conv2d(
            in_channels=max(channel // reduction, 2), 
            out_channels=channel, 
            kernel_size=1, stride=1, bias=False
        )
        self.F_w = nn.Conv2d(
            in_channels=max(channel // reduction, 2), 
            out_channels=channel, 
            kernel_size=1, stride=1, bias=False
        )
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        
        x_h = self.avg_pool_x(x, h).permute(0, 1, 3, 2)
        x_w = self.avg_pool_y(x, w)
        
        x_cat = torch.cat((x_h, x_w), 3)
        x_cat_conv = self.conv_1x1(x_cat)
        x_cat_conv_bn = self.bn(x_cat_conv)
        x_cat_conv_relu = self.relu(x_cat_conv_bn)
        
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))
        
        out = x * s_h * s_w
        return out + x
    
    def avg_pool_x(self, x, height):
        return F.adaptive_avg_pool2d(x, (height, 1))
    
    def avg_pool_y(self, x, width):
        return F.adaptive_avg_pool2d(x, (1, width))

# Cross Transformer: models cross-attention between two input features
class CrossTransformer(nn.Module):
    def __init__(self, dropout, d_model=128, n_head=2):
        super(CrossTransformer, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, n_head, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(negative_slope=0.1)

        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)

    def forward(self, input1, input2):
        dif = input2 - input1
        output_1 = self.cross(input1, dif)
        output_2 = self.cross(input2, dif)
        return output_1, output_2

    def cross(self, input, dif):
        attn_output, attn_weight = self.attention(input, dif, dif)
        output = input + self.dropout1(attn_output)
        output = self.norm1(output)
        ff_output = self.linear2(self.dropout2(self.activation(self.linear1(output))))
        output = output + self.dropout3(ff_output)
        output = self.norm2(output)
        return output

# Dynamic Feature Adapter: adapts dynamic input features to match target features
class DynamicFeatureAdapter(nn.Module):
    def __init__(self, in_channels, target_channels, reduction=16):
        super().__init__()
        self.target_channels = target_channels
        
        self.channel_mapper = nn.Sequential(
            StandardConv(in_channels, target_channels),
            nn.BatchNorm2d(target_channels),
            nn.LeakyReLU(inplace=True, negative_slope=0.1)
        )
        
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(target_channels, max(target_channels // reduction, 2), kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(max(target_channels // reduction, 2), target_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, target_feat):
        x = self.channel_mapper(x)
        if x.shape[2:] != target_feat.shape[2:]:
            x = F.interpolate(
                x, 
                size=target_feat.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        attn = self.attention(x)
        return target_feat + x * attn

# Main model for HRSI change detection with dynamic feature adaptation
class HRSICDWithDynamic(nn.Module):
    def __init__(self, fixed_in_channels, n_classes=2, img_size=64, bilinear=True,
                 dynamic_in_channels: Optional[List[int]] = None):
        super(HRSICDWithDynamic, self).__init__()
        self.n_channels = fixed_in_channels
        self.n_classes = n_classes
        self.img_size = img_size
        self.bilinear = bilinear
        
        self.inc = DoubleConv(self.n_channels, 16)
        self.PCM1 = PCM(16, 16)
        self.FE1 = DoubleConv(16, 32)
        self.PCM2 = PCM(32, 32)
        self.FE2 = DoubleConv(32, 64)
        self.PCM3 = PCM(64, 64)
        self.FE3 = DoubleConv(64, 128)
        self.PCM4 = PCM(128, 128)
        
        self.SPM1 = SP_Block(16)
        self.SPM2 = SP_Block(32)
        self.SPM3 = SP_Block(64)
        self.SPM4 = SP_Block(128)
        
        self.SFM = CrossTransformer(dropout=0.1, d_model=128, n_head=2)
        
        self.fusion1 = DoubleConv(32, 16)
        self.fusion2 = DoubleConv(64, 32)
        self.fusion3 = DoubleConv(128, 64)
        self.fusion4 = DoubleConv(256, 128)

        self.double2single1 = SingleConv(128, 64)
        self.sce1 = DoubleConv(128, 64)
        self.double2single2 = SingleConv(64, 32)
        self.sce2 = DoubleConv(64, 32)
        self.double2single3 = SingleConv(32, 16)
        self.sce3 = DoubleConv(32, 16)
        self.out = OutConv(16, n_classes)
        
        self.dynamic_branch_count = len(dynamic_in_channels) if dynamic_in_channels else 0
        self.dynamic_adapters = nn.ModuleList()
        if dynamic_in_channels:
            target_channels = [16, 32, 64, 128]
            for in_ch in dynamic_in_channels:
                level_adapters = nn.ModuleList([
                    DynamicFeatureAdapter(in_ch, target_channels[0]),
                    DynamicFeatureAdapter(in_ch, target_channels[1]),
                    DynamicFeatureAdapter(in_ch, target_channels[2]),
                    DynamicFeatureAdapter(in_ch, target_channels[3])
                ])
                self.dynamic_adapters.append(level_adapters)

    def forward(self, x1, x2, dynamic_inputs=None):
        B = x1.shape[0]
        opt_en_out = self.encoder(x1, dynamic_inputs, branch_id=0)
        sar_en_out = self.encoder(x2, dynamic_inputs, branch_id=1)

        h, w = opt_en_out[-1].shape[2], opt_en_out[-1].shape[3]
        
        opt_out, sar_out = self.SFM(
            opt_en_out[-1].view(B, 128, h * w).permute(0, 2, 1),
            sar_en_out[-1].view(B, 128, h * w).permute(0, 2, 1)
        )
        opt_en_out[-1] = opt_out.permute(0, 2, 1).view(B, 128, h, w)
        sar_en_out[-1] = sar_out.permute(0, 2, 1).view(B, 128, h, w)

        en_out = [torch.cat((opt_en_out[i], sar_en_out[i]), dim=1) for i in range(len(opt_en_out))]
        en_out = [self.fusion1(en_out[0]), self.fusion2(en_out[1]), self.fusion3(en_out[2]), self.fusion4(en_out[3])]

        opt_de_out = self.decoder(en_out)
        return self.out(opt_de_out)

    def encoder(self, x, dynamic_inputs=None, branch_id=0):
        if self.training:
            x1 = checkpoint(self.inc, x, use_reentrant=False)
            x1 = checkpoint(self.PCM1, x1, use_reentrant=False)
            
            x2 = checkpoint(self.FE1, x1, use_reentrant=False)
            x2 = checkpoint(self.PCM2, x2, use_reentrant=False)
            
            x3 = checkpoint(self.FE2, x2, use_reentrant=False)
            x3 = checkpoint(self.PCM3, x3, use_reentrant=False)
            
            x4 = checkpoint(self.FE3, x3, use_reentrant=False)
            x4 = checkpoint(self.PCM4, x4, use_reentrant=False)
            
            en_out = [
                checkpoint(self.SPM1, x1, use_reentrant=False),
                checkpoint(self.SPM2, x2, use_reentrant=False),
                checkpoint(self.SPM3, x3, use_reentrant=False),
                checkpoint(self.SPM4, x4, use_reentrant=False)
            ]
        else:
            x1 = self.inc(x)
            x1 = self.PCM1(x1)
            
            x2 = self.FE1(x1)
            x2 = self.PCM2(x2)
            
            x3 = self.FE2(x2)
            x3 = self.PCM3(x3)
            
            x4 = self.FE3(x3)
            x4 = self.PCM4(x4)
            
            en_out = [self.SPM1(x1), self.SPM2(x2), self.SPM3(x3), self.SPM4(x4)]
        
        if dynamic_inputs is not None and self.dynamic_adapters:
            assert len(dynamic_inputs) == len(self.dynamic_adapters), \
                f"Number of dynamic inputs {len(dynamic_inputs)} does not match number of adapters {len(self.dynamic_adapters)}"
            for adapter, dyn_inp in zip(self.dynamic_adapters, dynamic_inputs):
                if dyn_inp.shape[2:] != x.shape[2:]:
                    dyn_inp = F.interpolate(
                        dyn_inp, 
                        size=x.shape[2:], 
                        mode='bilinear', 
                        align_corners=False
                    )
                en_out[0] = adapter[0](dyn_inp, en_out[0])
                en_out[1] = adapter[1](dyn_inp, en_out[1])
                en_out[2] = adapter[2](dyn_inp, en_out[2])
                en_out[3] = adapter[3](dyn_inp, en_out[3])

        return en_out

    def decoder(self, x):
        out = self.double2single1(x[-1])
        out = self.double2single2(self.sce1(torch.cat((out, x[-2]), dim=1)))
        out = self.double2single3(self.sce2(torch.cat((out, x[1]), dim=1)))
        out = self.sce3(torch.cat((out, x[0]), dim=1))
        return out

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    import time
    start_time = time.time()

    print("Test 1: Standard input")
    x_before = torch.randn(2, 12, 64, 64)
    x_after = torch.randn(2, 12, 64, 64)
    dynamic_inputs = [
        torch.randn(2, 4, 64, 64),
        torch.randn(2, 3, 64, 64)
    ]
    model = HRSICDWithDynamic(fixed_in_channels=12, dynamic_in_channels=[4, 3])
    model.eval()
    with torch.no_grad():
        output = model(x_before, x_after, dynamic_inputs)
    params = count_parameters(model)
    print(f"Output shape: {output.shape} Expected: (2, 2, 64, 64)")
    print(f"Model parameters: {params:,}\n")

    print("Test 2: Different bands and sizes")
    x_before = torch.randn(3, 8, 128, 128)
    x_after = torch.randn(3, 8, 128, 128)
    dynamic_inputs = [torch.randn(3, 5, 128, 128)]
    model = HRSICDWithDynamic(fixed_in_channels=8, dynamic_in_channels=[5], img_size=128)
    model.eval()
    with torch.no_grad():
        output = model(x_before, x_after, dynamic_inputs)
    params = count_parameters(model)
    print(f"Output shape: {output.shape} Expected: (3, 2, 128, 128)")
    print(f"Model parameters: {params:,}\n")

    print("Test 3: No dynamic branches")
    x_before = torch.randn(1, 10, 64, 64)
    x_after = torch.randn(1, 10, 64, 64)
    model = HRSICDWithDynamic(fixed_in_channels=10)
    model.eval()
    with torch.no_grad():
        output = model(x_before, x_after)
    params = count_parameters(model)
    print(f"Output shape: {output.shape} Expected: (1, 2, 64, 64)")
    print(f"Model parameters: {params:,}\n")

    print("Test 4: Multiple dynamic branches")
    x_before = torch.randn(2, 12, 64, 64)
    x_after = torch.randn(2, 12, 64, 64)
    dynamic_inputs = [
        torch.randn(2, 1, 64, 64),
        torch.randn(2, 1, 64, 64),
        torch.randn(2, 1, 64, 64),
        torch.randn(2, 1, 64, 64),
        torch.randn(2, 1, 64, 64),
        torch.randn(2, 1, 64, 64),
        torch.randn(2, 1, 64, 64)
    ]
    model = HRSICDWithDynamic(fixed_in_channels=12, dynamic_in_channels=[1, 1, 1, 1, 1, 1, 1])
    model.eval()
    with torch.no_grad():
        output = model(x_before, x_after, dynamic_inputs)
    params = count_parameters(model)
    print(f"Output shape: {output.shape} Expected: (2, 2, 64, 64)")
    print(f"Model parameters: {params:,}\n")

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")