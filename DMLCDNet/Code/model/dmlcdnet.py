import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from mamba_ssm import Mamba
import math

def get_freq_indices(method):
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y

class LightweightMultiSpectralAttention(nn.Module):
    """Lightweight multi-spectral attention module"""
    def __init__(self, channel, dct_size=16, reduction=16, freq_sel_method='top8'):
        super().__init__()
        self.reduction = reduction
        self.dct_h = dct_size
        self.dct_w = dct_size
        
        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_size // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_size // 7) for temp_y in mapper_y]
        
        self.channel_compress = nn.Conv2d(channel, channel // 2, kernel_size=1)
        compressed_channel = channel // 2
        
        self.dct_layer = MultiSpectralDCTLayer(
            dct_size, dct_size, 
            mapper_x, mapper_y, 
            compressed_channel
        )
        
        self.fc = nn.Sequential(
            nn.Linear(compressed_channel, compressed_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(compressed_channel // reduction, compressed_channel, bias=False),
            nn.Sigmoid()
        )
        
        self.channel_restore = nn.Conv2d(compressed_channel, channel, kernel_size=1)

    def forward(self, x):
        n, c, h, w = x.shape
        
        x_compressed = self.channel_compress(x)
        x_pooled = F.adaptive_avg_pool2d(x_compressed, (self.dct_h, self.dct_w))
        y = self.dct_layer(x_pooled)
        y = self.fc(y).view(n, c//2, 1, 1)
        
        x_attended = x_compressed * y.expand_as(x_compressed)
        x_attended = self.channel_restore(x_attended)
        
        return x + x_attended

class MultiSpectralDCTLayer(nn.Module):
    """Multi-spectral DCT layer"""
    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super().__init__()
        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0
        
        self.num_freq = len(mapper_x)
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

    def forward(self, x):
        x = x * self.weight
        return torch.sum(x, dim=[2,3])

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        return result * math.sqrt(2) if freq != 0 else result
    
    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)
        c_part = channel // len(mapper_x)
        
        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part : (i+1)*c_part, t_x, t_y] = \
                        self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)
        return dct_filter

class LightweightMamba2D(nn.Module):
    """Adaptive visual Mamba module (H/W direction parallel modeling to address feature discontinuity and global dependency loss)"""
    def __init__(self, 
                 in_channels, 
                 hidden_dim=None, 
                 expansion=1.2,  
                 window_size=None,  
                 d_state=24, 
                 d_conv=3, 
                 expand=1.2):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim or int(in_channels * expansion)
        
        self.proj_in = nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1, bias=False)
        
        self.mamba_h = Mamba(d_model=self.hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_w = Mamba(d_model=self.hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand)
        
        self.local_h = nn.Conv2d(
            self.hidden_dim, self.hidden_dim,
            kernel_size=3, padding=1,
            groups=max(1, self.hidden_dim // (in_channels if in_channels > 0 else 1)),
            bias=False
        )
        self.local_w = nn.Conv2d(
            self.hidden_dim, self.hidden_dim,
            kernel_size=3, padding=1,
            groups=max(1, self.hidden_dim // (in_channels if in_channels > 0 else 1)),
            bias=False
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(2 * self.hidden_dim, self.hidden_dim, kernel_size=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1, bias=False)
        )
        
        self.proj_out = nn.Conv2d(self.hidden_dim, in_channels, kernel_size=1, bias=False)
        self.residual_adjust = nn.Identity()
        if self.hidden_dim != in_channels:
            self.residual_adjust = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        
        self.norm = nn.GroupNorm(
            num_groups=min(8, in_channels) if in_channels > 0 else 1,
            num_channels=in_channels
        )
        
        self.window_size = window_size
        self.register_buffer("auto_window", torch.tensor(8, dtype=torch.int))


    def forward(self, x):
        B, C, H, W = x.shape
        residual = self.residual_adjust(x)
        
        x = self.proj_in(x)
        
        window_size = self.window_size or min(max(H//8, 4), 16)
        self.auto_window = torch.tensor(window_size, device=x.device)
        
        x_h = self._sliding_sequence_model(x, dim=2, window_size=window_size, mamba=self.mamba_h)
        x_h = self.local_h(x_h)
        
        x_w = self._sliding_sequence_model(x, dim=3, window_size=window_size, mamba=self.mamba_w)
        x_w = self.local_w(x_w)
        
        x = torch.cat([x_h, x_w], dim=1)
        x = self.fusion(x)
        
        x = self.proj_out(x) + residual
        
        x = self.norm(x)
        
        return x


    def _sliding_sequence_model(self, x, dim, window_size, mamba):
        B, D, H, W = x.shape
        seq_dim = H if dim == 2 else W
        step = window_size // 2
        
        indices = []
        for i in range(0, seq_dim, step):
            end = min(i + window_size, seq_dim)
            start = max(0, end - window_size)
            indices.append((start, end))
        
        outputs = []
        for (start, end) in indices:
            if dim == 2:
                window = x[:, :, start:end, :]
                window = window.permute(0, 3, 2, 1).contiguous().view(B * W, window_size, D)
            else:
                window = x[:, :, :, start:end]
                window = window.permute(0, 2, 3, 1).contiguous().view(B * H, window_size, D)
            
            window_out = mamba(window)
            
            if dim == 2:
                window_out = window_out.view(B, W, window_size, D).permute(0, 3, 2, 1).contiguous()
            else:
                window_out = window_out.view(B, H, window_size, D).permute(0, 3, 1, 2).contiguous()
            outputs.append((window_out, start, end))
        
        fused = torch.zeros_like(x)
        counts = torch.zeros(seq_dim, device=x.device)
        for (out, start, end) in outputs:
            if dim == 2:
                fused[:, :, start:end, :] += out
                counts[start:end] += 1
            else:
                fused[:, :, :, start:end] += out
                counts[start:end] += 1
        counts = counts.clamp(min=1).view(1, 1, -1, 1) if dim == 2 else counts.clamp(min=1).view(1, 1, 1, -1)
        return fused / counts

class LightweightChannelAttention(nn.Module):
    """Lightweight channel attention with minimal parameters"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y

class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution with fewer parameters"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, 
            kernel_size=kernel_size, 
            padding=padding, 
            groups=in_channels,
            bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=1,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x

class DynamicFeatureExtractor(nn.Module):
    """Optimized dynamic branch feature extractor that extracts multi-scale features and provides multiple skip connections"""
    def __init__(self, in_channels: int, out_channels: int = 64):
        super().__init__()
        self.conv1 = nn.Sequential(
            DepthwiseSeparableConv(in_channels, 16),
            nn.ReLU(inplace=True)
        )
        self.attn1 = LightweightChannelAttention(16)
        self.residual_conv1 = nn.Conv2d(in_channels, 16, kernel_size=1, bias=False) if in_channels != 16 else nn.Identity()
        
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Sequential(
            DepthwiseSeparableConv(16, 32),
            nn.ReLU(inplace=True)
        )
        self.attn2 = LightweightChannelAttention(32)
        self.residual_conv2 = nn.Conv2d(16, 32, kernel_size=1, bias=False)
        
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Sequential(
            DepthwiseSeparableConv(32, 32),
            nn.ReLU(inplace=True)
        )
        self.attn3 = LightweightChannelAttention(32)
        self.residual_conv3 = nn.Conv2d(32, 32, kernel_size=1, bias=False)
        
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn_final = nn.BatchNorm2d(out_channels)
        
        self.pool_final = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        residual1 = self.residual_conv1(x)
        x1 = self.conv1(x)
        x1 = self.attn1(x1)
        x1 = x1 + residual1
        x1 = F.relu(x1)
        
        x2 = self.pool1(x1)
        residual2 = self.residual_conv2(x2)
        x2 = self.conv2(x2)
        x2 = self.attn2(x2)
        x2 = x2 + residual2
        x2 = F.relu(x2)
        
        x3 = self.pool2(x2)
        residual3 = self.residual_conv3(x3)
        x3 = self.conv3(x3)
        x3 = self.attn3(x3)
        x3 = x3 + residual3
        x3 = F.relu(x3)
        
        x_final = self.pool_final(x3)
        x_final = F.relu(self.bn_final(self.final_conv(x_final)))
        
        return x_final, [x1, x2, x3]

class FixedBranchEncoder(nn.Module):
    """Fixed branch encoder that supports arbitrary number of input bands"""
    def __init__(self, in_channels: int):
        super().__init__()
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Sequential(
            DepthwiseSeparableConv(64, 64),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            DepthwiseSeparableConv(64, 64),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.change_detector = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            LightweightMamba2D(64, expansion=0.5)
        )

    def forward(self, x_before: torch.Tensor, x_after: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        feat_before1 = self.shared_conv(x_before)
        feat_after1 = self.shared_conv(x_after)
        
        feat_before2 = self.pool1(feat_before1)
        feat_before2 = self.conv2(feat_before2)
        feat_after2 = self.pool1(feat_after1)
        feat_after2 = self.conv2(feat_after2)
        
        feat_before3 = self.pool2(feat_before2)
        feat_before3 = self.conv3(feat_before3)
        feat_after3 = self.pool2(feat_after2)
        feat_after3 = self.conv3(feat_after3)
        
        feat_before_final = self.pool3(feat_before3)
        feat_after_final = self.pool3(feat_after3)
        
        skip_feats = [
            feat_before1, feat_after1,
            feat_before2, feat_after2,
            feat_before3, feat_after3
        ]
        
        combined = torch.cat([feat_before_final, feat_after_final], dim=1)
        coarse_feat = self.change_detector(combined)
        return coarse_feat, skip_feats

class DynamicFusionModule(nn.Module):
    """Dynamic feature fusion module integrating lightweight multi-spectral attention"""
    def __init__(self, base_channels: int = 64):
        super().__init__()
        self.base_channels = base_channels
        self.attention = LightweightMultiSpectralAttention(
            channel=base_channels,
            dct_size=16,
            reduction=16,
            freq_sel_method='top8'
        )

    def forward(self, coarse_feat: torch.Tensor, dynamic_feats: List[torch.Tensor]) -> torch.Tensor:
        if dynamic_feats:
            fused_dynamic = torch.cat(dynamic_feats, dim=1)
            dyn_channels = fused_dynamic.shape[1]
            adapt_conv = nn.Conv2d(dyn_channels, self.base_channels, kernel_size=1).to(fused_dynamic.device)
            nn.init.kaiming_normal_(adapt_conv.weight, mode='fan_in', nonlinearity='relu')
            fused_dynamic = adapt_conv(fused_dynamic)
            
            attn_feat = self.attention(fused_dynamic)
            optimized_feat = coarse_feat + attn_feat
        else:
            optimized_feat = coarse_feat
        return optimized_feat

class Decoder(nn.Module):
    """Decoder supporting multi-scale skip connection fusion"""
    def __init__(self, out_channels: int = 2, dynamic_branch_count: int = 0):
        super().__init__()
        self.out_channels = out_channels
        self.dynamic_branch_count = dynamic_branch_count
        
        self.upconv3 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        
        scale3_fixed_channels = 64 * 2
        scale3_dynamic_channels = 32 * dynamic_branch_count
        scale2_fixed_channels = 64 * 2
        scale2_dynamic_channels = 32 * dynamic_branch_count
        scale1_fixed_channels = 64 * 2
        scale1_dynamic_channels = 16 * dynamic_branch_count
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64 + scale3_fixed_channels + scale3_dynamic_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64 + scale2_fixed_channels + scale2_dynamic_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(64 + scale1_fixed_channels + scale1_dynamic_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, skip_feats: List[torch.Tensor]) -> torch.Tensor:
        fixed_skip_count = 6
        fixed_skips = skip_feats[:fixed_skip_count]
        dynamic_skips = skip_feats[fixed_skip_count:]
        
        x = self.upconv3(x)
        scale3_skips = fixed_skips[4:6] + dynamic_skips[2::3]
        scale3_combined = torch.cat([x] + scale3_skips, dim=1)
        x = self.conv_block3(scale3_combined)
        
        x = self.upconv2(x)
        scale2_skips = fixed_skips[2:4] + dynamic_skips[1::3]
        scale2_combined = torch.cat([x] + scale2_skips, dim=1)
        x = self.conv_block2(scale2_combined)
        
        x = self.upconv1(x)
        scale1_skips = fixed_skips[0:2] + dynamic_skips[0::3]
        scale1_combined = torch.cat([x] + scale1_skips, dim=1)
        x = self.conv_block1(scale1_combined)
        
        return x

class DMLCDNet(nn.Module):
    """Fully adaptive dynamic multi-branch landslide detection network"""
    def __init__(self, fixed_in_channels: int, dynamic_in_channels: Optional[List[int]] = None):
        super().__init__()
        self.dynamic_branch_count = len(dynamic_in_channels) if dynamic_in_channels else 0
        
        self.fixed_encoder = FixedBranchEncoder(fixed_in_channels)
        
        self.dynamic_branches = nn.ModuleList()
        if dynamic_in_channels:
            for in_ch in dynamic_in_channels:
                self.dynamic_branches.append(DynamicFeatureExtractor(in_ch))
        
        self.fusion = DynamicFusionModule()
        
        self.decoder = Decoder(
            out_channels=2,
            dynamic_branch_count=self.dynamic_branch_count
        )

    def forward(self, 
                x_before: torch.Tensor, 
                x_after: torch.Tensor, 
                dynamic_inputs: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        coarse_feat, skip_feats = self.fixed_encoder(x_before, x_after)
        batch_size, _, input_height, input_width = x_before.shape
        
        dynamic_feats = []
        if dynamic_inputs and self.dynamic_branches:
            assert len(dynamic_inputs) == len(self.dynamic_branches), \
                f"Number of dynamic inputs {len(dynamic_inputs)} does not match number of branches {len(self.dynamic_branches)}"
                
            for branch, inp in zip(self.dynamic_branches, dynamic_inputs):
                if inp.shape[2:] != (input_height, input_width):
                    inp = F.interpolate(
                        inp, 
                        size=(input_height, input_width), 
                        mode='bilinear', 
                        align_corners=True
                    )
                dyn_feat, dyn_skips = branch(inp)
                dynamic_feats.append(dyn_feat)
                skip_feats.extend(dyn_skips)
        
        fused_feat = self.fusion(coarse_feat, dynamic_feats)
        output = self.decoder(fused_feat, skip_feats)
        
        assert output.shape == (batch_size, 2, input_height, input_width), \
            f"Output shape error: expected {(batch_size, 2, input_height, input_width)}, got {output.shape}"
            
        return output

def count_parameters(model):
    """Calculate the total number of trainable parameters in the model"""
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
    model = DMLCDNet(fixed_in_channels=12, dynamic_in_channels=[4, 3])
    output = model(x_before, x_after, dynamic_inputs)
    params = count_parameters(model)
    print(f"Output shape: {output.shape} Expected: (2, 2, 256, 256)")
    print(f"Model parameters: {params:,}\n")

    print("Test 2: Different bands and sizes")
    x_before = torch.randn(3, 8, 512, 512)
    x_after = torch.randn(3, 8, 512, 512)
    dynamic_inputs = [torch.randn(3, 5, 512, 512)]
    model = DMLCDNet(fixed_in_channels=8, dynamic_in_channels=[5])
    output = model(x_before, x_after, dynamic_inputs)
    params = count_parameters(model)
    print(f"Output shape: {output.shape} Expected: (3, 2, 512, 512)")
    print(f"Model parameters: {params:,}\n")

    print("Test 3: No dynamic branches")
    x_before = torch.randn(1, 10, 128, 128)
    x_after = torch.randn(1, 10, 128, 128)
    model = DMLCDNet(fixed_in_channels=10)
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
    model = DMLCDNet(fixed_in_channels=12, dynamic_in_channels=[1, 1, 1, 1, 1, 1, 1])
    output = model(x_before, x_after, dynamic_inputs)
    params = count_parameters(model)
    print(f"Output shape: {output.shape} Expected: (2, 2, 256, 256)")
    print(f"Model parameters: {params:,}\n")

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")
