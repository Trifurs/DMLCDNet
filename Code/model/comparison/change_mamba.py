import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint
from typing import Optional, Callable, Any, List

class LayerNorm2d(nn.LayerNorm):
    """Layer normalization for 2D tensors with channel-first format"""
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__(normalized_shape, eps=eps)
        
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)  # Convert back to (B, C, H, W)

class Linear2d(nn.Linear):
    """Linear layer adapted for 2D tensors with channel-first format"""
    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        x = x.view(B, H * W, C)  # (B, H*W, C)
        x = super().forward(x)  # (B, H*W, out_features)
        x = x.view(B, H, W, -1)  # (B, H, W, out_features)
        return x.permute(0, 3, 1, 2).contiguous()  # (B, out_features, H, W)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(self.weight.shape)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

class Mlp(nn.Module):
    """Multi-layer perceptron module"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or int(in_features * 1.5)

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class gMlp(nn.Module):
    """Gated multi-layer perceptron module"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., channels_first=False):
        super().__init__()
        self.channel_first = channels_first
        out_features = out_features or in_features
        hidden_features = hidden_features or int(in_features * 1.5)

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, 2 * hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
        x = self.fc2(x * self.act(z))
        x = self.drop(x)
        return x

class PatchMerging2D(nn.Module):
    """2D patch merging module for downsampling"""
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.out_dim = int(1.5 * dim) if out_dim < 0 else out_dim
        self.reduction = nn.Linear(4 * dim, self.out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]
        x1 = x[..., 1::2, 0::2, :]
        x2 = x[..., 0::2, 1::2, :]
        x3 = x[..., 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)
        return x

class VSSBlock(nn.Module):
    """Vision-Semantic-Spatial block combining SSM and MLP"""
    def __init__(self, hidden_dim, drop_path=0., norm_layer=nn.LayerNorm, channel_first=False,
                 ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer=nn.SiLU,
                 ssm_conv=1, ssm_conv_bias=True, ssm_drop_rate=0., ssm_init="v0",
                 forward_type="v1", mlp_ratio=1.5, mlp_act_layer=nn.GELU, mlp_drop_rate=0.,
                 gmlp=False, use_checkpoint=False):
        super().__init__()
        self.channel_first = channel_first
        self.use_checkpoint = use_checkpoint
        
        self.norm = norm_layer(hidden_dim)
        
        self.ssm = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=ssm_conv, padding=ssm_conv//2, bias=ssm_conv_bias) if channel_first else
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=ssm_conv, padding=ssm_conv//2, bias=ssm_conv_bias),
            ssm_act_layer(),
            nn.Dropout(ssm_drop_rate)
        )
        
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = gMlp(
            in_features=hidden_dim,
            hidden_features=mlp_hidden_dim,
            act_layer=mlp_act_layer,
            drop=mlp_drop_rate,
            channels_first=channel_first
        ) if gmlp else Mlp(
            in_features=hidden_dim,
            hidden_features=mlp_hidden_dim,
            act_layer=mlp_act_layer,
            drop=mlp_drop_rate,
            channels_first=channel_first
        )
        
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        def _forward(x):
            x = x + self.drop_path(self.ssm(self.norm(x)))
            x = x + self.drop_path(self.mlp(self.norm(x)))
            return x
        
        if self.use_checkpoint and x.requires_grad:
            return checkpoint(_forward, x)
        else:
            return _forward(x)

class Backbone_VSSM(nn.Module):
    """VSSM backbone for feature extraction"""
    def __init__(self, out_indices=(0, 1, 2, 3), pretrained=None, norm_layer="ln2d", in_channels=3,** kwargs):
        super().__init__()
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer_cls = _NORMLAYERS.get(norm_layer.lower(), nn.LayerNorm)
        
        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )
        
        ssm_act = kwargs.get('ssm_act_layer', 'silu')
        if isinstance(ssm_act, str):
            ssm_act = _ACTLAYERS.get(ssm_act.lower(), nn.SiLU)
        
        mlp_act = kwargs.get('mlp_act_layer', 'gelu')
        if isinstance(mlp_act, str):
            mlp_act = _ACTLAYERS.get(mlp_act.lower(), nn.GELU)
        
        vss_kwargs = {k: v for k, v in kwargs.items() if k not in ['distributed', 'ssm_act_layer', 'mlp_act_layer']}
        vss_kwargs['ssm_act_layer'] = ssm_act
        vss_kwargs['mlp_act_layer'] = mlp_act
        
        self.dims = [32, 64, 128, 256]
        self.out_indices = out_indices
        
        self.patch_embed = nn.Conv2d(in_channels, self.dims[0], kernel_size=4, stride=4)
        
        self.layers = nn.ModuleList()
        for i in range(4):
            layer_norm = norm_layer_cls(self.dims[i])
            block = VSSBlock(
                hidden_dim=self.dims[i], 
                channel_first=self.channel_first,
                norm_layer=lambda dim: layer_norm,
                **vss_kwargs
            )
            
            out_dim = self.dims[i+1] if i < 3 else self.dims[i]
            downsample = PatchMerging2D(self.dims[i], out_dim, norm_layer=nn.LayerNorm)
            
            self.layers.append(nn.Sequential(block, downsample))
        
        for i in out_indices:
            if self.channel_first:
                self.add_module(f'outnorm{i}', LayerNorm2d(self.dims[i]))
            else:
                self.add_module(f'outnorm{i}', nn.LayerNorm(self.dims[i]))
        
        self.load_pretrained(pretrained)

    def load_pretrained(self, ckpt=None, key="model"):
        if ckpt is None:
            return
        try:
            _ckpt = torch.load(ckpt, map_location=torch.device("cpu"))
            print(f"Successfully loaded checkpoint {ckpt}")
            incompatible_keys = self.load_state_dict(_ckpt.get(key, _ckpt), strict=False)
            print(incompatible_keys)
        except Exception as e:
            print(f"Failed loading checkpoint: {e}")

    def forward(self, x):
        x = self.patch_embed(x)
        
        outs = []
        for i, layer in enumerate(self.layers):
            blocks, downsample = layer[0], layer[1]
            x = blocks(x)
            
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(x)
                outs.append(out)
            
            if self.channel_first:
                x = x.permute(0, 2, 3, 1).contiguous()
            
            x = downsample(x)
            
            if self.channel_first:
                x = x.permute(0, 3, 1, 2).contiguous()
        
        return outs

class ResBlock(nn.Module):
    """Residual block with convolution layers"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.downsample(residual)
        out = self.relu(out)
        return out

class ChangeDecoder(nn.Module):
    """Decoder for change detection tasks"""
    def __init__(self, encoder_dims, channel_first, norm_layer, ssm_act_layer, mlp_act_layer,** kwargs):
        super().__init__()
        self.channel_first = channel_first
        self.encoder_dims = encoder_dims
        
        decoder_kwargs = {k: v for k, v in kwargs.items() if k not in ['distributed', 'ssm_act_layer', 'mlp_act_layer']}
        
        if channel_first:
            self.norm_layer_cls = LayerNorm2d
        else:
            self.norm_layer_cls = nn.LayerNorm
        
        self.st_blocks = nn.ModuleDict()
        
        self.st_blocks['block_41'] = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=512, out_channels=64),
            VSSBlock(
                hidden_dim=64,
                channel_first=channel_first,
                norm_layer=self.norm_layer_cls,
                **decoder_kwargs
            )
        )
        self.st_blocks['block_42'] = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=256, out_channels=64),
            VSSBlock(
                hidden_dim=64,
                channel_first=channel_first,
                norm_layer=self.norm_layer_cls,** decoder_kwargs
            )
        )
        self.st_blocks['block_43'] = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=256, out_channels=64),
            VSSBlock(
                hidden_dim=64,
                channel_first=channel_first,
                norm_layer=self.norm_layer_cls,
                **decoder_kwargs
            )
        )
        
        self.st_blocks['block_31'] = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=256, out_channels=64),
            VSSBlock(
                hidden_dim=64,
                channel_first=channel_first,
                norm_layer=self.norm_layer_cls,** decoder_kwargs
            )
        )
        self.st_blocks['block_32'] = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=128, out_channels=64),
            VSSBlock(
                hidden_dim=64,
                channel_first=channel_first,
                norm_layer=self.norm_layer_cls,
                **decoder_kwargs
            )
        )
        self.st_blocks['block_33'] = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=128, out_channels=64),
            VSSBlock(
                hidden_dim=64,
                channel_first=channel_first,
                norm_layer=self.norm_layer_cls,** decoder_kwargs
            )
        )
        
        self.st_blocks['block_21'] = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=128, out_channels=64),
            VSSBlock(
                hidden_dim=64,
                channel_first=channel_first,
                norm_layer=self.norm_layer_cls,
                **decoder_kwargs
            )
        )
        self.st_blocks['block_22'] = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=64, out_channels=64),
            VSSBlock(
                hidden_dim=64,
                channel_first=channel_first,
                norm_layer=self.norm_layer_cls,** decoder_kwargs
            )
        )
        self.st_blocks['block_23'] = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=64, out_channels=64),
            VSSBlock(
                hidden_dim=64,
                channel_first=channel_first,
                norm_layer=self.norm_layer_cls,
                **decoder_kwargs
            )
        )
        
        self.st_blocks['block_11'] = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=64, out_channels=64),
            VSSBlock(
                hidden_dim=64,
                channel_first=channel_first,
                norm_layer=self.norm_layer_cls,** decoder_kwargs
            )
        )
        self.st_blocks['block_12'] = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=32, out_channels=64),
            VSSBlock(
                hidden_dim=64,
                channel_first=channel_first,
                norm_layer=self.norm_layer_cls,
                **decoder_kwargs
            )
        )
        self.st_blocks['block_13'] = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=32, out_channels=64),
            VSSBlock(
                hidden_dim=64,
                channel_first=channel_first,
                norm_layer=self.norm_layer_cls,** decoder_kwargs
            )
        )
        
        self.fuse_layers = nn.ModuleList([
            nn.Sequential(nn.Conv2d(64*5, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU())
            for _ in range(4)
        ])
        
        self.smooth_layers = nn.ModuleList([
            ResBlock(64, 64) for _ in range(3)
        ])

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, pre_features, post_features):
        pre_feat = pre_features[::-1]
        post_feat = post_features[::-1]
        
        if pre_feat[0].shape[2:] != post_feat[0].shape[2:]:
            post_feat[0] = F.interpolate(
                post_feat[0], 
                size=pre_feat[0].shape[2:], 
                mode='bilinear', 
                align_corners=True
            )
        
        p41 = self.st_blocks['block_41'](torch.cat([pre_feat[0], post_feat[0]], dim=1))
        B, C, H, W = pre_feat[0].shape
        ct42 = torch.empty(B, C, H, 2*W, device=pre_feat[0].device)
        ct42[..., ::2] = pre_feat[0]
        ct42[..., 1::2] = post_feat[0]
        p42 = self.st_blocks['block_42'](ct42)
        ct43 = torch.cat([pre_feat[0], post_feat[0]], dim=3)
        p43 = self.st_blocks['block_43'](ct43)
        p4 = self.fuse_layers[0](torch.cat([
            p41, 
            p42[..., ::2], p42[..., 1::2],
            p43[..., :W], p43[..., W:]
        ], dim=1))
        
        if pre_feat[1].shape[2:] != post_feat[1].shape[2:]:
            post_feat[1] = F.interpolate(
                post_feat[1], 
                size=pre_feat[1].shape[2:], 
                mode='bilinear', 
                align_corners=True
            )
        
        p31 = self.st_blocks['block_31'](torch.cat([pre_feat[1], post_feat[1]], dim=1))
        B, C, H, W = pre_feat[1].shape
        ct32 = torch.empty(B, C, H, 2*W, device=pre_feat[1].device)
        ct32[..., ::2] = pre_feat[1]
        ct32[..., 1::2] = post_feat[1]
        p32 = self.st_blocks['block_32'](ct32)
        ct33 = torch.cat([pre_feat[1], post_feat[1]], dim=3)
        p33 = self.st_blocks['block_33'](ct33)
        p3 = self.fuse_layers[1](torch.cat([
            p31, 
            p32[..., ::2], p32[..., 1::2],
            p33[..., :W], p33[..., W:]
        ], dim=1))
        p3 = self._upsample_add(p4, p3)
        p3 = self.smooth_layers[0](p3)
        
        if pre_feat[2].shape[2:] != post_feat[2].shape[2:]:
            post_feat[2] = F.interpolate(
                post_feat[2], 
                size=pre_feat[2].shape[2:], 
                mode='bilinear', 
                align_corners=True
            )
        
        p21 = self.st_blocks['block_21'](torch.cat([pre_feat[2], post_feat[2]], dim=1))
        B, C, H, W = pre_feat[2].shape
        ct22 = torch.empty(B, C, H, 2*W, device=pre_feat[2].device)
        ct22[..., ::2] = pre_feat[2]
        ct22[..., 1::2] = post_feat[2]
        p22 = self.st_blocks['block_22'](ct22)
        ct23 = torch.cat([pre_feat[2], post_feat[2]], dim=3)
        p23 = self.st_blocks['block_23'](ct23)
        p2 = self.fuse_layers[2](torch.cat([
            p21, 
            p22[..., ::2], p22[..., 1::2],
            p23[..., :W], p23[..., W:]
        ], dim=1))
        p2 = self._upsample_add(p3, p2)
        p2 = self.smooth_layers[1](p2)
        
        if pre_feat[3].shape[2:] != post_feat[3].shape[2:]:
            post_feat[3] = F.interpolate(
                post_feat[3], 
                size=pre_feat[3].shape[2:], 
                mode='bilinear', 
                align_corners=True
            )
        
        p11 = self.st_blocks['block_11'](torch.cat([pre_feat[3], post_feat[3]], dim=1))
        B, C, H, W = pre_feat[3].shape
        ct12 = torch.empty(B, C, H, 2*W, device=pre_feat[3].device)
        ct12[..., ::2] = pre_feat[3]
        ct12[..., 1::2] = post_feat[3]
        p12 = self.st_blocks['block_12'](ct12)
        ct13 = torch.cat([pre_feat[3], post_feat[3]], dim=3)
        p13 = self.st_blocks['block_13'](ct13)
        p1 = self.fuse_layers[3](torch.cat([
            p11, 
            p12[..., ::2], p12[..., 1::2],
            p13[..., :W], p13[..., W:]
        ], dim=1))
        p1 = self._upsample_add(p2, p1)
        p1 = self.smooth_layers[2](p1)
        
        return p1

class ChangeMamba(nn.Module):
    """Main model for change detection using Mamba architecture"""
    def __init__(self, fixed_in_channels, output_nc=2, dynamic_in_channels: Optional[List[int]] = None, pretrained=None, **kwargs):
        super().__init__()
        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )
        
        ssm_act = kwargs.get('ssm_act_layer', 'silu')
        if isinstance(ssm_act, str):
            ssm_act = _ACTLAYERS.get(ssm_act.lower(), nn.SiLU)
        
        mlp_act = kwargs.get('mlp_act_layer', 'gelu')
        if isinstance(mlp_act, str):
            mlp_act = _ACTLAYERS.get(mlp_act.lower(), nn.GELU)
        
        clean_encoder_kwargs = {k: v for k, v in kwargs.items() if k not in ['ssm_act_layer', 'mlp_act_layer']}
        
        self.encoder = Backbone_VSSM(
            out_indices=(0, 1, 2, 3), 
            pretrained=pretrained, 
            in_channels=fixed_in_channels,
            ssm_act_layer=ssm_act,
            mlp_act_layer=mlp_act,** clean_encoder_kwargs
        )
        
        self.dynamic_branch_count = len(dynamic_in_channels) if dynamic_in_channels else 0
        
        self.dynamic_adapters = nn.ModuleList()
        if dynamic_in_channels:
            for in_ch in dynamic_in_channels:
                adapter = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(in_ch, self.encoder.dims[i], kernel_size=1, padding=0),
                        LayerNorm2d(self.encoder.dims[i]) if self.encoder.channel_first else nn.LayerNorm(self.encoder.dims[i])
                    ) for i in range(4)
                ])
                self.dynamic_adapters.append(adapter)
        
        self.fusion_gates = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2 * self.encoder.dims[i], self.encoder.dims[i], kernel_size=1),
                nn.Sigmoid()
            ) for i in range(4)
        ])
        
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )

        norm_layer = _NORMLAYERS.get(kwargs.get('norm_layer', 'ln2d').lower(), None)       
       
        clean_decoder_kwargs = {k: v for k, v in kwargs.items() if k not in ['norm_layer', 'ssm_act_layer', 'mlp_act_layer']}
        self.decoder = ChangeDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act,
            mlp_act_layer=mlp_act,
            **clean_decoder_kwargs
        )

        self.main_clf = nn.Conv2d(in_channels=64, out_channels=output_nc, kernel_size=1)
        self.distributed = kwargs.get('distributed', False)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, pre_data, post_data, dynamic_inputs=None):
        pre_features = self.encoder(pre_data)
        post_features = self.encoder(post_data)
        
        if dynamic_inputs is not None and self.dynamic_adapters:
            assert len(dynamic_inputs) == len(self.dynamic_adapters), \
                f"Number of dynamic inputs {len(dynamic_inputs)} does not match number of adapters {len(self.dynamic_adapters)}"
            
            all_dynamic_features = []
            for adapter, inp in zip(self.dynamic_adapters, dynamic_inputs):
                if inp.shape[2:] != pre_data.shape[2:]:
                    inp = F.interpolate(
                        inp, 
                        size=pre_data.shape[2:], 
                        mode='bilinear', 
                        align_corners=True
                    )
                dyn_feats = []
                for i in range(4):
                    feat = adapter[i](inp)
                    target_size = pre_features[i].shape[2:]
                    if feat.shape[2:] != target_size:
                        feat = F.interpolate(
                            feat, 
                            size=target_size, 
                            mode='bilinear', 
                            align_corners=True
                        )
                    dyn_feats.append(feat)
                all_dynamic_features.append(dyn_feats)
            
            enhanced_pre = []
            enhanced_post = []
            for i in range(4):
                dyn_feats_level = [dyn_feats[i] for dyn_feats in all_dynamic_features]
                aggregated_dyn = torch.mean(torch.stack(dyn_feats_level), dim=0)
                
                if aggregated_dyn.shape != pre_features[i].shape:
                    aggregated_dyn = F.interpolate(
                        aggregated_dyn,
                        size=pre_features[i].shape[2:],
                        mode='bilinear',
                        align_corners=True
                    )
                
                combined = torch.cat([pre_features[i], aggregated_dyn], dim=1)
                weight = self.fusion_gates[i](combined)
                
                enhanced_pre.append(pre_features[i] * weight + aggregated_dyn * (1 - weight))
                enhanced_post.append(post_features[i] * weight + aggregated_dyn * (1 - weight))
            
            pre_features, post_features = enhanced_pre, enhanced_post
        
        output = self.decoder(pre_features, post_features)
        
        output = self.main_clf(output)
        output = F.interpolate(output, size=pre_data.size()[-2:], mode='bilinear', align_corners=True)
        return output

    def prepare_for_distributed(self, device):
        if self.distributed:
            self.encoder = DDP(self.encoder.to(device), device_ids=[device])
            self.dynamic_adapters = DDP(self.dynamic_adapters.to(device), device_ids=[device])
            self.decoder = DDP(self.decoder.to(device), device_ids=[device])
            self.main_clf = DDP(self.main_clf.to(device), device_ids=[device])
        return self


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import time
    start_time = time.time()

    model_kwargs = {
        'norm_layer': 'ln2d',
        'ssm_act_layer': 'silu',
        'mlp_act_layer': 'gelu',
        'distributed': False
    }

    print("Test 1: Standard input")
    x_before = torch.randn(2, 12, 256, 256)
    x_after = torch.randn(2, 12, 256, 256)
    dynamic_inputs = [
        torch.randn(2, 4, 256, 256),
        torch.randn(2, 3, 256, 256)
    ]
    model = ChangeMamba(fixed_in_channels=12, dynamic_in_channels=[4, 3],** model_kwargs)
    output = model(x_before, x_after, dynamic_inputs)
    params = count_parameters(model)
    print(f"Output shape: {output.shape} Expected: (2, 2, 256, 256)")
    print(f"Model parameters: {params:,}\n")

    print("Test 2: Different bands and sizes")
    x_before = torch.randn(3, 8, 512, 512)
    x_after = torch.randn(3, 8, 512, 512)
    dynamic_inputs = [torch.randn(3, 5, 512, 512)]
    model = ChangeMamba(fixed_in_channels=8, dynamic_in_channels=[5], **model_kwargs)
    output = model(x_before, x_after, dynamic_inputs)
    params = count_parameters(model)
    print(f"Output shape: {output.shape} Expected: (3, 2, 512, 512)")
    print(f"Model parameters: {params:,}\n")

    print("Test 3: No dynamic branches")
    x_before = torch.randn(1, 10, 128, 128)
    x_after = torch.randn(1, 10, 128, 128)
    model = ChangeMamba(fixed_in_channels=10,** model_kwargs)
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
    model = ChangeMamba(fixed_in_channels=12, dynamic_in_channels=[1, 1, 1, 1, 1, 1, 1], **model_kwargs)
    output = model(x_before, x_after, dynamic_inputs)
    params = count_parameters(model)
    print(f"Output shape: {output.shape} Expected: (2, 2, 256, 256)")
    print(f"Model parameters: {params:,}\n")

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")