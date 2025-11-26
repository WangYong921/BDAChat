import torch
import torch.nn as nn
import re

# 新增MPLUG跳跃连接投影器
class MPLUGSkipProjector(nn.Module):
    def __init__(self, vision_size, text_size, num_layers=2):
        super().__init__()
        self.vision_size = vision_size
        self.text_size = text_size
        self.num_layers = num_layers  # 跳跃连接的层数

        # 视觉特征投影到文本空间（基础投影）
        self.vision_proj = nn.Linear(vision_size, text_size)

        # 跨模态交互层（每层含注意力和残差）
        self.cross_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(text_size),
                nn.MultiheadAttention(embed_dim=text_size, num_heads=4, batch_first=True),
                nn.GELU(),
                nn.Linear(text_size, text_size)
            ) for _ in range(num_layers)
        ])

        # 跳跃连接的残差层
        self.residual_proj = nn.Linear(text_size, text_size)

    def forward(self, vision_feat, text_feat):
        # 1. 视觉特征投影到文本维度
        vision_proj = self.vision_proj(vision_feat)  # (batch, seq_len_v, text_size)

        # 2. 初始化融合特征（视觉投影+文本特征残差）
        fusion_feat = vision_proj + self.residual_proj(text_feat)  # 首次跳跃连接

        # 3. 多阶段跨模态交互（每层都加入跳跃连接）
        for layer in self.cross_layers:
            # 跨模态注意力（用文本特征作为query，当前融合特征作为key/value）
            norm_feat = layer[0](fusion_feat)
            attn_out, _ = layer[1](query=text_feat, key=norm_feat, value=norm_feat)
            # 非线性变换+残差连接（跳跃连接）
            layer_out = layer[3](layer[2](attn_out))
            fusion_feat = fusion_feat + layer_out  # 累加残差

        return fusion_feat  # 输出融合后的特征（维度为text_size）

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
    
# 在build_visi
    if projector_type == 'mplug_skip':
        return MPLUGSkipProjector(
            vision_size=config.mm_hidden_size,
            text_size=config.hidden_size,
            num_layers=getattr(config, 'mplug_skip_layers', 2)  # 可通过配置指定层数
        )

    raise ValueError(f'Unknown projector type: {projector_type}')

