
import torch
import torch.nn as nn
import math
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class MSC_ResBlock(nn.Module):
    """多尺度残差块：融合3x3与5x5卷积感受野"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, in_channels, 5, padding=2)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.relu(self.bn(self.conv3x3(x) + self.conv5x5(x)))
        return identity + x


class CLIPVisionTower(nn.Module):
    """
    多尺度+灾种感知CLIP视觉编码器
    """
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)
            self._init_components(self.cfg_only.hidden_size)

    def _init_components(self, hidden_size):
        self.msc_block = MSC_ResBlock(hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8, batch_first=True),
            num_layers=2
        )
        self.num_disasters = 10
        self.disaster_embedding = nn.Embedding(self.num_disasters, hidden_size)

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)
        self._init_components(self.vision_tower.config.hidden_size)
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            return image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            return image_features
        raise ValueError(f'未知特征选择模式: {self.select_feature}')

    def _process_features(self, features, disaster_ids=None):
        features = self.transformer(features)
        B, N, C = features.shape
        h = w = int(math.sqrt(N))
        features = features.permute(0, 2, 1).view(B, C, h, w)
        features = self.msc_block(features)
        seq = features.flatten(2).permute(0, 2, 1)  # [B, N, C]

        if disaster_ids is not None:
            prompt = self.disaster_embedding(disaster_ids).unsqueeze(1)
            seq = seq + prompt  # 广播加和
        return seq

    @torch.no_grad()
    def forward(self, images, disaster_ids=None):
        if isinstance(images, list):
            image_features = []
            for img in images:
                feature = self.vision_tower(
                    img.unsqueeze(0).to(self.device, self.dtype),
                    output_hidden_states=True
                )
                feature = self.feature_select(feature)
                image_features.append(self._process_features(feature, disaster_ids))
            return torch.cat(image_features, dim=0)

        image_forward_outs = self.vision_tower(
            images.to(self.device, self.dtype),
            output_hidden_states=True
        )
        image_features = self.feature_select(image_forward_outs)
        return self._process_features(image_features, disaster_ids)

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        return self.vision_tower.config if self.is_loaded else self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
