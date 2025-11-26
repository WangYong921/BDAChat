#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_image_tower, build_video_tower
from .multimodal_projector.builder import build_vision_projector

from videollava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if getattr(config, "mm_image_tower", None) is not None:
            self.image_tower = build_image_tower(config, delay_load=True)
        if getattr(config, "mm_video_tower", None) is not None:
            self.video_tower = build_video_tower(config, delay_load=True)
        if getattr(config, "mm_image_tower", None) is not None or getattr(config, "mm_video_tower", None) is not None:
            self.mm_projector = build_vision_projector(config)

    def get_image_tower(self):
        image_tower = getattr(self, 'image_tower', None)
        if isinstance(image_tower, list):
            image_tower = image_tower[0]
        return image_tower

    def get_video_tower(self):
        video_tower = getattr(self, 'video_tower', None)
        if isinstance(video_tower, list):
            video_tower = video_tower[0]
        return video_tower

    def initialize_vision_modules(self, model_args, fsdp=None):

        image_tower = model_args.image_tower
        video_tower = model_args.video_tower
        assert image_tower is not None or video_tower is not None

        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_image_tower = image_tower
        if image_tower is not None:
            if self.get_image_tower() is None:
                image_tower = build_image_tower(model_args)
                if fsdp is not None and len(fsdp) > 0:
                    self.image_tower = [image_tower]
                else:
                    self.image_tower = image_tower
            else:
                image_tower = self.image_tower[0] if (fsdp is not None and len(fsdp) > 0) else self.image_tower
                image_tower.load_model()

        self.config.mm_video_tower = video_tower
        if video_tower is not None:
            if self.get_video_tower() is None:
                video_tower = build_video_tower(model_args)
                if fsdp is not None and len(fsdp) > 0:
                    self.video_tower = [video_tower]
                else:
                    self.video_tower = video_tower
            else:
                video_tower = self.video_tower[0] if (fsdp is not None and len(fsdp) > 0) else self.video_tower
                video_tower.load_model()


        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

 
        image_hidden_size = getattr(image_tower, 'hidden_size', -1) if image_tower is not None else -1
        video_hidden_size = getattr(video_tower, 'hidden_size', -1) if video_tower is not None else -1
        if image_tower is not None and video_tower is not None:
            assert image_hidden_size == video_hidden_size
            self.config.mm_hidden_size = image_hidden_size
        else:
            self.config.mm_hidden_size = max(image_hidden_size, video_hidden_size)
            assert self.config.mm_hidden_size > 0

       
        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
          
            for p in self.mm_projector.parameters():
                p.requires_grad = True

   
        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')

            def extract_weights(weights, keyword):
                return {k.split(f"{keyword}.")[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(extract_weights(mm_projector_weights, 'mm_projector'))


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_image_tower(self):
        return self.get_model().get_image_tower()

    def get_video_tower(self):
        return self.get_model().get_video_tower()

    def encode_images(self, images):

        image_tower = self.get_image_tower()
        assert image_tower is not None
        image_features = image_tower(images)  # [batch, seq_len, hidden_size]
        image_features = self.get_model().mm_projector(image_features)  # [batch, seq_len, proj_size]
        return image_features

    def encode_videos(self, videos):

        video_tower = self.get_video_tower()
        assert video_tower is not None
        batch_size, _, num_frames, _, _ = videos.shape
        video_features = video_tower(videos)
        video_features = self.get_model().mm_projector(video_features)
        return video_features

    def prepare_inputs_labels_for_multimodal(
            self, input_ids, position_ids, attention_mask, past_key_values, labels, images
    ):
        
        image_tower = self.get_image_tower()
        video_tower = self.get_video_tower()

        if (image_tower is None and video_tower is None) or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and (
                    image_tower is not None or video_tower is not None) and images is not None and input_ids.shape[
                1] == 1:
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device
                    )
                ), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        
        image_indices = [idx for idx, img in enumerate(images) if img.ndim == 3]  
        video_indices = [idx for idx, vid in enumerate(images) if vid.ndim == 4]  

 
        image_features = []
        if image_indices:
            images_batch = torch.stack([images[idx] for idx in image_indices])  # [img_batch, C, H, W]
            image_features_batch = self.encode_images(images_batch)  # [img_batch, seq_len, proj_size]
            image_features = [image_features_batch[i] for i in range(len(image_indices))]

   
        video_features = []
        if video_indices:
            videos_batch = torch.stack([images[idx] for idx in video_indices])  # [vid_batch, C, T, H, W]
            video_features_batch = self.encode_videos(videos_batch)  # [vid_batch, T, seq_len, proj_size]
            for i in range(len(video_indices)):
           
                video_features.extend([video_features_batch[i, t] for t in range(video_features_batch.shape[1])])

        all_media_features = []
        img_idx = vid_idx = 0
        for i in range(len(images)):
            if i in image_indices:
                all_media_features.append(image_features[img_idx])
                img_idx += 1
            elif i in video_indices:
      
                t = video_features_batch[vid_idx].shape[0]
                all_media_features.extend(video_features[vid_idx:vid_idx + t])
                vid_idx += t


        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        orig_labels = labels
        orig_position_ids = position_ids
        orig_attention_mask = attention_mask

        attention_mask = torch.ones_like(input_ids,
                                         dtype=torch.bool) if attention_mask is None else attention_mask.bool()
        position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long,
                                    device=input_ids.device) if position_ids is None else position_ids
        labels = torch.full_like(input_ids, IGNORE_INDEX) if labels is None else labels


        input_ids = [ids[mask] for ids, mask in zip(input_ids, attention_mask)]
        labels = [lbl[mask] for lbl, mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        media_idx = 0  

        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_media = (cur_input_ids == IMAGE_TOKEN_INDEX).sum().item()  

            if num_media == 0:
 
                cur_embeds = self.get_model().embed_tokens(cur_input_ids)
                new_input_embeds.append(cur_embeds)
                new_labels.append(labels[batch_idx])
                media_idx += 1  
                continue

            media_positions = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [len(cur_input_ids)]
            text_segments = []
            label_segments = []

            for i in range(len(media_positions) - 1):
                text_segments.append(cur_input_ids[media_positions[i] + 1: media_positions[i + 1]])
                label_segments.append(labels[batch_idx][media_positions[i] + 1: media_positions[i + 1]])

            text_embeds = self.get_model().embed_tokens(torch.cat(text_segments))
            text_embeds_split = torch.split(text_embeds, [len(seg) for seg in text_segments], dim=0)

            batch_embeds = []
            batch_labels = []

            for i in range(num_media + 1):
                batch_embeds.append(text_embeds_split[i])
                batch_labels.append(label_segments[i])

                if i < num_media:
                    if media_idx >= len(all_media_features):
                        raise IndexError
                    media_feat = all_media_features[media_idx]
                    media_idx += 1
                    batch_embeds.append(media_feat)
                    batch_labels.append(torch.full(
                        (media_feat.shape[0],),
                        IGNORE_INDEX,
                        device=labels[batch_idx].device,
                        dtype=labels[batch_idx].dtype
                    ))

            new_input_embeds.append(torch.cat(batch_embeds))
            new_labels.append(torch.cat(batch_labels))

        max_seq_len = getattr(self.config, 'tokenizer_model_max_length', None)
        if max_seq_len is not None:
            new_input_embeds = [emb[:max_seq_len] for emb in new_input_embeds]
            new_labels = [lbl[:max_seq_len] for lbl in new_labels]

        max_len = max(emb.shape[0] for emb in new_input_embeds) if new_input_embeds else 0
        batch_size = len(new_input_embeds)

        padded_embeds = []
        padded_labels = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device
        ) if new_labels else None
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=orig_attention_mask.dtype,
            device=orig_attention_mask.device
        ) if orig_attention_mask is not None else None
        position_ids = torch.zeros(
            (batch_size, max_len),
            dtype=orig_position_ids.dtype,
            device=orig_position_ids.device
        ) if orig_position_ids is not None else None

        for i, (emb, lbl) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = emb.shape[0]
            padding_side = getattr(self.config, 'tokenizer_padding_side', 'right')

            if padding_side == "left":
                pad_len = max_len - cur_len
                padded_emb = torch.cat((
                    torch.zeros((pad_len, emb.shape[1]), dtype=emb.dtype, device=emb.device),
                    emb
                ), dim=0)
                padded_embeds.append(padded_emb)

                if cur_len > 0 and padded_labels is not None:
                    padded_labels[i, -cur_len:] = lbl
                if attention_mask is not None:
                    attention_mask[i, -cur_len:] = True
                if position_ids is not None:
                    position_ids[i, -cur_len:] = torch.arange(cur_len, device=position_ids.device)
            else:
                pad_len = max_len - cur_len
                padded_emb = torch.cat((
                    emb,
                    torch.zeros((pad_len, emb.shape[1]), dtype=emb.dtype, device=emb.device)
                ), dim=0)
                padded_embeds.append(padded_emb)

                if cur_len > 0 and padded_labels is not None:
                    padded_labels[i, :cur_len] = lbl
                if attention_mask is not None:
                    attention_mask[i, :cur_len] = True
                if position_ids is not None:
                    position_ids[i, :cur_len] = torch.arange(cur_len, device=position_ids.device)


        new_input_embeds = torch.stack(padded_embeds, dim=0) if padded_embeds else None

        if orig_labels is None:
            padded_labels = None
        if orig_attention_mask is None:
            attention_mask = None
        if orig_position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, padded_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
        if model_args.mm_use_im_start_end:
            new_tokens = [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN]
            num_new_tokens = tokenizer.add_tokens(new_tokens, special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeds = self.get_input_embeddings().weight.data
                output_embeds = self.get_output_embeddings().weight.data if self.get_output_embeddings() is not None else None

                avg_input_embed = input_embeds[:-num_new_tokens].mean(dim=0, keepdim=True)
                input_embeds[-num_new_tokens:] = avg_input_embed

                if output_embeds is not None:
                    avg_output_embed = output_embeds[:-num_new_tokens].mean(dim=0, keepdim=True)
                    output_embeds[-num_new_tokens:] = avg_output_embed


            if model_args.pretrain_mm_mlp_adapter:
                adapter_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_weights = adapter_weights.get('model.embed_tokens.weight', None)
                if embed_weights is not None:
                    assert num_new_tokens == 2, 
                    if input_embeds.shape == embed_weights.shape:
                        input_embeds[-num_new_tokens:] = embed_weights[-num_new_tokens:]
                    elif embed_weights.shape[0] == num_new_tokens:
                        input_embeds[-num_new_tokens:] = embed_weights
                    else:
                        raise ValueError

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                if self.get_output_embeddings() is not None:
                    for p in self.get_output_embeddings().parameters():
                        p.requires_grad = False

        elif model_args.mm_use_im_patch_token and model_args.tune_mm_mlp_adapter:
            for p in self.get_input_embeddings().parameters():
                p.requires_grad = False
            if self.get_output_embeddings() is not None:
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
