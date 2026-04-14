#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Action Chunking Transformer Policy

As per Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (https://huggingface.co/papers/2304.13705).
The majority of changes here involve removing unused code, unifying naming, and adding helpful comments.
"""

import math
from collections import deque
from collections.abc import Callable
from itertools import chain
from typing import Any, Dict, cast, Optional, Tuple

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d
import transformers

from lerobot.common.policies.art.configuration_art import ARTConfig
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.configs.types import NormalizationMode


class ARTPolicy(PreTrainedPolicy):
    """
    Action Chunking Transformer Policy as per Learning Fine-Grained Bimanual Manipulation with Low-Cost
    Hardware (paper: https://huggingface.co/papers/2304.13705, code: https://github.com/tonyzhaozh/act)
    """

    config_class = ARTConfig
    name = "art"

    def __init__(
        self,
        config: ARTConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config




        
        if self.config.tokenize_actions:
            # self.action_tokenizer = SpatialActionTokenizer(num_bins=self.config.action_bins)
            self.action_tokenizer = PerDimKMeansTokenizer(centers_path=self.config.tokenizer_pth)
            if config.action_bins is None:
                config.action_bins = self.action_tokenizer.num_bins
            self.config.normalization_mapping["ACTION"] = NormalizationMode.IDENTITY
            
        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        
        self.model = ART(config)


        self.reset()

    def get_optim_params(self) -> dict:
        # TODO(aliberts, rcadene): As of now, lr_backbone == lr
        # Should we remove this and just `return self.parameters()`?
        return self.parameters()

    def reset(self):
        """This should be called whenever the environment is reset."""
        # now n_action_steps means the interval between two vision calls
        self.test_step_counter = 0
        self.model.reset_episode()

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()
        
        # print everything in the batch, with shape info
        # for k, v in batch.items():
        #     print(f"{k}: {v.shape}")
        
        if self.config.tokenize_actions and self.config.tokenize_delta_actions:
            # need cache unnormalized current state for delta action decoding
            cache_current_state = batch["observation.state"].clone()

        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )
            
        if self.test_step_counter % self.config.n_action_steps == 0:
            self.model.update_vision_prefix(batch)


        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        out_dict = self.model.generate_next_action(batch)  # [B, 1, Dim]
        
        # out_dict["action_out"] shape: [B, 1, Dim * Num_Bins]
        actions_raw = out_dict.get("action_out").squeeze(1) # [B, Dim * Num_Bins]

        if self.config.tokenize_actions:
            B = actions_raw.shape[0]
            Dim = self.action_tokenizer.action_dim
            Bins = self.action_tokenizer.num_bins
            
            # 1. Reshape: [B, D*K] -> [B, D, K]
            action_logits = actions_raw.view(B, Dim, Bins)
            
            # 2. Argmax per dimension: [B, D]
            action_tokens = torch.argmax(action_logits, dim=-1)
            
            # 3. Decode: [B, D] -> [B, D] continuous
            actions = self.action_tokenizer.decode(action_tokens)
            
            if self.config.tokenize_delta_actions:
                actions = actions + cache_current_state
        else:
            actions = self.unnormalize_outputs({"action": actions_raw})["action"]

        self.test_step_counter += 1
        return actions

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Run the batch through the model and compute the loss for training or validation."""
        if self.config.tokenize_actions and self.config.tokenize_delta_actions:
            # need cache unnormalized current state for delta action decoding
            cache_current_state = batch["observation.state"].clone()
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )
        batch = self.normalize_targets(batch)
        forward_dict = self.model(batch)
        actions_hat = forward_dict["action_out"]
        actions_fast = forward_dict["fast_out"]
        
        
        if self.config.tokenize_actions:
            gt_actions = batch["action"]
            action_offsets = forward_dict.get("action_offsets", None)
            
            with torch.no_grad(): # No gradients needed for target generation
                if self.config.tokenize_delta_actions:
                    # need to convert to delta actions
                    # print("gt_actions:", gt_actions)
                    # print("batch['observation.state']:", cache_current_state[:, -self.config.chunk_size:, :])
                    gt_actions = gt_actions - cache_current_state[:, -self.config.chunk_size:, :] # [B, Seq_Len, Dim]
                # Tokenizer returns [Batch, Seq, Dim]
                action_gt_tokens = self.action_tokenizer(gt_actions)
                
            B, S, _ = actions_hat.shape
            Dim = action_gt_tokens.shape[-1]
            Bins = self.config.action_bins

            # 2. View as 4D tensor: [Batch, Seq, Dim, Bins]
            actions_hat_4d = actions_hat.view(B, S, Dim, Bins)
            actions_hat_permuted = actions_hat_4d.permute(0, 3, 1, 2)
            
            actions_fast_4d = actions_fast.view(B, S, Dim, Bins)
            actions_fast_permuted = actions_fast_4d.permute(0, 3, 1, 2)
            
            
            # TODO: correct here
            
            mask = ~batch["action_is_pad"] # [B, S]
            mask = mask.unsqueeze(-1).expand(-1, -1, Dim) # [B, S, D]

                
                
                # print("action_tokens:", action_tokens)
            
            ar_loss = (
                F.cross_entropy(
                    actions_hat_permuted,
                    action_gt_tokens,  # (B, S)
                    reduction="none",
                ) * mask
            ).mean()
            
            fast_loss = (
                F.cross_entropy(
                    actions_fast_permuted,
                    action_gt_tokens,  # (B, S)
                    reduction="none",
                ) * mask
            ).mean()
            
            # if action_offsets is not None:
            #     offset_loss = (
            #         F.l1_loss(
            #             action_offsets,
            #             action_gt_offsets,
            #             reduction="none"
            #         ) * ~batch["action_is_pad"].unsqueeze(-1)
            #     ).mean()
                
            loss_dict = {"ar_loss": ar_loss.item(), "fast_loss": fast_loss.item(), "loss": ar_loss + fast_loss}
        
        else:
        

            ar_loss = (
                F.l1_loss(batch["action"], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)
            ).mean()
            
            fast_loss = (
                F.l1_loss(batch["action"], actions_fast, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)
            ).mean()

            loss_dict = {"ar_loss": ar_loss.item(), "fast_loss": fast_loss.item(),   "loss": ar_loss + fast_loss}
        
        # loss_dict["loss"] = l1_loss

        # loss = l1_loss

        return loss_dict

class ART(nn.Module):
    """ modified Action Chunking Transformer model, now with AR decoder and pizero style kv sharing.
    cache the kv features from vision encoder for use in decoder.
    deocder maintains its own cache for history states."""

    def __init__(self, config: ARTConfig):
        # BERT style VAE encoder with input tokens [cls, robot_state, *action_sequence].
        # The cls token forms parameters of the latent's distribution (like this [*means, *log_variances]).
        super().__init__()
        self.config = config
        
        self.roper = RotaryPositionalEncoding(config.dim_model // config.n_heads)


        # Backbone for image feature extraction.
        if self.config.image_features:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            # Note: The assumption here is that we are using a ResNet model (and hence layer4 is the final
            # feature map).
            # Note: The forward method of this returns a dict: {"feature_map": output}.
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        # Transformer (acts as VAE decoder when training with the variational objective).
        self.encoder = ARTEncoder(config)
        self.decoder = ARTDecoder(config)

        # Transformer encoder input projections. The tokens will be structured like
        # [latent, (robot_state), (env_state), (image_feature_map_pixels)].
        if self.config.robot_state_feature:
            self.state_input_proj = nn.Linear(
                self.config.robot_state_feature.shape[0], config.dim_model
            )
        if self.config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(
                self.config.env_state_feature.shape[0], config.dim_model
            )
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        if self.config.image_features:
            self.encoder_img_feat_input_proj = nn.Conv2d(
                backbone_model.fc.in_features, config.dim_model, kernel_size=1
            )
        # Transformer encoder positional embeddings.
        n_1d_tokens = 0  # no latent
        if self.config.robot_state_feature:
            n_1d_tokens += 1
        if self.config.env_state_feature:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if self.config.image_features:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        # Transformer decoder.
        # Learnable positional embedding for the transformer's decoder (in the style of DETR object queries).
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

        # Final action regression head on the output of the transformer's decoder.
        if not self.config.tokenize_actions:
            self.action_head = nn.Linear(config.dim_model, self.config.action_feature.shape[0])
            
            self.fast_head = nn.Linear(config.dim_model, self.config.action_feature.shape[0])
        else:
            self.action_head = nn.Linear(config.dim_model, config.action_bins * self.config.action_feature.shape[0])
            # self.offset_head = nn.Linear(config.dim_model, self.config.action_feature.shape[0])
            
            self.fast_head = nn.Linear(config.dim_model, config.action_bins * self.config.action_feature.shape[0])
            
        self.act_decoder = ACTDecoder(config)
        
        if config.crop_shape is not None:
            self.do_crop = True
            # Always use center crop for eval
            self.center_crop = torchvision.transforms.CenterCrop(config.crop_shape)
        else: 
            self.do_crop = False

        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier-uniform initialization of the transformer parameters as in the original code."""
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
    #     """A forward pass through the Action Chunking Transformer (with optional VAE encoder).

    #     `batch` should have the following structure:
    #     {
    #         [robot_state_feature] (optional): (B, state_dim) batch of robot states.

    #         [image_features]: (B, n_cameras, C, H, W) batch of images.
    #             AND/OR
    #         [env_state_feature]: (B, env_dim) batch of environment states.

    #         [action_feature] (optional, only if training with VAE): (B, chunk_size, action dim) batch of actions.
    #     }

    #     Returns:
    #         (B, chunk_size, action_dim) batch of action sequences
    #         Tuple containing the latent PDF's parameters (mean, log(σ²)) both as (B, L) tensors where L is the
    #         latent dimension.
    #     """
    #     if self.config.use_vae and self.training:
    #         assert (
    #             "action" in batch
    #         ), "actions must be provided when using the variational objective in training mode."

    #     batch_size = (
    #         batch["observation.images"]
    #         if "observation.images" in batch
    #         else batch["observation.environment_state"]
    #     ).shape[0]
        
    #     # print everything in the batch, with shape info
    #     # for k, v in batch.items():
    #     #     print(f"batch[{k}]: shape={v.shape}, dtype={v.dtype}, device={v.device}")


    #     # Prepare the latent for input to the transformer encoder.

    #         # When not using the VAE encoder, we set the latent to be all zeros.
    #     mu = log_sigma_x2 = None
    #     # TODO(rcadene, alexander-soare): remove call to `.to` to speedup forward ; precompute and use buffer
    #     latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(
    #         batch["observation.state"].device
    #     )

    #     # Prepare transformer encoder inputs.
    #     encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
    #     encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
    #     # Robot state token.
    #     # if self.config.robot_state_feature:
    #     #     encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch["observation.state"]))
    #     # Environment state token.
    #     if self.config.env_state_feature:
    #         encoder_in_tokens.append(
    #             self.encoder_env_state_input_proj(batch["observation.environment_state"])
    #         )

    #     # Camera observation features and positional embeddings.
    #     if self.config.image_features:
    #         all_cam_features = []
    #         all_cam_pos_embeds = []

    #         for cam_index in range(batch["observation.images"].shape[-4]):
    #             cam_features = self.backbone(batch["observation.images"][:, cam_index])["feature_map"]
    #             # TODO(rcadene, alexander-soare): remove call to `.to` to speedup forward ; precompute and use
    #             # buffer
    #             cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
    #             cam_features = self.encoder_img_feat_input_proj(cam_features)  # (B, C, h, w)
    #             all_cam_features.append(cam_features)
    #             all_cam_pos_embeds.append(cam_pos_embed)
    #         # Concatenate camera observation feature maps and positional embeddings along the width dimension,
    #         # and move to (sequence, batch, dim).
    #         all_cam_features = torch.cat(all_cam_features, axis=-1)
    #         encoder_in_tokens.extend(einops.rearrange(all_cam_features, "b c h w -> (h w) b c"))
    #         all_cam_pos_embeds = torch.cat(all_cam_pos_embeds, axis=-1)
    #         encoder_in_pos_embed.extend(einops.rearrange(all_cam_pos_embeds, "b c h w -> (h w) b c"))

    #     # Stack all tokens along the sequence dimension.
    #     encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
    #     encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)
        
    #     cache = transformers.DynamicCache()  # Initialize empty cache for KV caching

    #     # Forward pass through the transformer modules.
    #     encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed, past_key_value=cache)
    #     # TODO(rcadene, alexander-soare): remove call to `device` ; precompute and use buffer
    #     decoder_in = torch.zeros(
    #         (self.config.chunk_size, batch_size, self.config.dim_model),
    #         dtype=encoder_in_pos_embed.dtype,
    #         device=encoder_in_pos_embed.device,
    #     )
    #     decoder_in[:] = self.encoder_robot_state_input_proj(batch["observation.state"])
    #     decoder_out = self.decoder(
    #         decoder_in,
    #         encoder_out,
    #         encoder_pos_embed=encoder_in_pos_embed,
    #         decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
    #     )

    #     # Move back to (B, S, C).
    #     decoder_out = decoder_out.transpose(0, 1)

    #     actions = self.action_head(decoder_out)

    #     return actions, (mu, log_sigma_x2)

    def forward(self, batch: dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Training Forward:
        1. VLM Prefix -> Cache (Pos 0)
        2. States (History + Future) -> Decoder[hist:hist + future] -> Future Actions
        """
        
        # 1. Vision Encoder (Prefix)
        encoder_in_tokens = []
        encoder_in_pos_embed = []
        
        # print("batch keys:", batch.keys())
        # print("batch actions", batch["action"])
        
        # raise NotImplementedError("This forward method is incomplete and for illustration only.")
        
        
        if self.config.robot_state_feature and self.config.encode_current_state_in_prefix:
            encoder_in_tokens.append(self.state_input_proj(batch["observation.state"][:,self.config.history_length:self.config.history_length+1,:]))
            encoder_in_pos_embed.append(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        if self.config.image_features:
            all_cam_features = []
            all_cam_pos_embed = []
            for cam_index in range(batch["observation.images"].shape[-4]):
                img = batch["observation.images"][:, cam_index] # [B, C, H, W]
                if self.do_crop:
                    img = self.center_crop(img)
                feat = self.backbone(img)["feature_map"]
                feat = self.encoder_img_feat_input_proj(feat)
                pos = self.encoder_cam_feat_pos_embed(feat).to(dtype=feat.dtype)
                all_cam_features.append(feat)
                all_cam_pos_embed.append(pos)
            
            all_cam_features = torch.cat(all_cam_features, dim=-1)
            encoder_in_tokens.append(einops.rearrange(all_cam_features, "b c h w -> b (h w) c"))
            all_cam_pos_embed = torch.cat(all_cam_pos_embed, dim=-1)
            encoder_in_pos_embed.append(einops.rearrange(all_cam_pos_embed, "b c h w -> b (h w) c"))

        encoder_in = torch.cat(encoder_in_tokens, dim=1)
        encoder_pos_embed = torch.cat(encoder_in_pos_embed, dim=1)
        
        # Populate Cache (Prefix)
        cache = transformers.DynamicCache()
        encoder_out = self.encoder(encoder_in, pos_emb=encoder_pos_embed, past_key_value=cache)
        

        

        
        
        prefix_len = cache[0][0].shape[-2]
        
        # detach cache to avoid gradients flowing into vision encoder
        for i, _ in enumerate(cache):
            cache.layers[i].keys = cache.layers[i].keys.detach()
            cache.layers[i].values = cache.layers[i].values.detach()
        # print cache shapes, 2nd dim is 0,1 for k,v
        # for i, layer_cache in enumerate(cache):
        #     print(f"Layer {i}:")
        #     print(f"  Key shape: {layer_cache[0].shape}")
        #     print(f"  Value shape: {layer_cache[1].shape}")

        # 2. State Projection (History + Future combined)
        # We don't slice history/future separately; we treat it as one sequence.
        all_states = batch["observation.state"] # [B, Seq_Len, Dim]
        
        # get state padding mask
        state_padding_mask = batch.get("observation.state_is_pad", None)
        
        # print("state_padding_mask:", state_padding_mask)
        
        decoder_input = self.state_input_proj(all_states)
        
        batch_size, seq_len, _ = decoder_input.shape
        device = decoder_input.device
        hist_len = self.config.history_length
        
        assert seq_len == hist_len + self.config.chunk_size, ValueError(
            f"Sequence length {seq_len} does not match history {hist_len} + chunk size {self.config.chunk_size}"
        )

        # 3. Construct RoPE Positions
        # Sequence indices: [0, 1, 2, ... L]
        # Shift by history: [-H, -H+1, ... -1, 0, 1 ... F]
        position_ids = torch.arange(seq_len, device=device) - hist_len
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # 4. Build Mask
        # Rules:
        # - Indices < 0 (History): Cannot see Prefix. Causal Self.
        # - Indices >= 0 (Future): See Prefix. Causal Self.
        mask = self._build_pizero_mask(batch_size, prefix_len, hist_len, seq_len, state_padding_mask, device)

        # 5. Run Decoder
        # DynamicCache automatically appends to the end, so we don't strictly need cache_position for training
        # unless we were doing something non-contiguous.
        decoder_out = self.decoder(
            decoder_input,
            attn_mask=mask,
            cache=cache,
            attn_kwargs={
                'query_position_indices': position_ids,
                'key_position_indices': position_ids,
                'cache_kwargs': {'cache_position': None} # Default: Append to end
            }
        )
        
        # for i, layer_cache in enumerate(cache):
        #     print(f"Layer {i}:")
        #     print(f"final  Key shape: {layer_cache[0].shape}")
        #     print(f"final  Value shape: {layer_cache[1].shape}")

        # 6. Extract Future (Indices corresponding to >= 0)
        future_out = decoder_out[:, hist_len:]
        
        # 7. fast heads
        fast_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_pos_embed.dtype,
            device=encoder_pos_embed.device,
        )
        encoder_out = encoder_out.transpose(0, 1)  # (B, S, C) -> (S, B, C)
        encoder_pos_embed = encoder_pos_embed.transpose(0, 1)  # (B, S, C) -> (S, B, C)
        
        # print("fast_in shape:", fast_in.shape)
        # print("encoder_out shape:", encoder_out.shape)
        # print("encoder_pos_embed shape:", encoder_pos_embed.shape)
        # print("decoder_pos_embed shape:", self.decoder_pos_embed.weight.unsqueeze(1).shape)
        
        fast_out = self.act_decoder(
            fast_in,
            encoder_out,
            encoder_pos_embed=encoder_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

        # Move back to (B, S, C).
        fast_out = fast_out.transpose(0, 1)

        fast_actions = self.fast_head(fast_out)
        
        return {
            "action_out": self.action_head(future_out),
            "fast_out": fast_actions,
            # "action_offsets": self.offset_head(future_out) if self.config.tokenize_actions else None
        }
        
        # return {a}self.action_head(future_out), fast_actions
    
    def reset_episode(self):
        """Reset all counters and cache for a new episode."""
        self.inference_cache = None
        # set rope idx as a random long int.
        self.inference_rope_idx = torch.randint(0, 10000, (1,)).item()
        self.inference_write_offset = 0
        self.prefix_len = 0

    @torch.inference_mode()
    def update_vision_prefix(self, batch: dict):
        """
        Refreshes the Vision Prefix in the Static Cache.
        CRITICAL: Applies RoPE to the new Vision Keys based on the CURRENT timestep.
        """
        # 1. Run Vision Encoder to get raw features
        encoder_in_tokens = []
        encoder_in_pos_embed = []
        
        if self.config.robot_state_feature and self.config.encode_current_state_in_prefix:
            encoder_in_tokens.append(self.state_input_proj(batch["observation.state"].unsqueeze(1)))
            encoder_in_pos_embed.append(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        if self.config.image_features:
            all_cam_features = []
            all_cam_pos_embed = []
            for cam_index in range(batch["observation.images"].shape[-4]):
                img = batch["observation.images"][:, cam_index]
                if self.do_crop:
                    img = self.center_crop(img)
                feat = self.backbone(img)["feature_map"]
                feat = self.encoder_img_feat_input_proj(feat)
                pos = self.encoder_cam_feat_pos_embed(feat).to(dtype=feat.dtype)
                all_cam_pos_embed.append(pos)
                all_cam_features.append(feat)
            
            all_cam_features = torch.cat(all_cam_features, dim=-1)
            encoder_in_tokens.append(einops.rearrange(all_cam_features, "b c h w -> b (h w) c"))
            encoder_in_pos_embed.append(einops.rearrange(torch.cat(all_cam_pos_embed, dim=-1), "b c h w -> b (h w) c"))

        encoder_in = torch.cat(encoder_in_tokens, dim=1)
        encoder_pos_embed = torch.cat(encoder_in_pos_embed, dim=1)
        
        # 2. Capture raw K/V via a temporary DynamicCache
        temp_cache = transformers.DynamicCache()
        _ = self.encoder(encoder_in, pos_emb=encoder_pos_embed, past_key_value=temp_cache)
        
        # Extract params from the fresh encode
        # k shape: [Batch, Heads, Seq_Len, Head_Dim], take from layer 0's k as example
        example_k = temp_cache[0][0]
        batch_size, num_heads, seq_len, head_dim = example_k.shape
        device = example_k.device
        self.prefix_len = seq_len

        # 3. Initialize Static Cache (if first run)
        if self.inference_cache is None:
            # Capacity: Prefix + History + Horizon + Buffer
            max_len = self.prefix_len + self.config.test_time_history
            
            self.inference_cache = transformers.StaticCache(
                config=transformers.PretrainedConfig(
                    num_hidden_layers=len(self.decoder.layers),
                    num_attention_heads=num_heads,
                    num_key_value_heads=num_heads,
                    hidden_size=self.config.dim_model,
                ),
                max_batch_size=batch_size,
                max_cache_len=max_len,
                device=device,
                dtype=example_k.dtype
            )
            # Counters start at 0
            self.inference_rope_idx = 0 

        # 4. Prepare for Insertion
        # We overwrite cache indices [0 ... prefix_len]
        cache_positions = torch.arange(self.prefix_len, device=device)
        
        # 5. RoPE Calculation for Vision Keys
        # "In test time we need to rope adding to the k of the visual kv to the current timestep"
        # We treat the entire vision prefix as existing at `self.inference_rope_idx`.
        rope_pos_ids = torch.full(
            (1, self.prefix_len), 
            self.inference_rope_idx, 
            device=device, 
            dtype=torch.long
        )
        
        # Pre-compute cos/sin for this timestep (Optimization)
        # Note: We pass dtype=k.dtype so we don't have mismatch errors
        cos, sin = self.roper(rope_pos_ids, device=device, dtype=example_k.dtype)

        # 6. Update Cache Layer by Layer
        for i in range(len(self.decoder.layers)):
            k_raw = temp_cache[i][0]
            v_raw = temp_cache[i][1]
            
            # Apply RoPE to Keys ONLY
            # K is [B, H, L, D], cos/sin is [1, 1, L, D]
            k_rotated = apply_rotary_pos_emb(k_raw, cos, sin)
            
            # Update StaticCache
            # We explicitly update the prefix region
            self.inference_cache.update(
                k_rotated, 
                v_raw, # Values are not rotated
                i, 
                cache_kwargs={'cache_position': cache_positions}
            )
            
            
    @torch.inference_mode()
    def generate_next_action(self, batch: Dict) -> Dict[str, Tensor]:
        """
        Generates the next action based on current state + cache.
        Incrementally updates cache and counters.
        """
        # 0. add one seq dim, [B, 1, D]
        all_states = batch["observation.state"] # [B, Dim]
        current_state = all_states.unsqueeze(1)
        # 1. Project State
        state_token = self.state_input_proj(current_state) # [B, 1, D]
        batch_size = state_token.shape[0]
        device = state_token.device

        # 2. Prepare Indices
        # RoPE: Increases naturally (0, 1, 2...)
        rope_ids = torch.tensor([self.inference_rope_idx], device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Cache Write: Points to the end of the sequence [Prefix + History + Current]
        cache_pos = torch.tensor([self.prefix_len + self.inference_write_offset], device=device, dtype=torch.long)

        # 3. Run Decoder
        # Mask=None -> StaticCache implies Causal Attention to all valid past data
        decoder_out = self.decoder(
            state_token,
            attn_mask=None,
            cache=self.inference_cache,
            attn_kwargs={
                'query_position_indices': rope_ids, # Q uses current time
                'key_position_indices': rope_ids,   # K uses current time
                'cache_kwargs': {'cache_position': cache_pos}
            }
        )

        # 4. Update Counters
        self.inference_rope_idx += 1
        self.inference_write_offset += 1
        if self.inference_write_offset == self.config.test_time_history:
            self.inference_write_offset = 0

        # 5. Output
        return {"action_out": self.action_head(decoder_out),
                # "offset_out": self.offset_head(decoder_out) if self.config.tokenize_actions else None}
        }
    
    
    def _build_pizero_mask(self, batch_size, prefix_len, hist_len, seq_len, state_padding_mask, device):
        """
        Masking Logic:
        Q: [Seq_Len] (History + Future)
        K: [Prefix + Seq_Len]
        """
        k_len = prefix_len + seq_len
        mask = torch.zeros(batch_size, 1, seq_len, k_len, dtype=torch.bool, device=device)
        
        # Slices
        c_prefix = slice(0, prefix_len)
        c_seq = slice(prefix_len, k_len) # The AR part of Keys
        
        r_future = slice(hist_len, seq_len)
        
        # 1. Causal Self-Attention for the whole sequence
        # This handles History->History and Future->Future and Future->History
        causal = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        mask[:, :, :, c_seq] = causal
        
        # 1.1. Apply State Padding Mask (if exists)
        # state_padding_mask is [Batch, Seq_Len] where True = Pad.
        # We need to block attention to any Key that is Padded.
        if state_padding_mask is not None:
            # Invert: True = Valid (Keep), False = Pad (Block)
            is_valid = ~state_padding_mask 
            
            # Broadcast: [B, Seq] -> [B, 1, 1, Seq] to match [B, 1, Q, K_seq]
            valid_key_mask = is_valid.unsqueeze(1).unsqueeze(1)
            
            # Apply AND: Keep true only if it was Causal AND it is Valid
            mask[:, :, :, c_seq] &= valid_key_mask
        
        # 1.2 for the history part, random mask out some history to future attention
        # given that this part is already causal,
        if hist_len > 0:
            prob = self.config.history_mask_prob  # chance to mask out
            keep_mask = torch.rand((seq_len - hist_len, hist_len), device=device) > prob
            mask[:, :, hist_len:, prefix_len:prefix_len+hist_len] &= keep_mask
        
        # 2. Handle Vision Prefix Visibility
        # History (r_hist) -> Prefix: BLOCKED (False) -> Default is False, so do nothing.
        # Future (r_future) -> Prefix: VISIBLE (True)
        if prefix_len > 0:
            mask[:, :, r_future, c_prefix] = True
            
        return mask
    



class ARTEncoder(nn.Module):
    def __init__(self, config: ARTConfig, is_vae_encoder: bool = False):
        super().__init__()
        # Reusing ARTDecoderLayer logic but for Encoding
        num_layers = config.n_vae_encoder_layers if is_vae_encoder else config.n_encoder_layers
        self.layers = nn.ModuleList([
            ARTDecoderLayer(config, layer_idx=i) for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

    def forward(self, x: Tensor, pos_emb: Tensor | None = None, past_key_value: transformers.Cache | None = None) -> Tensor:
        for layer in self.layers:
            # Encoder call: No RoPE indices, No Mask (Full Attention)
            x = layer(
                x,
                pos_emb=pos_emb,
                attention_mask=None,
                cache=past_key_value,
                attn_kwargs=None # No RoPE indices -> No Rotation applied
            )
        x = self.norm(x)
        return x


class ARTDecoder(nn.Module):
    def __init__(self, config: ARTConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            ARTDecoderLayer(config, layer_idx=i) for i in range(config.n_decoder_layers)
        ])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(self, x, attn_mask=None, cache=None, attn_kwargs=None):
        for layer in self.layers:
            x = layer(x, attention_mask=attn_mask, cache=cache, attn_kwargs=attn_kwargs)
        x = self.norm(x)
        return x


class ARTDecoderLayer(nn.Module):
    def __init__(self, config: ARTConfig, layer_idx: int):
        super().__init__()
        # Shared Attention Class
        self.self_attn = ARTMaskAttention(config, layer_idx=layer_idx)
        
        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = get_activation_fn(config.feedforward_activation)

    def forward(
        self,
        x: torch.Tensor,
        pos_emb: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cache: Optional[transformers.Cache] = None,
        attn_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        
        residual = x
        x = self.norm1(x)
        
        # Shared Attention Logic
        x = self.self_attn(
            x, 
            pos_emb=pos_emb,
            attn_mask=attention_mask, 
            cache=cache, 
            attn_kwargs=attn_kwargs
        )
        x = residual + self.dropout(x)

        residual = x
        x = self.norm2(x)
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = residual + self.dropout(x)
        
        return x


class ARTMaskAttention(nn.Module):
    """
    Unified Attention Block.
    - If attn_kwargs is None (Encoder): Standard Self-Attention, No RoPE.
    - If attn_kwargs is Set (Decoder): RoPE + Cache indexing.
    - Expects BATCH FIRST inputs (B, L, D).
    """
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.dim_model
        self.num_heads = config.n_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.layer_idx = layer_idx

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        self.rotary_emb = RotaryPositionalEncoding(self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        pos_emb: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        cache: Optional[transformers.Cache] = None,
        attn_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        
        # 1. Input is [B, L, D] (Batch First)
        bsz, q_len, _ = hidden_states.size()

        # 2. Project
        if pos_emb is not None:
            query_states = self.q_proj(hidden_states+pos_emb)
            key_states = self.k_proj(hidden_states+pos_emb)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # 3. Reshape for SDPA: [B, H, L, D]
        # view: [B, L, H, D] -> transpose: [B, H, L, D]
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 4. RoPE (Only if kwargs provided)
        if attn_kwargs is not None:
            q_pos_ids = attn_kwargs.get('query_position_indices')
            k_pos_ids = attn_kwargs.get('key_position_indices')
            
            if q_pos_ids is not None:
                cos_q, sin_q = self.rotary_emb(q_pos_ids, device=query_states.device, dtype=query_states.dtype)
                query_states = apply_rotary_pos_emb(query_states, cos_q, sin_q)
            
            if k_pos_ids is not None:
                if q_pos_ids is not None and torch.equal(q_pos_ids, k_pos_ids):
                    key_states = apply_rotary_pos_emb(key_states, cos_q, sin_q)
                else:
                    cos_k, sin_k = self.rotary_emb(k_pos_ids, device=key_states.device, dtype=key_states.dtype)
                    key_states = apply_rotary_pos_emb(key_states, cos_k, sin_k)

        # 5. Cache
        if cache is not None:
            cache_args = attn_kwargs.get('cache_kwargs', {}) if attn_kwargs else {}
            # Update cache and get FULL sequence
            key_states, value_states = cache.update(
                key_states, value_states, self.layer_idx, cache_kwargs=cache_args
            )

        # 6. Attention
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states, 
            attn_mask=attn_mask, 
            dropout_p=self.config.dropout if self.training else 0.0
        )

        # 7. Reshape Back: [B, L, D]
        # transpose: [B, L, H, D] -> view: [B, L, H*D]
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        
        return self.out_proj(attn_output)


class RotaryPositionalEncoding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, position_ids, device=None, dtype=None):
        if device is None: device = self.inv_freq.device
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        with torch.autocast(device_type=device.type if device.type != 'mps' else 'cpu', enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        if dtype is not None: return cos.to(dtype=dtype), sin.to(dtype=dtype)
        return cos, sin

def apply_rotary_pos_emb(x, cos, sin):
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    if cos.ndim == 3: cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return (x * cos) + (rotate_half(x) * sin)

def get_activation_fn(activation: str) -> Callable:
    if activation == "relu": return F.relu
    if activation == "gelu": return F.gelu
    if activation == "glu": return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")

# Keep existing embedding utils
class ACTSinusoidalPositionEmbedding2d(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        self._temperature = 10000

    def forward(self, x: Tensor) -> Tensor:
        not_mask = torch.ones_like(x[0, :1])
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)
        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi
        inverse_frequency = self._temperature ** (2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension)
        x_range = x_range.unsqueeze(-1) / inverse_frequency
        y_range = y_range.unsqueeze(-1) / inverse_frequency
        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)
        return pos_embed
    
class FastMixer(nn.Module):
    def __init__(self, config: ARTConfig, dropout=0.0):
        super().__init__()
        in_seq = config.image_tokens + 1
        out_seq = config.chunk_size
        in_dim = config.dim_model
        out_dim = config.action_feature.shape[0]
        
        # 1. Time Mixing Block (Compression: 300 -> 20)
        # We use a Sequential block to bundle the Linear + Activation + Norm
        self.time_mixer = nn.Sequential(
            nn.Linear(in_seq, out_seq),
            nn.GELU(),             # <--- Vital Non-linearity
            nn.LayerNorm(out_seq), # <--- Stabilizes the new time axis
            nn.Dropout(dropout)    # <--- Prevents overfitting
        )
        
        # 2. Channel Mixing Block (Compression: 512 -> 7)
        self.channel_mixer = nn.Sequential(
            nn.Linear(in_dim, out_dim)
            # We usually DO NOT put an activation after the final layer 
            # if these are your final action logits.
        )

    def forward(self, x):
        # Input x: [Batch, 300, 512]
    
        
        # --- Step 1: Mix Time ---
        # We need to apply Linear to the dimension of size 300.
        # PyTorch Linear applies to the LAST dimension.
        # So we swap (Batch, Seq, Dim) -> (Batch, Dim, Seq)
        x = x.permute(0, 2, 1)      # [B, 512, 300]
        
        
        x = self.time_mixer(x)      # [B, 512, 300] -> [B, 512, 20]
        
        # --- Step 2: Mix Channels ---
        # Now we need to apply Linear to the dimension of size 512.
        # We swap back: (Batch, Dim, Seq) -> (Batch, Seq, Dim)
        x = x.permute(0, 2, 1)      # [B, 20, 512]
        
        x = self.channel_mixer(x)   # [B, 20, 512] -> [B, 20, 7]
        
        return x
    
    
class ACTDecoder(nn.Module):
    def __init__(self, config: ARTConfig):
        """Convenience module for running multiple decoder layers followed by normalization."""
        super().__init__()
        self.layers = nn.ModuleList([ACTDecoderLayer(config) for _ in range(config.n_fast_decoder_layers)])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(
                x, encoder_out, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed
            )
        if self.norm is not None:
            x = self.norm(x)
        return x


class ACTDecoderLayer(nn.Module):
    def __init__(self, config: ARTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.multihead_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        # Feed forward layers.
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def maybe_add_pos_embed(self, tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x: (Decoder Sequence, Batch, Channel) tensor of input tokens.
            encoder_out: (Encoder Sequence, B, C) output features from the last layer of the encoder we are
                cross-attending with.
            decoder_pos_embed: (ES, 1, C) positional embedding for keys (from the encoder).
            encoder_pos_embed: (DS, 1, C) Positional_embedding for the queries (from the decoder).
        Returns:
            (DS, B, C) tensor of decoder output features.
        """
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]  # select just the output, not the attention weights
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.multihead_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]  # select just the output, not the attention weights
        x = skip + self.dropout2(x)
        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)
        return x


import torch
import torch.nn as nn
from typing import Tuple, Optional, Union

class SpatialActionTokenizer(nn.Module):
    """
    N-Dimensional Torch-native tokenizer that discretizes continuous actions into a grid-based vocabulary.
    
    Features:
    - Dimension Agnostic: Works for 2D, 3D, or N-D actions automatically.
    - Vectorized Encoding: Uses stride arithmetic for fast tokenization.
    - Buffer Management: Automatically moves with model to GPU/CPU.
    """
    
    def __init__(
        self,
        num_bins: int = 32,
        action_min: Union[list, tuple] = (12.0, 25.0),
        action_max: Union[list, tuple] = (511.0, 511.0),
    ):
        super().__init__()
        self.num_bins = num_bins
        
        # Convert inputs to tensors
        _min = torch.tensor(action_min, dtype=torch.float32)
        _max = torch.tensor(action_max, dtype=torch.float32)
        
        assert _min.shape == _max.shape, "Action min and max must have same dimension"
        self.action_dim = len(_min)
        self.vocab_size = num_bins ** self.action_dim
        
        # Register bounds as buffers
        self.register_buffer('action_min', _min)
        self.register_buffer('action_max', _max)
        self.register_buffer('action_range', _max - _min)
        
        # Pre-compute decoding table (Vocab Size, Action Dim)
        # This creates a lookup table where index i -> [x, y, ...] continuous values
        self._create_vocab_embedding()
        
        # Pre-compute encoding basis strides for flattening N-D coordinates to 1D tokens
        # Example for 2D (size 100): basis is [100, 1]. dot([y, x], basis) -> token_id
        # We use powers of num_bins: [num_bins^(D-1), ..., num_bins^1, num_bins^0]
        powers = [num_bins ** i for i in reversed(range(self.action_dim))]
        self.register_buffer('stride_basis', torch.tensor(powers, dtype=torch.long))

    def _create_vocab_embedding(self):
        """
        Creates a (Vocab_Size, Action_Dim) tensor containing the center value
        of every bin. This acts like a fixed Embedding layer.
        """
        # 1. Create linspace for each dimension
        grids = []
        for i in range(self.action_dim):
            # Centers are: min + step/2 + k*step
            # Or simply linspace over the range
            dim_centers = torch.linspace(
                self.action_min[i], 
                self.action_max[i], 
                self.num_bins
            )
            grids.append(dim_centers)
            
        # 2. Create meshgrid (N-Dimensional)
        # indexing='ij' ensures correct order for flattening
        mesh = torch.meshgrid(*grids, indexing='ij')
        
        # 3. Stack and flatten to (Vocab_Size, Action_Dim)
        # Stack dim -1 puts the coordinates in the last dimension
        vocab = torch.stack(mesh, dim=-1).reshape(-1, self.action_dim)
        
        self.register_buffer('vocab_centers', vocab)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """Alias for encode."""
        return self.encode(actions)

    def encode(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Continuous (..., D) -> Token IDs (...,)
        """
        # 1. Normalize to [0, 1]
        # Clamp inputs to ensure they stay within bounds
        clamped = torch.clamp(actions, min=self.action_min, max=self.action_max)
        norm = (clamped - self.action_min) / self.action_range
        
        # 2. Scale to [0, num_bins - 1] and round
        # We subtract a tiny epsilon to prevent checking edge case at exactly 1.0
        bin_coords = (norm * (self.num_bins - 1)).round().long()
        
        # 3. Flatten N-D coords to 1D token ID using dot product with strides
        # Input: (..., D), Basis: (D) -> Output: (...,)
        # We simply sum(coords * strides) along the last dimension
        tokens = (bin_coords * self.stride_basis).sum(dim=-1)
        
        return tokens

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Token IDs (...,) -> Continuous (..., D)
        """
        # Use standard PyTorch embedding lookup semantics
        # F.embedding or direct indexing works. Direct indexing is simpler for fixed buffers.
        return self.vocab_centers[tokens]

    def extra_repr(self):
        return (f"dim={self.action_dim}, bins={self.num_bins}, "
                f"vocab={self.vocab_size}")


class KMeansTokenizer(nn.Module):
    def __init__(self, centers_path="kmeans_centers.pt"):
        super().__init__()
        
        # Load the pre-calculated centroids
        # Shape: (Vocab_Size, 2)
        centers = torch.load(centers_path)
        
        self.vocab_size = centers.shape[0]
        self.action_dim = centers.shape[1] # Should be 2
        
        # Register as a buffer so it moves to GPU automatically with the model
        # but is not updated during backprop (frozen codebook)
        self.register_buffer('vocab_centers', centers)

    def encode(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Finds the nearest centroid for each action.
        Input: (Batch, ..., 2)
        Output: (Batch, ...)
        """
        # 1. Flatten input to (N, 2) for cdist
        input_shape = actions.shape
        flat_actions = actions.view(-1, self.action_dim)
        
        # 2. Calculate distances
        # torch.cdist computes euclidean distance between every row in A and every row in B
        # Input: (N, 2), Vocab: (1024, 2) -> Output: (N, 1024)
        # Note: cdist is heavily optimized (uses matrix multiplication under the hood)
        dists = torch.cdist(flat_actions, self.vocab_centers)
        
        # 3. Find closest centroid (Argmin)
        tokens = torch.argmin(dists, dim=1)
        
        # 4. Reshape back to original batch dimensions
        return tokens.view(input_shape[:-1])

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Look up the centroid coordinates.
        Input: (Batch, ...)
        Output: (Batch, ..., 2)
        """
        # Simple embedding lookup
        return self.vocab_centers[tokens]

    def forward(self, actions):
        return self.encode(actions)

    def extra_repr(self):
        return f"vocab_size={self.vocab_size}, dim={self.action_dim}"

def spatial_action_loss(logits: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy loss for spatial action tokens.
    
    Args:
        logits: (batch_size, seq_len, vocab_size) 
        target_tokens: (batch_size, seq_len) with token IDs
    
    Returns:
        loss: scalar loss value
    """
    # Reshape for cross entropy: (batch_size * seq_len, vocab_size) and (batch_size * seq_len,)
    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = target_tokens.view(-1)
    
    return nn.functional.cross_entropy(logits_flat, targets_flat)

class PerDimKMeansTokenizer(nn.Module):
    """
    Performs 1D quantization per dimension using pre-computed centroids.
    
    Expected pt file shape: (Action_Dim, Num_Bins)
    Meaning: centers[d, k] is the value of the k-th bin center for dimension d.
    """
    def __init__(self, centers_path: str):
        super().__init__()
        
        # Load centroids: Expecting shape [Action_Dim, Num_Bins]
        # Example: 14 dims, 100 bins -> [14, 100]
        centers = torch.load(centers_path)
        
        if centers.ndim != 2:
            raise ValueError(f"Expected centroids shape (Dim, Bins), got {centers.shape}")
            
        self.action_dim = centers.shape[0]
        self.num_bins = centers.shape[1]
        self.vocab_size = self.num_bins * self.action_dim
        
        # Register as buffer so it moves to GPU with model
        self.register_buffer('centers', centers)
        self.register_buffer('dim_offsets', torch.arange(self.action_dim) * self.num_bins)

    def encode(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Input:  [Batch, Seq, Dim] (Continuous values)
        Output: [Batch, Seq, Dim] (Token IDs 0..Num_Bins-1)
        """
        # 1. Expand actions to broadcast against bins
        # actions: [B, S, D] -> [B, S, D, 1]
        actions_expanded = actions.unsqueeze(-1)
        
        # 2. Expand centers to broadcast against batch/seq
        # centers: [D, K] -> [1, 1, D, K]
        centers_expanded = self.centers.unsqueeze(0).unsqueeze(0)
        
        # 3. Calculate absolute distance for every bin in every dimension
        # dists: [B, S, D, K]
        dists = torch.abs(actions_expanded - centers_expanded)
        
        # 4. Argmin to find closest bin index
        tokens = torch.argmin(dists, dim=-1) # [B, S, D]
        
        return tokens

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Input:  [Batch, Dim] (Token IDs)
        Output: [Batch, Dim] (Continuous values)
        """
        # We need to gather values from self.centers [D, K]
        # tokens is [B, S, D] containing indices k.
        
        # 1. Expand centers to match batch size for gather
        # centers: [D, K] -> [1, 1, D, K] -> expand to [B, S, D, K]
        # This can be memory intensive, so let's do it smarter:
        
        # Method: Gather acts on the last dimension.
        # We assume tokens are indices into the last dim of centers.
        
        # tokens: [B, S, D]
        batch, dim = tokens.shape
        
        # We want to map tokens[b,s,d] -> centers[d, tokens[b,s,d]]
        # Simplest way in PyTorch without massive broadcasting:
        
        flat_tokens = tokens.view(-1, dim) # [N, D]
        
        # Result placeholder
        # decoded = torch.zeros_like(flat_tokens, dtype=self.centers.dtype)
        
        # Loop over dimensions (since D is usually small, e.g., 14, this is fast enough)
        # and significantly saves memory compared to expanding centers to [B,S,D,K]
        
        
        flat_indices = flat_tokens + self.dim_offsets.view(1, -1)
        decoded = self.centers.view(-1)[flat_indices]
             
        return decoded.view(batch, dim)

    def forward(self, actions):
        return self.encode(actions)