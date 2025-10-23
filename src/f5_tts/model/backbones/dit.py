"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from x_transformers.x_transformers import RotaryEmbedding

from f5_tts.model.modules import (
    AdaLayerNorm_Final,
    ConvNeXtV2Block,
    ConvPositionEmbedding,
    DiTBlock,
    TimestepEmbedding,
    get_pos_embed_indices,
    precompute_freqs_cis,
)


# Text embedding


class TextEmbedding(nn.Module):
    def __init__(
        self, text_num_embeds, text_dim, mask_padding=True, average_upsampling=False, conv_layers=0, conv_mult=2, use_mas=False
    ):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)  # use 0 as filler token

        self.mask_padding = mask_padding  # mask filler and batch padding tokens or not
        self.average_upsampling = average_upsampling  # zipvoice-style text late average upsampling (after text encoder)
        if average_upsampling:
            assert mask_padding, "text_embedding_average_upsampling requires text_mask_padding to be True"

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096  # ~44s of 24khz audio
            self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
            self.text_blocks = nn.Sequential(*[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)])
        else:
            self.extra_modeling = False

        # MAS (Monotonic Alignment Search) components for gradual mixing
        self.use_mas = use_mas
        if use_mas:
            # Similarity projection for text-mel alignment (identity initialized for smooth adaptation)
            self.similarity_proj = nn.Linear(text_dim, text_dim)
            nn.init.eye_(self.similarity_proj.weight)
            nn.init.zeros_(self.similarity_proj.bias)

            # Gradual mixing parameters (start with pure V0/V1 behavior)
            self.register_buffer("mas_alpha", torch.tensor(0.0))  # 0=V0/V1 only, 1=MAS only
            self.register_buffer("mas_temperature", torch.tensor(10.0))  # High=soft, Low=sharp attention
        else:
            self.similarity_proj = None
            self.mas_alpha = None
            self.mas_temperature = None

    def average_upsample_text_by_mask(self, text, text_mask, audio_mask):
        batch, text_len, text_dim = text.shape

        if audio_mask is None:
            audio_mask = torch.ones_like(text_mask, dtype=torch.bool)
        valid_mask = audio_mask & text_mask
        audio_lens = audio_mask.sum(dim=1)  # [batch]
        valid_lens = valid_mask.sum(dim=1)  # [batch]

        upsampled_text = torch.zeros_like(text)

        for i in range(batch):
            audio_len = audio_lens[i].item()
            valid_len = valid_lens[i].item()

            if valid_len == 0:
                continue

            valid_ind = torch.where(valid_mask[i])[0]
            valid_data = text[i, valid_ind, :]  # [valid_len, text_dim]

            base_repeat = audio_len // valid_len
            remainder = audio_len % valid_len

            indices = []
            for j in range(valid_len):
                repeat_count = base_repeat + (1 if j >= valid_len - remainder else 0)
                indices.extend([j] * repeat_count)

            indices = torch.tensor(indices[:audio_len], device=text.device, dtype=torch.long)
            upsampled = valid_data[indices]  # [audio_len, text_dim]

            upsampled_text[i, :audio_len, :] = upsampled

        return upsampled_text

    def forward(
        self,
        text: int["b nt"],  # noqa: F722
        seq_len,
        drop_text=False,
        audio_mask: bool["b n"] | None = None,  # noqa: F722
        mel_features: torch.Tensor | None = None,  # [b, n, d] - for MAS similarity
        text_lens: torch.Tensor | None = None,  # [b] - actual text lengths
        mel_lens: torch.Tensor | None = None,  # [b] - actual mel lengths
        duration_pred: torch.Tensor | None = None,  # [b, nt] - predicted durations for inference
    ):
        """
        Forward pass with optional MAS support for gradual mixing or duration-based upsampling.

        Args:
            text: [b, nt] - text token indices
            seq_len: target sequence length (mel frames)
            drop_text: whether to drop text for CFG
            audio_mask: [b, n] - audio mask
            mel_features: [b, n, d] - mel features for MAS similarity (if use_mas=True during training)
            text_lens: [b] - actual text lengths (if use_mas=True during training)
            mel_lens: [b] - actual mel lengths (if use_mas=True during training)
            duration_pred: [b, nt] - predicted durations for each text token (inference only)

        Returns:
            text_embed: [b, n, d] - text embeddings aligned to mel frames
            attn: [b, nt, n] - attention matrix from MAS (None if MAS not used or duration_pred provided)
            text_embed_raw: [b, nt, d] - raw text embeddings before upsampling
        """
        text = text + 1  # use 0 as filler token. preprocess of batch pad -1, see list_str_to_idx()
        text = text[:, :seq_len]  # curtail if character tokens are more than the mel spec tokens
        batch, text_len = text.shape[0], text.shape[1]

        # Save raw text tokens for MAS (before padding)
        text_tokens_valid = text[:, :text_len].clone()  # [b, text_len]

        text = F.pad(text, (0, seq_len - text_len), value=0)  # (opt.) if not self.average_upsampling:
        if self.mask_padding:
            text_mask = text == 0

        if drop_text:  # cfg for text
            text = torch.zeros_like(text)

        # Get text embeddings
        text_embed = self.text_embed(text)  # b n -> b n d

        # Store raw embeddings (before extra modeling) for MAS
        text_embed_raw = self.text_embed(text_tokens_valid)  # [b, text_len, d]

        # ===== V0/V1 Path: Existing behavior (always computed) =====
        text_v0_v1 = text_embed.clone()  # Start with padded embeddings

        # possible extra modeling
        if self.extra_modeling:
            # sinus pos emb
            batch_start = torch.zeros((batch,), device=text_v0_v1.device, dtype=torch.long)
            pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
            text_pos_embed = self.freqs_cis[pos_idx]
            text_v0_v1 = text_v0_v1 + text_pos_embed

            # convnextv2 blocks
            if self.mask_padding:
                text_v0_v1 = text_v0_v1.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text_v0_v1.size(-1)), 0.0)
                for block in self.text_blocks:
                    text_v0_v1 = block(text_v0_v1)
                    text_v0_v1 = text_v0_v1.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text_v0_v1.size(-1)), 0.0)
            else:
                text_v0_v1 = self.text_blocks(text_v0_v1)

        if self.average_upsampling:
            text_v0_v1 = self.average_upsample_text_by_mask(text_v0_v1, ~text_mask, audio_mask)

        # ===== MAS Path: New alignment-based behavior =====
        attn = None
        text_mas = None

        if self.use_mas:
            # Compute similarity between text and mel features
            text_proj = self.similarity_proj(text_embed_raw)  # [b, text_len, d]
            similarity = torch.bmm(text_proj, mel_features.transpose(1, 2))  # [b, text_len, n]

            assert self.mas_temperature is not None
            # Apply temperature for soft/hard attention
            similarity = similarity / self.mas_temperature

            # Run MAS to get optimal alignment
            from f5_tts.model.mas import monotonic_alignment_search

            attn = monotonic_alignment_search(similarity, text_lens, mel_lens)  # [b, text_len, n]

            # Apply alignment to distribute text embeddings across mel frames
            text_mas = torch.bmm(attn.transpose(1, 2), text_embed_raw)  # [b, n, d]

        # ===== Duration Prediction Path: For inference with learned durations =====
        if duration_pred is not None:
            # Use predicted durations to construct alignment matrix (inference mode)
            # duration_pred: [b, nt] - number of mel frames for each text token
            from f5_tts.model.mas import generate_path

            # Create masks for valid regions
            text_mask = torch.arange(text_len, device=text_embed_raw.device)[None, :, None] < text_lens[:, None, None]
            mel_mask = (
                torch.arange(seq_len, device=text_embed_raw.device)[None, None, :]
                < torch.tensor([seq_len] * batch, device=text_embed_raw.device)[:, None, None]
            )
            attn_mask = text_mask & mel_mask  # [b, nt, n]

            # Generate alignment matrix from predicted durations
            # This creates the same type of alignment used during training!
            attn = generate_path(duration_pred, attn_mask.float())  # [b, nt, n]

            # Apply alignment to upsample text embeddings (SAME AS TRAINING!)
            text_final = torch.bmm(attn.transpose(1, 2), text_embed_raw)  # [b, n, d]

        # ===== Gradual Mixing: Blend V0/V1 and MAS paths =====
        elif text_mas is not None:
            assert self.mas_alpha is not None
            # Mix between V0/V1 (alpha=0) and MAS (alpha=1)
            text_final = (1 - self.mas_alpha) * text_v0_v1 + self.mas_alpha * text_mas
        else:
            # Pure V0/V1 path
            text_final = text_v0_v1

        return text_final, attn, text_embed_raw


# noised input audio and context mixing embedding


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, x: float["b n d"], cond: float["b n d"], text_embed: float["b n d"], drop_audio_cond=False):  # noqa: F722
        if drop_audio_cond:  # cfg for cond audio
            cond = torch.zeros_like(cond)

        x = self.proj(torch.cat((x, cond, text_embed), dim=-1))
        x = self.conv_pos_embed(x) + x
        return x


# Transformer backbone using DiT blocks


class DiT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100,
        text_num_embeds=256,
        text_dim=None,
        text_mask_padding=True,
        text_embedding_average_upsampling=False,
        qk_norm=None,
        conv_layers=0,
        pe_attn_head=None,
        attn_backend="torch",  # "torch" | "flash_attn"
        attn_mask_enabled=False,
        long_skip_connection=False,
        checkpoint_activations=False,
        use_mas=False,  # Enable MAS for alignment
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        if text_dim is None:
            text_dim = mel_dim
        self.text_embed = TextEmbedding(
            text_num_embeds,
            text_dim,
            mask_padding=text_mask_padding,
            average_upsampling=text_embedding_average_upsampling,
            conv_layers=conv_layers,
            use_mas=use_mas,
        )
        # Text and attention cache for CFG
        self.text_cond, self.text_uncond = None, None
        self.text_cond_attn, self.text_uncond_attn = None, None
        self.text_cond_raw, self.text_uncond_raw = None, None
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    qk_norm=qk_norm,
                    pe_attn_head=pe_attn_head,
                    attn_backend=attn_backend,
                    attn_mask_enabled=attn_mask_enabled,
                )
                for _ in range(depth)
            ]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        self.norm_out = AdaLayerNorm_Final(dim)  # final modulation
        self.proj_out = nn.Linear(dim, mel_dim)

        self.checkpoint_activations = checkpoint_activations

        self.initialize_weights()

    def initialize_weights(self):
        # Zero-out AdaLN layers in DiT blocks:
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def ckpt_wrapper(self, module):
        # https://github.com/chuanyangjin/fast-DiT/blob/main/models.py
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs

        return ckpt_forward

    def get_input_embed(
        self,
        x: torch.Tensor,  # [b, n, d]
        cond: torch.Tensor,  # [b, n, d]
        text: torch.Tensor,  # [b, nt]
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        cache: bool = True,
        audio_mask: torch.BoolTensor | None = None,  # [b, n]
        mel_features: torch.Tensor | None = None,  # [b, n, d] - for MAS
        text_lens: torch.Tensor | None = None,  # [b] - for MAS
        mel_lens: torch.Tensor | None = None,  # [b] - for MAS
        duration_pred: torch.Tensor | None = None,  # [b, nt] - predicted durations for inference only
    ):
        seq_len = x.shape[1]

        if cache:
            if drop_text:
                if self.text_uncond is None:
                    # Cache miss - compute and store
                    self.text_uncond, self.text_uncond_attn, self.text_uncond_raw = self.text_embed(
                        text,
                        seq_len,
                        drop_text=True,
                        audio_mask=audio_mask,
                        mel_features=mel_features,
                        text_lens=text_lens,
                        mel_lens=mel_lens,
                        duration_pred=duration_pred,
                    )
                # Cache hit - reuse cached values
                text_embed = self.text_uncond
                attn = self.text_uncond_attn
                text_embed_raw = self.text_uncond_raw
            else:
                if self.text_cond is None:
                    # Cache miss - compute and store
                    self.text_cond, self.text_cond_attn, self.text_cond_raw = self.text_embed(
                        text,
                        seq_len,
                        drop_text=False,
                        audio_mask=audio_mask,
                        mel_features=mel_features,
                        text_lens=text_lens,
                        mel_lens=mel_lens,
                        duration_pred=duration_pred,
                    )
                # Cache hit - reuse cached values
                text_embed = self.text_cond
                attn = self.text_cond_attn
                text_embed_raw = self.text_cond_raw
        else:
            # No cache - always compute fresh
            text_embed, attn, text_embed_raw = self.text_embed(
                text,
                seq_len,
                drop_text=drop_text,
                audio_mask=audio_mask,
                mel_features=mel_features,
                text_lens=text_lens,
                mel_lens=mel_lens,
                duration_pred=duration_pred,
            )

        x = self.input_embed(x, cond, text_embed, drop_audio_cond=drop_audio_cond)

        return x, attn, text_embed_raw  # Return raw embeddings for duration predictor

    def clear_cache(self):
        self.text_cond, self.text_uncond = None, None
        self.text_cond_attn, self.text_uncond_attn = None, None
        self.text_cond_raw, self.text_uncond_raw = None, None

    def forward(
        self,
        x: float["b n d"],  # nosied input audio  # noqa: F722
        cond: float["b n d"],  # masked cond audio  # noqa: F722
        text: int["b nt"],  # text  # noqa: F722
        time: float["b"] | float[""],  # time step  # noqa: F821 F722
        mask: bool["b n"] | None = None,  # noqa: F722
        drop_audio_cond: bool = False,  # cfg for cond audio
        drop_text: bool = False,  # cfg for text
        cfg_infer: bool = False,  # cfg inference, pack cond & uncond forward
        cache: bool = False,
        mel_features: torch.Tensor | None = None,  # [b, n, d] - for MAS
        text_lens: torch.Tensor | None = None,  # [b] - for MAS
        mel_lens: torch.Tensor | None = None,  # [b] - for MAS
        duration_pred: torch.Tensor | None = None,  # [b, nt] - predicted durations for inference
        returns_text_embed: bool = False,  # whether to return text embeddings and attention
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        # t: conditioning time, text: text, x: noised audio + cond audio + text
        t = self.time_embed(time)
        attn = None
        text_embed_for_return = None  # Track text_embed for returns_text_embed

        if cfg_infer:  # pack cond & uncond forward: b n d -> 2b n d
            x_cond, attn, text_embed_cond = self.get_input_embed(
                x,
                cond,
                text,
                drop_audio_cond=False,
                drop_text=False,
                cache=cache,
                audio_mask=mask,
                mel_features=mel_features,
                text_lens=text_lens,
                mel_lens=mel_lens,
                duration_pred=duration_pred,
            )
            x_uncond, _, _ = self.get_input_embed(
                x,
                cond,
                text,
                drop_audio_cond=True,
                drop_text=True,
                cache=cache,
                audio_mask=mask,
                mel_features=mel_features,
                text_lens=text_lens,
                mel_lens=mel_lens,
                duration_pred=duration_pred,
            )
            text_embed_for_return = text_embed_cond  # Use conditional text embed
            x = torch.cat((x_cond, x_uncond), dim=0)
            t = torch.cat((t, t), dim=0)
            mask = torch.cat((mask, mask), dim=0) if mask is not None else None
        else:
            x, attn, text_embed_for_return = self.get_input_embed(
                x,
                cond,
                text,
                drop_audio_cond=drop_audio_cond,
                drop_text=drop_text,
                cache=cache,
                audio_mask=mask,
                mel_features=mel_features,
                text_lens=text_lens,
                mel_lens=mel_lens,
                duration_pred=duration_pred,
            )

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        for block in self.transformer_blocks:
            if self.checkpoint_activations:
                # https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, t, mask, rope, use_reentrant=False)
            else:
                x = block(x, t, mask=mask, rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x)

        if returns_text_embed:
            assert text_embed_for_return is not None
            assert attn is not None
            # For training with duration predictor - use the text_embed we computed
            return output, text_embed_for_return, attn

        return output
