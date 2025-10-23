"""
Duration Predictor for predicting mel frame durations from text embeddings.

This module predicts log-duration (number of mel frames) for each text token
based on learned features. It's trained using ground-truth durations extracted
from MAS alignments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from f5_tts.model.modules import LayerNorm


class DurationPredictor(nn.Module):
    """
    Predicts log-duration for each text token.

    Architecture:
        text_embed → Conv1d → ReLU → LayerNorm → Dropout →
        Conv1d → ReLU → LayerNorm → Dropout → Conv1d(out=1) → log_duration
    """

    def __init__(
        self,
        in_channels=512,      # text embedding dimension
        filter_channels=256,  # hidden dimension
        kernel_size=3,        # convolution kernel size
        p_dropout=0.5,        # dropout probability
    ):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        # Layer 1: in_channels → filter_channels
        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = LayerNorm(filter_channels)
        self.drop_1 = nn.Dropout(p_dropout)

        # Layer 2: filter_channels → filter_channels
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = LayerNorm(filter_channels)
        self.drop_2 = nn.Dropout(p_dropout)

        # Projection: filter_channels → 1 (log duration)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

    def forward(
        self,
        x: torch.Tensor,     # [b, nt, d] - text embeddings
        x_mask: torch.Tensor # [b, nt] - text mask (1 for valid, 0 for padding)
    ) -> torch.Tensor:
        """
        Predict log-duration for each text token.

        Args:
            x: [b, nt, d] - text embeddings from transformer
            x_mask: [b, nt] - binary mask (1 for valid tokens, 0 for padding)

        Returns:
            log_duration: [b, nt, 1] - predicted log-duration for each token
        """
        # Transpose to [b, d, nt] for Conv1d
        x = x.transpose(1, 2)  # [b, d, nt]
        x_mask = x_mask.unsqueeze(1)  # [b, 1, nt]

        # Layer 1
        x = self.conv_1(x * x_mask)
        x = F.relu(x)
        x = self.norm_1(x)
        x = self.drop_1(x)

        # Layer 2
        x = self.conv_2(x * x_mask)
        x = F.relu(x)
        x = self.norm_2(x)
        x = self.drop_2(x)

        # Project to log-duration
        x = self.proj(x * x_mask)

        # Transpose back to [b, nt, 1]
        x = x.transpose(1, 2)

        # Mask padding tokens
        x = x * x_mask.transpose(1, 2)

        return x