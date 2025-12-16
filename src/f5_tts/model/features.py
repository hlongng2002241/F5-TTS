from dataclasses import asdict, dataclass
from typing import Any, Dict

import numpy as np
import torch

from lhotse.features.base import FeatureExtractor, register_extractor
from lhotse.utils import Seconds, compute_num_frames


EPSILON = 1e-7


@dataclass
class VocosMelSpectrogramConfig:
    """Configuration for Vocos mel spectrogram feature extractor."""

    sample_rate: int = 24000
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 100
    padding: str = "center"  # "center" or "same"

    def __post_init__(self):
        if self.padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "VocosMelSpectrogramConfig":
        return VocosMelSpectrogramConfig(**data)


@register_extractor
class VocosMelSpectrogram(FeatureExtractor):
    """Vocos mel spectrogram feature extractor.

    This extractor uses magnitude spectrum (power=1) followed by log compression,
    and supports both 'center' and 'same' padding modes.
    """

    name = "vocos_mel_spec"
    config_type = VocosMelSpectrogramConfig

    def __init__(self, config: VocosMelSpectrogramConfig = None):
        super().__init__(config or VocosMelSpectrogramConfig())
        self.config = config or VocosMelSpectrogramConfig()

        # Initialize mel spectrogram transform
        self.mel_spec = None  # Will be lazily initialized on first call

    def _init_mel_spec(self):
        """Lazily initialize the mel spectrogram transform."""
        if self.mel_spec is None:
            import torchaudio.transforms

            self.mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.config.sample_rate,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                n_mels=self.config.n_mels,
                center=self.config.padding == "center",
                power=1,  # Magnitude spectrum
            )

    def extract(self, samples: np.ndarray, sampling_rate: int) -> np.ndarray:
        """Extract Vocos mel spectrogram features.

        Args:
            samples: Audio samples as numpy array
            sampling_rate: Sampling rate of the audio

        Returns:
            Log mel spectrogram features
        """
        assert sampling_rate == self.config.sample_rate, (
            f"Sampling rate mismatch: expected {self.config.sample_rate}, " f"got {sampling_rate}"
        )

        self._init_mel_spec()

        # Convert to torch tensor
        audio = torch.from_numpy(samples).float()
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        # Apply padding for "same" mode
        if self.config.padding == "same":
            pad = self.config.win_length - self.config.hop_length
            audio = torch.nn.functional.pad(audio, (pad // 2, pad // 2), mode="reflect")

        # Compute mel spectrogram
        mel = self.mel_spec(audio)
        mel = torch.log(torch.clamp(mel, min=EPSILON))

        # Convert back to numpy and transpose to (time, freq)
        mel = mel.squeeze(0).t()

        num_frames = compute_num_frames(audio.size(1) / self.config.sample_rate, self.frame_shift, self.config.sample_rate)
        if mel.shape[0] > num_frames:
            mel = mel[:num_frames]
        elif mel.shape[0] < num_frames:
            mel = mel.unsqueeze(0)
            mel = torch.nn.functional.pad(mel, (0, 0, 0, num_frames - mel.shape[1]), mode="replicate").squeeze(0)

        return mel.numpy()

    def feature_dim(self, sampling_rate: int) -> int:
        """Return the feature dimension."""
        return self.config.n_mels

    @property
    def frame_shift(self) -> Seconds:
        """Return the frame shift in seconds."""
        return self.config.hop_length / self.config.sample_rate

    @staticmethod
    def mix(features_a: np.ndarray, features_b: np.ndarray, energy_scaling_factor_b: float) -> np.ndarray:
        """Mix two feature matrices with energy scaling.

        Since features are in log domain, we convert to linear, mix, then back to log.
        """
        return np.log(
            np.maximum(
                EPSILON,
                np.exp(features_a) + energy_scaling_factor_b * np.exp(features_b),
            )
        )

    @staticmethod
    def compute_energy(features: np.ndarray) -> float:
        """Compute total energy from log-domain features."""
        return float(np.sum(np.exp(features)))
