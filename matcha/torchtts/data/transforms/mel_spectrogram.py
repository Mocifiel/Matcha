import torch
import torchaudio
from torchaudio import transforms
from packaging.version import parse as V


class SpectralNormalization(torch.nn.Module):
    def __init__(self, min_level_db, max_abs_value):
        super().__init__()
        self.min_level_db = min_level_db
        self.max_abs_value = max_abs_value

    def forward(self, input):
        # Conver to dB before normalize
        input = 20 * torch.log10(torch.clamp(input, min=1e-5))
        # Symmetric normalize
        input = (2 * self.max_abs_value) * ((input - self.min_level_db) / (-self.min_level_db)) - self.max_abs_value

        return input


class MelSpectrogram(torch.nn.Module):
    """On-the-fly mel spectrogram extractor."""

    def __init__(
        self,
        input_freq=24000,
        resample_freq=24000,
        n_fft=2048,
        win_length=1200,
        hop_length=300,
        f_min=0,
        f_max=None,
        n_mels=128,
        power=1,
        normalized=False,
        mel_scale="slaney",
        norm="slaney",
        min_level_db=-100,
        max_abs_value=1.0,
    ):
        super().__init__()
        self.resample = transforms.Resample(orig_freq=input_freq, new_freq=resample_freq)

        if V(torchaudio.__version__) >= V("0.9.0"):
            self.mel_spec = transforms.MelSpectrogram(
                sample_rate=resample_freq,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                f_min=f_min,
                f_max=f_max,
                n_mels=n_mels,
                mel_scale=mel_scale,
                normalized=normalized,
                power=power,
                norm=norm,
            )
        else:
            if mel_scale != "htk":
                raise ValueError(f"Only support htk for mel_scale in torchaudio {torchaudio.__version__}")

            self.mel_spec = transforms.MelSpectrogram(
                sample_rate=resample_freq,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                f_min=f_min,
                f_max=f_max,
                n_mels=n_mels,
                mel_scale=mel_scale,
                normalized=normalized,
                power=power,
                norm=norm,
            )

        self.norm = SpectralNormalization(min_level_db, max_abs_value)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # Resample the input
        resampled = self.resample(waveform)
        # Convert to mel spectrogram
        mel_spec = self.mel_spec(resampled)
        # Normalize to db
        mel_spec = self.norm(mel_spec)

        return mel_spec
