import torch
from torchaudio import transforms


class Spectrogram(torch.nn.Module):
    """On-the-fly spectrogram extractor."""

    def __init__(
        self,
        input_freq=24000,
        resample_freq=24000,
        n_fft=2048,
        win_length=1200,
        hop_length=300,
        power=1,
    ):
        super().__init__()
        self.resample = transforms.Resample(orig_freq=input_freq, new_freq=resample_freq)

        self.spec = transforms.Spectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length, power=power)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # Resample the input
        resampled = self.resample(waveform)

        # Convert to mel spectrogram
        spec = self.spec(resampled)

        return spec
