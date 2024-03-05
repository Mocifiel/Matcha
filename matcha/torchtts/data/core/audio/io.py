import logging
import wave
from pathlib import Path
from typing import Union

import librosa
import soundfile as sf
import numpy as np

logger = logging.getLogger(__name__)


def load_wav(filename, sample_rate=None, offset=0.0, duration=None, res_type="kaiser_best", dtype=np.float32):
    if isinstance(filename, Path):
        filename = str(filename)

    # To preserve the native sampling rate of the file, we need to set sr to
    # None here. We also set mono to False here to bypass the bug of librosa
    # that we cannot set dtype to other types except for float.
    audio_segment, sr = librosa.load(filename, sr=None, mono=False, offset=offset, duration=duration, dtype=dtype)

    # Resample doesn't work for types other than float.
    if sample_rate is not None and sr != sample_rate:
        if dtype != np.float32:
            raise ValueError(f"Resample from {sr} to {sample_rate} only supports fp32 audio")
        else:
            audio_segment = librosa.resample(audio_segment, orig_sr=sr, target_sr=sample_rate, res_type=res_type)

    return audio_segment, sr


def save_wav(filename, data, sample_rate):
    if isinstance(filename, Path):
        filename = str(filename)
    sf.write(file=filename, data=data, samplerate=sample_rate)


def get_audio_length(filename: Union[str, Path]):
    # wave module only support str filename
    if isinstance(filename, Path):
        filename = str(filename)
    with wave.open(filename) as f:
        return f.getnframes() / f.getframerate()
