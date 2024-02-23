import abc
import logging
import numpy as np
import io
import os
import soundfile as sf
import librosa

from torchtts.data.core.features import tensor_feature

logger = logging.getLogger(__name__)


class _AudioDecoder(abc.ABC):
    """Utils which encode/decode audios."""

    def __init__(self, dtype, shape, sample_rate):
        self._dtype = dtype
        self._shape = shape
        self._sample_rate = sample_rate
        self._channels = shape[1] if len(shape) > 1 else 1

    @abc.abstractmethod
    def encode_audio(self, fobj):
        raise NotImplementedError

    @abc.abstractmethod
    def decode_audio(self, audio):
        raise NotImplementedError


def _load_audio(fobj, dtype=np.float32, mono=True, sample_rate=None):
    audio, sr = sf.read(fobj, dtype=dtype)
    audio = audio.T
    if mono:
        audio = librosa.to_mono(audio)
    if sample_rate is not None and sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate, res_type="soxr_hq")
    return audio


class _LazyDecoder(_AudioDecoder):
    """Read audio during decoding."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode_audio(self, fobj):
        return np.array(fobj.read(), dtype=np.object_)

    def decode_audio(self, audio):
        fobj = io.BytesIO(audio.item())
        return _load_audio(fobj, dtype=self._dtype, mono=self._channels == 1,
                           sample_rate=self._sample_rate)


class _EagerDecoder(_AudioDecoder):
    """Read audio during decoding."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode_audio(self, fobj):
        return _load_audio(fobj, dtype=self._dtype, mono=self._channels == 1,
                           sample_rate=self._sample_rate)

    def decode_audio(self, audio):
        return audio


class Audio(tensor_feature.Tensor):
    """`FeatureConnector` for audio."""

    def __init__(self, *, shape=(None,), dtype=np.float32, sample_rate=None, lazy_decode=False):
        if lazy_decode:
            serialized_dtype = np.object_
            serialized_shape = ()
            decoder_cls = _LazyDecoder
        else:
            serialized_dtype = None
            serialized_shape = None
            decoder_cls = _EagerDecoder

        super().__init__(shape=shape, dtype=dtype, serialized_dtype=serialized_dtype,
                         serialized_shape=serialized_shape)

        self._audio_decoder = decoder_cls(dtype=dtype, shape=shape, sample_rate=sample_rate)

    def encode_example(self, audio_or_path_or_fobj):
        if isinstance(audio_or_path_or_fobj, (np.ndarray, list)):
            audio = audio_or_path_or_fobj
        elif isinstance(audio_or_path_or_fobj, (str, os.PathLike)):
            filename = os.fspath(audio_or_path_or_fobj)
            with open(filename, 'rb') as audio_f:
                audio = self._audio_decoder.encode_audio(audio_f)
        else:
            audio = self._audio_decoder.encode_audio(audio_or_path_or_fobj)

        return super().encode_example(audio)

    def decode_example(self, audio_data):
        audio = super().decode_example(audio_data)
        return self._audio_decoder.decode_audio(audio)
