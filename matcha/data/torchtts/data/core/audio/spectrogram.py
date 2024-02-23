import librosa
import numpy as np
from scipy import signal


def stft(y, num_freq, frame_shift_ms, frame_length_ms, sample_rate):
    n_fft, hop_length, win_length = _stft_parameters(num_freq, frame_shift_ms, frame_length_ms, sample_rate)
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def mel_spectrogram(
    audio,
    pre_emphasis_coeff,
    num_freq,
    frame_shift_ms,
    frame_length_ms,
    sample_rate,
    num_mels,
    min_mel_freq,
    max_mel_freq,
    ref_level_db,
    symmetric_specs,
    max_abs_value,
    min_level_db,
    clip_norm=True,
    **kwargs
):
    audio = _pre_emphasize(audio, coeff=pre_emphasis_coeff)
    spec = stft(
        y=audio,
        num_freq=num_freq,
        frame_shift_ms=frame_shift_ms,
        frame_length_ms=frame_length_ms,
        sample_rate=sample_rate,
    )
    mel_spec = _linear_to_mel(
        spec=np.abs(spec),
        num_freq=num_freq,
        sample_rate=sample_rate,
        num_mels=num_mels,
        min_mel_freq=min_mel_freq,
        max_mel_freq=max_mel_freq,
    )
    db_mel_spec = _amp_to_db(mel_spec) - ref_level_db
    if clip_norm:
        db_mel_spec = _normalize(db_mel_spec, symmetric_specs, max_abs_value, min_level_db)
    return db_mel_spec.astype(np.float32).T


def _pre_emphasize(x, coeff):
    assert len(x.shape) == 1
    return signal.lfilter([1, -coeff], [1], x)


def _stft_parameters(num_freq, frame_shift_ms, frame_length_ms, sample_rate):
    n_fft = (num_freq - 1) * 2
    hop_length = int(frame_shift_ms / 1000 * sample_rate)
    win_length = int(frame_length_ms / 1000 * sample_rate)
    return n_fft, hop_length, win_length


def _amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))


def _normalize(spec, symmetric_specs, max_abs_value, min_level_db):
    if symmetric_specs:
        return np.clip(
            (2 * max_abs_value) * ((spec - min_level_db) / (-min_level_db)) - max_abs_value,
            -max_abs_value,
            max_abs_value,
        )
    else:
        return np.clip(max_abs_value * ((spec - min_level_db) / (-min_level_db)), 0, max_abs_value)


def _build_mel_basis(num_freq, sample_rate, num_mels, min_mel_freq, max_mel_freq):
    n_fft = (num_freq - 1) * 2
    return librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=num_mels, fmin=min_mel_freq, fmax=max_mel_freq)


def _linear_to_mel(spec, num_freq, sample_rate, num_mels, min_mel_freq, max_mel_freq):
    _mel_basis = _build_mel_basis(num_freq, sample_rate, num_mels, min_mel_freq, max_mel_freq)
    return np.dot(_mel_basis, spec)
