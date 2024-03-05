import csv
import json
import os
import re
import random
from itertools import chain
from more_itertools import split_before, windowed

import numpy as np
import logging
from pathlib import Path

from torchtts.data.core import features
from torchtts.data.core.datapipes import IterDataPipe
from torchtts.data.core.dataset_builder import GeneratorBasedBuilder
from torchtts.data.core.dataset_info import DatasetInfo
from torchtts.utils.data_utils import get_bucket_scheme, lowercase_dict_keys


logger = logging.getLogger(__name__)


class UnifiedTtsDataset(GeneratorBasedBuilder):
    LOCALE_PREFIX_WHITELIST = re.compile(r"^(<s>|</s>|-|sil|br[0-4]|/|symbol|t[1-6])$|^punc")

    def _info(self):
        sample_rate = self._config.get("sample_rate", 24000)
        lazy_decode = self._config.get("lazy_decode", False)

        feature_dict = {
            "phone_id": features.Tensor(shape=(None,), dtype=np.int64),
            "duration": features.Tensor(shape=(None,), dtype=np.int64),
            "speech": features.Audio(sample_rate=sample_rate, lazy_decode=lazy_decode),
            "mel": features.Tensor(shape=(None, 80), dtype=np.float32),
            "f0": features.Tensor(shape=(None,), dtype=np.float32),
            "uv": features.Tensor(shape=(None,), dtype=np.float32),
            "speaker_id": features.Tensor(shape=(), dtype=np.int64),
            "locale_id": features.Tensor(shape=(), dtype=np.int64),
            "style_id": features.Tensor(shape=(), dtype=np.int64),
        }

        if self._config.get('with_stat_data', False):
            feature_dict.update({
                'stat': features.Tensor(shape=(None, 6), dtype=np.float32),
                'syl_phone_num': features.Tensor(shape=(None,), dtype=np.int64),
            })

        if self._config.get('with_text_data', False):
            feature_dict.update({"text": features.Text()})

        if self._config.get('with_context_info', False):
            feature_dict.update({
                'has_left_context': features.Tensor(shape=(), dtype=bool),
                'has_right_context': features.Tensor(shape=(), dtype=bool),
            })

        return DatasetInfo(
            builder=self, description="FastSpeech dataset builder", features=features.FeaturesDict(feature_dict)
        )

    def _split_generators(self):
        path = self._config.get("raw_data", None)
        if path is None:
            raise ValueError("You should specify raw_data in dataset builder")
        return {k: self._raw_data_generator(split=v, path=Path(path)) for k, v in self.split_type.items()}

    def _raw_data_generator(self, split, path: Path):
        # Map to id for training
        with open(self._config["phone_set_path"], "r", encoding="utf-8") as f:
            phone_set = json.load(f)
        max_phone_id = max(phone_set.values())

        with open(self._config["speaker_set_path"], "r", encoding="utf-8") as f:
            speaker_set = json.load(f)
        with open(self._config["locale_set_path"], "r", encoding="utf-8") as f:
            locale_set = json.load(f)
        with open(self._config["style_set_path"], "r", encoding="utf-8") as f:
            style_set = json.load(f)
            lowercase_dict_keys(style_set)

        # Registry.csv works as centralized index for voice collections
        registry_file = path / "registry.csv"
        if not registry_file.exists():
            raise ValueError(f"Cannot find registry.csv in {path}")

        # Allow frame mismatch to be less than frame_tolerance
        frame_tolerance = self._config.get("frame_tolerance", 5)
        add_locale_prefix = self._config.get("add_locale_prefix", False)

        with open(registry_file, "r", newline="", encoding="utf-8") as reg_f:
            registry_reader = csv.DictReader(reg_f, delimiter="|")
            for voice in registry_reader:
                metadata_file = path / voice["metadata_path"]
                with open(metadata_file, "r", newline="", encoding="utf-8") as meta_f:
                    metadata_root = metadata_file.parent
                    metadata_reader = csv.DictReader(meta_f, delimiter="|")
                    for metadata in metadata_reader:
                        phone_id = []
                        for p in metadata["phones"].split():
                            if add_locale_prefix:
                                if self.LOCALE_PREFIX_WHITELIST.match(p):
                                    phone = p
                                else:
                                    phone = f"{metadata['locale']}_{p}"
                            else:
                                phone = p
                            # Automatically insert incremental phone id to phone set
                            p_id = phone_set.get(phone, None)
                            if p_id is None:
                                max_phone_id += 1
                                p_id = max_phone_id
                                phone_set[phone] = p_id
                                logger.warning(f"Missing phone {phone} in phone set, set that to {p_id} automatically")
                            phone_id.append(p_id)
                        duration = list(map(int, metadata["durations"].split()))

                        total_frames = sum(duration)

                        mel = np.load(metadata_root / metadata["mel_path"])
                        mel = self._handle_length_mismatch(mel, total_frames, frame_tolerance)
                        f0 = np.load(metadata_root / metadata["sf_path"])
                        f0 = self._handle_length_mismatch(f0, total_frames, frame_tolerance)
                        uv = np.load(metadata_root / metadata["uv_path"])
                        uv = self._handle_length_mismatch(uv, total_frames, frame_tolerance)
                        # The number of mismatch frames are larger than the tolerance.
                        if mel is None or f0 is None or uv is None:
                            continue

                        sid = metadata["sid"]
                        speaker = metadata["speaker"]
                        locale = metadata["locale"]
                        style = metadata["style"].lower()
                        speaker_id = metadata["speaker_id"] if "speaker_id" in metadata else speaker_set[speaker]

                        # Generate a unique sid
                        example_id = f"{locale}_{speaker}_{style}_{sid}"
                        ret = {
                            "phone_id": phone_id,
                            "duration": duration,
                            "speech": metadata_root / metadata["speech_path"],
                            "mel": mel,
                            "f0": f0,
                            "uv": uv,
                            "speaker_id": speaker_id,
                            "locale_id": locale_set[locale],
                            "style_id": style_set[style],
                        }

                        if self._config.get('with_stat_data', False):
                            stat = np.load(metadata_root / metadata['stat_path'])
                            ret['stat'] = stat

                            syl_phone_num = list(map(int, metadata["syl_phone_num"].split()))
                            ret['syl_phone_num'] = syl_phone_num

                        if self._config.get('with_text_data', False):
                            ret['text'] = metadata['text']

                        if self._config.get('with_context_info', False):
                            ret['has_left_context'] = np.array(int(metadata['has_left_context']), dtype='bool')
                            ret['has_right_context'] = np.array(int(metadata['has_right_context']), dtype='bool')

                        yield example_id, ret

            with open(os.path.join(os.getcwd(), "phone.txt"), "w", encoding="utf-8") as fin:
                phonemap_list = ['"' + key + '" : "' + str(value) + '"' for key, value in phone_set.items()]
                fin.write("\n".join(phonemap_list))

    def _data_pipeline(self, datapipe, shuffle=True):
        # On-the-fly context should be added before shuffle
        if self._config.get('with_context_info', False):
            datapipe = ContextDataPipe(datapipe, window_size=self._config.get("window_size", 5))

        if not self._config.get("need_speech", True):
            datapipe = datapipe.map(self._release_unused_item, fn_kwargs={"unused_key": "speech"})

        if shuffle:
            datapipe = datapipe.shuffle(buffer_size=100)

        # filter negative duration
        datapipe = datapipe.filter(self._filter_negative_duration)

        # filter by max frame numbers
        max_frames = self._config.get("max_frames", None)
        if max_frames is not None:
            datapipe = datapipe.filter(self._filter_mel_len, fn_kwargs={"max_mel_len": max_frames})

        # filter by locale id
        if self._config.get("filter_locale_id", None):
            locale_str = self._config["filtered_locale"]
            locale_id_list = list(map(int, str(locale_str).split("-")))
            datapipe = datapipe.filter(self._filter_locale_id, fn_kwargs={"locale_id_list": locale_id_list})

        # filter by speaker id
        filter_speaker_id = self._config.get("filter_speaker_id", None)
        if filter_speaker_id is not None:
            speaker_id_list = list(map(int, str(filter_speaker_id).split("-")))
            datapipe = datapipe.filter(self._filter_speaker_id, fn_kwargs={"speaker_id_list": speaker_id_list})

        # filter by style id
        filter_style_id = self._config.get("filter_style_id", None)
        if filter_style_id is not None:
            style_id_list = list(map(int, str(filter_style_id).split("-")))
            logger.info(f"This run will only train style id in {style_id_list}")
            datapipe = datapipe.filter(self._filter_style_id, fn_kwargs={"style_id_list": style_id_list})

        # Data balance
        if self._config.get("balance_training", None):
            speaker_str = self._config["filtered_speaker"]
            balance_ratio = self._config["balance_ratio"]
            datapipe = datapipe.filter(
                self._balance_training, fn_kwargs={"speaker_id": speaker_str, "balance_ratio": balance_ratio}
            )

        if self._config.get("speakerid_map", None):
            speakerid_map = json.load(open(self._config["speakerid_map"], "r"))
            datapipe = datapipe.map(self._speakerid_replace, fn_kwargs={"speakerid_map": speakerid_map})

        if self._config.get("break_map", None):
            # e.g., break_map.json: {"4144": "4145", "4146": "4145"}, 4144/4145/4146 are phone id of br0, br1, br2
            # in phone_set.json, this map means map br0/br2 to br1
            break_map = json.load(open(self._config["break_map"], "r"))
            datapipe = datapipe.map(self._break_replace, fn_kwargs={"break_map": break_map})

        pitch_norm = self._config["pitch_norm"]
        if pitch_norm["type"] == "min_max":
            datapipe = datapipe.map(
                self._min_max_normalize, fn_kwargs={"f0_min": pitch_norm["f0_min"], "f0_max": pitch_norm["f0_max"]}
            )
        elif pitch_norm["type"] == "mean_var":
            datapipe = datapipe.map(
                self._mean_var_normalize, fn_kwargs={"f0_mean": pitch_norm["f0_mean"], "f0_var": pitch_norm["f0_var"]}
            )
        else:
            raise ValueError("pitch_norm type should be one of min_max or mean_var")

        # Add phone level pitch and energy
        datapipe = datapipe.map(self._add_phone_pitch)
        datapipe = datapipe.map(self._add_phone_energy)

        # Align audio length to frame length
        if self._config.get("align_len", True):
            hop_length = int(self._config.get("frame_shift_ms", 12.5) / 1000 * self._config.get("sample_rate", 24000))
            datapipe = datapipe.map(self._align_len, fn_kwargs={"hop_length": hop_length})

        # Scale mel range to [-1, 1]
        if self._config.get("scale_mel", False):
            datapipe = datapipe.map(self._scale_mel)

        # Random clip speech and mel
        if self._config.get("random_clip", False):
            logger.warning(
                "random_clip only works for speech and mel, take care to use other fields "
                "since they may not align with the clipped speech and mel. (like phone_ids)"
            )
            hop_length = int(self._config["frame_shift_ms"] / 1000 * self._config["sample_rate"])
            segment_length = self._config["segment_length"]
            datapipe = datapipe.map(
                self._random_clip, fn_kwargs={"hop_length": hop_length, "segment_length": segment_length}
            )

        # Add length info before batch and padding
        datapipe = datapipe.map(self._add_length)

        if self._config.get("add_wav_length", False):
            datapipe = datapipe.map(self._add_wav_length)

        if self._config.get("dynamic_batch", True):
            # Dynamic batching with bucket
            batch_size = self._config.get("batch_size", 6000)
            bucket_step = self._config.get("bucket_step", 1.1)
            bucket_scheme = get_bucket_scheme(batch_size, 8, bucket_step)
            datapipe = datapipe.dynamic_batch(
                group_key_fn=self.get_frames,
                bucket_boundaries=bucket_scheme["boundaries"],
                batch_sizes=bucket_scheme["batch_sizes"],
            )
        else:
            datapipe = datapipe.batch(self._config["batch_size"])

        # Shuffle on batch
        if shuffle:
            datapipe = datapipe.shuffle(buffer_size=32)

        repeat_epoch = self._config.get("repeat_epoch", -1)
        if repeat_epoch != -1:
            datapipe = datapipe.repeat(repeat_epoch)  # this datapipe must be last data pipeline

        # Apply padding and then convert to pytorch tensor
        datapipe = datapipe.collate(
            fn_kwargs={
                "padding_axes": {
                    "phone_id": -1,
                    "duration": -1,
                    "speech": 0,
                    "mel": 0,
                    "f0": 0,
                    "uv": 0,
                    "phone_f0": 0,
                    "phone_energy": 0,
                    'stat': 0,
                    'syl_phone_num': -1
                },
                "padding_values": {"phone_id": 0, "duration": 0, "speech": 0.0, "mel": -4.0, "uv": 0.0, 'stat': 0.0,
                                   'syl_phone_num': 0},
                "blacklist": ["text_context"],
            }
        )
        return datapipe

    @staticmethod
    def _release_unused_item(data, unused_key):
        del data[unused_key]
        return data

    @staticmethod
    def _handle_length_mismatch(mutable, target_length, tolerance):
        mismatch = target_length - len(mutable)
        if mismatch != 0:
            if abs(mismatch) > tolerance:
                return None
            else:
                if mismatch > 0:
                    pad_width = ((0, mismatch), *((0, 0) for _ in range(mutable.ndim - 1)))
                    mutable = np.pad(mutable, pad_width, mode="edge")
                else:
                    mutable = mutable[:mismatch]
        return mutable

    @staticmethod
    def _filter_negative_duration(data):
        return bool((data["duration"] >= 0).all())

    @staticmethod
    def _filter_mel_len(data, max_mel_len):
        return bool(len(data["mel"]) < max_mel_len)

    @staticmethod
    def _filter_speaker_id(data, speaker_id_list):
        return int(data["speaker_id"]) in speaker_id_list

    @staticmethod
    def _filter_style_id(data, style_id_list):
        return int(data["style_id"]) in style_id_list

    @staticmethod
    def _balance_training(data, speaker_id, balance_ratio):
        rand = np.random.random_sample()

        if int(data["speaker_id"]) != speaker_id:
            if rand <= balance_ratio:
                return False
        return True

    @staticmethod
    def _filter_locale_id(data, locale_id_list):
        return int(data["locale_id"]) in locale_id_list

    @staticmethod
    def _min_max_normalize(data, f0_min, f0_max):
        data["f0"] = (data["f0"] - f0_min) / (f0_max - f0_min)
        return data

    @staticmethod
    def _mean_var_normalize(data, f0_mean, f0_var):
        data["f0"] = (data["f0"] - f0_mean) / f0_var
        return data

    @staticmethod
    def _add_length(data):
        data["num_phones"] = data["phone_id_length"] = len(data["phone_id"])
        data["num_frames"] = data["mel_length"] = len(data["mel"])
        return data

    @staticmethod
    def _add_wav_length(data):
        data['wavs_length'] = len(data['speech'])
        return data

    @staticmethod
    def _add_phone_energy(data):
        duration = data["duration"]
        phone_boundary = np.cumsum(np.pad(duration, (1, 0)))
        # Padding because the last phone has duration 0

        p_mel = data["mel"].mean(-1)

        phone_energy = np.add.reduceat(np.pad(p_mel, (0, 1)), phone_boundary[:-1])
        phone_energy[duration == 0] = np.min(p_mel)
        phone_energy /= np.where(duration == 0, 1, duration)
        data["phone_energy"] = phone_energy.astype(np.float32)

        return data

    @staticmethod
    def _add_phone_pitch(data):
        duration = data["duration"]
        phone_boundary = np.cumsum(np.pad(duration, (1, 0)))
        # Padding because the last phone has duration 0
        phone_f0 = np.add.reduceat(np.pad(data["f0"], (0, 1)), phone_boundary[:-1])
        phone_f0[duration == 0] = np.min(data["f0"])
        phone_f0 /= np.where(duration == 0, 1, duration)
        data["phone_f0"] = phone_f0.astype(np.float32)
        return data

    @staticmethod
    def get_frames(data):
        return len(data["mel"])

    @staticmethod
    def _align_len(data, hop_length):
        frame_num = len(data["mel"])
        sample_num = len(data["speech"])

        expected_sample_num = frame_num * hop_length
        if expected_sample_num > sample_num:
            data["speech"] = np.pad(
                data["speech"],
                (0, expected_sample_num - sample_num),
                "constant",
                constant_values=(0, data["speech"][-1]),
            )
        else:
            data["speech"] = data["speech"][:expected_sample_num]

        return data

    @staticmethod
    def _speakerid_replace(data, speakerid_map):
        data["speaker_id"] = int(speakerid_map[str(data["speaker_id"])])
        return data

    @staticmethod
    def _break_replace(data, break_map):
        for k, v in dict(break_map).items():
            data["phone_id"] = np.where(data["phone_id"] == int(k), int(v), data["phone_id"])

        return data

    @staticmethod
    def _random_clip(x, hop_length, segment_length):
        if segment_length <= len(x["speech"]):
            max_wav_start = len(x["speech"]) - segment_length
            wav_start = random.randint(0, max_wav_start)
            wav_start = wav_start // hop_length * hop_length
            x["speech"] = x["speech"][wav_start : wav_start + segment_length]
            start_frame = wav_start // hop_length
            x["mel"] = x["mel"][start_frame : start_frame + segment_length // hop_length, :]
        else:
            x["speech"] = np.pad(
                x["speech"], [0, segment_length - len(x["speech"])], mode="constant", constant_values=0.0
            )
            x["mel"] = np.pad(
                x["mel"],
                ((0, int(segment_length // hop_length - len(x["mel"]))), (0, 0)),
                mode="constant",
                constant_values=-4.0,
            )
        return x

    @staticmethod
    def _scale_mel(x):
        x["mel"] = x["mel"] / 4.0
        return x


def _load_csv(filename: Path):
    with open(filename, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            yield row


class ContextDataPipe(IterDataPipe):

    def __init__(self, datapipe: IterDataPipe, *, window_size: int):
        super().__init__()
        self.datapipe = datapipe
        self.window_size = window_size

    def __iter__(self):
        for paragraph in split_before(self.datapipe, lambda x: not x["has_left_context"]):
            left_padding = [paragraph[0]] * (self.window_size // 2)
            right_padding = [paragraph[-1]] * (self.window_size // 2)
            for context in windowed(chain(left_padding, paragraph, right_padding), self.window_size):
                current_sent = context[self.window_size // 2]
                current_sent["text_context"] = list(map(lambda x: x["text"], context))
                yield current_sent
