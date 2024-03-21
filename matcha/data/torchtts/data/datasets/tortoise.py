import csv
import json
import os
import re
import random
from itertools import chain
# from more_itertools import split_before, windowed

import numpy as np
import logging
from pathlib import Path
from matcha.text.cleaners import english_cleaners
from matcha.text import text_to_sequence
from matcha.utils.utils import intersperse

from torchtts.data.core import features
from torchtts.data.core.datapipes import IterDataPipe
from torchtts.data.core.dataset_builder import GeneratorBasedBuilder
from torchtts.data.core.dataset_info import DatasetInfo
from torchtts.utils.data_utils import get_bucket_scheme, lowercase_dict_keys

# from data.audio.voice_tokenizer import VoiceBpeTokenizer
import librosa 
import torch

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


logger = logging.getLogger(__name__)

def remove_extraneous_punctuation(word):
    replacement_punctuation = {
        '{': '(', '}': ')',
        '[': '(', ']': ')',
        '`': '\'', '—': '-',
        '—': '-', '`': '\'',
        'ʼ': '\''
    }
    replace = re.compile("|".join([re.escape(k) for k in sorted(replacement_punctuation, key=len, reverse=True)]), flags=re.DOTALL)
    word = replace.sub(lambda x: replacement_punctuation[x.group(0)], word)

    # TODO: some of these are spoken ('@', '%', '+', etc). Integrate them into the cleaners.
    extraneous = re.compile(r'^[@#%_=\$\^&\*\+\\]$')
    word = extraneous.sub('', word)
    return word

class VoiceBpeTokenizer:
    def __init__(self, vocab_file):
        if vocab_file is not None:
            self.tokenizer = Tokenizer.from_file(vocab_file)

    def preprocess_text(self, txt):
        txt = english_cleaners(txt)
        txt = remove_extraneous_punctuation(txt)
        return txt

    def encode(self, txt):
        txt = self.preprocess_text(txt)
        txt = txt.replace(' ', '[SPACE]')
        return self.tokenizer.encode(txt).ids

    def decode(self, seq):
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()
        txt = self.tokenizer.decode(seq, skip_special_tokens=False).replace(' ', '')
        txt = txt.replace('[SPACE]', ' ')
        txt = txt.replace('[STOP]', '')
        txt = txt.replace('[UNK]', '')

        return txt


class TortoiseDataset(GeneratorBasedBuilder):
    def _info(self):
        sample_rate = self._config.get("sample_rate", 22050)
        lazy_decode = self._config.get("lazy_decode", True)

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
            builder=self, description="Tortoise dataset builder", features=features.FeaturesDict(feature_dict)
        )

    def _split_generators(self):
        path = self._config.get("raw_data", None)
        if path is None:
            raise ValueError("You should specify raw_data in dataset builder")
        return {k: self._raw_data_generator(split=v, path=Path(path)) for k, v in self.split_type.items()}

    def _raw_data_generator(self, split, path: Path):
        raise NotImplementedError("Not implemented yet, please use unified_tts dataset")

    def _data_pipeline(self, datapipe, shuffle=True):
        # On-the-fly context should be added before shuffle
        self.load_condition = self._config.get("load_condition", True)
        self.conditioning_candidates = self._config.get("conditioning_candidates", 1)
        self.conditioning_length = self._config.get("conditioning_length", 360)
        self.load_aligned_codes = self._config.get("load_aligned_codes", False)
        self.aligned_codes_to_audio_ratio = self._config.get("aligned_codes_ratio", 443)

        self.tokenizer = VoiceBpeTokenizer(self._config["vocab_path"])

        datapipe = datapipe.map(self._process_wav, fn_kwargs = {"conditioning_length": self.conditioning_length})
        datapipe = datapipe.map(self._tokenize, fn_kwargs={"tokenizer": self.tokenizer})
        datapipe = datapipe.filter(self._filter_unk_text)
        datapipe = datapipe.filter(self._filter_unk_spks)
        datapipe = datapipe.filter(self._filter_long_sentence, fn_kwargs={"max_text_tokens": self._config.get("max_text_tokens", 400),
                                                                          "max_audio_length": self._config.get("max_audio_length", 600 / 22 * 22050)})
        datapipe = datapipe.map(self._rename_and_resize)
        datapipe = datapipe.map(self._release_unnecessary_data)

        if shuffle:
            datapipe = datapipe.shuffle(buffer_size=100)

        if self._config.get("break_map", None):
            # e.g., break_map.json: {"4144": "4145", "4146": "4145"}, 4144/4145/4146 are phone id of br0, br1, br2
            # in phone_set.json, this map means map br0/br2 to br1
            break_map = json.load(open(self._config["break_map"], "r"))
            datapipe = datapipe.map(self._break_replace, fn_kwargs={"break_map": break_map})


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
                    "text": -1,
                    "wav": -1,
                    "cond": -1,
                    "phone_id":-1,
                    "y":-1,
                    "x":-1
                },
                "padding_values": {
                    "text": 0, 
                    "wav": 0, 
                    "cond": 0,
                    "phone_id":0,
                    "y":0,
                    "x":0
                },
            }
        )
        datapipe = datapipe.map(self._modify_data_after_batch)
        return datapipe

    @staticmethod
    def _process_wav(data, conditioning_length):
        audio = data["speech"]
        mel = data["mel"].T
        data["wav"] = np.expand_dims(audio,axis=0)
        data["wav_lengths"] = torch.LongTensor([len(audio)])
        #data["conditioning"] = np.expand_dims(audio, axis=0)
        # gap = audio.shape[-1] - conditioning_length
        gap = mel.shape[-1] - conditioning_length
        if gap>0:
            rand_start = random.randint(0, gap)
            # cond = audio[rand_start:rand_start+conditioning_length]
            cond = mel[:,rand_start:rand_start+conditioning_length]
            data["cond"] = cond
        else:
            # data["conditioning"] = audio
            data["cond"] = mel
        
        # data["conditioning"] = np.expand_dims(data["conditioning"], axis=0)


        return data

    @staticmethod
    def _tokenize(data, tokenizer):
        text = data["text"]
        # phone_norm = text_to_sequence(text, ['english_cleaners2'])
        phone_norm = data["phone_id"]
        phone_norm = intersperse(phone_norm, 0)
        phone_norm = torch.IntTensor(phone_norm)

        tokens = tokenizer.encode(text)
        tokens = torch.IntTensor(tokens)

        # if torch.any(tokens <= 1):
        #     logger.warning(f"Found <unk> or <pad> in {text}")
          
        # assert not torch.any(tokens <= 1) # assert no <unk>, start, end token
        data["text"] = tokens
        data["text_lengths"] = torch.LongTensor([len(tokens)])
        data["x"] = phone_norm
        data["x_lengths"] = phone_norm.shape[-1]
        return data

    @staticmethod
    def _filter_unk_text(data):
        text = data["text"]
        if torch.any(text <= 1):
            return False
        return True
    
    @staticmethod
    def _filter_unk_spks(data):
        spks = data["speaker_id"].item()
        if spks<359366 or spks>361791:
            return False
        return True
    
    @staticmethod
    def _filter_long_sentence(data, max_text_tokens, max_audio_length):
        text = data["text"]
        wav = data["wav"]
        if text.shape[-1] > max_text_tokens or wav.shape[-1] > max_audio_length:
            return False
        return True
    
    @staticmethod
    def _rename_and_resize(data):
        # data["text"] = np.squeeze(data["text"])
        # data["wav_lengths"] = np.squeeze(data["wav_lengths"])
        # data["text_lengths"] = np.squeeze(data["text_lengths"])
        # data["skipped_items"] = 0
        # data["conditioning_contains_self"]= 1
        data["y"] = data["mel"].T
        data["y_lengths"] = data["mel"].shape[0]
        data["spks"] = data["speaker_id"]-359366


        return data

    @staticmethod
    def _release_unnecessary_data(data):
        # necessary_keys = ["wav", "wav_lengths", "text", "text_lengths", "conditioning","skipped_items", "conditioning_contains_self"]
        necessary_keys = ["x","x_lengths","y","y_lengths","spks","cond"]
        for key in list(data.keys()):
            if key not in necessary_keys:
                del data[key]
        return data

    @staticmethod
    def _modify_data_after_batch(data):
        # data["spks"] = None
        if data["y"].shape[-1] % 2 ==1:
            data["y"] = torch.nn.functional.pad(data["y"],(0,1))


        return data


   