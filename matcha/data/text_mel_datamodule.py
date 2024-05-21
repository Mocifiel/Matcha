import random
from typing import Any, Dict, Optional

import torch
import numpy as np
import torchaudio as ta
from lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader

# from matcha.text import text_to_sequence
# from matcha.utils.audio import mel_spectrogram
# from matcha.utils.model import fix_len_compatibility, normalize
# from matcha.utils.utils import intersperse

from torchtts.data.datasets import TortoiseDataset as D
from torchtts.data.core.datapipe_loader import DataPipeLoader


import re



def parse_filelist(filelist_path, split_char="|"):
    with open(filelist_path, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(split_char) for line in f]
    return filepaths_and_text

def compute_data_statistics(data_loader: torch.utils.data.DataLoader, out_channels: int):
    """Generate data mean and standard deviation helpful in data normalisation

    Args:
        data_loader (torch.utils.data.Dataloader): _description_
        out_channels (int): mel spectrogram channels
    """
    total_mel_sum = 0
    total_mel_sq_sum = 0
    total_mel_len = 0

    for batch in data_loader:
        mels = batch["y"]
        mel_lengths = batch["y_lengths"]

        total_mel_len += torch.sum(mel_lengths)
        total_mel_sum += torch.sum(mels)
        total_mel_sq_sum += torch.sum(torch.pow(mels, 2))

    data_mean = total_mel_sum / (total_mel_len * out_channels)
    data_std = torch.sqrt((total_mel_sq_sum / (total_mel_len * out_channels)) - torch.pow(data_mean, 2))

    return {"mel_mean": data_mean.item(), "mel_std": data_std.item()}




# class TextMelDataModule(LightningDataModule):
#     def __init__(  # pylint: disable=unused-argument
#         self,
#         name,
#         train_filelist_path,
#         valid_filelist_path,
#         batch_size,
#         num_workers,
#         pin_memory,
#         cleaners,
#         add_blank,
#         n_spks,
#         n_fft,
#         n_feats,
#         sample_rate,
#         hop_length,
#         win_length,
#         f_min,
#         f_max,
#         data_statistics,
#         seed,
#     ):
#         super().__init__()

#         # this line allows to access init params with 'self.hparams' attribute
#         # also ensures init params will be stored in ckpt
#         self.save_hyperparameters(logger=False)

#     def setup(self, stage: Optional[str] = None):  # pylint: disable=unused-argument
#         """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

#         This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
#         careful not to execute things like random split twice!
#         """
#         # load and split datasets only if not loaded already

#         self.trainset = TextMelDataset(  # pylint: disable=attribute-defined-outside-init
#             self.hparams.train_filelist_path,
#             self.hparams.n_spks,
#             self.hparams.cleaners,
#             self.hparams.add_blank,
#             self.hparams.n_fft,
#             self.hparams.n_feats,
#             self.hparams.sample_rate,
#             self.hparams.hop_length,
#             self.hparams.win_length,
#             self.hparams.f_min,
#             self.hparams.f_max,
#             self.hparams.data_statistics,
#             self.hparams.seed,
#         )
#         self.validset = TextMelDataset(  # pylint: disable=attribute-defined-outside-init
#             self.hparams.valid_filelist_path,
#             self.hparams.n_spks,
#             self.hparams.cleaners,
#             self.hparams.add_blank,
#             self.hparams.n_fft,
#             self.hparams.n_feats,
#             self.hparams.sample_rate,
#             self.hparams.hop_length,
#             self.hparams.win_length,
#             self.hparams.f_min,
#             self.hparams.f_max,
#             self.hparams.data_statistics,
#             self.hparams.seed,
#         )

#     def train_dataloader(self):
#         return DataLoader(
#             dataset=self.trainset,
#             batch_size=self.hparams.batch_size,
#             num_workers=self.hparams.num_workers,
#             pin_memory=self.hparams.pin_memory,
#             shuffle=True,
#             collate_fn=TextMelBatchCollate(self.hparams.n_spks),
#         )

#     def val_dataloader(self):
#         return DataLoader(
#             dataset=self.validset,
#             batch_size=self.hparams.batch_size,
#             num_workers=self.hparams.num_workers,
#             pin_memory=self.hparams.pin_memory,
#             shuffle=False,
#             collate_fn=TextMelBatchCollate(self.hparams.n_spks),
#         )

#     def teardown(self, stage: Optional[str] = None):
#         """Clean up after fit or test."""
#         pass  # pylint: disable=unnecessary-pass

#     def state_dict(self):  # pylint: disable=no-self-use
#         """Extra things to save to checkpoint."""
#         return {}

#     def load_state_dict(self, state_dict: Dict[str, Any]):
#         """Things to do when loading checkpoint."""
#         pass  # pylint: disable=unnecessary-pass


# class TextMelDataset(torch.utils.data.Dataset):
#     def __init__(
#         self,
#         filelist_path,
#         n_spks,
#         cleaners,
#         add_blank=True,
#         n_fft=1024,
#         n_mels=80,
#         sample_rate=22050,
#         hop_length=256,
#         win_length=1024,
#         f_min=0.0,
#         f_max=8000,
#         data_parameters=None,
#         seed=None,
#     ):
#         self.filepaths_and_text = parse_filelist(filelist_path)
#         self.n_spks = n_spks
#         self.cleaners = cleaners
#         self.add_blank = add_blank
#         self.n_fft = n_fft
#         self.n_mels = n_mels
#         self.sample_rate = sample_rate
#         self.hop_length = hop_length
#         self.win_length = win_length
#         self.f_min = f_min
#         self.f_max = f_max
#         if data_parameters is not None:
#             self.data_parameters = data_parameters
#         else:
#             self.data_parameters = {"mel_mean": 0, "mel_std": 1}
#         random.seed(seed)
#         random.shuffle(self.filepaths_and_text)

#     def get_datapoint(self, filepath_and_text):
#         if self.n_spks > 1:
#             filepath, spk, text = (
#                 filepath_and_text[0],
#                 int(filepath_and_text[1]),
#                 filepath_and_text[2],
#             )
#         else:
#             filepath, text = filepath_and_text[0], filepath_and_text[1]
#             spk = None

#         text = self.get_text(text, add_blank=self.add_blank)
#         mel = self.get_mel(filepath)

#         return {"x": text, "y": mel, "spk": spk}

#     def get_mel(self, filepath):
#         audio, sr = ta.load(filepath)
#         assert sr == self.sample_rate
#         mel = mel_spectrogram(
#             audio,
#             self.n_fft,
#             self.n_mels,
#             self.sample_rate,
#             self.hop_length,
#             self.win_length,
#             self.f_min,
#             self.f_max,
#             center=False,
#         ).squeeze()
#         mel = normalize(mel, self.data_parameters["mel_mean"], self.data_parameters["mel_std"])
#         return mel

#     def get_text(self, text, add_blank=True):
#         text_norm = text_to_sequence(text, self.cleaners)
#         if self.add_blank:
#             text_norm = intersperse(text_norm, 0)
#         text_norm = torch.IntTensor(text_norm)
#         return text_norm

#     def __getitem__(self, index):
#         datapoint = self.get_datapoint(self.filepaths_and_text[index])
#         return datapoint

#     def __len__(self):
#         return len(self.filepaths_and_text)


# class TextMelBatchCollate:
#     def __init__(self, n_spks):
#         self.n_spks = n_spks

#     def __call__(self, batch):
#         B = len(batch)
#         y_max_length = max([item["y"].shape[-1] for item in batch])
#         y_max_length = fix_len_compatibility(y_max_length)
#         x_max_length = max([item["x"].shape[-1] for item in batch])
#         n_feats = batch[0]["y"].shape[-2]

#         y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
#         x = torch.zeros((B, x_max_length), dtype=torch.long)
#         y_lengths, x_lengths = [], []
#         spks = []
#         for i, item in enumerate(batch):
#             y_, x_ = item["y"], item["x"]
#             y_lengths.append(y_.shape[-1])
#             x_lengths.append(x_.shape[-1])
#             y[i, :, : y_.shape[-1]] = y_
#             x[i, : x_.shape[-1]] = x_
#             spks.append(item["spk"])

#         y_lengths = torch.tensor(y_lengths, dtype=torch.long)
#         x_lengths = torch.tensor(x_lengths, dtype=torch.long)
#         spks = torch.tensor(spks, dtype=torch.long) if self.n_spks > 1 else None

#         return {"x": x, "x_lengths": x_lengths, "y": y, "y_lengths": y_lengths, "spks": spks}

class TextMelTorchTTSDataModule(LightningDataModule):
    def __init__(  # pylint: disable=unused-argument
        self,
        name,
        raw_data,
        data_dir,
        shard_format,
        shard_masks,
        shard_name,
        split,
        shard_size,
        n_workers,
        pin_memory,
        dynamic_batch,
        batch_size,
        num_samples,
        vocab_path,
        with_stat_data,
        with_text_data,
        with_context_info,
        sample_rate,
        lazy_decode,
        conditioning_length,
        max_text_tokens,
        max_audio_length,
        n_spks,
        filter_unk_spks,
        data_statistics,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
    
    def setup(self, stage: Optional[str] = None):  # pylint: disable=unused-argument
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        dataset = D(**self.hparams)
        dataset.prepare_dataset()
        self.dataset = dataset.as_data_pipeline()[self.hparams.split]

    def train_dataloader(self):
        return DataPipeLoader(
            dataset=self.dataset, 
            num_workers=self.hparams.n_workers, 
            pin_memory=self.hparams.pin_memory)
    



if __name__ == '__main__':

    # data_module = TextMelDataModule(
    #     name='ljspeech',
    #     train_filelist_path='/data/chong/Matcha-TTS/data/filelists/ljs_audio_text_train_filelist.txt',
    #     valid_filelist_path = '/data/chong/Matcha-TTS/data/filelists/ljs_audio_text_val_filelist.txt',
    #     batch_size= 32,
    #     num_workers=20,
    #     pin_memory=True,
    #     cleaners=['english_cleaners2'],
    #     add_blank=True,
    #     n_spks=1,
    #     n_fft=1024,
    #     n_feats=80,
    #     sample_rate=22050,
    #     hop_length=256,
    #     win_length=1024,
    #     f_min=0,
    #     f_max=8000,
    #     data_statistics={'mel_mean':-5.517436981201172,
    #                         'mel_std': 2.0643768310546875},
    #     seed=1234)
    # data_module.setup()
    # data_loader = data_module.train_dataloader()
    # for data in data_loader:
    #     for key in data:
    #         print(key)
    #     print(data['x'].shape)
    #     print(data['x'][0,:])
    #     print(data['y'].shape)
    #     print(data['spks'])
    #     print(data['x_lengths'])
    #     print(data['y_lengths'])
    #     break

    data_torchtts_module = TextMelTorchTTSDataModule(
        name='sydney',
        raw_data='/data/yanzhen/LibriTTS',
        # data_dir='/data/chong/sydney',
        data_dir = '/data2/chong/libri',
        shard_format='tar',
        # shard_masks='en-us_EnUSSydney_*.tar',
        shard_masks='en-us_libriTTSR_*.tar',
        shard_name='shards',
        split='train',
        shard_size=2000,
        n_workers=2,
        pin_memory=True,
        dynamic_batch=False,
        batch_size=16,
        num_samples=290000,
        vocab_path='/data/chong/Matcha-TTS/bpe_lowercase_asr_256.json',
        with_stat_data=True,
        with_text_data=True,
        with_context_info=True,
        sample_rate=22050,
        lazy_decode=True,
        conditioning_length=360,
        max_text_tokens=400,
        max_audio_length=441000,
        n_spks=1,
        data_statistics={'mel_mean':0.06798957288265228,
                            'mel_std': 1.9658503532409668},
    )
    data_torchtts_module.setup()
    phone_min=10000
    phone_max=-10000
    i = 0
    y_lengths = []
    for data in data_torchtts_module.train_dataloader():
        
        # print(data['x'][0])
        # print(data['x_lengths'])
        # print(data['y_lengths'])
        print(data.keys())
        print(f'phone shape = {data["x"].shape}')
        print(f'mel shape = {data["y"].shape}')
        print(f'cond shape = {data["cond"].shape}')
        print(f'cond_wav shape = {data["cond_wav"].shape}')
        break
        # print(data['y'].shape)
        # print(data['spks'])
        # y_lengths.append(data['y_lengths'])
        # spks.append(data['spks'])
    # spks = torch.cat(spks,dim=-1)
    # print(spks.max())
    # print(spks.min())
    # spks = spks.tolist()
    # sorted_unique_spks = sorted(set(spks))
    # print(sorted_unique_spks)


        # i +=1
        # if i==2:
            # break
    #     for key in data:
    #         print(f'{key} is {data[key].shape}')
    #         print(data[key])
    #         phone_min = min(phone_min, data['x'].min().item())
    #         phone_max = max(phone_max, data['x'].max().item())
    #     break
    # print(f'phone_min = {phone_min}')
    # print(f'phone_max = {phone_max}')

    # data_loader = data_torchtts_module.train_dataloader()
    # stats = compute_data_statistics(data_loader,80)
    # print(f'mean={stats["mel_mean"]},std={stats["mel_std"]}')
    # print('finish')
