import datetime as dt
import math
import random
import inspect
from abc import ABC
from typing import Any, Dict
import torch

import matcha.utils.monotonic_align as monotonic_align
from matcha import utils
from matcha.models.baselightningmodule import BaseLightningClass
from matcha.models.components.flow_matching import CFM
from matcha.models.components.text_encoder import TextEncoder
from matcha.models.components.attention import AttentionBlock, normalization
from matcha.models.components.WavLM import WavLM, WavLMConfig
from matcha.models.components.ecapa_tdnn import ECAPA_TDNN_SMALL
from matcha.utils.model import (
    denormalize,
    duration_loss,
    fix_len_compatibility,
    generate_path,
    sequence_mask,
)

log = utils.get_pylogger(__name__)


class MatchaTTS(BaseLightningClass):  # 🍵
    def __init__(
        self,
        n_vocab,
        n_spks,
        spk_emb_dim,
        n_feats,
        encoder,
        decoder,
        cfm,
        data_statistics,
        out_size,
        unconditioned_percentage=0,
        optimizer=None,
        scheduler=None,
        prior_loss=True,
        cond_wave=True,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.n_vocab = n_vocab
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.n_feats = n_feats
        self.out_size = out_size
        self.prior_loss = prior_loss
        self.unconditioned_percentage = unconditioned_percentage

        if n_spks > 1:
            # self.spk_emb = torch.nn.Embedding(n_spks, spk_emb_dim)
        
            self.cond_embedder = self.contextual_embedder = torch.nn.Sequential(torch.nn.Conv1d(80,spk_emb_dim,3,padding=1,stride=2),
                                                        torch.nn.Conv1d(spk_emb_dim, spk_emb_dim,3,padding=1,stride=2),
                                                        AttentionBlock(spk_emb_dim, num_heads=4, relative_pos_embeddings=True, do_checkpoint=False),
                                                        AttentionBlock(spk_emb_dim, num_heads=4, relative_pos_embeddings=True, do_checkpoint=False),
                                                        AttentionBlock(spk_emb_dim, num_heads=4, relative_pos_embeddings=True, do_checkpoint=False),
                                                        AttentionBlock(spk_emb_dim, num_heads=4, relative_pos_embeddings=True, do_checkpoint=False),
                                                        AttentionBlock(spk_emb_dim, num_heads=4, relative_pos_embeddings=True, do_checkpoint=False))
            self.uncond_emb = torch.nn.Parameter(torch.randn(1,spk_emb_dim))

        self.encoder = TextEncoder(
            encoder.encoder_type,
            encoder.encoder_params,
            encoder.duration_predictor_params,
            n_vocab,
            n_spks,
            spk_emb_dim,
        )

        # self.decoder = CFM(
        #     in_channels=2 * encoder.encoder_params.n_feats,
        #     out_channel=encoder.encoder_params.n_feats,
        #     cfm_params=cfm,
        #     decoder_params=decoder,
        #     n_spks=n_spks,
        #     spk_emb_dim=spk_emb_dim,
        # )
        self.decoder = CFM(
            in_channels=2 * encoder.encoder_params.n_feats,
            out_channel=encoder.encoder_params.n_feats,
            cfm_params=cfm,
            decoder_params=decoder,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )
        

        if cond_wave:
            # WavLM init
            # wavelm_checkpoint = torch.load('/data2/chong/wavelm/WavLM-Large.pt')
            # self.wavelm_cfg = WavLMConfig(wavelm_checkpoint['cfg'])
            # self.wavelm = WavLM(self.wavelm_cfg)
            # self.wavelm.load_state_dict(wavelm_checkpoint['model'])
            # self.wavelm.eval()
            # self.wavelm.requires_grad_(False)
            self.wavelmmodel = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large', config_path=None, update_extract=False)
            # state_dict = torch.load('/data2/chong/wavelm/wavlm_large_finetune.pth', map_location='cpu')
            state_dict = torch.load("/datablob/bohli/spkemb/wavlm_large_finetune.pth", map_location='cpu')
            self.wavelmmodel.load_state_dict(state_dict['model'],strict=False)
            self.wavelmmodel.eval()
            self.wavelmmodel.requires_grad_(False)
        


        self.update_data_statistics(data_statistics)
        
        # self.load_all_except_decoder_from_ckpt('/data/chong/matcha/models/cfg-mean-80.ckpt')
        self.load_all_except_decoder_from_ckpt('/datablob/v-chongzhang/cfg-mean-80.ckpt')
        # self.decoder.load_from_ckpt('/data/chong/matcha/models/cfg-mean-80.ckpt')
        self.decoder.load_from_ckpt('/datablob/v-chongzhang/cfg-mean-80.ckpt')

    @torch.inference_mode()
    def synthesise(self, x, x_lengths, n_timesteps, temperature=1.0, spks=None, cond=None,length_scale=1.0,cfk=0.5,cond_wav=None):
        """
        Generates mel-spectrogram from text. Returns:
            1. encoder outputs
            2. decoder outputs
            3. generated alignment

        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
                shape: (batch_size, max_text_length)
            x_lengths (torch.Tensor): lengths of texts in batch.
                shape: (batch_size,)
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            temperature (float, optional): controls variance of terminal distribution.
            spks (bool, optional): speaker ids.
                shape: (batch_size,)
            cond (torch.Tensor, optional): condition mel of the speaker.
                shape: (batch_size, n_feats, cond_mel_length)
            length_scale (float, optional): controls speech pace.
                Increase value to slow down generated speech and vice versa.
            cfk (float, optional): classifier-free guidance
                Increase value to increase the effect of condition
            cond_wav (torch.Tensor, optional): conditioning wav.
                shape: (batch_size, 1, wav_length)

        Returns:
            dict: {
                "encoder_outputs": torch.Tensor, shape: (batch_size, n_feats, max_mel_length),
                # Average mel spectrogram generated by the encoder
                "decoder_outputs": torch.Tensor, shape: (batch_size, n_feats, max_mel_length),
                # Refined mel spectrogram improved by the CFM
                "attn": torch.Tensor, shape: (batch_size, max_text_length, max_mel_length),
                # Alignment map between text and mel spectrogram
                "mel": torch.Tensor, shape: (batch_size, n_feats, max_mel_length),
                # Denormalized mel spectrogram
                "mel_lengths": torch.Tensor, shape: (batch_size,),
                # Lengths of mel spectrograms
                "rtf": float,
                # Real-time factor
        """
        # For RTF computation
        t = dt.datetime.now()

        if self.n_spks > 1:
            # Get speaker embedding
            # spks = self.spk_emb(spks.long())
            # Get cond embedding
            # spks = self.cond_embedder(cond)[:,:,0] # (batch_size, spk_emb_dim)
            spks = self.cond_embedder(cond).mean(dim=-1)

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spks)

        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w) * length_scale
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = y_lengths.max()
        y_max_length_ = fix_len_compatibility(y_max_length)

        # Using obtained durations `w` construct alignment map `attn`
        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

        # Align encoded text and get mu_y
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)
        encoder_outputs = mu_y[:, :, :y_max_length]

        # Generate sample tracing the probability flow
        uncond_spks = self.uncond_emb.repeat(mu_y.shape[0],1) if cfk>0 else None#(batch_size, spk_emb_dim)
        if cond_wav is not None:
            cond_wav = self.wavlm_feature2(cond_wav)
        print(f'cond_wav.shape after wavlm={cond_wav.shape}')

        
        decoder_outputs = self.decoder(mu_y, y_mask, n_timesteps, temperature, spks,uncond_spks=uncond_spks,cfk=cfk,cond_wav=cond_wav)
        decoder_outputs = decoder_outputs[:, :, :y_max_length]

        t = (dt.datetime.now() - t).total_seconds()
        rtf = t * 22050 / (decoder_outputs.shape[-1] * 256)

        return {
            "encoder_outputs": encoder_outputs,
            "decoder_outputs": decoder_outputs,
            "attn": attn[:, :, :y_max_length],
            "mel": denormalize(decoder_outputs, self.mel_mean, self.mel_std),
            "mel_lengths": y_lengths,
            "rtf": rtf,
        }
    
    @torch.no_grad()
    def wavlm_feature(self,cond_wav=None):
        """
        cond_wav (torch.Tensor, optional): conditioning wav.
                shape: (batch_size, 1, wav_length)
        """
        
        # extract the representation of each layer
        if self.wavelm_cfg.normalize:
            cond_wav = torch.nn.functional.layer_norm(cond_wav.squeeze() , (cond_wav.shape[-1],))

        rep, layer_results = self.wavelm.extract_features(cond_wav, output_layer=self.wavelm.cfg.encoder_layers, ret_layer_results=True)[0]
        layer_reps = [x.transpose(0, 1) for x, _ in layer_results]
        layer_reps = torch.stack(layer_reps,dim=-1).mean(dim=-1) #
        return layer_reps

    @torch.no_grad()
    def wavlm_feature2(self,cond_wav=None):
        """
        cond_wav (torch.Tensor, optional): conditioning wav.
                shape: (batch_size, 1, wav_length)
        """

        wavemb = self.wavelmmodel(cond_wav.squeeze(1), fix=True)
        wavemb = wavemb.reshape(-1, 4, 64)

        # wavemb = wavemb.reshape(-1, 2, 256)
        # wavemb = wavemb.mean(1)
        # wavemb = wavemb / torch.norm(wavemb, dim=1, keepdim=True)
        # spkembnew = wavemb#.detach()



        return wavemb

    def forward(self, x, x_lengths, y, y_lengths, spks=None, out_size=None, cond=None, cond_wav=None):
        """
        Computes 3 losses:
            1. duration loss: loss between predicted token durations and those extracted by Monotinic Alignment Search (MAS).
            2. prior loss: loss between mel-spectrogram and encoder outputs.
            3. flow matching loss: loss between mel-spectrogram and decoder outputs.

        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
                shape: (batch_size, max_text_length)
            x_lengths (torch.Tensor): lengths of texts in batch.
                shape: (batch_size,)
            y (torch.Tensor): batch of corresponding mel-spectrograms.
                shape: (batch_size, n_feats, max_mel_length)
            y_lengths (torch.Tensor): lengths of mel-spectrograms in batch.
                shape: (batch_size,)
            out_size (int, optional): length (in mel's sampling rate) of segment to cut, on which decoder will be trained.
                Should be divisible by 2^{num of UNet downsamplings}. Needed to increase batch size.
            spks (torch.Tensor, optional): speaker ids.
                shape: (batch_size,)
            cond (torch.Tensor, optional): condition mel of the speaker.
                shape: (batch_size, n_feats, cond_mel_length)
            cond_wav (torch.Tensor, optional): conditioning wav.
                shape: (batch_size, 1, wav_length)
        """
        if self.n_spks > 1:
            # Get speaker embedding
            # spks = self.spk_emb(spks) # (batch_size, spk_emb_dim)
            # Get cond embedding
            # spks = self.cond_embedder(cond)[:,:,0] # (batch_size, spk_emb_dim)
            spks = self.cond_embedder(cond).mean(dim=-1) # (batch_size, spk_emb_dim)
        if cond_wav is not None:
            cond_wav = self.wavlm_feature2(cond_wav)
        
        if self.unconditioned_percentage > 0:
            unconditioned_batches = torch.rand((x.shape[0],1),device=x.device)<self.unconditioned_percentage
            unconditioned_batches2 = torch.rand((x.shape[0],1,1),device=x.device)<self.unconditioned_percentage
            spks = torch.where(unconditioned_batches,self.uncond_emb.repeat(x.shape[0],1),spks) # (batch_size, spk_emb_dim)
            cond_wav = torch.where(unconditioned_batches2,torch.zeros_like(cond_wav),cond_wav) # (batch_size, 4, wavlm_emb)
        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spks)
        y_max_length = y.shape[-1]

        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask) #(batch_size, 1, y_max_length)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2) #(batch, 1, max_text_length, y_max_length)

        # Use MAS to find most likely alignment `attn` between text and mel-spectrogram
        with torch.no_grad():
            const = -0.5 * math.log(2 * math.pi) * self.n_feats
            factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
            y_square = torch.matmul(factor.transpose(1, 2), y**2)
            y_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)
            mu_square = torch.sum(factor * (mu_x**2), 1).unsqueeze(-1)
            log_prior = y_square - y_mu_double + mu_square + const

            attn = monotonic_align.maximum_path(log_prior, attn_mask.squeeze(1))
            attn = attn.detach()

        # Compute loss between predicted log-scaled durations and those obtained from MAS
        # refered to as prior loss in the paper
        logw_ = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
        dur_loss = duration_loss(logw, logw_, x_lengths)

        # Cut a small segment of mel-spectrogram in order to increase batch size
        #   - "Hack" taken from Grad-TTS, in case of Grad-TTS, we cannot train batch size 32 on a 24GB GPU without it
        #   - Do not need this hack for Matcha-TTS, but it works with it as well
        if not isinstance(out_size, type(None)):
            max_offset = (y_lengths - out_size).clamp(0)
            offset_ranges = list(zip([0] * max_offset.shape[0], max_offset.cpu().numpy()))
            out_offset = torch.LongTensor(
                [torch.tensor(random.choice(range(start, end)) if end > start else 0) for start, end in offset_ranges]
            ).to(y_lengths)
            attn_cut = torch.zeros(attn.shape[0], attn.shape[1], out_size, dtype=attn.dtype, device=attn.device)
            y_cut = torch.zeros(y.shape[0], self.n_feats, out_size, dtype=y.dtype, device=y.device)

            y_cut_lengths = []
            for i, (y_, out_offset_) in enumerate(zip(y, out_offset)):
                y_cut_length = out_size + (y_lengths[i] - out_size).clamp(None, 0)
                y_cut_lengths.append(y_cut_length)
                cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
                y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
                attn_cut[i, :, :y_cut_length] = attn[i, :, cut_lower:cut_upper]

            y_cut_lengths = torch.LongTensor(y_cut_lengths)
            y_cut_mask = sequence_mask(y_cut_lengths).unsqueeze(1).to(y_mask)

            attn = attn_cut
            y = y_cut
            y_mask = y_cut_mask

        # Align encoded text with mel-spectrogram and get mu_y segment
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)

        # Compute loss of the decoder
        diff_loss, _ = self.decoder.compute_loss(x1=y, mask=y_mask, mu=mu_y, spks=spks,cond=None,cond_wav=cond_wav) # flow_matching.py:CFM

        if self.prior_loss:
            prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
            prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)
        else:
            prior_loss = 0

        return dur_loss, prior_loss, diff_loss
    
    def configure_optimizers(self) -> Any:
        optimizer = self.hparams.optimizer(params=self.decoder.controlnet.parameters())
        if self.hparams.scheduler not in (None, {}):
            scheduler_args = {}
            # Manage last epoch for exponential schedulers
            if "last_epoch" in inspect.signature(self.hparams.scheduler.scheduler).parameters:
                if hasattr(self, "ckpt_loaded_epoch"):
                    current_epoch = self.ckpt_loaded_epoch - 1
                else:
                    current_epoch = -1

            scheduler_args.update({"optimizer": optimizer})
            scheduler = self.hparams.scheduler.scheduler(**scheduler_args)
            scheduler.last_epoch = current_epoch
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": self.hparams.scheduler.lightning_args.interval,
                    "frequency": self.hparams.scheduler.lightning_args.frequency,
                    "name": "learning_rate",
                },
            }

        return {"optimizer": optimizer}
    
    def load_all_except_decoder_from_ckpt(self,ckpt_path):
        # load the state dict for any params other than decoder
        checkpoint = torch.load(ckpt_path)
        old_state_dict = checkpoint['state_dict']

        all_except_decoder_state_dict = {}
        for name, param in self.state_dict().items():
            if not name.startswith('decoder') and name in old_state_dict:
                all_except_decoder_state_dict[name] = old_state_dict[name]
            else:
                all_except_decoder_state_dict[name] = param
        
        self.load_state_dict(all_except_decoder_state_dict)


if __name__ == "__main__":
    print("test")