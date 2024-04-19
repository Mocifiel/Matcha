from abc import ABC

import torch
import torch.nn.functional as F

from matcha.models.components.decoder import Decoder
from matcha.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class BASECFM(torch.nn.Module, ABC):
    def __init__(
        self,
        n_feats,
        cfm_params,
        n_spks=1,
        spk_emb_dim=128,
    ):
        super().__init__()
        self.n_feats = n_feats
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.solver = cfm_params.solver
        if hasattr(cfm_params, "sigma_min"):
            self.sigma_min = cfm_params.sigma_min
        else:
            self.sigma_min = 1e-4

        self.estimator = None

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None, uncond_spks=None,cfk=0,cond_wav=None):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
            uncond_spks (torch.Tensor, optional): unconditional speaker ids. Defauts to None. 
                Used to implement classifier-free guidance
                shape: (batch_size, spk_emb_dim)
            cfk: (float, optional): classifier-free guidance coefficient
            cond_wav (torch.Tensor, optional): wavlm embedding of conditioning wav.
                shape: (batch_size, 4, 64)

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        z = torch.randn_like(mu) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond, uncond_spks=uncond_spks, cfk=cfk,cond_wav=cond_wav)

    def solve_euler(self, x, t_span, mu, mask, spks, cond, uncond_spks=None, cfk=0,cond_wav=None):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

            cond_wav (torch.Tensor, optional): wavlm embedding of conditioning wav.
                shape: (batch_size, 4, 64)
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []

        for step in range(1, len(t_span)):
            dphi_dt = self.estimator(x, mask, mu, t, spks, cond, cond_wav=cond_wav)
            if uncond_spks is not None and cfk>0:
                dphi_dt_uncond0 = self.estimator(x, mask, mu, t, uncond_spks, cond,cond_wav=torch.zeros_like(cond_wav))
                dphi_dt_uncond1 = self.estimator(x, mask, mu, t, spks, cond,cond_wav=torch.zeros_like(cond_wav))
                dphi_dt_uncond2 = self.estimator(x, mask, mu, t, uncond_spks, cond,cond_wav=cond_wav)

                # dphi_dt = dphi_dt +cfk*(dphi_dt-dphi_dt_uncond0) +cfk*(dphi_dt_uncond1+dphi_dt_uncond2-2*dphi_dt_uncond0)
            # x = x + dt * dphi_dt
            x = x + dt * (dphi_dt+cfk*(dphi_dt_uncond2-dphi_dt_uncond0))
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1]

    def compute_loss(self, x1, mask, mu, spks=None, cond=None, cond_wav=None):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond_wav (torch.Tensor, optional): WaveLM feature. Defaults to None.
                shape: (batch_size, seq_len, wavelm_emb_dim=1024) 

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = mu.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z
        # debugging
        # print(f'y.shape={y.shape}')
        # print(f'mask.shape={mask.shape}')
        # print(f'mu.shape={mu.shape}')
        # print(f't.shape={t.shape}')
        # print(f'u.shape={u.shape}')

        loss = F.mse_loss(self.estimator(y, mask, mu, t.squeeze(), spks, cond, cond_wav), u, reduction="sum") / (
            torch.sum(mask) * u.shape[1]
        )
        return loss, y


class CFM(BASECFM):
    def __init__(self, in_channels, out_channel, cfm_params, decoder_params, n_spks=1, spk_emb_dim=64):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )

        in_channels = in_channels + (spk_emb_dim if n_spks > 1 else 0)
        # Just change the architecture of the estimator here
        self.estimator = Decoder(in_channels=in_channels, out_channels=out_channel, **decoder_params)
