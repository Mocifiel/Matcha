from abc import ABC

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.func import jvp
import numpy as np

from matcha.models.components.decoder import Decoder
from matcha.utils.pylogger import get_pylogger
from einops import rearrange

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
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None, uncond_spks=None,cfk=0):
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

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        z = torch.randn_like(mu) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond, uncond_spks=uncond_spks, cfk=cfk)

    def solve_euler(self, x, t_span, mu, mask, spks, cond, uncond_spks=None, cfk=0):
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
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []

        for step in range(1, len(t_span)):
            dphi_dt = self.estimator(x, mask, mu, t, spks, cond)
            if uncond_spks is not None and cfk>0:
                dphi_dt_uncond = self.estimator(x, mask, mu, t, uncond_spks, cond)
                dphi_dt = (1+cfk)*dphi_dt -cfk*dphi_dt_uncond
            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1]

    def compute_loss(self, x1, mask, mu, spks=None, cond=None):
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

        loss = F.mse_loss(self.estimator(y, mask, mu, t.squeeze(), spks, cond), u, reduction="sum") / (
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

class NFDM(BASECFM):
    def __init__(self, in_channels, out_channel, cfm_params, decoder_params, n_spks=1, spk_emb_dim=64):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )

        in_channels = in_channels + (spk_emb_dim if n_spks > 1 else 0)
        # Just change the architecture of the estimator here
        self.x_predictor = Decoder(in_channels=in_channels, out_channels=out_channel, **decoder_params) # x_theta_
        in_channels = 160
        self.mean_predictor = Decoder(in_channels=in_channels, out_channels=out_channel, **decoder_params)
        decoder_params["use_softplus"]=True
        self.var_predictor = Decoder(in_channels=in_channels, out_channels=out_channel, **decoder_params)
        self.g_phi = nn.Sequential(nn.Linear(1,16),
                                   nn.ReLU(True),
                                   nn.Linear(16,8),
                                   nn.ReLU(True),
                                   nn.Linear(8,1),
                                   nn.Softplus())
        self.delta = 1e-2
    
    
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None, uncond_spks=None,cfk=0):
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

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        z = torch.randn_like(mu) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond, uncond_spks=uncond_spks, cfk=cfk)
    
    def forward_phi(self, eps, t, x, mu, mask):
        """
        The marginal distribution q_phi(zt|x) of the forward process
        where x is the data point, zt is the intermediate latent variable
        zt is reparameteriazed by eps, a standard gaussian variable ~ N(0, I)
        
        Args:
            eps (torch.Tensor):  should be a standard gaussian variable ~ N(0, I)
                shape: (batch_size, n_feats, mel_timesteps)
            t (torch.Tensor): timesteps
                shape (batch_size)
            x (torch.Tensor): the original data point
                shape (batch_size, n_feats, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
        
        Returns:
            z (torch.Tensor): the intermediate latent variable
                shape (batch_size, n_feats, mel_timesteps)
        """
        t_expand = rearrange(t, 'b -> b 1 1')
        mean = t_expand*x + t_expand*(1-t_expand)*self.mean_predictor(x, mask, mu, t)
        var = self.delta + ((1-t_expand)*
                            self.var_predictor(t_expand*x, mask, mu, (1-t))/
                            self.var_predictor(torch.zeros_like(x),mask,mu,torch.ones_like(t)))
        z = mean + eps*var
        return z
    
    def forward_sde(self, z, t, x, mu, mask):
        """
        The forward sde
        
        Args:
            z (torch.Tensor):  the intermediate latent variable
                shape: (batch_size, n_feats, mel_timesteps)
            t (torch.Tensor): timesteps
                shape (batch_size)
            x (torch.Tensor): the original data point
                shape (batch_size, n_feats, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
        
        Returns:
            dzdt (torch.Tensor): the intermediate latent variable
                shape (batch_size, n_feats, mel_timesteps)
        """
        t_expand = rearrange(t, 'b -> b 1 1')
        mean = t_expand*x + t_expand*(1-t_expand)*self.mean_predictor(x, mask, mu, t)
        var = self.delta + ((1-t_expand)*
                            self.var_predictor(t_expand*x, mask, mu, (1-t))/
                            self.var_predictor(torch.zeros_like(x),mask,mu,torch.ones_like(t)))
        eps = (z-mean)/var
        _, dzdt = jvp(self.forward_phi,(eps, t, x, mu, mask),(torch.zeros_like(eps),
                                                              torch.ones_like(t),
                                                              torch.zeros_like(x),
                                                              torch.zeros_like(mu),
                                                              torch.zeros_like(mask)))
        # too much function evaluation, can it be reduced? (later)


        g_phi_t = self.g_phi(t_expand)

        log_q_phi = self.log_q_phi(z,t,x,mu,mask)
        score = torch.autograd.grad(outputs=log_q_phi,inputs=z,create_graph=True)
        drift = dzdt + g_phi_t**2/2 * score
    
    def log_q_phi(self, z, t, x, mu, mask):
        '''
        log(q_phi(z_t | x)) = log(q(eps)) - log|J_F|

        Args:
            z (torch.Tensor):  the intermediate latent variable
                shape: (batch_size, n_feats, mel_timesteps)
            t (torch.Tensor): timesteps
                shape (batch_size)
            x (torch.Tensor): the original data point
                shape (batch_size, n_feats, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
        
        Returns:
            logp (torch.Tensor): the intermediate latent variable
                shape (batch_size, )
        '''

        t_expand = rearrange(t, 'b -> b 1 1')
        mean = t_expand*x + t_expand*(1-t_expand)*self.mean_predictor(x, mask, mu, t)
        var = self.delta + ((1-t_expand)*
                            self.var_predictor(t_expand*x, mask, mu, (1-t))/
                            self.var_predictor(torch.zeros_like(x),mask,mu,torch.ones_like(t)))
        eps = (z-mean)/var

        D = np.prod(z.shape[1:])
        
        return -D/2. *np.log(2*np.pi) - torch.sum(eps**2,dim=(1,2))/2 -torch.sum(torch.log(var),dim=(1,2))
    
    def compute_loss(self, x1, mask, mu, spks=None, cond=None):
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

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = mu.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        # sample noise p(x_0)
        z0 = torch.randn_like(x1)

        # sample zt
        mean = t*x1 + t*(1-t)*self.mean_predictor(x1, mask, mu, t.squeeze())
        std = self.delta + ((1-t)*self.var_predictor(t*x1, mask, mu, (1-t.squeeze()))/self.var_predictor(torch.zeros_like(x1),mask,mu,torch.ones_like(t.squeeze())))
        zt = mean + z0*std

        # compute flow term and score term
        _, flow = jvp(self.forward_phi,(z0, t.squeeze(), x1, mu, mask),(torch.zeros_like(z0),
                                                              torch.ones_like(t.squeeze()),
                                                              torch.zeros_like(x1),
                                                              torch.zeros_like(mu),
                                                              torch.zeros_like(mask)))
        
        g_phi_t = self.g_phi(t)
        log_q_phi = self.log_q_phi(zt,t.squeeze(),x1,mu,mask)
        print(f'log_q_phi.shape={log_q_phi.shape}')
        
        score = torch.autograd.grad(outputs=log_q_phi.sum(),inputs=zt,create_graph=True) # can be reduced to closed-form later.
        print(f'score.shape={score[0].shape}')

        # compute reverse SDE drift term, which is the training target
        drift_target = flow - g_phi_t**2/2 *score[0]

        # compute the reverse SDE drift term incorporating the prediction of x:
        x_pred = self.x_predictor(zt, mask, mu, t.squeeze(), spks, cond)
        # compute flow term and score term
        _, flow = jvp(self.forward_phi,(z0, t.squeeze(), x_pred, mu, mask),(torch.zeros_like(z0),
                                                              torch.ones_like(t.squeeze()),
                                                              torch.zeros_like(x_pred),
                                                              torch.zeros_like(mu),
                                                              torch.zeros_like(mask)))
        
        log_q_phi = self.log_q_phi(zt,t.squeeze(),x_pred,mu,mask)
        score = torch.autograd.grad(outputs=log_q_phi.sum(),inputs=zt,create_graph=True)

        drift = flow -g_phi_t**2/2 *score[0]
        loss = F.mse_loss(drift, drift_target, reduction="sum") / (torch.sum(mask) * drift_target.shape[1])
        return loss, zt
    
    


if __name__ == "__main__":
    import yaml
    from easydict import EasyDict
    # 读取YAML文件内容
    with open("/home/chong/Matcha-TTS/configs/model/decoder/default.yaml", "r") as file:
        decoder_params = yaml.safe_load(file)
    decoder_params['cross_attention_dim']=None
    cfm_params  = EasyDict({'name': 'CFM','solver':'euler','sigma_min':1e-4})
    nfdm = NFDM(in_channels=160,
              out_channel=80,
              cfm_params=cfm_params,
              decoder_params=decoder_params,
              n_spks=2426,
              spk_emb_dim=64)
    # cfm.load_from_ckpt('/data/chong/matcha/models/cfg-mean-80.ckpt')
    # print(cfm.controlnet.state_dict()['input_control_block.bias'])
    # print(list(cfm.controlnet.state_dict().keys()))
    # for key in cfm.estimator.state_dict():
    #     print(key)
    print('start')
    x = torch.randn(4,80,74)
    mu = torch.randn(4,80,74)
    mask = torch.ones(4,1,74)
    t = torch.rand(4,)
    spks = torch.rand(4,64)
    
    nfdm.compute_loss(x,mask,mu,spks)
    









        
