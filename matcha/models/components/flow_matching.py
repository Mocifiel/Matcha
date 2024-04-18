from abc import ABC

import torch
import torch.nn.functional as F

from matcha.models.components.decoder import Decoder, ControlledDecoder, ControlNet
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
        self.controlnet = None

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
                dphi_dt_uncond = self.estimator(x, mask, mu, t, uncond_spks, cond,cond_wav=cond_wav)
                dphi_dt = (1+cfk)*dphi_dt -cfk*dphi_dt_uncond
            x = x + dt * dphi_dt
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
        
        controls = self.controlnet(y, mask, mu, t.squeeze(), spks, cond_wav=cond_wav)
        outs = self.estimator(y, mask, mu, t.squeeze(), spks, control=controls)
        loss = F.mse_loss(outs, u, reduction="sum") / (torch.sum(mask) * u.shape[1])

        # loss = F.mse_loss(self.estimator(y, mask, mu, t.squeeze(), spks, cond, cond_wav), u, reduction="sum") / (
        #     torch.sum(mask) * u.shape[1]
        # )
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
        # self.estimator = Decoder(in_channels=in_channels, out_channels=out_channel, **decoder_params)
        self.estimator = ControlledDecoder(in_channels=in_channels, out_channels=out_channel, **decoder_params)
        self.controlnet = ControlNet(in_channels=in_channels, out_channels=out_channel, **decoder_params)
    
    def load_from_ckpt(self,ckpt_path):
        checkpoint = torch.load(ckpt_path)
        # 获取decoder.estimator模型的参数字典
        estimator_state_dict = {}
        for name, param in checkpoint['state_dict'].items():
            if name.startswith('decoder.estimator'):  # 选择只包含decoder的参数
                estimator_state_dict[name.replace('decoder.estimator.','')] = param
        controlnet_state_dict = {}
        for name, param in self.controlnet.state_dict().items():
            if name in estimator_state_dict:
                controlnet_state_dict[name]= estimator_state_dict[name]
            else:
                controlnet_state_dict[name] = param
        # 加载参数到decoder模型
        self.estimator.load_state_dict(estimator_state_dict)
        self.controlnet.load_state_dict(controlnet_state_dict)
        


if __name__ == "__main__":
    import yaml
    from easydict import EasyDict
    # 读取YAML文件内容
    with open("/home/chong/Matcha-TTS/configs/model/decoder/default.yaml", "r") as file:
        decoder_params = yaml.safe_load(file)
    decoder_params['cross_attention_dim']=None
    cfm_params  = EasyDict({'name': 'CFM','solver':'euler','sigma_min':1e-4})
    cfm = CFM(in_channels=160,
              out_channel=80,
              cfm_params=cfm_params,
              decoder_params=decoder_params,
              n_spks=2426,
              spk_emb_dim=64)
    cfm.load_from_ckpt('/data/chong/matcha/models/cfg-mean-80.ckpt')
    
    
    # checkpoint = torch.load('/data/chong/matcha/models/cfg-mean-80.ckpt')
    # # 获取decoder.estimator模型的参数字典
    # estimator_state_dict = {}
    # for name, param in checkpoint['state_dict'].items():
    #     if name.startswith('decoder.estimator'):  # 选择只包含decoder的参数
    #         estimator_state_dict[name.replace('decoder.estimator.','')] = param
    # controlnet_state_dict = {}
    # for name, param in cfm.controlnet.state_dict().items():
    #     if name in estimator_state_dict:
    #         controlnet_state_dict[name]= estimator_state_dict[name]
    #     else:
    #         controlnet_state_dict[name] = param



    # # 加载参数到decoder模型
    # cfm.estimator.load_state_dict(estimator_state_dict)
    # cfm.controlnet.load_state_dict(controlnet_state_dict)
    print(cfm.controlnet.state_dict()['input_control_block.bias'])
    # print(list(cfm.controlnet.state_dict().keys()))
    print('start')
    x = torch.randn(4,80,74)
    mu = torch.randn(4,80,74)
    mask = torch.ones(4,1,74)
    t = torch.rand(4,)
    spks = torch.rand(4,64)
    cond_wav = torch.rand(4,4,64)

    outs = cfm.controlnet.forward(x,mask,mu,t,spks=spks,cond_wav=cond_wav)
    print(outs)
    final_outs = cfm.estimator.forward(x,mask,mu,t,spks,control=outs)

    print(final_outs.shape)
    


