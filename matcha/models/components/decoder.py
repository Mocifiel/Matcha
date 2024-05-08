import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from conformer import ConformerBlock
from diffusers.models.activations import get_activation
from einops import pack, rearrange, repeat

from matcha.models.components.transformer import BasicTransformerBlock
from matcha.models.components.attention import AttentionBlock, normalization
# from attention import AttentionBlock

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0, "SinusoidalPosEmb requires dim to be even"

    def forward(self, x, scale=1000):
        if x.ndim < 1:
            x = x.unsqueeze(0)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Block1D(torch.nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv1d(dim, dim_out, 3, padding=1),
            torch.nn.GroupNorm(groups, dim_out),
            nn.Mish(),
        )

    def forward(self, x, mask):
        output = self.block(x * mask)
        return output * mask


class ResnetBlock1D(torch.nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super().__init__()
        self.mlp = torch.nn.Sequential(nn.Mish(), torch.nn.Linear(time_emb_dim, dim_out))

        self.block1 = Block1D(dim, dim_out, groups=groups)
        self.block2 = Block1D(dim_out, dim_out, groups=groups)

        self.res_conv = torch.nn.Conv1d(dim, dim_out, 1)

    def forward(self, x, mask, time_emb):
        h = self.block1(x, mask)
        h += self.mlp(time_emb).unsqueeze(-1)
        h = self.block2(h, mask)
        output = h + self.res_conv(x * mask)
        return output


class Downsample1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = torch.nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim=None,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        self.act = get_activation(act_fn)

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out)

        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = get_activation(post_act_fn)

    def forward(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


class Upsample1D(nn.Module):
    """A 1D upsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        use_conv_transpose (`bool`, default `False`):
            option to use a convolution transpose.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
    """

    def __init__(self, channels, use_conv=False, use_conv_transpose=True, out_channels=None, name="conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name

        self.conv = None
        if use_conv_transpose:
            self.conv = nn.ConvTranspose1d(channels, self.out_channels, 4, 2, 1)
        elif use_conv:
            self.conv = nn.Conv1d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, inputs):
        assert inputs.shape[1] == self.channels
        if self.use_conv_transpose:
            return self.conv(inputs)

        outputs = F.interpolate(inputs, scale_factor=2.0, mode="nearest")

        if self.use_conv:
            outputs = self.conv(outputs)

        return outputs


class ConformerWrapper(ConformerBlock):
    def __init__(  # pylint: disable=useless-super-delegation
        self,
        *,
        dim,
        dim_head=64,
        heads=8,
        ff_mult=4,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        attn_dropout=0,
        ff_dropout=0,
        conv_dropout=0,
        conv_causal=False,
    ):
        super().__init__(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            ff_mult=ff_mult,
            conv_expansion_factor=conv_expansion_factor,
            conv_kernel_size=conv_kernel_size,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            conv_dropout=conv_dropout,
            conv_causal=conv_causal,
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        timestep=None,
    ):
        return super().forward(x=hidden_states, mask=attention_mask.bool())


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        channels=(256, 256),
        dropout=0.05,
        attention_head_dim=64,
        n_blocks=1,
        num_mid_blocks=2,
        num_heads=4,
        cross_attention_dim=None,
        act_fn="snake",
        down_block_type="transformer",
        mid_block_type="transformer",
        up_block_type="transformer",
        use_cond = False
    ):
        
        super().__init__()
        channels = tuple(channels)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.time_embeddings = SinusoidalPosEmb(in_channels)
        time_embed_dim = channels[0] * 4
        self.time_mlp = TimestepEmbedding(
            in_channels=in_channels,
            time_embed_dim=time_embed_dim,
            act_fn="silu",
        )
        
        if use_cond:
            self.contextual_embedder = nn.Sequential(nn.Conv1d(in_channels//2,in_channels//2,3,padding=1,stride=2),
                                                    nn.Conv1d(in_channels//2, in_channels,3,padding=1,stride=2),
                                                    AttentionBlock(in_channels, num_heads, relative_pos_embeddings=True, do_checkpoint=False),
                                                    AttentionBlock(in_channels, num_heads, relative_pos_embeddings=True, do_checkpoint=False),
                                                    AttentionBlock(in_channels, num_heads, relative_pos_embeddings=True, do_checkpoint=False),
                                                    AttentionBlock(in_channels, num_heads, relative_pos_embeddings=True, do_checkpoint=False),
                                                    AttentionBlock(in_channels, num_heads, relative_pos_embeddings=True, do_checkpoint=False))
            self.latent_conditioner = nn.Sequential(nn.Conv1d(in_channels//2, in_channels//2, 3, padding=1),
                                                    AttentionBlock(in_channels//2, num_heads, relative_pos_embeddings=True),
                                                    AttentionBlock(in_channels//2, num_heads, relative_pos_embeddings=True),
                                                    AttentionBlock(in_channels//2, num_heads, relative_pos_embeddings=True),
                                                    AttentionBlock(in_channels//2, num_heads, relative_pos_embeddings=True),)
            self.cond_norm = normalization(in_channels//2)

        self.down_blocks = nn.ModuleList([])
        self.mid_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        output_channel = in_channels
        for i in range(len(channels)):  # pylint: disable=consider-using-enumerate
            input_channel = output_channel
            output_channel = channels[i]
            is_last = i == len(channels) - 1
            resnet = ResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)
            transformer_blocks = nn.ModuleList(
                [
                    self.get_block(
                        down_block_type,
                        output_channel,
                        attention_head_dim,
                        num_heads,
                        dropout,
                        act_fn,
                        cross_attention_dim=None,
                    )
                    for _ in range(n_blocks)
                ]
            )
            downsample = (
                Downsample1D(output_channel) if not is_last else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            )

            self.down_blocks.append(nn.ModuleList([resnet, transformer_blocks, downsample]))

        for i in range(num_mid_blocks):
            input_channel = channels[-1]
            # out_channels = channels[-1] # should be corrected as output_channel
            output_channel = channels[-1]

            resnet = ResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)

            transformer_blocks = nn.ModuleList(
                [
                    self.get_block(
                        mid_block_type,
                        output_channel,
                        attention_head_dim,
                        num_heads,
                        dropout,
                        act_fn,
                        cross_attention_dim=cross_attention_dim,
                    )
                    for _ in range(n_blocks)
                ]
            )

            self.mid_blocks.append(nn.ModuleList([resnet, transformer_blocks]))

        channels = channels[::-1] + (channels[0],)
        for i in range(len(channels) - 1):
            input_channel = channels[i]
            output_channel = channels[i + 1]
            is_last = i == len(channels) - 2

            resnet = ResnetBlock1D(
                dim=2 * input_channel,
                dim_out=output_channel,
                time_emb_dim=time_embed_dim,
            )
            transformer_blocks = nn.ModuleList(
                [
                    self.get_block(
                        up_block_type,
                        output_channel,
                        attention_head_dim,
                        num_heads,
                        dropout,
                        act_fn,
                        cross_attention_dim=None
                    )
                    for _ in range(n_blocks)
                ]
            )
            upsample = (
                Upsample1D(output_channel, use_conv_transpose=True)
                if not is_last
                else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            )

            self.up_blocks.append(nn.ModuleList([resnet, transformer_blocks, upsample]))

        self.final_block = Block1D(channels[-1], channels[-1])
        self.final_proj = nn.Conv1d(channels[-1], self.out_channels, 1)

        self.initialize_weights()
        # nn.init.normal_(self.final_proj.weight)

    @staticmethod
    def get_block(block_type, dim, attention_head_dim, num_heads, dropout, act_fn, cross_attention_dim=None):
        if block_type == "conformer":
            block = ConformerWrapper(
                dim=dim,
                dim_head=attention_head_dim,
                heads=num_heads,
                ff_mult=1,
                conv_expansion_factor=2,
                ff_dropout=dropout,
                attn_dropout=dropout,
                conv_dropout=dropout,
                conv_kernel_size=31,
            )
        elif block_type == "transformer":
            block = BasicTransformerBlock(
                dim=dim,
                num_attention_heads=num_heads,
                attention_head_dim=attention_head_dim,
                dropout=dropout,
                cross_attention_dim=cross_attention_dim,
                activation_fn=act_fn,
            )
        else:
            raise ValueError(f"Unknown block type {block_type}")

        return block

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def timestep_independent(self, mu, cond):
        """Compute conditions using mu and cond

        Args:
            mu (torch.Tensor): shape (batch_size, n_feats, mel_timesteps)
            cond (torch.Tensor): shape (batch_size, n_feats, cond_mel_timesteps)

        Raises:

        Returns:
            cond_emb (torch.Tensor): shape (batch_size, n_feats, mel_timesteps)
        """
        cond = self.contextual_embedder(cond) # (batch_size, 2*n_feats, cond_mel_timesteps)
        cond_emb = cond.mean(dim=-1) #(batch_size, 2*n_feats)
        cond_scale, cond_shift = torch.chunk(cond_emb, 2, dim=1) #(batch_size, n_feats) for each
        cond_emb = self.latent_conditioner(mu) #(batch_size, n_feats, mel_timesteps)
        cond_emb = self.cond_norm(cond_emb) * (1 + cond_scale.unsqueeze(-1)) + cond_shift.unsqueeze(-1) # (batch_size, n_feats, mel_timesteps)
        return cond_emb

    def forward(self, x, mask, mu, t, spks=None, cond=None, cond_wav=None):
        """Forward pass of the UNet1DConditional model.

        Args:
            x (torch.Tensor): shape (batch_size, n_feats, mel_timesteps)
            mask (_type_): shape (batch_size, 1, mel_timesteps)
            t (_type_): shape (batch_size)
            spks (_type_, optional): shape: (batch_size, condition_channels). Defaults to None.
            cond (_type_, optional): placeholder for future use. Defaults to None.
            cond_wav (torch.tensor, optional): WaveLM feature. Defaults to None.
                shape (batch_size, seq_len, wavelm_emb_dim)

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        assert not (spks is not None and cond is not None)  # These two are mutually exclusive.

        t = self.time_embeddings(t) #(batch_size, in_channels)
        t = self.time_mlp(t)
        
        if cond is not None:
            mu = self.timestep_independent(mu,cond) #(batch_size, n_feats, mel_timesteps)

        x = pack([x, mu], "b * t")[0]

        if spks is not None:
            spks = repeat(spks, "b c -> b c t", t=x.shape[-1])
            x = pack([x, spks], "b * t")[0]

        hiddens = []
        masks = [mask]
        for resnet, transformer_blocks, downsample in self.down_blocks:
            mask_down = masks[-1]
            x = resnet(x, mask_down, t)
            x = rearrange(x, "b c t -> b t c")
            mask_down = rearrange(mask_down, "b 1 t -> b t")
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=mask_down,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t")
            mask_down = rearrange(mask_down, "b t -> b 1 t")
            hiddens.append(x)  # Save hidden states for skip connections
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]

        for resnet, transformer_blocks in self.mid_blocks:
            x = resnet(x, mask_mid, t)
            x = rearrange(x, "b c t -> b t c")
            mask_mid = rearrange(mask_mid, "b 1 t -> b t")
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=mask_mid,
                    encoder_hidden_states=cond_wav,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t")
            mask_mid = rearrange(mask_mid, "b t -> b 1 t")

        for resnet, transformer_blocks, upsample in self.up_blocks:
            mask_up = masks.pop()
            # debugging
            # print('decoder.py')
            # print(f'x.shape={x.shape}')
            # print(f'hiddens[-1].shape={hiddens[-1].shape}')
            # print(f'pack.shape={pack([x, hiddens[-1]], "b * t")[0].shape}')
            # print(f'mask_up.shape={mask_up.shape}')
            # print(f't.shape={t.shape}')

            x = resnet(pack([x, hiddens.pop()], "b * t")[0], mask_up, t)
            x = rearrange(x, "b c t -> b t c")
            mask_up = rearrange(mask_up, "b 1 t -> b t")
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=mask_up,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t")
            mask_up = rearrange(mask_up, "b t -> b 1 t")
            x = upsample(x * mask_up)

        x = self.final_block(x, mask_up)
        output = self.final_proj(x * mask_up)

        return output * mask

class ControlNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        channels=(256, 256),
        dropout=0.05,
        attention_head_dim=64,
        n_blocks=1,
        num_mid_blocks=2,
        num_heads=4,
        cross_attention_dim=64,
        act_fn="snake",
        down_block_type="transformer",
        mid_block_type="transformer",
        up_block_type="transformer",
        use_cond = False,
    ):
        super().__init__()
        channels = tuple(channels)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.time_embeddings = SinusoidalPosEmb(in_channels)
        time_embed_dim = channels[0] * 4
        self.time_mlp = TimestepEmbedding(
            in_channels=in_channels,
            time_embed_dim=time_embed_dim,
            act_fn="silu",
        )
        self.input_control_block = zero_module(nn.Linear(256, in_channels))
        self.zero_convs = nn.ModuleList([])
        
        self.down_blocks = nn.ModuleList([])
        self.mid_blocks = nn.ModuleList([])


        output_channel = in_channels
        for i in range(len(channels)):  # pylint: disable=consider-using-enumerate
            input_channel = output_channel
            output_channel = channels[i]
            is_last = i == len(channels) - 1
            resnet = ResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)
            transformer_blocks = nn.ModuleList(
                [
                    self.get_block(
                        down_block_type,
                        output_channel,
                        attention_head_dim,
                        num_heads,
                        dropout,
                        act_fn,
                        cross_attention_dim=None,
                    )
                    for _ in range(n_blocks)
                ]
            )
            downsample = (
                Downsample1D(output_channel) if not is_last else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            )

            self.down_blocks.append(nn.ModuleList([resnet, transformer_blocks, downsample]))
            self.zero_convs.append(zero_module(nn.Conv1d(output_channel, output_channel, 1, padding=0)))

        for i in range(num_mid_blocks):
            input_channel = channels[-1]
            output_channel = channels[-1]

            resnet = ResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)

            transformer_blocks = nn.ModuleList(
                [
                    self.get_block(
                        mid_block_type,
                        output_channel,
                        attention_head_dim,
                        num_heads,
                        dropout,
                        act_fn,
                        cross_attention_dim=cross_attention_dim,
                    )
                    for _ in range(n_blocks)
                ]
            )

            self.mid_blocks.append(nn.ModuleList([resnet, transformer_blocks]))

        self.mid_block_out = zero_module(nn.Conv1d(output_channel, output_channel, 1, padding=0))
        # self.initialize_weights()
        # nn.init.normal_(self.final_proj.weight)

    @staticmethod
    def get_block(block_type, dim, attention_head_dim, num_heads, dropout, act_fn, cross_attention_dim=None):
        if block_type == "conformer":
            block = ConformerWrapper(
                dim=dim,
                dim_head=attention_head_dim,
                heads=num_heads,
                ff_mult=1,
                conv_expansion_factor=2,
                ff_dropout=dropout,
                attn_dropout=dropout,
                conv_dropout=dropout,
                conv_kernel_size=31,
            )
        elif block_type == "transformer":
            block = BasicTransformerBlock(
                dim=dim,
                num_attention_heads=num_heads,
                attention_head_dim=attention_head_dim,
                dropout=dropout,
                cross_attention_dim=cross_attention_dim,
                activation_fn=act_fn,
            )
        else:
            raise ValueError(f"Unknown block type {block_type}")

        return block

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def timestep_independent(self, mu, cond):
        """Compute conditions using mu and cond

        Args:
            mu (torch.Tensor): shape (batch_size, n_feats, mel_timesteps)
            cond (torch.Tensor): shape (batch_size, n_feats, cond_mel_timesteps)

        Raises:

        Returns:
            cond_emb (torch.Tensor): shape (batch_size, n_feats, mel_timesteps)
        """
        cond = self.contextual_embedder(cond) # (batch_size, 2*n_feats, cond_mel_timesteps)
        cond_emb = cond.mean(dim=-1) #(batch_size, 2*n_feats)
        cond_scale, cond_shift = torch.chunk(cond_emb, 2, dim=1) #(batch_size, n_feats) for each
        cond_emb = self.latent_conditioner(mu) #(batch_size, n_feats, mel_timesteps)
        cond_emb = self.cond_norm(cond_emb) * (1 + cond_scale.unsqueeze(-1)) + cond_shift.unsqueeze(-1) # (batch_size, n_feats, mel_timesteps)
        return cond_emb


    def forward(self, x, mask, mu, t, spks=None, cond=None, cond_wav=None):
        """Forward pass of the UNet1DConditional model.

        Args:
            x (torch.Tensor): shape (batch_size, n_feats, mel_timesteps)
            mask (_type_): shape (batch_size, 1, mel_timesteps)
            t (_type_): shape (batch_size)
            spks (_type_, optional): shape: (batch_size, condition_channels). Defaults to None.
            cond (_type_, optional): placeholder for future use. Defaults to None.
            cond_wav (torch.tensor, optional): WaveLM feature. Defaults to None.
                shape (batch_size, seq_len, wavelm_emb_dim)

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        assert not (spks is not None and cond is not None)  # These two are mutually exclusive.

        t = self.time_embeddings(t) #(batch_size, in_channels)
        t = self.time_mlp(t)
        
        if cond is not None:
            mu = self.timestep_independent(mu,cond) #(batch_size, n_feats, mel_timesteps)

        x = pack([x, mu], "b * t")[0]

        if spks is not None:
            spks = repeat(spks, "b c -> b c t", t=x.shape[-1])
            x = pack([x, spks], "b * t")[0]

        
        control = self.input_control_block(cond_wav.reshape(cond_wav.shape[0],-1)).unsqueeze(-1) #(batch_size, in_channels,1)
        x = x + control 

        hiddens = []
        masks = [mask]
        for down_block,zero_conv in zip(self.down_blocks,self.zero_convs):
            resnet, transformer_blocks, downsample = down_block
            
            mask_down = masks[-1]
            x = resnet(x, mask_down, t)
            x = rearrange(x, "b c t -> b t c")
            mask_down = rearrange(mask_down, "b 1 t -> b t")
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=mask_down,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t")
            mask_down = rearrange(mask_down, "b t -> b 1 t")
            hiddens.append(zero_conv(x))  # Save hidden states for skip connections
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]

        for resnet, transformer_blocks in self.mid_blocks:
            x = resnet(x, mask_mid, t)
            x = rearrange(x, "b c t -> b t c")
            mask_mid = rearrange(mask_mid, "b 1 t -> b t")
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=mask_mid,
                    encoder_hidden_states=cond_wav,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t")
            mask_mid = rearrange(mask_mid, "b t -> b 1 t")
        
        hiddens.append(self.mid_block_out(x))



        return hiddens

class ControlledDecoder(Decoder):
    def forward(self, x, mask, mu, t, spks=None, cond=None, control=None, only_mid_control=False, cond_wav=None):
        """Forward pass of the controlled UNet1DConditional model.

        Args:
            x (torch.Tensor): shape (batch_size, n_feats, mel_timesteps)
            mask (_type_): shape (batch_size, 1, mel_timesteps)
            t (_type_): shape (batch_size)
            spks (_type_, optional): shape: (batch_size, condition_channels). Defaults to None.
            cond (_type_, optional): placeholder for future use. Defaults to None.
            control (torch.tensor, optional):
            only_mid_control (bool, optional):
            cond_wav (torch.tensor, optional): WaveLM feature. Defaults to None.
                shape (batch_size, seq_len, wavelm_emb_dim)

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        assert not (spks is not None and cond is not None)  # These two are mutually exclusive.
        
        with torch.no_grad():
            t = self.time_embeddings(t) #(batch_size, in_channels)
            t = self.time_mlp(t)
            
            if cond is not None:
                mu = self.timestep_independent(mu,cond) #(batch_size, n_feats, mel_timesteps)

            x = pack([x, mu], "b * t")[0]

            if spks is not None:
                spks = repeat(spks, "b c -> b c t", t=x.shape[-1])
                x = pack([x, spks], "b * t")[0]

            hiddens = []
            masks = [mask]
            for resnet, transformer_blocks, downsample in self.down_blocks:
                mask_down = masks[-1]
                x = resnet(x, mask_down, t)
                x = rearrange(x, "b c t -> b t c")
                mask_down = rearrange(mask_down, "b 1 t -> b t")
                for transformer_block in transformer_blocks:
                    x = transformer_block(
                        hidden_states=x,
                        attention_mask=mask_down,
                        timestep=t,
                    )
                x = rearrange(x, "b t c -> b c t")
                mask_down = rearrange(mask_down, "b t -> b 1 t")
                hiddens.append(x)  # Save hidden states for skip connections
                x = downsample(x * mask_down)
                masks.append(mask_down[:, :, ::2])

            masks = masks[:-1]
            mask_mid = masks[-1]

            for resnet, transformer_blocks in self.mid_blocks:
                x = resnet(x, mask_mid, t)
                x = rearrange(x, "b c t -> b t c")
                mask_mid = rearrange(mask_mid, "b 1 t -> b t")
                for transformer_block in transformer_blocks:
                    x = transformer_block(
                        hidden_states=x,
                        attention_mask=mask_mid,
                        encoder_hidden_states=cond_wav,
                        timestep=t,
                    )
                x = rearrange(x, "b t c -> b c t")
                mask_mid = rearrange(mask_mid, "b t -> b 1 t")
        
        if control is not None:
            x = x + control.pop()

        for resnet, transformer_blocks, upsample in self.up_blocks:
            mask_up = masks.pop()
            if only_mid_control or control is None:
                x = resnet(pack([x, hiddens.pop()], "b * t")[0], mask_up, t)
            else:
                x = resnet(pack([x, hiddens.pop()+control.pop()], "b * t")[0], mask_up, t)
            x = rearrange(x, "b c t -> b t c")
            mask_up = rearrange(mask_up, "b 1 t -> b t")
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=mask_up,
                    timestep=t,
                )
            x = rearrange(x, "b t c -> b c t")
            mask_up = rearrange(mask_up, "b t -> b 1 t")
            x = upsample(x * mask_up)

        x = self.final_block(x, mask_up)
        output = self.final_proj(x * mask_up)

        return output * mask
    
if __name__ == "__main__":
    control_net = ControlNet(in_channels=224,
                             out_channels=80,
                             act_fn="snakebeta")
    control_dec = ControlledDecoder(in_channels=224,
                                    out_channels=80,
                                    cross_attention_dim=64,
                                    act_fn="snakebeta")
    print('start')
    x = torch.randn(4,80,74)
    mu = torch.randn(4,80,74)
    mask = torch.ones(4,1,74)
    t = torch.rand(4,)
    spks = torch.rand(4,64)
    cond_wav = torch.rand(4,4,64)

    outs = control_net.forward(x,mask,mu,t,spks=spks,cond_wav=cond_wav)
    final_outs = control_dec.forward(x,mask,mu,t,spks,control=outs,cond_wav=cond_wav)

    print(final_outs.shape)

    