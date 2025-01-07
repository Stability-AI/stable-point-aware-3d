from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from spar3d.models.transformers.attention import BasicTransformerBlock
from spar3d.models.utils import BaseModule


class Transformer1D(BaseModule):
    """
    A 1D Transformer model for sequence data.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to use in feed-forward.
        attention_bias (`bool`, *optional*):
            Configure if the `TransformerBlocks` attention should contain a bias parameter.
    """

    @dataclass
    class Config(BaseModule.Config):
        num_attention_heads: int = 16
        attention_head_dim: int = 88
        in_channels: Optional[int] = None
        out_channels: Optional[int] = None
        num_layers: int = 1
        norm_num_groups: int = 32
        attention_bias: bool = False
        activation_fn: str = "geglu"
        norm_elementwise_affine: bool = True
        residual: bool = True
        input_layer_norm: bool = True
        norm_eps: float = 1e-5

    cfg: Config

    def configure(self) -> None:
        self.num_attention_heads = self.cfg.num_attention_heads
        self.attention_head_dim = self.cfg.attention_head_dim
        inner_dim = self.num_attention_heads * self.attention_head_dim

        linear_cls = nn.Linear

        # 2. Define input layers
        self.in_channels = self.cfg.in_channels

        self.norm = torch.nn.GroupNorm(
            num_groups=self.cfg.norm_num_groups,
            num_channels=self.cfg.in_channels,
            eps=self.cfg.norm_eps,
            affine=True,
        )
        self.proj_in = linear_cls(self.cfg.in_channels, inner_dim)

        # 3. Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    self.num_attention_heads,
                    self.attention_head_dim,
                    activation_fn=self.cfg.activation_fn,
                    attention_bias=self.cfg.attention_bias,
                    norm_elementwise_affine=self.cfg.norm_elementwise_affine,
                    norm_eps=self.cfg.norm_eps,
                )
                for d in range(self.cfg.num_layers)
            ]
        )

        # 4. Define output layers
        self.out_channels = (
            self.cfg.in_channels
            if self.cfg.out_channels is None
            else self.cfg.out_channels
        )

        self.proj_out = linear_cls(inner_dim, self.cfg.in_channels)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        The [`Transformer1DModel`] forward method.

        Args:
            hidden_states (`torch.LongTensor` of shape `(batch size, num latent pixels)` if discrete, `torch.FloatTensor` of shape `(batch size, channel, height, width)` if continuous):
                Input `hidden_states`.
            encoder_hidden_states ( `torch.FloatTensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            attention_mask ( `torch.Tensor`, *optional*):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            encoder_attention_mask ( `torch.Tensor`, *optional*):
                Cross-attention mask applied to `encoder_hidden_states`. Two formats supported:

                    * Mask `(batch, sequence_length)` True = keep, False = discard.
                    * Bias `(batch, 1, sequence_length)` 0 = keep, -10000 = discard.

                If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (
                1 - encoder_attention_mask.to(hidden_states.dtype)
            ) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. Input
        batch, _, seq_len = hidden_states.shape
        residual = hidden_states

        if self.cfg.input_layer_norm:
            hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 1).reshape(
            batch, seq_len, inner_dim
        )
        hidden_states = self.proj_in(hidden_states)

        # 2. Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
            )

        # 3. Output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = (
            hidden_states.reshape(batch, seq_len, inner_dim)
            .permute(0, 2, 1)
            .contiguous()
        )

        if self.cfg.residual:
            output = hidden_states + residual
        else:
            output = hidden_states

        return output
