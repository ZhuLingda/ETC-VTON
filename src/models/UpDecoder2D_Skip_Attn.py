from wsgiref import headers
import torch.nn.functional as F
from torch import nn
import torch

from diffusers.utils import logging
from diffusers.models.attention_processor import Attention, AttnAddedKVProcessor, AttnAddedKVProcessor2_0
from diffusers.models.resnet import ResnetBlock2D, Upsample2D

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class UpDecoderBlock2D_skip_attn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # default, spatial
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_upsample=True,
        temb_channels=None,
        attention_head_dim=1,
        cross_attention_dim=1280,
        only_cross_attention=True,
        cross_attention_norm=None,
        heads=8,
        dim_head=64,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

            processor = (
                AttnAddedKVProcessor2_0() if hasattr(F, "scaled_dot_product_attention") else AttnAddedKVProcessor()
            )

        self.resnets = nn.ModuleList(resnets)

        self.conv = nn.Sequential(
                nn.Conv2d(cross_attention_dim, cross_attention_dim, kernel_size=3, bias=True, padding=1, stride=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(cross_attention_dim, cross_attention_dim, kernel_size=3, bias=True, padding=1, stride=1),
            )

        self.attention_head_dim = attention_head_dim    
        self.num_heads = out_channels // self.attention_head_dim

        # Add the Skip-Attention Block here
        self.attention = Attention(
            query_dim=out_channels,
            cross_attention_dim=out_channels,
            heads=heads,
            dim_head=dim_head,
            added_kv_proj_dim=cross_attention_dim,
            norm_num_groups=resnet_groups,
            bias=True,
            upcast_softmax=True,
            only_cross_attention=only_cross_attention,
            cross_attention_norm=cross_attention_norm,
            processor=processor,
        )
        
        self.fuseConv = nn.Conv2d(in_channels=out_channels+cross_attention_dim, out_channels=out_channels, kernel_size=1)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

    def forward(self, hidden_states, int_feat=None, temb=None):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=temb)
        res_states = hidden_states

        if int_feat is not None:
            int_feat = self.conv(int_feat)
            _int_feat = int_feat.view(int_feat.shape[0], int_feat.shape[1], -1).transpose(1, 2)
            hidden_states = self.attention(hidden_states, _int_feat)
            hidden_states = torch.cat((hidden_states, int_feat), dim=1)
            hidden_states = self.fuseConv(hidden_states)
            hidden_states += res_states

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states

def get_up_block(
    up_block_type,
    num_layers,
    in_channels,
    out_channels,
    prev_output_channel,
    temb_channels,
    add_upsample,
    resnet_eps,
    resnet_act_fn,
    transformer_layers_per_block=1,
    heads=None,
    dim_head=None,
    num_attention_heads=None,
    resnet_groups=None,
    cross_attention_dim=None,
    dual_cross_attention=False,
    use_linear_projection=False,
    only_cross_attention=False,
    upcast_attention=False,
    resnet_time_scale_shift="default",
    resnet_skip_time_act=False,
    resnet_out_scale_factor=1.0,
    cross_attention_norm=None,
    attention_head_dim=None,
    upsample_type=None,
):
    # If attn head dim is not defined, we default it to the number of heads
    if attention_head_dim is None:
        logger.warn(
            f"It is recommended to provide `attention_head_dim` when calling `get_up_block`. Defaulting `attention_head_dim` to {num_attention_heads}."
        )
        attention_head_dim = num_attention_heads

    up_block_type = up_block_type[7:] if up_block_type.startswith("UNetRes") else up_block_type
    if up_block_type == "UpDecoderBlock2D":
        return UpDecoderBlock2D_skip_attn(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            cross_attention_dim=cross_attention_dim,
            heads=heads,
            dim_head=dim_head,
        )