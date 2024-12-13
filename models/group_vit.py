# -------------------------------------------------------------------------
# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual
# property and proprietary rights in and to this software, related
# documentation and any modifications thereto.  Any use, reproduction,
# disclosure or distribution of this software and related documentation
# without an express license agreement from NVIDIA CORPORATION is strictly
# prohibited.
#
# Written by Jiarui Xu
# -------------------------------------------------------------------------
# Modified by Qingsong Zhao
# -------------------------------------------------------------------------


from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
from .builder import MODELS
from .misc import Result, interpolate_pos_encoding
# from builder import MODELS
# from misc import Result, interpolate_pos_encoding
# from ipdb import set_trace
import clip
import cv2
from IPython import embed
from transformers import AutoModel
from .vision_transformer import VisionTransformer


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MixerMlp(Mlp):

    def forward(self, x):
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


def hard_softmax(logits, dim):
    y_soft = logits.softmax(dim)
    # Straight through.
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft

    return ret


def gumbel_softmax(logits: torch.Tensor, tau: float = 1, hard: bool = False, dim: int = -1) -> torch.Tensor:
    gumbel_dist = torch.distributions.gumbel.Gumbel(
        torch.tensor(0., device=logits.device, dtype=logits.dtype),
        torch.tensor(1., device=logits.device, dtype=logits.dtype))
    gumbels = gumbel_dist.sample(logits.shape)

    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


class AssignAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=1,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 hard=True,
                 gumbel=False,
                 gumbel_tau=1.,
                 sum_assign=False,
                 assign_eps=1.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.hard = hard
        self.gumbel = gumbel

        if (not self.gumbel) and (not self.hard):
            self.attn_dim = -1
        else:
            self.attn_dim = -2

        self.gumbel_tau = gumbel_tau
        self.sum_assign = sum_assign
        self.assign_eps = assign_eps

    def get_attn(self, attn, gumbel=None, hard=None, attn_dim=-2):
        # embed()
        # exit()
        if gumbel is None:
            gumbel = self.gumbel

        if hard is None:
            hard = self.hard

        # attn_dim = -2
        # attn_dim = attn_dim

        if gumbel and self.training:
            attn = gumbel_softmax(attn, dim=attn_dim, hard=hard, tau=self.gumbel_tau)
        else:
            if hard:
                attn = hard_softmax(attn, dim=attn_dim)  # torch.Size([16, 1, 8, 196])
            else:
                attn = F.softmax(attn, dim=attn_dim)

        return attn
        
    def forward(self, query, key=None, *, value=None, return_attn=False, mask=None):
        B, N, C = query.shape
        if key is None:
            key = query
        if value is None:
            value = key
        S = key.size(1)
        # [B, nh, N, C//nh]
        q = rearrange(self.q_proj(query), 'b n (h c)-> b h n c', h=self.num_heads, b=B, n=N, c=C // self.num_heads)
        # [B, nh, S, C//nh]
        k = rearrange(self.k_proj(key), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)
        # [B, nh, S, C//nh]
        v = rearrange(self.v_proj(value), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)

        # [B, nh, N, S]
        raw_attn = (q @ k.transpose(-2, -1)) * self.scale
        # embed()
        # exit()

        attn = self.get_attn(raw_attn, attn_dim=self.attn_dim)  # attn: [256, 1, 2, 32], raw_attn: [256, 1, 2, 32]
        if return_attn:
            hard_attn = attn.clone()
            soft_attn = self.get_attn(raw_attn, gumbel=False, hard=False, attn_dim=self.attn_dim)  # soft_attn: [256, 1, 2, 32]
            attn_dict = {'hard': hard_attn, 'soft': soft_attn, 'rawk': key, 'rawq':query, 'k':k, 'q':q}
        else:
            attn_dict = None

        if not self.sum_assign:
            attn = attn / (attn.sum(dim=-1, keepdim=True) + self.assign_eps)
        attn = self.attn_drop(attn)
        assert attn.shape == (B, self.num_heads, N, S)

        # [B, nh, N, C//nh] <- [B, nh, N, S] @ [B, nh, S, C//nh]
        out = rearrange(attn @ v, 'b h n c -> b n (h c)', h=self.num_heads, b=B, n=N, c=C // self.num_heads)

        out = self.proj(out)
        out = self.proj_drop(out)
        return out, attn_dict

    def extra_repr(self):
        return f'num_heads: {self.num_heads}, \n' \
               f'hard: {self.hard}, \n' \
               f'gumbel: {self.gumbel}, \n' \
               f'sum_assign={self.sum_assign}, \n' \
               f'gumbel_tau: {self.gumbel_tau}, \n' \
               f'assign_eps: {self.assign_eps}'


class GroupingBlock(nn.Module):
    """Grouping Block to group similar segments together.

    Args:
        dim (int): Dimension of the input.
        out_dim (int): Dimension of the output.
        num_heads (int): Number of heads in the grouping attention.
        num_output_group (int): Number of output groups.
        norm_layer (nn.Module): Normalization layer to use.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        hard (bool): Whether to use hard or soft assignment. Default: True
        gumbel (bool): Whether to use gumbel softmax. Default: True
        sum_assign (bool): Whether to sum assignment or average. Default: False
        assign_eps (float): Epsilon to avoid divide by zero. Default: 1
        gum_tau (float): Temperature for gumbel softmax. Default: 1
    """

    def __init__(self,
                 *,
                 dim,
                 out_dim,
                 num_heads,
                 num_group_token,
                 num_output_group,
                 norm_layer,
                 mlp_ratio=(0.5, 4.0),  # TODO 可能需要修改 groupvit，384-192，in_f64, out_f64/8
                 hard=True,
                 gumbel=True,
                 sum_assign=False,
                 assign_eps=1.,
                 gumbel_tau=1.,
                 attn_drop=0.,
                 ):
        super(GroupingBlock, self).__init__()
        self.dim = dim
        self.hard = hard
        self.gumbel = gumbel
        self.sum_assign = sum_assign
        self.num_output_group = num_output_group
        # norm on group_tokens
        self.norm_tokens = norm_layer(dim)
        tokens_dim, channels_dim = [int(x * dim) for x in to_2tuple(mlp_ratio)]
        self.mlp_inter = Mlp(num_group_token, tokens_dim, num_output_group)
        self.norm_post_tokens = norm_layer(dim)
        # norm on x
        self.norm_x = norm_layer(dim)
        self.pre_assign_attn = CrossAttnBlock(
            dim=dim, num_heads=num_heads, mlp_ratio=4, qkv_bias=True, norm_layer=norm_layer, post_norm=True)

        self.assign = AssignAttention(
            dim=dim,
            num_heads=1,
            qkv_bias=True,
            hard=hard,
            gumbel=gumbel,
            gumbel_tau=gumbel_tau,
            sum_assign=sum_assign,
            assign_eps=assign_eps,
            attn_drop=attn_drop,
            )
        self.norm_new_x = norm_layer(dim)
        self.mlp_channels = Mlp(dim, channels_dim, out_dim)
        if out_dim is not None and dim != out_dim:
            self.reduction = nn.Sequential(norm_layer(dim), nn.Linear(dim, out_dim, bias=False))
        else:
            self.reduction = nn.Identity()

    def extra_repr(self):
        return f'hard={self.hard}, \n' \
               f'gumbel={self.gumbel}, \n' \
               f'sum_assign={self.sum_assign}, \n' \
               f'num_output_group={self.num_output_group}, \n '

    def project_group_token(self, group_tokens):
        """
        Args:
            group_tokens (torch.Tensor): group tokens, [B, S_1, C]

        inter_weight (torch.Tensor): [B, S_2, S_1], S_2 is the new number of
            group tokens, it's already softmaxed along dim=-1

        Returns:
            projected_group_tokens (torch.Tensor): [B, S_2, C]
        """
        # [B, S_2, C] <- [B, S_1, C]
        projected_group_tokens = self.mlp_inter(group_tokens.transpose(1, 2)).transpose(1, 2)
        projected_group_tokens = self.norm_post_tokens(projected_group_tokens)
        return projected_group_tokens

    def forward(self, x, group_tokens, return_attn=False):
        """
        Args:
            x (torch.Tensor): image tokens, [B, L, C] [256, 32, 768]
            group_tokens (torch.Tensor): group tokens, [B, S_1, C] [256, 8, 768]
            return_attn (bool): whether to return attention map

        Returns:
            new_x (torch.Tensor): [B, S_2, C], S_2 is the new number of
                group tokens
        """
        group_tokens = self.norm_tokens(group_tokens)  # group_tokens: [256, 8, 768]
        x = self.norm_x(x)  # x: [256, 32, 768]
        # [B, S_2, C]
        projected_group_tokens = self.project_group_token(group_tokens)  # group_tokens: [256, 8, 768], projected_group_tokens: [256, 2, 768]
        projected_group_tokens = self.pre_assign_attn(projected_group_tokens, x) # projected_group_tokens: [256, 2, 768], x: [256, 32, 768]

        new_x, attn_dict = self.assign(projected_group_tokens, x, return_attn=return_attn)  

        new_x += projected_group_tokens
        new_x = self.reduction(new_x) + self.mlp_channels(self.norm_new_x(new_x))

        return new_x, attn_dict


class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 out_dim=None,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 qkv_fuse=False,):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv_fuse = qkv_fuse

        if qkv_fuse:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        else:
            self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def extra_repr(self):
        return f'num_heads={self.num_heads}, \n' \
               f'qkv_bias={self.scale}, \n' \
               f'qkv_fuse={self.qkv_fuse}'

    def forward(self, query, key=None, *, value=None, mask=None):
        if self.qkv_fuse:
            assert key is None
            assert value is None
            x = query
            B, N, C = x.shape
            S = N
            # [3, B, nh, N, C//nh]
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            # [B, nh, N, C//nh]
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        else:
            B, N, C = query.shape
            if key is None:
                key = query
            if value is None:
                value = key
            S = key.size(1)
            # [B, nh, N, C//nh]
            q = rearrange(self.q_proj(query), 'b n (h c)-> b h n c', h=self.num_heads, b=B, n=N, c=C // self.num_heads)
            # [B, nh, S, C//nh]
            k = rearrange(self.k_proj(key), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)
            # [B, nh, S, C//nh]
            v = rearrange(self.v_proj(value), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)

        if mask is not None:
            attn = q @ k.transpose(-2, -1)
            attn.masked_fill_(mask, -np.inf)
            attn *= self.scale
            # print(f"i am not none")
            # embed()
            # exit()
        else:
            # [B, nh, N, S]
            attn = (q @ k.transpose(-2, -1)) * self.scale
            # print(f"i am none")

        attn = attn.softmax(dim=-1)
        
        attn = self.attn_drop(attn)
        assert attn.shape == (B, self.num_heads, N, S)

        # [B, nh, N, C//nh] -> [B, N, C]
        # out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = rearrange(attn @ v, 'b h n c -> b n (h c)', h=self.num_heads, b=B, n=N, c=C // self.num_heads)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class CrossAttnBlock(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 post_norm=False):
        super().__init__()
        if post_norm:
            self.norm_post = norm_layer(dim)
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()
        else:
            self.norm_q = norm_layer(dim)
            self.norm_k = norm_layer(dim)
            self.norm_post = nn.Identity()
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, query, key, *, mask=None):
        x = query
        x = x + self.drop_path(self.attn(self.norm_q(query), self.norm_k(key), mask=mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = self.norm_post(x)
        return x


class TransformerDecoderBlock(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 post_norm=False):
        super().__init__()
        if post_norm:
            self.norm_post = norm_layer(dim)
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()
        else:
            self.norm_q = norm_layer(dim)
            self.norm_k = norm_layer(dim)
            self.norm_post = nn.Identity()
        self.self_attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = norm_layer(dim)

        self.cross_attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, query, key, *, mask=None):
        x = self.self_attn(self.norm_q(query), self.norm_q(query), mask=mask)
        x = query + self.drop_path1(x)
        x = self.norm1(x)

        x = x + self.drop_path2(self.cross_attn(x, self.norm_k(key), mask=mask))
        x = x + self.drop_path3(self.mlp(self.norm2(x)))
        x = self.norm_post(x)
        return x
    


class AttnBlock(nn.Module):

    def __init__(self,
                 tune_config,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            qkv_fuse=True,)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.tune_config = tune_config

        if self.tune_config is not None:
            from .adapter import Adapter
            self.adaptmlp = Adapter(tune_config, dropout=0.1, bottleneck=tune_config.ffn_num,
                                    init_option=tune_config.ffn_adapter_init_option,
                                    adapter_scalar=tune_config.ffn_adapter_scalar,
                                    adapter_layernorm_option=tune_config.ffn_adapter_layernorm_option,
                                    )
        # embed()
        # exit()

    def forward(self, x, mask=None):
        if self.tune_config is not None and self.tune_config.ffn_option == 'serial':
            x = x + self.adaptmlp(self.drop_path(self.attn(self.norm1(x), mask=mask)))
            x = x + self.adaptmlp(self.drop_path(self.mlp(self.norm2(x))))
        elif self.tune_config is not None and self.tune_config.ffn_option == 'parallel':
            x = x + self.drop_path(self.attn(self.norm1(x), mask=mask))
            adapt_x = self.adaptmlp(x, add_residual=False)
            residual = x
            x = self.drop_path(self.mlp(self.norm2(x)))
            x = x + adapt_x
            x = residual + x
        else:
            x = x + self.drop_path(self.attn(self.norm1(x), mask=mask))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        # x = x + self.drop_path(self.attn(self.norm1(x), mask=mask))
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class GroupingLayer(nn.Module):
    """A Transformer layer with Grouping Block for one stage.

    Args:
        dim (int): Number of input channels.
        num_input_token (int): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer.
            In GroupViT setting, Grouping Block serves as the downsampling layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        group_projector (nn.Module | None, optional): Projector for the grouping layer. Default: None.
        zero_init_group_token (bool): Whether to initialize the grouping token to 0. Default: False.
    """

    def __init__(self,
                 dim,
                 num_input_token,
                 depth,
                 num_heads,
                 num_group_token,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 group_projector=None,
                 zero_init_group_token=False,
                 mask_flag=False,
                 only_mask_short=False,
                 enc_steps=128,
                 tune_config=None,
                 use_enc_feat=False,
                 enc_feat_lsxattn=False,
                 long_term_detach=False,
                 long_term_compress=None,
                 i_layer=None,
                 ):

        super().__init__()
        self.dim = dim
        self.input_length = num_input_token
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.num_group_token = num_group_token
        if num_group_token > 0:
            self.group_token = nn.Parameter(torch.zeros(1, num_group_token, dim))
            if not zero_init_group_token:
                trunc_normal_(self.group_token, std=.02)
        else:
            self.group_token = None

        # build blocks
        self.depth = depth
        self.i_layer = i_layer
        blocks = []
        for i in range(depth):
            blocks.append(
                AttnBlock(
                    tune_config=tune_config,
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i],
                    norm_layer=norm_layer,))
        self.blocks = nn.ModuleList(blocks)  # normal ViT self attention layers

        self.downsample = downsample
        self.input_resolution = num_input_token
        self.use_checkpoint = use_checkpoint

        self.group_projector = group_projector

        self.use_enc_feat = use_enc_feat
        self.enc_feat_lsxattn = enc_feat_lsxattn
        self.long_term_detach = long_term_detach
        self.long_term_compress = long_term_compress

        if self.enc_feat_lsxattn and self.i_layer == 0:  # 只在第一层计算长短时attn
            self.short_term_len = 8 # lazy code TODO
            # self.long_short_xattn = TransformerDecoderBlock(dim=dim, num_heads=num_heads, mlp_ratio=4, qkv_bias=True, norm_layer=norm_layer, post_norm=True)
            self.long_short_xattn = CrossAttnBlock(dim=dim, num_heads=num_heads, mlp_ratio=4, qkv_bias=True, norm_layer=norm_layer, post_norm=True)

            if self.long_term_compress == "conv1d":
                self.long_term_compress_fn = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)
            elif self.long_term_compress == "dw_conv1d":
                self.long_term_compress_fn = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1, groups=768//3)
            elif self.long_term_compress == "causal_conv1d":  # TODO after dw_conv1d works !!
                self.long_term_compress_fn = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1, groups=768//2)
            else:
                self.long_term_compress_fn = nn.Identity()

        self.mask = None
        if mask_flag: 
            if only_mask_short and self.enc_feat_lsxattn: # disconnect the long short memory, and only mask short term frames
                short_term_shape = [1, 1, self.short_term_len, self.short_term_len]
                long_term_shape = [1, 1, enc_steps-self.short_term_len, enc_steps-self.short_term_len]
                long_x_short_shape = [1, 1, enc_steps-self.short_term_len, self.short_term_len]
                # short_x_long_shape = [1, 1, self.short_term_len, enc_steps-self.short_term_len]
                enc_shape = [1, 1, enc_steps, enc_steps]
                with torch.no_grad():
                    self.mask = torch.zeros(enc_shape, dtype=torch.bool).cuda()  # [1, 1, 32, 32], all false
                    long_mask = torch.zeros(long_term_shape, dtype=torch.bool).cuda()  # [1, 1, 24, 24], all false
                    short_mask = torch.triu(torch.ones(short_term_shape, dtype=torch.bool), diagonal=1).cuda()  # [1, 1, 8, 8], 右上为true，左下为false
                    long_x_short_mask = torch.ones(long_x_short_shape, dtype=torch.bool).cuda()
                    self.mask[:,:, :-self.short_term_len, :-self.short_term_len] = long_mask
                    self.mask[:,:, -self.short_term_len:, -self.short_term_len:] = short_mask
                    self.mask[:,:, :-self.short_term_len, -self.short_term_len:] = long_x_short_mask  # [1, 1, 24, 8]
            else: 
                full_shape = [1, 1, enc_steps+num_group_token, enc_steps+num_group_token]
                mask_shape = [1, 1, enc_steps, enc_steps]
                with torch.no_grad():
                    self.mask = torch.zeros(full_shape, dtype=torch.bool).cuda()
                    _mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1)
                    self.mask[:,:, :enc_steps, :enc_steps] = _mask
    @property
    def with_group_token(self):
        return self.group_token is not None

    def extra_repr(self):
        return f'dim={self.dim}, \n' \
               f'input_resolution={self.input_resolution}, \n' \
               f'depth={self.depth}, \n' \
               f'num_group_token={self.num_group_token}, \n'

    def split_x(self, x):
        if self.with_group_token:
            return x[:, :-self.num_group_token], x[:, -self.num_group_token:]
        else:
            return x, None

    def concat_x(self, x, group_token=None):
        if group_token is None:
            return x
        return torch.cat([x, group_token], dim=1)

    def forward(self, x, prev_group_token=None, return_attn=False):
        """
        Args:
            x (torch.Tensor): image tokens, [B, L, C]
            prev_group_token (torch.Tensor): group tokens, [B, S_1, C]
            return_attn (bool): whether to return attention maps
        """
        if self.with_group_token:
            group_token = self.group_token.expand(x.size(0), -1, -1)
            if self.group_projector is not None:
                group_token = group_token + self.group_projector(prev_group_token)
        else:
            group_token = None
        # print(f"x.shape {x.shape}, group_token.shape {group_token.shape if group_token is not None else None}")
        B, L, C = x.shape
        cat_x = self.concat_x(x, group_token)  # cat_x [16, 260, 384]

        for blk_idx, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                cat_x = checkpoint.checkpoint(blk, cat_x)  
            else:
                cat_x = blk(cat_x, mask=self.mask)

        x, group_token = self.split_x(cat_x)  # x [256, 32, 768], group_token [16, 64, 384]
        enc_x = None
        if self.use_enc_feat and self.i_layer == 0:
            if self.enc_feat_lsxattn:
                if self.long_term_detach:  
                    long_term_feats, short_term_feats = x[:, :-self.short_term_len, :].clone().detach().contiguous(), x[:, -self.short_term_len:, :].clone().detach().contiguous()
                else:
                    long_term_feats, short_term_feats = x[:, :-self.short_term_len, :].clone().contiguous(), x[:, -self.short_term_len:, :].clone().contiguous()
                short_term_feats = self.long_short_xattn(query=short_term_feats, key=long_term_feats)
                enc_x = short_term_feats[:,-1,:].contiguous()
            else:
                if self.long_term_detach:

                    enc_x = x[:,-1,:].clone().detach().contiguous()
                else:
                    enc_x = x[:,-1,:].clone().contiguous() # last enc frame [256, 768 ] 

        attn_dict = None
        if self.downsample is not None:
            x, attn_dict = self.downsample(x, group_token, return_attn=return_attn)
        return x, group_token, attn_dict, enc_x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self, img_size=224, kernel_size=7, stride=4, padding=2, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.img_size = img_size
        self.patches_resolution = (
            int((img_size[1] + 2 * padding[1] - kernel_size[1]) / stride[1] + 1),
            int((img_size[0] + 2 * padding[0] - kernel_size[0]) / stride[0] + 1),
        )

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    @property
    def num_patches(self):
        return self.patches_resolution[1] * self.patches_resolution[0]

    def forward(self, x):
        B, C, H, W = x.shape
        if self.training:
            # FIXME look at relaxing size constraints
            assert H == self.img_size[0] and W == self.img_size[1], \
                f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        hw_shape = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x, hw_shape


@MODELS.register_module()
class GroupViT(nn.Module):
    r""" Group Vision Transformer
        A PyTorch impl of : `GroupViT: Semantic Segmentation Emerges from Text Supervision`  -
          https://arxiv.org/pdf/2202.11094.pdf

    Args:
        img_size (int | tuple[int]): Input image size. Default 224
        patch_size (int | tuple[int]): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 0
        embed_dim (int): Patch embedding dimension. Default: 384
        embed_factors (list[int]): Embedding dim multipliers for each stage.
        depths (list[int]): Depth of each stage
        num_heads (list[int]): Number of heads for each stage
        num_group_tokens (list[int]): Number of group tokens for each stage
        num_output_group (list[int]): Number of output groups for each stage
        hard_assignment (bool): Whether to use hard assignment or not. Default: True
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        pos_embed_type (str): Type of positional embedding. Default: 'simple'
        freeze_patch_embed (bool): Whether to freeze patch embedding. Default: False
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=0,
                 embed_dim=384,
                 embed_factors=[1, 1, 1],
                 depths=[6, 3, 3],
                 num_heads=[6, 6, 6],
                 num_group_tokens=[64, 8, 0],
                 num_output_groups=[64, 8],
                 hard_assignment=True,
                 gumbel_assignment=True,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 patch_norm=True,
                 use_checkpoint=False,
                 pos_embed_type='simple',
                 freeze_patch_embed=False,
                 imgnet_pretrained=None,
                 pretrained=True,
                 fixed=False,
                 parallel = False,
                 imgnet_pretrained_checkpoint='/mnt/petrelfs/xxx/checkpoints/dino_vitbase16_pretrain.pth',
                 no_patch_embed=True,
                 enc_steps=128,
                 long_term_steps=0,   # default: -1, not use long term memory! others for long term frames.
                 dec_steps=8,
                 mask_attn_layers=None,
                 only_mask_short=False,
                 tune_config=None,
                 pre_proj="linear",
                 use_enc_feat=False, 
                 enc_feat_lsxattn=False,
                 long_term_detach=False,
                 long_term_compress=None,
                 switch_off_layer0=False,
                 ):
        super().__init__()
        assert patch_size in [4, 8, 16]
        self.num_classes = num_classes
        assert len(embed_factors) == len(depths) == len(num_group_tokens)
        assert all(_ == 0 for _ in num_heads) or len(depths) == len(num_heads)
        # assert len(depths) - 1 == len(num_output_groups)
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * embed_factors[len(depths) - 1])
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.num_group_tokens = num_group_tokens
        self.num_output_groups = num_output_groups
        self.pos_embed_type = pos_embed_type
        assert pos_embed_type in ['simple', 'fourier']
        self.freeze_backbone = fixed
        self.no_patch_embed = no_patch_embed
        self.enc_steps = enc_steps
        self.long_term_steps = long_term_steps
        self.dec_steps = dec_steps
        self.pre_proj = pre_proj
        self.use_enc_feat = use_enc_feat
        self.enc_feat_lsxattn = enc_feat_lsxattn
        self.long_term_detach = long_term_detach
        self.long_term_compress = long_term_compress
        self.switch_off_layer0 = switch_off_layer0

        self.tune_config = tune_config
        from easydict import EasyDict
        if self.tune_config is not None:
            self.tune_config = EasyDict(
                ffn_adapt=tune_config['ffn_adapt'],
                ffn_option=tune_config['ffn_option'],
                ffn_adapter_layernorm_option=tune_config['ffn_adapter_layernorm_option'],
                ffn_adapter_init_option=tune_config['ffn_adapter_init_option'],
                ffn_adapter_scalar=tune_config['ffn_adapter_scalar'],
                ffn_num=tune_config['ffn_num'],
                d_model=tune_config['d_model'],)
        
        self.vision_backbone = VisionTransformer(input_resolution=img_size, patch_size=patch_size,
                                                 width=embed_dim,layers=12, heads=3, output_dim=512,
                                                 pretrained=pretrained, fixed = fixed, 
                                                 tune_config = self.tune_config)

        norm_layer = nn.LayerNorm

        if self.pre_proj == 'linear': # TODO
            self.pre_projector = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim),
                                                norm_layer(self.embed_dim),)
        elif 'adapter' in self.pre_proj:            
            from .adapter import Adapter
            self.pre_projector = Adapter(config=None,
                                    dropout=0.1, 
                                    d_model=self.embed_dim,
                                    bottleneck=128,
                                    init_option="lora",
                                    adapter_scalar="0.1",
                                    adapter_layernorm_option="none",
                                    )
        else: # default
            self.pre_projector = nn.Identity()
        
        if self.no_patch_embed:
            self.patch_embed = None
            self.num_patches = enc_steps
            self.patches_resolution = None
            pass
        else:  # default: False
            # split image into non-overlapping patches
            self.patch_embed = PatchEmbed(
                img_size=img_size,
                kernel_size=patch_size,
                stride=patch_size,
                padding=0,
                in_chans=in_chans,
                embed_dim=embed_dim,
                norm_layer=norm_layer if self.patch_norm else None)

            num_patches = self.patch_embed.num_patches
            patches_resolution = self.patch_embed.patches_resolution
            self.patches_resolution = patches_resolution

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        if pos_embed_type == 'simple':  # default
            self.pos_embed = self.build_simple_position_embedding()  # [128 x Dim]
        elif pos_embed_type == 'fourier':
            self.pos_embed = self.build_2d_sincos_position_embedding()
        else:
            raise ValueError

        if freeze_patch_embed:  # default : False
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            self.pos_embed.requires_grad = False

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        num_input_token = self.num_patches
        num_output_token = num_input_token

        # build layers
        self.layers = nn.ModuleList()
        self.mask_attn_layers = mask_attn_layers

        for i_layer in range(self.num_layers):
            mask_flag = False
            if i_layer in self.mask_attn_layers: # 只有layer 0 添加 direc mask
                mask_flag = True
            dim = int(embed_dim * embed_factors[i_layer])
            downsample = None
            
            if i_layer < self.num_layers - 1:
                if num_output_groups[i_layer] > 0:
                    out_dim = embed_dim * embed_factors[i_layer + 1]
                    downsample = GroupingBlock(
                        dim=dim,
                        out_dim=out_dim,
                        num_heads=num_heads[i_layer],
                        num_group_token=num_group_tokens[i_layer],
                        num_output_group=num_output_groups[i_layer],
                        norm_layer=norm_layer,
                        hard=hard_assignment,
                        gumbel=gumbel_assignment,
                        attn_drop=attn_drop_rate,
                        )
                    num_output_token = num_output_groups[i_layer]
            
            
            if i_layer > 0 and num_group_tokens[i_layer] > 0: 
                if num_output_groups[i_layer-1] > 0:
                    prev_dim = int(embed_dim * embed_factors[i_layer - 1])
                    group_projector = nn.Sequential(
                        norm_layer(prev_dim),
                        MixerMlp(num_group_tokens[i_layer - 1], prev_dim // 2, num_group_tokens[i_layer]))

                    if dim != prev_dim:
                        group_projector = nn.Sequential(group_projector, norm_layer(prev_dim),
                                                        nn.Linear(prev_dim, dim, bias=False))
                else:
                    group_projector = None
            else:
                group_projector = None

            # print(mask_flag)

            layer = GroupingLayer(
                dim=dim,
                num_input_token=num_input_token,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                num_group_token=num_group_tokens[i_layer],
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=downsample,
                use_checkpoint=use_checkpoint,
                group_projector=group_projector,
                # only zero init group token if we have a projection
                zero_init_group_token=group_projector is not None,
                mask_flag=mask_flag,
                only_mask_short=only_mask_short,
                enc_steps=enc_steps,
                tune_config=self.tune_config,
                use_enc_feat=self.use_enc_feat,
                enc_feat_lsxattn=self.enc_feat_lsxattn,
                long_term_detach=long_term_detach,
                long_term_compress=long_term_compress,
                i_layer=i_layer,
                )
            self.layers.append(layer)
            if i_layer < self.num_layers - 1:
                num_input_token = num_output_token

        self.norm = norm_layer(self.num_features)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        self.imgnet_pretrained = imgnet_pretrained
        self.proj = None

        if imgnet_pretrained is not None:  # default: False
            ### add cls_token to enable params loading ###
            self.pos_embed = self.build_simple_position_embedding_with_cls_token() 
            self.init_backbone_with_imagenet_weights(imgnet_pretrained_checkpoint)
            ### drop cls_token ###
            self.pos_embed = nn.Parameter(self.pos_embed[0, 1:])

    def init_backbone_with_imagenet_weights(self, checkpoint_path):
        if self.imgnet_pretrained == 'imgnet':
            from timm.models import vit_base_patch16_224
            net = vit_base_patch16_224(pretrained=True)
            state_dict = net.state_dict()
        elif self.imgnet_pretrained in ['dino', 'dinob8', 'dinos16', 'dinos8']:
            state_dict = torch.load(checkpoint_path)
        elif self.imgnet_pretrained == 'clip':
            clip_model, _ = clip.load('ViT-B/16', device='cuda', jit=False)
            state_dict = clip_model.visual.state_dict()
        elif self.imgnet_pretrained == 'viclip':
            state_dict = torch.load(checkpoint_path)['model']
        elif self.imgnet_pretrained == 'videomae':
            state_dict = torch.load(checkpoint_path)['model']
        elif self.imgnet_pretrained == 'timesformer':
            state_dict = torch.load(checkpoint_path)['model_state']
        elif self.imgnet_pretrained == 'bert':
            state_dict = AutoModel.from_pretrained(checkpoint_path).state_dict()
            
        print('Initializing ImageNet-pretrained weights')
        print('$' * 100)
        newdict = {}
        if self.imgnet_pretrained in ['dino', 'dinob8', 'dinos16', 'dinos8', 'imgnet']:
            if self.num_layers == 2:
                for kk, vv in state_dict.items():
                    newkey = kk
                    if kk.startswith('blocks.'):
                        layerid = int(kk.split('.')[1])
                        if 0 <= layerid < 6:
                            newkey = 'layers.0.' + kk
                        elif 6 <= layerid < 12:
                            old_prefix = 'blocks.' + str(layerid) + '.'
                            new_prefix = 'blocks.' + str(layerid - 6) + '.'
                            suffix = kk.split(old_prefix)[1]
                            newkey = 'layers.1.' + new_prefix + suffix
                    newdict[newkey] = vv
            elif self.num_layers == 3:
                for kk, vv in state_dict.items():
                    newkey = kk
                    if kk.startswith('blocks.'):
                        layerid = int(kk.split('.')[1])
                        if 0 <= layerid < 6:
                            newkey = 'layers.0.' + kk
                        elif 6 <= layerid < 9:
                            old_prefix = 'blocks.' + str(layerid) + '.'
                            new_prefix = 'blocks.' + str(layerid - 6) + '.'
                            suffix = kk.split(old_prefix)[1]
                            newkey = 'layers.1.' + new_prefix + suffix
                        elif 9 <= layerid < 12:
                            old_prefix = 'blocks.' + str(layerid) + '.'
                            new_prefix = 'blocks.' + str(layerid - 9) + '.'
                            suffix = kk.split(old_prefix)[1]
                            newkey = 'layers.2.' + new_prefix + suffix
                    newdict[newkey] = vv
        elif self.imgnet_pretrained == 'clip':
            for kk, vv in state_dict.items():
                newkey = kk
                newkey = newkey.replace('transformer.','')
                newkey = newkey.replace('resblocks', 'blocks')
                
                newkey = newkey.replace('attn.in_proj_weight','attn.qkv.weight')
                newkey = newkey.replace('attn.in_proj_bias','attn.qkv.bias')
                newkey = newkey.replace('attn.out_proj.weight','attn.proj.weight')
                newkey = newkey.replace('attn.out_proj.bias','attn.proj.bias')
                
                newkey = newkey.replace('ln_1.weight','norm1.weight')
                newkey = newkey.replace('ln_1.bias','norm1.bias')
                newkey = newkey.replace('ln_2.weight','norm2.weight')
                newkey = newkey.replace('ln_2.bias','norm2.bias')
                
                newkey = newkey.replace('mlp.c_fc.weight','mlp.fc1.weight')
                newkey = newkey.replace('mlp.c_fc.bias', 'mlp.fc1.bias')
                newkey = newkey.replace('mlp.c_proj.weight','mlp.fc2.weight')
                newkey = newkey.replace('mlp.c_proj.bias', 'mlp.fc2.bias')
                
                newkey = newkey.replace('ln_post.weight', 'norm.weight')
                newkey = newkey.replace('ln_post.bias', 'norm.bias')
                
                newkey = newkey.replace('positional_embedding', 'pos_embed')
                # newkey = newkey.replace('conv1.weight', 'patch_embed.proj.weight')
                
                kk = newkey
                if newkey == 'proj':
                    # self.proj = nn.Parameter(torch.zeros(vv.shape[0], vv.shape[1]))
                    self.proj = None

                if newkey == 'pos_embed':
                    vv = vv.unsqueeze(0)
                if kk.startswith('blocks.'):
                    layerid = int(kk.split('.')[1])
                    if 0 <= layerid < 6:
                        newkey = 'layers.0.' + kk
                    elif 6 <= layerid < 12:
                        old_prefix = 'blocks.' + str(layerid) + '.'
                        new_prefix = 'blocks.' + str(layerid - 6) + '.'
                        suffix = kk.split(old_prefix)[1]
                        newkey = 'layers.1.' + new_prefix + suffix
                newdict[newkey] = vv
        elif self.imgnet_pretrained == 'viclip':
            for kk, vv in state_dict.items():
                if 'vision_encoder' in kk:
                    newkey = kk
                    newkey = newkey.replace('vision_encoder.','')
                    newkey = newkey.replace('transformer.','')
                    newkey = newkey.replace('resblocks', 'blocks')

                    newkey = newkey.replace('attn.in_proj_weight','attn.qkv.weight')
                    newkey = newkey.replace('attn.in_proj_bias','attn.qkv.bias')
                    newkey = newkey.replace('attn.out_proj.weight','attn.proj.weight')
                    newkey = newkey.replace('attn.out_proj.bias','attn.proj.bias')

                    newkey = newkey.replace('ln_1.weight','norm1.weight')
                    newkey = newkey.replace('ln_1.bias','norm1.bias')
                    newkey = newkey.replace('ln_2.weight','norm2.weight')
                    newkey = newkey.replace('ln_2.bias','norm2.bias')

                    newkey = newkey.replace('mlp.c_fc.weight','mlp.fc1.weight')
                    newkey = newkey.replace('mlp.c_fc.bias', 'mlp.fc1.bias')
                    newkey = newkey.replace('mlp.c_proj.weight','mlp.fc2.weight')
                    newkey = newkey.replace('mlp.c_proj.bias', 'mlp.fc2.bias')

                    newkey = newkey.replace('ln_post.weight', 'norm.weight')
                    newkey = newkey.replace('ln_post.bias', 'norm.bias')

                    newkey = newkey.replace('temporal_positional_embedding', 'pos_embed') # we use temporal pos. embed
                    newkey = newkey.replace('conv1.weight', 'patch_embed.proj.weight')  # we have no patch embedding

                    kk = newkey
                    if newkey == 'proj':
                        # self.proj = nn.Parameter(torch.zeros(vv.shape[0], vv.shape[1]))
                        self.proj = None # we have no proj

                    if kk.startswith('blocks.'):
                        layerid = int(kk.split('.')[1])
                        if 0 <= layerid < 6:
                            newkey = 'layers.0.' + kk
                        elif 6 <= layerid < 12:
                            old_prefix = 'blocks.' + str(layerid) + '.'
                            new_prefix = 'blocks.' + str(layerid - 6) + '.'
                            suffix = kk.split(old_prefix)[1]
                            newkey = 'layers.1.' + new_prefix + suffix
                    newdict[newkey] = vv
        elif self.imgnet_pretrained == 'videomae':
            if self.num_layers == 2:
                for kk, vv in state_dict.items():
                    # newkey = kk
                    if kk.startswith('encoder.blocks.'):
                        kk = kk.replace('encoder.','')
                        layerid = int(kk.split('.')[1])
                        if 0 <= layerid < 6:
                            newkey = 'layers.0.' + kk
                        elif 6 <= layerid < 12:
                            old_prefix = 'blocks.' + str(layerid) + '.'
                            new_prefix = 'blocks.' + str(layerid - 6) + '.'
                            suffix = kk.split(old_prefix)[1]
                            newkey = 'layers.1.' + new_prefix + suffix
                        if '.q_bias' in newkey:
                            # embed()
                            # exit() 
                            suffix = kk.split('.q_bias')[0]
                            v_bias = state_dict['encoder.'+suffix+'.v_bias']
                            k_bias = nn.Parameter(torch.zeros(vv.shape[0],device=v_bias.device))
                            vv = torch.cat((vv,k_bias,v_bias), dim=0)
                            newkey = newkey.replace('.q_bias','.qkv.bias')
                        newdict[newkey] = vv
        elif self.imgnet_pretrained == 'timesformer':
            for kk, vv in state_dict.items():
                newkey = kk
                newkey = newkey.replace('model.','')
                if newkey.startswith('blocks.') and '.attn.' not in newkey:
                    newkey = newkey.replace('temporal_attn.qkv.weight','attn.qkv.weight')   
                    newkey = newkey.replace('temporal_attn.qkv.bias','attn.qkv.bias')
                    newkey = newkey.replace('temporal_attn.proj.weight','attn.proj.weight')   
                    newkey = newkey.replace('temporal_attn.proj.bias','attn.proj.bias')
                    
                    layerid = int(newkey.split('.')[1])
                    if 0 <= layerid < 6:
                        newkey = 'layers.0.' + newkey
                    elif 6 <= layerid < 12:
                        old_prefix = 'blocks.' + str(layerid) + '.'
                        new_prefix = 'blocks.' + str(layerid - 6) + '.'
                        suffix = newkey.split(old_prefix)[1]
                        newkey = 'layers.1.' + new_prefix + suffix
                    newdict[newkey] = vv
                if newkey in ['time_embed', 'norm.weight', 'norm.bias']:
                    newkey = newkey.replace('time_embed', 'pos_embed')
                    newdict[newkey] = vv
        elif self.imgnet_pretrained == 'bert':
            for kk, vv in state_dict.items():
                newkey = kk
                newkey = newkey.replace('encoder.','')

                if newkey.startswith('layer.'):
                    newkey = newkey.replace('layer.','blocks.')  

                    newkey = newkey.replace('attention.output.LayerNorm.weight','norm1.weight')   
                    newkey = newkey.replace('attention.output.LayerNorm.bias','norm1.bias')
                    #  here
                    newkey = newkey.replace('attention.output.dense.weight','attn.proj.weight')   
                    newkey = newkey.replace('attention.output.dense.bias','attn.proj.bias')

                    newkey = newkey.replace('output.LayerNorm.weight', 'norm2.weight')
                    newkey = newkey.replace('output.LayerNorm.bias', 'norm2.bias')

                    newkey = newkey.replace('intermediate.dense.weight', 'mlp.fc1.weight')
                    newkey = newkey.replace('intermediate.dense.bias', 'mlp.fc1.bias')

                    newkey = newkey.replace('output.dense.weight', 'mlp.fc2.weight')
                    newkey = newkey.replace('output.dense.bias', 'mlp.fc2.bias')

                    layerid = int(newkey.split('.')[1])
                    if 0 <= layerid < 6:
                        newkey = 'layers.0.' + newkey
                    elif 6 <= layerid < 12:
                        old_prefix = 'blocks.' + str(layerid) + '.'
                        new_prefix = 'blocks.' + str(layerid - 6) + '.'
                        suffix = newkey.split(old_prefix)[1]
                        newkey = 'layers.1.' + new_prefix + suffix
                        
                    if '.query.weight' in newkey:
                        suffix = kk.split('.query')[0]
                        v_weight = state_dict[suffix+'.value.weight'] # [768, 768]
                        k_weight = state_dict[suffix+'.key.weight'] # [768, 768]
                        vv = torch.cat((vv,k_weight, v_weight), dim=0)
                        newkey = newkey.replace('attention.self.query.weight','attn.qkv.weight')
                    if '.query.bias' in newkey:
                        # embed()
                        # exit() 
                        suffix = kk.split('.query')[0]
                        v_bias = state_dict[suffix+'.value.bias'] # [768, 768]
                        k_bias = state_dict[suffix+'.key.bias'] # [768, 768]
                        vv = torch.cat((vv, k_bias, v_bias), dim=0)
                        newkey = newkey.replace('attention.self.query.bias','attn.qkv.bias')    
                    newdict[newkey] = vv
                if 'position_embeddings' in newkey:
                    newkey = newkey.replace('embeddings.position_embeddings.weight', 'pos_embed')
                    newdict[newkey] = vv.unsqueeze(0)

        ### init all self-attn/pos_embed/patch_embed layers ###
        msg = self.load_state_dict(newdict, strict=False)

        if self.freeze_backbone:  # TODO： 尝试只放开第一个layer.0 的参数
            for n, p in self.named_parameters():
                if n in newdict and n != "pos_embed": # TODO open pos enc
                    p.requires_grad = False
                    print('Freezing parameter: ', n)

        print(msg)
        print('$' * 100)

    def load_state_dict(self, state_dict: 'OrderedDict[str, torch.Tensor]', strict: bool = True):

        if self.pos_embed_type == 'simple' and 'pos_embed' in state_dict:
            load_pos_embed = state_dict['pos_embed']  # [1, 197, 768]
            pos_embed = self.pos_embed  # [1, 128, 768]
            length_pos_embed = pos_embed.shape[1]
            if load_pos_embed.shape != pos_embed.shape:
                load_pos_embed = load_pos_embed.transpose(1, 2).contiguous()
                load_pos_embed = F.interpolate(load_pos_embed, size=(length_pos_embed), mode='nearest')
                load_pos_embed = load_pos_embed.transpose(1, 2).contiguous()
                state_dict['pos_embed'] = load_pos_embed

        return super().load_state_dict(state_dict, strict)
        

    def build_simple_position_embedding(self):
        # pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, self.embed_dim))
        pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))
        trunc_normal_(pos_embed, std=.02)
        return pos_embed # 3D

    def build_simple_position_embedding_with_cls_token(self):
        pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.embed_dim))
        trunc_normal_(pos_embed, std=.02)
        return pos_embed

    def build_2d_sincos_position_embedding(self, temperature=10000.):
        h, w = self.patch_embed.patches_resolution
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert self.embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

        pos_embed = nn.Parameter(pos_emb)
        pos_embed.requires_grad = False
        return pos_embed

    @property
    def width(self):
        return self.num_features

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_pos_embed(self, B, H, W):
        if self.training:
            return self.pos_embed
        pos_embed = self.pos_embed
        pos_embed = interpolate_pos_encoding(pos_embed, H, W)
        return pos_embed
    
    def get_time_pos_encoding(self, T):
        if self.training:
            return self.pos_embed
        pos_embed = self.pos_embed

        ##### problems might occur here######, 
        ##### N = pos.embed.shape[0] and num_patches could be N - 1
        if pos_embed.ndim == 2:
            pos_embed = pos_embed.unsqueeze(0)
        N = pos_embed.shape[1]
        # N = pos_embed.shape[1]
        if N == T:
            return pos_embed

        pos_embed = pos_embed.transpose(1, 2).contiguous()
        pos_embed = F.interpolate(pos_embed, size=(T), mode='nearest')
        pos_embed = pos_embed.transpose(1, 2).contiguous()
        return pos_embed

        
    def forward_features(self, x, *, return_attn=False):
        if len(x.shape) == 5:
            B, N, C, H, W= x.shape
            x = x.reshape(B*N, C ,H ,W)
            x = self.vision_backbone(x)
            x = x.reshape(B,N,x.shape[-1])
        else:
            x = self.vision_backbone(x)
    
        if "parallel" in self.pre_proj:
            x = x + self.pre_projector(x)
        else:
            x = self.pre_projector(x)

        if self.no_patch_embed:
            x = x + self.get_time_pos_encoding(self.num_patches)
        else:
            x, hw_shape = self.patch_embed(x)
            x = x + self.get_pos_embed(B, *hw_shape)

        ## TODO: nothing todo about the frames features?
        x = self.pos_drop(x)

        group_token = None
        attn_dict_list = []
        enc_feats = []

        for i, layer in enumerate(self.layers):
            if self.switch_off_layer0 and i == 0:
                _x, _group_token, attn_dict, enc_x = layer(x, group_token, return_attn=return_attn)
            else:
                x, group_token, attn_dict, enc_x = layer(x, group_token, return_attn=return_attn)
            if attn_dict is not None:
                attn_dict_list.append(attn_dict)
            if enc_x is not None:
                enc_feats.append(self.norm(enc_x))

        x = self.norm(x)
        return x, group_token, attn_dict_list, enc_feats

    def forward_image_head(self, x):
        """

        Args:
            x: shape [B, L, C]

        Returns:

        """
        # [B, L, C]
        x = self.avgpool(x.transpose(1, 2).contiguous())  # B C 1
        x = torch.flatten(x, 1).contiguous()
        x = self.head(x) if self.proj is None else x @ self.proj
        return x

    def forward(self, x, *, return_feat=False, return_attn=False, as_dict=False, sampled_noun_indices=None):
        x, group_token, attn_dicts, enc_feats = self.forward_features(x, return_attn=return_attn)
        x_feat = x if return_feat else None

        outs = Result(as_dict=as_dict)
        outs.append(self.forward_image_head(x), name='x')
        if return_feat:
            outs.append(x_feat if self.proj is None else x_feat @ self.proj, name='feat')
            outs.append(enc_feats, name='enc_feats')

        if return_attn:
            outs.append(attn_dicts, name='attn_dicts')
        return outs.as_return()


if __name__ == "__main__":
    from IPython import embed
    image = torch.randn(16,128,384).cuda()
    img_encoder = GroupViT(num_heads=[8, 8], depths=[6, 6], embed_factors=[1, 1], num_group_tokens=[64, 0], num_output_groups=[8], batch_size=16).cuda()

    img_encoder.eval()
    img_outs = img_encoder(image, return_feat=True, as_dict=True, return_attn=True)
    
