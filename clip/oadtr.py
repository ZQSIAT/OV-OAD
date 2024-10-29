import os.path as osp
from collections import OrderedDict
import math
import numpy as np
import torch
import torch.nn as nn
from itertools import repeat
import collections.abc
from IPython import embed


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)


# take from timm/models/vit
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


# take from timm/models/vit
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


# take from timm/models/vit
class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma



class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# review from timm/models/vit, add attn mask
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # self.attn = None

    def forward(self, x):
        B, N, C = x.shape  # x: [128, 196, 768]
        # embed()
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)  # q: [128, 8, 196, 96]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [128, 8, 196, 196]
        # self.attn = attn
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # [128, 196, 8, 96] --> [128, 196, 768]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x  # , self.attn


class TriangularCausalMask(object):
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


# cross attn wheels
class CrossAttention(nn.Module):

    def __init__(self,  dim, num_heads=8, qkv_bias=False,
                 cross_attn_flag=False, mask_flag=False, attn_drop=0.1, proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.mask_flag = mask_flag
        self.cross_attn_flag = cross_attn_flag

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        pass

    def forward_linear(self, x, y):
        B, N, C = x.shape  # B 8 C  decoder input
        _, L, _ = y.shape  # B 32 C  encoder input
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads)

        if not self.cross_attn_flag:  # self attn
            k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads)
            v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads)
        else:  # cross attn
            k = self.k(y).reshape(B, L, self.num_heads, C // self.num_heads)
            v = self.v(y).reshape(B, L, self.num_heads, C // self.num_heads)
            pass
        return q, k, v, B, N, C, L  # q,k,v, batch size, decoder seq length, dim, encoder seq length

    def forward(self, x, y, attn_mask=None):
        # linear layers
        q, k, v, B, N, C, L= self.forward_linear(x, y)

        # compute attn weight
        attn = torch.einsum("bnhd,blhd->bhnl", q, k)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, N, device=q.device)
            attn.masked_fill_(attn_mask.mask, -np.inf)
        attn = self.attn_drop(torch.softmax(self.scale * attn, dim=-1))
        # attn @ v
        out_x = torch.einsum("bhnl,blhd->bnhd", attn, v).reshape(B, N, C).contiguous()
        # projection and dropout
        out_x = self.proj(out_x)
        out_x = self.proj_drop(out_x)
        return out_x


# take from timm/models/vit
class Block(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # embed()
        # exit()
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


# add cross attn to self-attn block
class CrossAttnBlock(Block):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 init_values=None,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 directional_mask=True,
                 ):
        super().__init__(dim,
                         num_heads,
                         mlp_ratio,
                         qkv_bias,
                         drop,
                         attn_drop,
                         init_values,
                         drop_path,
                         act_layer,
                         norm_layer)

        # layer1 self attn with mask
        self.attn = CrossAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, cross_attn_flag=False,
                                   mask_flag=directional_mask, attn_drop=attn_drop, proj_drop=drop)
        self.norm1_y = norm_layer(dim)

        # layer2 cross attn without mask
        self.norm3 = norm_layer(dim)
        self.norm3_y = norm_layer(dim)
        self.cross_attn = CrossAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, cross_attn_flag=True,
                                         mask_flag=False, attn_drop=attn_drop, proj_drop=drop)
        self.ls3 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # layer3 FFN can be found in Block.__init__()

    def forward(self, inputs):
        x, y = inputs
        assert y is not None, 'cross attn need input x and y tensors!'
        # layer1
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), self.norm1_y(y), attn_mask=None)))
        # layer2
        x = x + self.drop_path3(self.ls3(self.cross_attn(self.norm3(x), self.norm3_y(y), attn_mask=None)))
        # layer3
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        outputs = (x, y)
        return outputs


class OadTransformer(nn.Module):
    def __init__(
        self,
        cfg,
        num_class,  # aka, out_dim 22
        num_tokens,  # 32/64 input previous frames
        decoder_query_frames=8,  # 8 prediction current/feature frames
        encoder_embedding_dim=512,  # 512
        decoder_embedding_dim=512,  # 512/1024
        encoder_num_heads=8,  # 8
        decoder_num_heads=4,  # 4
        encoder_layers=3,  # 3
        decoder_layers=5,  # 5
        encoder_mlp_ratio=4.0,  # 1.0/4.0
        decoder_mlp_ratio=4.0,  # 1.0/4.0
        dropout_rate=0.1,  # 0.1
        encoder_drop_path_rate=0.1,  # 0.1
        decoder_drop_path_rate=0.1,  # 0.1
        decoder_attn_dp=0.1,  # 0.1
        encoder_attn_dp=0.1,  # 0.1
        qkv_bias=True,
        class_token=True,
        directional_mask=True,
        encoder_block_fn=Block,
        decoder_block_fn=CrossAttnBlock,
    ):
        super(OadTransformer, self).__init__()

        assert encoder_embedding_dim % encoder_num_heads == 0 and decoder_embedding_dim % decoder_num_heads == 0
        self.zero_shot = cfg.zero_shot
        self.num_class = num_class
        self.add_fuse = cfg.add_fuse
        self.encoder_embedding_dim = encoder_embedding_dim
        self.decoder_embedding_dim = decoder_embedding_dim
        self.encoder_num_heads = encoder_num_heads
        self.decoder_num_heads = decoder_num_heads
        self.dropout_rate = dropout_rate
        self.encoder_attn_dp = encoder_attn_dp
        self.decoder_attn_dp = decoder_attn_dp

        self.global_tokens = 1 if class_token else 0
        self.seq_length = num_tokens + self.global_tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_embedding_dim))

        # Encoder
        self.encoder_position_encoding = nn.Parameter(torch.randn(1, self.seq_length, encoder_embedding_dim) * .02)
        self.encoder_pos_drop = nn.Dropout(p=dropout_rate)
        self.pre_head_ln = nn.LayerNorm(encoder_embedding_dim)

        # stochastic depth decay rule
        encoder_dpr = [x.item() for x in torch.linspace(0, encoder_drop_path_rate, encoder_layers)]
        self.encoder = nn.Sequential(*[
            encoder_block_fn(
                dim=encoder_embedding_dim, num_heads=encoder_num_heads, mlp_ratio=encoder_mlp_ratio, qkv_bias=qkv_bias,
                init_values=None, drop=dropout_rate, attn_drop=encoder_attn_dp, drop_path=encoder_dpr[i])
            for i in range(encoder_layers)])

        # Decoder
        self.decoder_position_encoding = nn.Parameter(torch.randn(1, self.seq_length, decoder_embedding_dim) * .02)
        self.decoder_pos_drop = nn.Dropout(p=dropout_rate)

        # stochastic depth decay rule
        decoder_dpr = [x.item() for x in torch.linspace(0, decoder_drop_path_rate, decoder_layers)]
        self.decoder = nn.Sequential(*[
            decoder_block_fn(
                dim=decoder_embedding_dim, num_heads=decoder_num_heads, mlp_ratio=decoder_mlp_ratio, qkv_bias=qkv_bias,
                init_values=None, drop=dropout_rate, attn_drop=decoder_attn_dp, drop_path=decoder_dpr[i],
                directional_mask=directional_mask)
            for i in range(decoder_layers)])

        self.decoder_cls_token = nn.Parameter(torch.zeros(1, decoder_query_frames, decoder_embedding_dim))

        if not self.zero_shot:
            self.classifier = nn.Linear(decoder_embedding_dim, self.num_class)  # for out logis
            self.loss_weight = cfg.criterion.loss_weight
            self.cross_entropy = nn.CrossEntropyLoss()

        self.after_dropout = nn.Dropout(p=dropout_rate)

        # encoder_embedding_dim/num_class for features or logits
        if self.zero_shot:
            if self.add_fuse:
                self.mlp_head = nn.Linear(encoder_embedding_dim + decoder_embedding_dim, encoder_embedding_dim)
            else:
                self.mlp_head = nn.Identity()
        else:
            self.mlp_head = nn.Linear(encoder_embedding_dim + decoder_embedding_dim, self.num_class)


    def forward(self, inputs):
        x, targets = inputs  # (x, labels)
        # x size: [1, 32, 512]
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((x, cls_tokens), dim=1)  # [1, 33, 512]
        x = self.encoder_pos_drop(x + self.encoder_position_encoding)

        # encoder
        x = self.encoder(x)
        x = self.pre_head_ln(x)  # [1, 33, 512]

        # decoder
        decoder_cls_token = self.decoder_cls_token.expand(x.shape[0], -1, -1)  # [1, 8, 512]
        decoder_inputs = (decoder_cls_token, x)
        dec, _ = self.decoder(decoder_inputs)   # [1, 8, 512]
        dec = self.after_dropout(dec)  # [1, 8, 512]

        dec_for_token = dec.mean(dim=1)  # AP [BS, num_class] [1, 512]

        if not self.zero_shot:
            dec_cls_out = self.classifier(dec)
            dec_cls_out_ = dec_cls_out.view(-1, self.num_class)  # prediction [BS, query_frames, num_class]  


        if self.zero_shot and not self.add_fuse:
            x = x[:, -1] + dec_for_token
        else:
            x = torch.cat((x[:, -1], dec_for_token), dim=1)  # [BS, enc_dim + dec_dim] [1, 1024]

        x = self.mlp_head(x)  # [BS, dim] / [B, Cls]

        # embed()
        # exit()
        if not self.zero_shot:
            if not self.training:
                return torch.cat((x.unsqueeze(dim=1), dec_cls_out), dim=1) # [B, 9, 21]
            losses_enc = self.cross_entropy(x, targets[:,0,:]) * self.loss_weight.enc_vtc
            losses_dec = self.cross_entropy(dec_cls_out_, targets[:,1:,:].reshape(-1, self.num_class)) * self.loss_weight.dec_vtc
            out = dict(loss_enc_vtc=losses_enc, loss_dec_vtc=losses_dec,)
            return out  
        else:
            # [BS, cur_1 + query_frames, dim][1, 9, 512]
            out = torch.cat((x.unsqueeze(dim=1), dec), dim=1)
            return out


if __name__ == "__main__":

    model = OadTransformer()
    x_train = torch.randn(1, 32, 512).cuda()  # BxTxC
    out = model(x_train)
    print(out.shape)
    pass


