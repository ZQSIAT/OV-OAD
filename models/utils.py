# -------------------------------------------------------------------------
# MIT License
#
# Copyright (c) 2021 OpenAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Modified by Jiarui Xu
# -------------------------------------------------------------------------

from collections import OrderedDict

import torch
from torch import nn
from .adapter import Adapter

class QuickGELU(nn.Module):

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):

    def __init__(self, tune_config, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ('c_fc', nn.Linear(d_model, d_model * 4)), 
            ('gelu', QuickGELU()),
            ('c_proj', nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.tune_config = tune_config
        if self.tune_config is not None:
            self.adaptmlp = Adapter(tune_config, dropout=0.1, bottleneck=tune_config.ffn_num,
                                    init_option=tune_config.ffn_adapter_init_option,
                                    adapter_scalar=tune_config.ffn_adapter_scalar,
                                    adapter_layernorm_option=tune_config.ffn_adapter_layernorm_option,
                                    )

    def attention(self, x: torch.Tensor, key_padding_mask: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask, key_padding_mask=key_padding_mask)[0]

    def forward(self, x: torch.Tensor, key_padding_mask=None):
        x = x + self.attention(self.ln_1(x), key_padding_mask=key_padding_mask)
        if self.tune_config is not None and self.tune_config.ffn_option == 'parallel':
            adapt_x = self.adaptmlp(x, add_residual=False)
        residual = x
        x = self.mlp(self.ln_2(x))
        if self.tune_config is not None and self.tune_config.ffn_option == 'parallel':
            x = x + adapt_x
        x = residual + x
        # x = x + self.mlp(self.ln_2(x))
        return x
