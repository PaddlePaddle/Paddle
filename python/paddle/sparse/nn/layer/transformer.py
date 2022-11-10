#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# TODO: define the classes of Transformer neural network

import copy
import collections
import numpy as np

import paddle
from paddle.nn import Linear, Dropout, LayerNorm
from .. import functional as F
from paddle import tensor
from paddle.nn import Layer, LayerList
from paddle.framework import ParamAttr
from paddle.fluid.data_feeder import convert_dtype

__all__ = []


def _convert_param_attr_to_list(param_attr, n):
    """
    If `param_attr` is a list or tuple, convert every element in it to a
    ParamAttr instance. Otherwise, repeat `param_attr` `n` times to
    construct a list, and rename every one by appending a increasing index
    suffix to avoid having same names when `param_attr` contains a name.

    Parameters:
        param_attr (list|tuple|ParamAttr): A list, tuple or something can be
            converted to a ParamAttr instance by `ParamAttr._to_attr`.
        n (int): The times to repeat to construct a list when `param_attr`
            is not a list or tuple.

    Returns:
        list: A list composed of each including cell's `param_attr`.
    """
    if isinstance(param_attr, (list, tuple)):
        assert len(param_attr) == n, (
            "length of param_attr should be %d when it is a list/tuple" % n
        )
        param_attrs = []
        for attr in param_attr:
            if isinstance(attr, bool):
                if attr:
                    param_attrs.append(ParamAttr._to_attr(None))
                else:
                    param_attrs.append(False)
            else:
                param_attrs.append(ParamAttr._to_attr(attr))
        # param_attrs = [ParamAttr._to_attr(attr) for attr in param_attr]
    elif isinstance(param_attr, bool):
        param_attrs = []
        if param_attr:
            param_attrs = [ParamAttr._to_attr(None) for i in range(n)]
        else:
            param_attrs = [False] * n
    else:
        param_attrs = []
        attr = ParamAttr._to_attr(param_attr)
        for i in range(n):
            attr_i = copy.deepcopy(attr)
            if attr.name:
                attr_i.name = attr_i.name + "_" + str(i)
            param_attrs.append(attr_i)
    return param_attrs


def _convert_attention_mask(attn_mask, dtype):
    """
    Convert the attention mask to the target dtype we expect.

    Parameters:
        attn_mask (Tensor, optional): A tensor used in multi-head attention
                to prevents attention to some unwanted positions, usually the
                paddings or the subsequent positions. It is a tensor with shape
                broadcasted to `[batch_size, n_head, sequence_length, sequence_length]`.
                When the data type is bool, the unwanted positions have `False`
                values and the others have `True` values. When the data type is
                int, the unwanted positions have 0 values and the others have 1
                values. When the data type is float, the unwanted positions have
                `-INF` values and the others have 0 values. It can be None when
                nothing wanted or needed to be prevented attention to. Default None.
        dtype (VarType): The target type of `attn_mask` we expect.

    Returns:
        Tensor: A Tensor with shape same as input `attn_mask`, with data type `dtype`.
    """
    if attn_mask is not None and attn_mask.dtype != dtype:
        attn_mask_dtype = convert_dtype(attn_mask.dtype)
        if attn_mask_dtype == 'bool' or 'int' in attn_mask_dtype:
            attn_mask = (paddle.cast(attn_mask, dtype) - 1.0) * 1e9
        else:
            attn_mask = paddle.cast(attn_mask, dtype)
    return attn_mask


class SparseSelfAttention(Layer):
    """ """

    Cache = collections.namedtuple("Cache", ["k", "v"])
    StaticCache = collections.namedtuple("StaticCache", ["k", "v"])

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        kdim=None,
        vdim=None,
        need_weights=False,
        weight_attr=None,
        bias_attr=None,
    ):
        super(SparseSelfAttention, self).__init__()

        assert embed_dim > 0, (
            "Expected embed_dim to be greater than 0, "
            "but received {}".format(embed_dim)
        )
        assert num_heads > 0, (
            "Expected num_heads to be greater than 0, "
            "but received {}".format(num_heads)
        )

        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.need_weights = need_weights

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.q_proj = Linear(
            embed_dim, embed_dim, weight_attr, bias_attr=bias_attr
        )
        self.k_proj = Linear(
            self.kdim, embed_dim, weight_attr, bias_attr=bias_attr
        )
        self.v_proj = Linear(
            self.vdim, embed_dim, weight_attr, bias_attr=bias_attr
        )
        self.out_proj = Linear(
            embed_dim, embed_dim, weight_attr, bias_attr=bias_attr
        )

    def _prepare_qkv(self, query, key, value, cache=None):
        r""" """
        q = self.q_proj(query)
        q = tensor.reshape(x=q, shape=[0, 0, self.num_heads, self.head_dim])
        q = tensor.transpose(x=q, perm=[0, 2, 1, 3])

        k, v = self.compute_kv(key, value)

        return (q, k, v)

    def compute_kv(self, key, value):
        r""" """
        k = self.k_proj(key)
        v = self.v_proj(value)
        k = tensor.reshape(x=k, shape=[0, 0, self.num_heads, self.head_dim])
        k = tensor.transpose(x=k, perm=[0, 2, 1, 3])
        v = tensor.reshape(x=v, shape=[0, 0, self.num_heads, self.head_dim])
        v = tensor.transpose(x=v, perm=[0, 2, 1, 3])
        return k, v

    def forward(
        self,
        query,
        key=None,
        value=None,
        sparse_mask=None,
        attn_mask=None,
        cache=None,
    ):
        r""" """
        key = query if key is None else key
        value = query if value is None else value
        # compute q ,k ,v
        q, k, v = self._prepare_qkv(query, key, value, cache)

        # print(q.shape, k.shape, v.shape, sparse_mask.shape)
        attn_out = F.attention(
            q, k, v, sparse_mask, key_padding_mask=attn_mask.squeeze()
        )

        attn_out = tensor.transpose(attn_out, perm=[0, 2, 1, 3])
        attn_out = tensor.reshape(
            x=attn_out, shape=[0, 0, attn_out.shape[2] * attn_out.shape[3]]
        )

        return attn_out


class SparseTransformerEncoderLayer(Layer):
    """ """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        dropout=0.1,
        activation="relu",
        attn_dropout=None,
        act_dropout=None,
        normalize_before=False,
        weight_attr=None,
        bias_attr=None,
    ):
        self._config = locals()
        self._config.pop("self")
        self._config.pop("__class__", None)  # py3

        super(SparseTransformerEncoderLayer, self).__init__()

        assert (
            d_model > 0
        ), "Expected d_model to be greater than 0, " "but received {}".format(
            d_model
        )
        assert (
            nhead > 0
        ), "Expected nhead to be greater than 0, " "but received {}".format(
            nhead
        )
        assert dim_feedforward > 0, (
            "Expected dim_feedforward to be greater than 0, "
            "but received {}".format(dim_feedforward)
        )

        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        weight_attrs = _convert_param_attr_to_list(weight_attr, 2)
        bias_attrs = _convert_param_attr_to_list(bias_attr, 2)

        self.self_attn = SparseSelfAttention(
            d_model,
            nhead,
            dropout=attn_dropout,
            weight_attr=weight_attrs[0],
            bias_attr=bias_attrs[0],
        )
        self.linear1 = Linear(
            d_model, dim_feedforward, weight_attrs[1], bias_attr=bias_attrs[1]
        )
        self.dropout = Dropout(act_dropout, mode="upscale_in_train")
        self.linear2 = Linear(
            dim_feedforward, d_model, weight_attrs[1], bias_attr=bias_attrs[1]
        )
        self.pre_norm = LayerNorm(d_model)
        self.post_norm = LayerNorm(d_model)
        self.dropout = Dropout(dropout, mode="upscale_in_train")
        self.activation = getattr(paddle.nn.functional, activation)

    def forward(self, src, sparse_mask, src_mask=None, cache=None):
        r""" """
        src_mask = _convert_attention_mask(src_mask, src.dtype)

        residual = src

        src = self.pre_norm(src)
        # Add cache for encoder for the usage like UniLM
        attn_out = self.self_attn(src, src, src, sparse_mask, src_mask)

        intermediate_input = residual + attn_out

        intermediate_norm = self.post_norm(intermediate_input)

        # intermediate
        intermediate_out = self.linear1(intermediate_norm)
        intermediate_out = self.activation(intermediate_out)
        # bert output
        layer_out = self.linear2(intermediate_out)
        layer_out = self.dropout(layer_out)

        return layer_out + intermediate_input


class SparseTransformerEncoder(Layer):
    """ """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(SparseTransformerEncoder, self).__init__()
        self.layers = LayerList(
            [
                (
                    encoder_layer
                    if i == 0
                    else type(encoder_layer)(**encoder_layer._config)
                )
                for i in range(num_layers)
            ]
        )
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, sparse_mask, src_mask=None, cache=None):
        r""" """
        src_mask = _convert_attention_mask(src_mask, src.dtype)

        output = src
        new_caches = []
        for i, mod in enumerate(self.layers):
            output = mod(output, sparse_mask, src_mask=src_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output if cache is None else (output, new_caches)


class SparseTransformer(Layer):
    """ """

    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        attn_dropout=None,
        act_dropout=None,
        normalize_before=False,
        weight_attr=None,
        bias_attr=None,
        custom_encoder=None,
        custom_decoder=None,
    ):
        super(SparseTransformer, self).__init__()

        assert (
            d_model > 0
        ), "Expected d_model to be greater than 0, " "but received {}".format(
            d_model
        )
        assert (
            nhead > 0
        ), "Expected nhead to be greater than 0, " "but received {}".format(
            nhead
        )
        assert dim_feedforward > 0, (
            "Expected dim_feedforward to be greater than 0, "
            "but received {}".format(dim_feedforward)
        )

        if isinstance(bias_attr, (list, tuple)):
            if len(bias_attr) == 1:
                encoder_bias_attr = [bias_attr[0]] * 2
                decoder_bias_attr = [bias_attr[0]] * 3
            elif len(bias_attr) == 2:
                encoder_bias_attr = bias_attr
                decoder_bias_attr = [bias_attr[0], bias_attr[0], bias_attr[-1]]
            elif len(bias_attr) == 3:
                encoder_bias_attr = [bias_attr[0], bias_attr[-1]]
                decoder_bias_attr = bias_attr
            else:
                assert (
                    False
                ), "length of bias_attr should be 1 or 2 or 3 when it is a list/tuple"
        else:
            encoder_bias_attr = bias_attr
            decoder_bias_attr = bias_attr

        if isinstance(weight_attr, (list, tuple)):
            if len(weight_attr) == 1:
                encoder_weight_attr = [weight_attr[0]] * 2
                decoder_weight_attr = [weight_attr[0]] * 3
            elif len(weight_attr) == 2:
                encoder_weight_attr = weight_attr
                decoder_weight_attr = [
                    weight_attr[0],
                    weight_attr[0],
                    weight_attr[-1],
                ]
            elif len(weight_attr) == 3:
                encoder_weight_attr = [weight_attr[0], weight_attr[-1]]
                decoder_weight_attr = weight_attr
            else:
                assert (
                    False
                ), "length of weight_attr should be 1 or 2 or 3 when it is a list/tuple"
        else:
            encoder_weight_attr = weight_attr
            decoder_weight_attr = weight_attr

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(
                d_model,
                nhead,
                dim_feedforward,
                dropout,
                activation,
                attn_dropout,
                act_dropout,
                normalize_before,
                encoder_weight_attr,
                encoder_bias_attr,
            )
            encoder_norm = LayerNorm(d_model)
            self.encoder = TransformerEncoder(
                encoder_layer, num_encoder_layers, encoder_norm
            )

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(
                d_model,
                nhead,
                dim_feedforward,
                dropout,
                activation,
                attn_dropout,
                act_dropout,
                normalize_before,
                decoder_weight_attr,
                decoder_bias_attr,
            )
            decoder_norm = LayerNorm(d_model)
            self.decoder = TransformerDecoder(
                decoder_layer, num_decoder_layers, decoder_norm
            )

        self.d_model = d_model
        self.nhead = nhead

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        r""" """
        src_mask = _convert_attention_mask(src_mask, src.dtype)
        memory = self.encoder(src, src_mask=src_mask)

        tgt_mask = _convert_attention_mask(tgt_mask, tgt.dtype)
        memory_mask = _convert_attention_mask(memory_mask, memory.dtype)
        output = self.decoder(
            tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask
        )
        return output

    def generate_square_subsequent_mask(self, length):
        """ """
        return paddle.tensor.triu(
            paddle.full(
                shape=[length, length],
                fill_value=-np.inf,
                dtype=paddle.get_default_dtype(),
            ),
            1,
        )
