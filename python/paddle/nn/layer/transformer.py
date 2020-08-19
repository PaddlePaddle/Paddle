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
# __all__ = [ ]

import copy
import collections

import numpy as np

from ...fluid import layers
from ...fluid.param_attr import ParamAttr
from ...fluid.dygraph import Layer, Linear, Dropout, LayerNorm, LayerList
from .. import functional as F
from ...fluid.layers import utils
from ...fluid.layers.utils import map_structure


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
            "length of param_attr should be %d when it is a list/tuple" % n)
        param_attrs = [ParamAttr._to_attr(attr) for attr in param_attr]
    else:
        param_attrs = []
        attr = ParamAttr._to_attr(param_attr)
        for i in range(n):
            attr_i = copy.deepcopy(attr)
            if attr.name:
                attr_i.name = attr_i.name + "_" + str(i)
            param_attrs.append(attr_i)
    return param_attrs


class MultiheadAttention(Layer):
    """
    Attention mapps queries and a set of key-value pairs to outputs, and
    Multi-Head Attention performs multiple parallel attention to jointly attending
    to information from different representation subspaces.

    Please refer to `Attention Is All You Need <https://arxiv.org/pdf/1706.03762.pdf>`_
    for more details.

    Parameters:
        embed_dim (int): The expected feature size in the input and output.
        num_heads (int): The number of heads in multi-head attention.
        dropout (float, optional): The dropout probability used on attention
            weights to drop some attention targets. 0 for no dropout. Default 0
        kdim (int, optional): The feature size in key. If None, assumed equal to
            `embed_dim`. Default None.
        vdim (int, optional): The feature size in value. If None, assumed equal to
            `embed_dim`. Default None.
        need_weights (bool, optional): Indicate whether to return the attention
            weights. Default False.
        param_attr(ParamAttr, optional):  To specify the weight parameter property.
            Default: None, which means the default weight parameter property is used.
            See usage for details in :ref:`api_fluid_ParamAttr` .
        bias_attr (ParamAttr, optional): To specify the bias parameter property.
            Default: None, which means the default bias parameter property is used.
            See usage for details in :ref:`api_fluid_ParamAttr` .
         
    Examples:

        .. code-block:: python

            import paddle

            # encoder input: [batch_size, sequence_length, d_model]
            query = paddle.rand((2, 4, 128))
            # self attention mask: [batch_size, num_heads, query_len, query_len]
            attn_mask = paddle.rand((2, 2, 4, 4))
            multi_head_attn = paddle.MultiheadAttention(64, 64, 128, n_head=2)
            output = multi_head_attn(query, attn_mask=attn_mask)  # [2, 4, 128]
    """

    Cache = collections.namedtuple("Cache", ["k", "v"])
    StaticCache = collections.namedtuple("StaticCache", ["k", "v"])

    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.,
                 kdim=None,
                 vdim=None,
                 need_weights=False,
                 param_attr=None,
                 bias_attr=None):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.need_weights = need_weights

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = Linear(
            input_dim=embed_dim,
            output_dim=embed_dim,
            param_attr=param_attr,
            bias_attr=bias_attr)
        self.k_proj = Linear(
            input_dim=self.kdim,
            output_dim=embed_dim,
            param_attr=param_attr,
            bias_attr=bias_attr)
        self.v_proj = Linear(
            input_dim=self.vdim,
            output_dim=embed_dim,
            param_attr=param_attr,
            bias_attr=bias_attr)
        self.out_proj = Linear(
            input_dim=embed_dim,
            output_dim=embed_dim,
            param_attr=param_attr,
            bias_attr=bias_attr)

    def _prepare_qkv(self, query, key, value, cache=None):
        """
        Prapares linear projected queries, keys and values for usage of subsequnt
        multiple parallel attention. If `cache` is not None, using cached results
        to reduce redundant calculations.

        Parameters:
            query (Variable): The queries for multi-head attention. It is a
                tensor with shape `[batch_size, query_length, embed_dim]`. The
                data type should be float32 or float64.
            key (Variable): The keys for multi-head attention. It is
                a tensor with shape `[batch_size, key_length, kdim]`. The
                data type should be float32 or float64. If None, use `query` as
                `key`.
            value (Variable): The values for multi-head attention. It
                is a tensor with shape `[batch_size, value_length, vdim]`.
                The data type should be float32 or float64. If None, use `query` as
                `value`.
            cache (MultiheadAttention.Cache|MultiheadAttention.StaticCache, optional):
                It is a namedtuple with `k` and `v` as fields, and stores tensors
                shaped `[batch_size, num_heads, length, embed_dim]` which are results
                of linear projection, reshape and transpose calculations in
                MultiheadAttention. If is an instance of `Cache`, `k` and `v`
                fields reserve intermediate results of previous positions, which
                mostly used for decoder self attention. If it is an instance of
                `StaticCache`, `key` and `value` args would be ignored, `k` and
                `v` fields would be used as calculated results on `key` and
                `value`, which mostly used for decoder-encoder cross attention.
                It is only used for inference and should be None for training.
                Default None.

        Returns:
            tuple: A tuple including linear projected keys and values. These two \
                tensors have shapes `[batch_size, n_head, sequence_length, d_key]` \
                and `[batch_size, n_head, sequence_length, d_value]` separately, \
                and their data types are same as inputs.
        """
        q = self.q_proj(query)
        q = layers.reshape(x=q, shape=[0, 0, self.n_head, self.d_key])
        q = layers.transpose(x=q, perm=[0, 2, 1, 3])

        if isinstance(cache, self.StaticCache):
            # for encoder-decoder attention in inference and has cached
            k, v = cache.k, cache.v
        else:
            k, v = self.cal_kv(key, value)

        if isinstance(cache, self.Cache):
            # for decoder self-attention in inference
            k = layers.concat([cache.k, k], axis=2)
            v = layers.concat([cache.v, v], axis=2)
            cache = self.Cache(k, v)

        return (q, k, v) if cache is None else (q, k, v, cache)

    def cal_kv(self, key, value):
        """
        Applies linear projection on input keys and values, then splits heads
        (reshape and transpose) to get keys and values from different representation
        subspaces. The results are used as key-values pairs for subsequent multiple
        parallel attention.
        
        It is part of calculations in multi-head attention, and is provided as
        a method to pre-compute and prefetch these results, thus we can use them
        to construct cache for inference.

        Parameters:
            key (Variable, optional): The keys for multi-head attention. It is
                a tensor with shape `[batch_size, sequence_length, kdim]`. The
                data type should be float32 or float64.
            value (Variable, optional): The values for multi-head attention. It
                is a tensor with shape `[batch_size, sequence_length, vdim]`.
                The data type should be float32 or float64.

        Returns:
            tuple: A tuple including transformed keys and values. Their shapes \
                both are `[batch_size, num_heads, sequence_length, embed_dim // num_heads]`, \
                and their data types are same as inputs.
        """
        k = self.k_proj(key)
        v = self.v_proj(value)
        k = layers.reshape(x=k, shape=[0, 0, self.num_heads, self.head_dim])
        k = layers.transpose(x=k, perm=[0, 2, 1, 3])
        v = layers.reshape(x=v, shape=[0, 0, self.num_heads, self.head_dim])
        v = layers.transpose(x=v, perm=[0, 2, 1, 3])
        return k, v

    def gen_cache(self, key, value=None, type=MultiheadAttention.Cache):
        """
        Generates cache for `forward` usage in inference accroding to arguments.
        The generated cache is an instance of `MultiheadAttention.Cache` or an
        instance of `MultiheadAttention.StaticCache`.

        `Cache` or `StaticCache` is namedtuple with `k` and `v` as fields,
        and it stores tensors shaped `[batch_size, num_heads, length, embed_dim]`
        which are results of linear projection, reshape and transpose calculations
        in MultiheadAttention.
        
        If the generated cache is an instance of `Cache`, `k` and `v` fields
        reserve intermediate result tensors of previous positions, and the tensors
        are incremental among decoding steps, which mostly are used for decoder
        decoder self attention.
        
        If the generated cache is an instance of `StaticCache`, `k` and `v` fields
        would be used as calculated result tensors on keys an values in `forward`,
        and the tensors keep unchanged among decoding steps, which are mostly used
        for decoder-encoder cross attention.

        The cache is generated as follows:

        1. If `type` is `StaticCache`, apply `cal_kv(key, value)` and use the results
        to create an instance of `StaticCache`.
        
        2. If `type` is `Cache` and `value` is None, generate empty tensors shaped
        `[batch_size, num_heads, 0, embed_dim // num_heads]` and use the results
        to create an instance of `Cache`, where `batch_size` is from the first
        dimension of `key`.

        3. If `type` is `Cache` and `value` is not None, use `key`, `value` to create
        an instance of `Cache`.

        Parameters:
            key (Variable): The keys for multi-head attention. It is
                a tensor with shape `[batch_size, key_length, kdim]`. The
                data type should be float32 or float64. If `value` is None,
                it is only for batch size and data type reference.
            value (Variable, optional): The values for multi-head attention. It
                is a tensor with shape `[batch_size, value_length, vdim]`.
                The data type should be float32 or float64. If None, `key` is only
                for batch size reference. Default None.
            type (type): It should be `MultiheadAttention.StaticCache` or
                `MultiheadAttention.Cache` to indicate the cache type to generate.
        
        Returns:
            namedtuple: an instance of `Cache` or `StaticCache` accordingly.
        """
        if type == MultiheadAttention.StaticCache:  # static_kv
            k, v = self.cal_kv(key, value)
            return self.StaticCache(k, v)
        elif value is None:  # incremental_state
            k = layers.fill_constant_batch_size_like(
                input=key,
                shape=[-1, self.num_heads, 0, self.head_dim],
                dtype=key.dtype,
                value=0)
            v = layers.fill_constant_batch_size_like(
                input=key,
                shape=[-1, self.num_heads, 0, self.head_dim],
                dtype=key.dtype,
                value=0)
            return self.Cache(k, v)
        else:
            # incremental_state with initial value, mainly for usage like UniLM
            return self.Cache(key, value)

    def forward(self, query, key, value, attn_mask=None, cache=None):
        """
        Applies multi-head attention to map queries and a set of key-value pairs
        to outputs.

        Parameters:
            query (Variable): The queries for multi-head attention. It is a
                tensor with shape `[batch_size, query_length, embed_dim]`. The
                data type should be float32 or float64.
            key (Variable, optional): The keys for multi-head attention. It is
                a tensor with shape `[batch_size, key_length, kdim]`. The
                data type should be float32 or float64. If None, use `query` as
                `key`. Default None.
            value (Variable, optional): The values for multi-head attention. It
                is a tensor with shape `[batch_size, value_length, vdim]`.
                The data type should be float32 or float64. If None, use `query` as
                `value`. Default None.
            attn_mask (Variable, optional): A tensor used in multi-head attention
                to prevents attention to some unwanted positions, usually the
                paddings or the subsequent positions. It is a tensor with shape
                broadcasted to `[batch_size, n_head, sequence_length, sequence_length]`,
                where the unwanted positions have `-INF` values and the others
                have 0 values. The data type should be float32 or float64. It can
                be None when nothing wanted or needed to be prevented attention to.
                Default None
            cache (MultiheadAttention.Cache|MultiheadAttention.StaticCache, optional):
                It is a namedtuple with `k` and `v` as fields, and stores tensors
                shaped `[batch_size, num_heads, length, embed_dim]` which are results
                of linear projection, reshape and transpose calculations in
                MultiheadAttention. If it is an instance of `Cache`, `k` and `v`
                fields reserve intermediate results of previous positions, which
                mostly used for decoder self attention. If it is an instance of
                `StaticCache`, `key` and `value` args would be ignored, `k` and
                `v` fields would be used as calculated results on `key` and
                `value`, which mostly used for decoder-encoder cross attention.
                It is only used for inference and should be None for training.
                Default None.

        Returns:
            Variable|tuple: It is a tensor that has the same shape and data type \
                as `query`, representing attention output. Or a tuple if \
                `need_weights` is True or `cache` is not None. If `need_weights` \
                is True, except for attention output, the tuple also includes \
                the attention weights tensor shaped `[batch_size, num_heads, query_length, key_length]`. \
                If `cache` is not None, the tuple then includes the new cache \
                having the same type as `cache`, and if it is `StaticCache`, it \
                is same as the input `cache`, if it is `Cache`, the new cache \
                reserves tensors concatanating raw tensors with intermediate \
                results of current query.
        """
        key = query if key is None else key
        value = query if value is None else value
        # compute q ,k ,v
        if cache is None:
            q, k, v = self._prepare_qkv(query, key, value, cache)
        else:
            q, k, v, cache = self._prepare_qkv(query, key, value, cache)

        # scale dot product attention
        product = layers.matmul(
            x=q, y=k, transpose_y=True, alpha=self.head_dim**-0.5)
        if attn_mask is not None:
            # TODO(guosheng): support bool mask
            product = product + attn_mask
        weights = layers.softmax(product)
        if self.dropout:
            weights = layers.dropout(
                weights,
                dropout_prob=self.dropout,
                dropout_implementation="upscale_in_train",
                is_test=False)

        out = layers.matmul(weights, v)

        # combine heads
        out = layers.transpose(out, perm=[0, 2, 1, 3])
        out = layers.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

        # project to output
        out = self.out_proj(out)

        outs = [out]
        if self.need_weights:
            outs.append(weights)
        if cache is not None:
            outs.append(cache)
        return out if len(outs) else tuple(outs)


class TransformerEncoderLayer(Layer):
    """
    TransformerEncoderLayer is composed of two sub-layers which are self (multi-head)
    attention and feedforward network. Before and after each sub-layer, pre-process
    and post-precess would be applied on the input and output accordingly. If
    `normalize_before` is True, pre-process is layer normalization and post-precess
    includes dropout, residual connection. Otherwise, no pre-process and post-precess
    includes dropout, residual connection, layer normalization.

    Parameters:
        d_model (int): The expected feature size in the input and output.
        nhead (int): The number of heads in multi-head attention(MHA).
        dim_feedforward (int): The hidden layer size in the feedforward network(FFN).
        dropout (float, optional): The dropout probability used in pre-process
            and post-precess of MHA and FFN sub-layer. Default 0.1
        activation (str, optional): The activation function in the feedforward
            network. Default relu.
        attn_dropout (float, optional): The dropout probability used
            in MHA to drop some attention target. If None, use the value of
            `dropout`. Default None
        act_dropout (float, optional): The dropout probability used after FFN
            activition.  If None, use the value of `dropout`. Default None
        normalize_before (bool, optional): Indicate whether to put layer normalization
            into preprocessing of MHA and FFN sub-layers. If True, pre-process is layer
            normalization and post-precess includes dropout, residual connection.
            Otherwise, no pre-process and post-precess includes dropout, residual
            connection, layer normalization. Default False
        param_attr(ParamAttr|tuple, optional): To specify the weight parameter property.
            If it is a tuple, `param_attr[0]` would be used as `param_attr` for
            MHA, and `param_attr[1]` would be used as `param_attr` for linear in FFN.
            Otherwise, MHA and FFN both use it as `param_attr` to create parameters.
            Default: None, which means the default weight parameter property is used.
            See usage for details in :ref:`api_fluid_ParamAttr` . 
        bias_attr (ParamAttr|tuple, optional): To specify the bias parameter property.
            If it is a tuple, `bias_attr[0]` would be used as `bias_attr` for
            MHA, and `bias_attr[1]` would be used as `bias_attr` for linear in FFN.
            Otherwise, MHA and FFN both use it as `bias_attr` to create parameters.
            Default: None, which means the default bias parameter property is used.
            See usage for details in :ref:`api_fluid_ParamAttr` .

    Examples:

        .. code-block:: python

            import paddle
            from paddle import TransformerEncoderLayer

            # encoder input: [batch_size, src_len, d_model]
            enc_input = paddle.rand((2, 4, 128))
            # self attention mask: [batch_size, n_head, src_len, src_len]
            attn_mask = paddle.rand((2, 2, 4, 4))
            encoder_layer = TransformerEncoderLayer(128, 2, 512)
            enc_output = encoder_layer(enc_input, attn_mask)  # [2, 4, 128]
    """

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward,
                 dropout=0.1,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False,
                 param_attr=None,
                 bias_attr=None):
        self._config = locals()
        self._config.pop("self")

        super(TransformerEncoderLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        param_attrs = _convert_param_attr_to_list(param_attr, 2)
        bias_attrs = _convert_param_attr_to_list(bias_attr, 2)

        self.self_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=attn_dropout,
            param_attr=param_attrs[0],
            bias_attr=bias_attrs[0])
        self.linear1 = Linear(
            d_model,
            dim_feedforward,
            param_attr=param_attrs[1],
            bias_attr=bias_attrs[1])
        self.dropout = Dropout(
            act_dropout, dropout_implementation="upscale_in_train")
        self.linear2 = Linear(
            dim_feedforward,
            d_model,
            param_attr=param_attrs[1],
            bias_attr=bias_attrs[1])
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(
            dropout, dropout_implementation="upscale_in_train")
        self.dropout2 = Dropout(
            dropout, dropout_implementation="upscale_in_train")
        self.activation = getattr(layers, activation)

    def forward(self, src, src_mask=None):
        """
        Applies a Transformer encoder layer on the input.

        Parameters:
            src (Variable): The input of Transformer encoder layer. It is
                a tensor with shape `[batch_size, sequence_length, d_model]`.
                The data type should be float32 or float64.
            src_mask (Variable, optional): A tensor used in multi-head attention
                to prevents attention to some unwanted positions, usually the
                paddings or the subsequent positions. It is a tensor with shape
                broadcasted to `[batch_size, n_head, sequence_length, sequence_length]`,
                where the unwanted positions have `-INF` values and the others
                have 0 values. The data type should be float32 or float64. It can
                be None when nothing wanted or needed to be prevented attention to.
                Default None

        Returns:
            Variable: The output of Transformer encoder layer. It is a tensor that \
                has the same shape and data type as `enc_input`.
        """
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        # TODO(guosheng): Add cache for encoder for the usage like UniLM
        src = self.self_attn(src, src, src, src_mask)
        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(Layer):
    """
    TransformerEncoder is a stack of N encoder layers. 

    Parameters:
        encoder_layer (Layer): an instance of the `TransformerEncoderLayer`. It
            would be used as the first layer, and the other layers would be created
            according to the configurations of it.
        num_layers (int): The number of encoder layers to be stacked.
        norm (LayerNorm, optional): the layer normalization component. If provided,
            apply layer normalization on the output of last encoder layer.

    Examples:

        .. code-block:: python

            import paddle
            from paddle import TransformerEncoderLayer, TransformerEncoder

            # encoder input: [batch_size, src_len, d_model]
            enc_input = paddle.rand((2, 4, 128))
            # self attention mask: [batch_size, n_head, src_len, src_len]
            attn_mask = paddle.rand((2, 2, 4, 4))
            encoder_layer = TransformerEncoderLayer(2, 64, 64, 128, 512)
            encoder = TransformerEncoder(encoder_layer, 2)
            enc_output = encoder(enc_input, attn_mask)  # [2, 4, 128]
    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = LayerList([(encoder_layer if i == 0 else
                                  type(encoder_layer)(encoder_layer._config))
                                 for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None):
        """
        Applies a stack of N Transformer encoder layers on inputs. If `norm` is
        provided, also applies layer normalization on the output of last encoder
        layer.

        Parameters:
            src (Variable): The input of Transformer encoder. It is a tensor
                with shape `[batch_size, sequence_length, d_model]`. The data
                type should be float32 or float64.
            src_mask (Variable, optional): A tensor used in multi-head attention
                to prevents attention to some unwanted positions, usually the
                paddings or the subsequent positions. It is a tensor with shape
                broadcasted to `[batch_size, n_head, sequence_length, sequence_length]`,
                where the unwanted positions have `-INF` values and the others
                have 0 values. The data type should be float32 or float64. It can
                be None when nothing wanted or needed to be prevented attention to.
                Default None

        Returns:
            Variable: The output of Transformer encoder. It is a tensor that \
                has the same shape and data type as `src`.
        """
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=src_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoderLayer(Layer):
    """
    TransformerDecoderLayer is composed of three sub-layers which are decoder
    self (multi-head) attention, decoder-encoder cross attention and feedforward
    network. Before and after each sub-layer, pre-process and post-precess would
    be applied on the input and output accordingly. If `normalize_before` is True,
    pre-process is layer normalization and post-precess includes dropout, residual
    connection. Otherwise, no pre-process and post-precess includes dropout, residual
    connection, layer normalization.

    Parameters:
        d_model (int): The expected feature size in the input and output.
        nhead (int): The number of heads in multi-head attention(MHA).
        dim_feedforward (int): The hidden layer size in the feedforward network(FFN).
        dropout (float, optional): The dropout probability used in pre-process
            and post-precess of MHA and FFN sub-layer. Default 0.1
        activation (str, optional): The activation function in the feedforward
            network. Default relu.
        attn_dropout (float, optional): The dropout probability used
            in MHA to drop some attention target. If None, use the value of
            `dropout`. Default None
        act_dropout (float, optional): The dropout probability used after FFN
            activition.  If None, use the value of `dropout`. Default None
        normalize_before (bool, optional): Indicate whether to put layer normalization
            into preprocessing of MHA and FFN sub-layers. If True, pre-process is layer
            normalization and post-precess includes dropout, residual connection.
            Otherwise, no pre-process and post-precess includes dropout, residual
            connection, layer normalization. Default False
        param_attr(ParamAttr|tuple, optional): To specify the weight parameter property.
            If it is a tuple, `param_attr[0]` would be used as `param_attr` for
            self attention, `param_attr[1]` would be used as `param_attr` for
            cross attention, and `param_attr[2]` would be used as `param_attr`
            for linear in FFN. Otherwise, the three sub-layers all uses it as
            `param_attr` to create parameters. Default: None, which means the
            default weight parameter property is used. See usage for details
            in :ref:`api_fluid_ParamAttr` . 
        bias_attr (ParamAttr|tuple, optional): To specify the bias parameter property.
            If it is a tuple, `bias_attr[0]` would be used as `bias_attr` for
            self attention, `bias_attr[1]` would be used as `bias_attr` for
            cross attention, and `bias_attr[2]` would be used as `bias_attr`
            for linear in FFN. Otherwise, the three sub-layers all uses it as
            `bias_attr` to create parameters. Default: None, which means the
            default bias parameter property is used. See usage for details
            in :ref:`api_fluid_ParamAttr` .

    Examples:

        .. code-block:: python

            import paddle
            from paddle import TransformerDecoderLayer

            # decoder input: [batch_size, tgt_len, d_model]
            dec_input = paddle.rand((2, 4, 128))
            # encoder output: [batch_size, src_len, d_model]
            enc_output = paddle.rand((2, 6, 128))
            # self attention mask: [batch_size, n_head, tgt_len, tgt_len]
            self_attn_mask = paddle.rand((2, 2, 4, 4))
            # cross attention mask: [batch_size, n_head, tgt_len, src_len]
            cross_attn_mask = paddle.rand((2, 2, 4, 6))
            decoder_layer = TransformerDecoderLayer(128, 2, 512)
            output = decoder_layer(dec_input,
                                   enc_output,
                                   self_attn_mask,
                                   cross_attn_mask)  # [2, 4, 128]
    """

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward,
                 dropout=0.1,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False,
                 param_attr=None,
                 bias_attr=None):
        self._config = locals()
        self._config.pop("self")

        super(TransformerDecoderLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        param_attrs = _convert_param_attr_to_list(param_attr, 3)
        bias_attrs = _convert_param_attr_to_list(bias_attr, 3)

        self.self_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=attn_dropout,
            param_attr=param_attrs[0],
            bias_attr=bias_attrs[0])
        self.cross_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=attn_dropout,
            param_attr=param_attrs[1],
            bias_attr=bias_attrs[1])
        self.linear1 = Linear(
            d_model,
            dim_feedforward,
            param_attr=param_attrs[2],
            bias_attr=bias_attrs[2])
        self.dropout = Dropout(
            act_dropout, dropout_implementation="upscale_in_train")
        self.linear2 = Linear(
            dim_feedforward,
            d_model,
            param_attr=param_attrs[2],
            bias_attr=bias_attrs[2])
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(
            dropout, dropout_implementation="upscale_in_train")
        self.dropout2 = Dropout(
            dropout, dropout_implementation="upscale_in_train")
        self.dropout3 = Dropout(
            dropout, dropout_implementation="upscale_in_train")
        self.activation = getattr(layers, activation)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, cache=None):
        """
        Applies a Transformer decoder layer on the input.

        Parameters:
            tgt (Variable): The input of Transformer decoder layer. It is a tensor
                with shape `[batch_size, target_length, d_model]`. The data type
                should be float32 or float64.
            memory (Variable): The output of Transformer encoder. It is a tensor
                with shape `[batch_size, source_length, d_model]`. The data type
                should be float32 or float64.
            tgt_mask (Variable, optional): A tensor used in self attention
                to prevents attention to some unwanted positions, usually the
                the subsequent positions. It is a tensor with shape broadcasted
                to `[batch_size, n_head, target_length, target_length]`,
                where the unwanted positions have `-INF` values and the others
                have 0 values. The data type should be float32 or float64. It can
                be None when nothing wanted or needed to be prevented attention to.
                Default None
            memory_mask (Variable, optional): A tensor used in decoder-encoder
                cross attention to prevents attention to some unwanted positions,
                usually the paddings. It is a tensor with shape broadcasted to
               `[batch_size, n_head, target_length, source_length]`, where the
                unwanted positions have `-INF` values and the others have 0 values.
                The data type should be float32 or float64. It can be None when
                nothing wanted or needed to be prevented attention to. Default None
            cache (tuple, optional): It is a tuple( :code:`(incremental_cache, static_cache)` ),
                `incremental_cache` is an instance of `MultiheadAttention.Cache`,
                `static_cache` is an instance of `MultiheadAttention.StaticCache.
                See `TransformerDecoderLayer.gen_cache` for more details. It is
                only used for inference and should be None for training. Default
                None.

        Returns:
            Variable|tuple: It is a tensor that has the same shape and data type \
                as `tgt`, representing the output of Transformer decoder layer. \
                Or a tuple if `cache` is not None, except for decoder layer output, \
                the tuple includes the new cache which is same as input `cache` \
                argument but `incremental_cache` in it has an incremental length. \
                See `MultiheadAttention.gen_cache` and `MultiheadAttention.forward` \
                for more details.
        """
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)
        if cache is None:
            tgt = self.self_attn(tgt, tgt, tgt, tgt_mask, None)
        else:
            tgt, incremental_cache = self.self_attn(tgt, tgt, tgt, tgt_mask,
                                                    cache[0])
        tgt = residual + self.dropout1(tgt)
        if not self.normalize_before:
            tgt = self.norm1(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm2(tgt)
        if cache is None:
            tgt = self.cross_attn(tgt, memory, memory, memory_mask, None)
        else:
            tgt, static_cache = self.cross_attn(tgt, memory, memory,
                                                memory_mask, cache[1])
        tgt = residual + self.dropout2(tgt)
        if not self.normalize_before:
            tgt = self.norm2(tgt)

        residual = tgt
        if self.normalize_before:
            tgt = self.norm3(tgt)
        tgt = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = residual + self.dropout3(tgt)
        if not self.normalize_before:
            tgt = self.norm3(tgt)
        return tgt if cache is None else (tgt, (incremental_cache,
                                                static_cache))

    def gen_cache(self, memory):
        """
        Generates cache for `forward` usage. The generated cache is a tuple
        composed of an instance of `MultiheadAttention.Cache` and an instance
        of `MultiheadAttention.StaticCache`.

        Parameters:
            memory (Variable): The output of Transformer encoder. It is a tensor
                with shape `[batch_size, source_length, d_model]`. The data type
                should be float32 or float64.

        Returns:
            tuple: It is a tuple( :code:`(incremental_cache, static_cache)` ). \
                `incremental_cache` is an instance of `MultiheadAttention.Cache` \
                produced by `self_attn.gen_cache(memory, MultiheadAttention.Cache)`, \
                it reserves two tensors shaped `[batch_size, nhead, 0, d_model // nhead]`. \
                `static_cache` is an instance of `MultiheadAttention.StaticCache` \
                produced by `cross_attn.gen_cache(memory, MultiheadAttention.StaticCache)`, \
                it reserves two tensors shaped `[batch_size, nhead, source_length, d_model // nhead]`.
                See `MultiheadAttention.gen_cache` and `MultiheadAttention.forward` \
                for more details.
        """
        incremental_cache = self.self_attn.gen_cache(
            memory, type=self.self_attn.Cache)
        static_cache = self.cross_attn.gen_cache(
            memory, memory, type=self.cross_attn.StaticCache)
        return incremental_cache, static_cache


class TransformerDecoder(Layer):
    """
    TransformerDecoder is a stack of N decoder layers. 

    Parameters:
        decoder_layer (Layer): an instance of the `TransformerDecoderLayer`. It
            would be used as the first layer, and the other layers would be created
            according to the configurations of it.
        num_layers (int): The number of decoder layers to be stacked.
        norm (LayerNorm, optional): the layer normalization component. If provided,
            apply layer normalization on the output of last encoder layer.

    Examples:

        .. code-block:: python

            import paddle
            from paddle import TransformerDecoderLayer, TransformerDecoder

            # decoder input: [batch_size, tgt_len, d_model]
            dec_input = paddle.rand((2, 4, 128))
            # encoder output: [batch_size, src_len, d_model]
            enc_output = paddle.rand((2, 6, 128))
            # self attention mask: [batch_size, n_head, tgt_len, tgt_len]
            self_attn_mask = paddle.rand((2, 2, 4, 4))
            # cross attention mask: [batch_size, n_head, tgt_len, src_len]
            cross_attn_mask = paddle.rand((2, 2, 4, 6))
            decoder_layer = TransformerDecoderLayer(128, 2, 512)
            decoder = TransformerDecoder(decoder_layer, 2)
            output = decoder(dec_input,
                             enc_output,
                             self_attn_mask,
                             cross_attn_mask)  # [2, 4, 128]
    """

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = LayerList([(decoder_layer if i == 0 else
                                  type(decoder_layer)(decoder_layer._config))
                                 for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, cache=None):
        """
        Applies a stack of N Transformer decoder layers on inputs. If `norm` is
        provided, also applies layer normalization on the output of last decoder
        layer.

        Parameters:
            tgt (Variable): The input of Transformer decoder. It is a tensor
                with shape `[batch_size, target_length, d_model]`. The data type
                should be float32 or float64.
            memory (Variable): The output of Transformer encoder. It is a tensor
                with shape `[batch_size, source_length, d_model]`. The data type
                should be float32 or float64.
            tgt_mask (Variable, optional): A tensor used in self attention
                to prevents attention to some unwanted positions, usually the
                the subsequent positions. It is a tensor with shape broadcasted
                to `[batch_size, n_head, target_length, target_length]`,
                where the unwanted positions have `-INF` values and the others
                have 0 values. The data type should be float32 or float64. It can
                be None when nothing wanted or needed to be prevented attention to.
                Default None
            memory_mask (Variable, optional): A tensor used in decoder-encoder
                cross attention to prevents attention to some unwanted positions,
                usually the paddings. It is a tensor with shape broadcasted to
               `[batch_size, n_head, target_length, source_length]`, where the
                unwanted positions have `-INF` values and the others have 0 values.
                The data type should be float32 or float64. It can be None when
                nothing wanted or needed to be prevented attention to. Default None
            cache (list, optional): It is a list, and each element in the list
                is a tuple( :code:`(incremental_cache, static_cache)` ). See
                `TransformerDecoder.gen_cache` for more details. It is only
                used for inference and should be None for training. Default None.

        Returns:
            Variable|tuple: It is a tensor that has the same shape and data type \
                as `tgt`, representing the output of Transformer decoder. \
                Or a tuple if `cache` is not None, except for decoder output, \
                the tuple includes the new cache which is same as input `cache` \
                argument but `incremental_cache` in it has an incremental length. \
                See `MultiheadAttention.gen_cache` and `MultiheadAttention.forward` \
                for more details.
        """
        output = tgt
        new_caches = []
        for i, mod in enumerate(self.layers):
            if cache is None:
                output = mod(output,
                             memory,
                             tgt_mask=tgt_mask,
                             memory_mask=memory_mask,
                             cache=None)
            else:
                output, new_cache = mod(output,
                                        memory,
                                        tgt_mask=tgt_mask,
                                        memory_mask=memory_mask,
                                        cache=cache[i])
                new_caches.append(new_cache)

        if self.norm is not None:
            output = self.norm(output)

        return output if cache is None else (output, new_caches)

    def gen_cache(self, memory):
        """
        Generates cache for `forward` usage. The generated cache is a list, and
        each element in it is a tuple( :code:`(incremental_cache, static_cache)` )
        produced by `TransformerDecoderLayer.gen_cache`. See `TransformerDecoderLayer.gen_cache`
        for more details.


        Parameters:
            memory (Variable): The output of Transformer encoder. It is a tensor
                with shape `[batch_size, source_length, d_model]`. The data type
                should be float32 or float64.

        Returns:
            list: It is a list, and each element in the list is a tuple produced \
                by `TransformerDecoderLayer.gen_cache(memory)`. See `TransformerDecoderLayer.gen_cache` \
                for more details.
        """
        return [layer.gen_cache(memory) for layer in self.layers]


class Transformer(Layer):
    """
    A Transformer model composed of an instance of `TransformerEncoder` and an
    instance of `TransformerDecoder`. While the embedding layer and output layer
    are not included.

    Please refer to `Attention is all you need <http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf>`_ ,
    and see `TransformerEncoder` and `TransformerDecoder` for more details.
    
    Users can configurate the model architecture with corresponding parameters.
    Note the usage of `normalize_before` representing where to apply layer
    normalization (in pre-process or post-precess of multi-head attention or FFN),
    and some transformer like models are different on this, such as
    `BERT <https://arxiv.org/abs/1810.04805>`_ and `GPT2 <https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf>`_ . 
    The default architecture here places layer normalization in pre-process and
    applies another layer normalization on the output of last encoder/decoder layer.

    Parameters:
        d_model (int): The expected feature size in the encoder/decoder input
            and output.
        nhead (int): The number of heads in multi-head attention(MHA).
        num_encoder_layers (int): The number of layers in encoder.
        num_encoder_layers (int): The number of layers in decoder.
        dim_feedforward (int): The hidden layer size in the feedforward network(FFN).
        dropout (float, optional): The dropout probability used in pre-process
            and post-precess of MHA and FFN sub-layer. Default 0.1
        activation (str, optional): The activation function in the feedforward
            network. Default relu.
        attn_dropout (float, optional): The dropout probability used
            in MHA to drop some attention target. If None, use the value of
            `dropout`. Default None
        act_dropout (float, optional): The dropout probability used after FFN
            activition.  If None, use the value of `dropout`. Default None
        normalize_before (bool, optional): Indicate whether to put layer normalization
            into preprocessing of MHA and FFN sub-layers. If True, pre-process is layer
            normalization and post-precess includes dropout, residual connection.
            Otherwise, no pre-process and post-precess includes dropout, residual
            connection, layer normalization. Default False
        param_attr(ParamAttr|tuple, optional): To specify the weight parameter property.
            If it is a tuple, `param_attr[0]` would be used as `param_attr` for
            self attention, `param_attr[1]` would be used as `param_attr` for
            cross attention, and `param_attr[2]` would be used as `param_attr`
            for linear in FFN. Otherwise, the three sub-layers all uses it as
            `param_attr` to create parameters. Default: None, which means the
            default weight parameter property is used. See usage for details
            in :ref:`api_fluid_ParamAttr` . 
        bias_attr (ParamAttr|tuple, optional): To specify the bias parameter property.
            If it is a tuple, `bias_attr[0]` would be used as `bias_attr` for
            self attention, `bias_attr[1]` would be used as `bias_attr` for
            cross attention, and `bias_attr[2]` would be used as `bias_attr`
            for linear in FFN. Otherwise, the three sub-layers all uses it as
            `bias_attr` to create parameters. Default: None, which means the
            default bias parameter property is used. See usage for details
            in :ref:`api_fluid_ParamAttr` .
        custom_encoder (Layer): If custom encoder is provided, use it as the encoder.
            Default None
        custom_decoder (Layer): If custom decoder is provided, use it as the decoder.
            Default None

    Examples:

        .. code-block:: python

            import paddle
            from paddle import Transformer

            # src: [batch_size, tgt_len, d_model]
            enc_input = paddle.rand((2, 4, 128))
            # tgt: [batch_size, src_len, d_model]
            dec_input = paddle.rand((2, 6, 128))
            # src_mask: [batch_size, n_head, src_len, src_len]
            enc_self_attn_mask = paddle.rand((2, 2, 4, 4))
            # tgt_mask: [batch_size, n_head, tgt_len, tgt_len]
            dec_self_attn_mask = paddle.rand((2, 2, 6, 6))
            # memory_mask: [batch_size, n_head, tgt_len, src_len]
            cross_attn_mask = paddle.rand((2, 2, 6, 4))
            transformer = Transformer(128, 2, 4, 4, 512)
            output = transformer(dec_input,
                                 enc_output,
                                 enc_self_attn_mask,
                                 dec_self_attn_mask,
                                 cross_attn_mask)  # [2, 6, 128]
    """

    def __init__(self,
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
                 param_attr=None,
                 bias_attr=None,
                 custom_encoder=None,
                 custom_decoder=None):
        super(Transformer, self).__init__()

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout, activation,
                attn_dropout, act_dropout, normalize_before, param_attr,
                bias_attr)
            encoder_norm = LayerNorm(d_model)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers,
                                              encoder_norm)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(
                d_model, nhead, dim_feedforward, dropout, activation,
                attn_dropout, act_dropout, normalize_before, param_attr,
                bias_attr)
            decoder_norm = LayerNorm(d_model)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers,
                                              decoder_norm)

        self.d_model = d_model
        self.nhead = nhead

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        """
        Applies a Transformer model on the inputs.

        Parameters:
            src (Variable): The input of Transformer encoder. It is a tensor
                with shape `[batch_size, source_length, d_model]`. The data type
                should be float32 or float64.
            tgt (Variable): The input of Transformer decoder. It is a tensor
                with shape `[batch_size, target_length, d_model]`. The data type
                should be float32 or float64.
            memory (Variable): The output of Transformer encoder. It is a tensor
                with shape `[batch_size, source_length, d_model]`. The data type
                should be float32 or float64.
            tgt_mask (Variable, optional): A tensor used in self attention
                to prevents attention to some unwanted positions, usually the
                the subsequent positions. It is a tensor with shape broadcasted
                to `[batch_size, n_head, target_length, target_length]`,
                where the unwanted positions have `-INF` values and the others
                have 0 values. The data type should be float32 or float64. It can
                be None when nothing wanted or needed to be prevented attention to.
                Default None
            memory_mask (Variable, optional): A tensor used in decoder-encoder
                cross attention to prevents attention to some unwanted positions,
                usually the paddings. It is a tensor with shape broadcasted to
               `[batch_size, n_head, target_length, source_length]`, where the
                unwanted positions have `-INF` values and the others have 0 values.
                The data type should be float32 or float64. It can be None when
                nothing wanted or needed to be prevented attention to. Default None

        Returns:
            Variable: It is a tensor that has the same shape and data type \
                as `tgt`, representing the output of Transformer decoder.
        """
        memory = self.encoder(src, mask=src_mask)
        output = self.decoder(
            tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return output


class TransformerDecoderCell(Layer):
    """
    TransformerDecoderCell wraps a Transformer decoder producing logits from
    `inputs` composed by ids and position.

    Parameters:
        decoder(callable): A TransformerDecoder instance. Or a wrapper of it that
            includes a embedding layer accepting ids and positions instead of embeddings
            and includes a output layer transforming decoder output features to logits.
        embedding_fn(callable, optional): A callable that accepts ids and position
            as arguments and return embeddings as input of `decoder`. It can be
            None if `decoder` includes a embedding layer. Default None.
        output_fn(callable, optional): A callable applid on `decoder` output to
            transform decoder output features to get logits. Mostly it is a Linear
            layer with vocabulary size. It can be None if `decoder` includes a
            output layer. Default None.

    Examples:

        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            from paddle.fluid.dygraph import Embedding, Linear
            from paddle.incubate.hapi.text import TransformerDecoder
            from paddle.incubate.hapi.text import TransformerCell
            from paddle.incubate.hapi.text import TransformerBeamSearchDecoder
            from paddle.incubate.hapi.text import DynamicDecode

            paddle.disable_static()

            class Embedder(fluid.dygraph.Layer):
                def __init__(self):
                    super(Embedder, self).__init__()
                    self.word_embedder = Embedding(size=[1000, 128])
                    self.pos_embedder = Embedding(size=[500, 128])

                def forward(self, word, position):
                    return self.word_embedder(word) + self.pos_embedder(position)

            embedder = Embedder()
            output_layer = Linear(128, 1000)
            decoder = TransformerDecoder(2, 2, 64, 64, 128, 512)
            transformer_cell = TransformerCell(decoder, embedder, output_layer)
            dynamic_decoder = DynamicDecode(
                TransformerBeamSearchDecoder(
                    transformer_cell,
                    start_token=0,
                    end_token=1,
                    beam_size=4,
                    var_dim_in_state=2),
                max_step_num=10,
                is_test=True)
            
            enc_output = paddle.rand((2, 4, 128))
            # cross attention bias: [batch_size, n_head, tgt_len, src_len]
            trg_src_attn_bias = paddle.rand((2, 2, 1, 4))
            # inputs for beam search on Transformer
            caches = transformer_cell.get_initial_states(enc_output)
            enc_output = TransformerBeamSearchDecoder.tile_beam_merge_with_batch(
                enc_output, beam_size=4)
            trg_src_attn_bias = TransformerBeamSearchDecoder.tile_beam_merge_with_batch(
                trg_src_attn_bias, beam_size=4)
            static_caches = decoder.prepare_static_cache(enc_output)
            outputs = dynamic_decoder(
                inits=caches,
                enc_output=enc_output,
                trg_src_attn_bias=trg_src_attn_bias,
                static_caches=static_caches)
    """

    def __init__(self, decoder, embedding_fn=None, output_fn=None):
        super(TransformerDecoderCell, self).__init__()
        self.decoder = decoder
        self.embedding_fn = embedding_fn
        self.output_fn = output_fn

    def forward(self,
                inputs,
                states=None,
                enc_output=None,
                trg_slf_attn_bias=None,
                trg_src_attn_bias=None,
                static_caches=[]):
        """
        Produces logits from `inputs` composed by ids and positions.

        Parameters:
            inputs(tuple): A tuple includes target ids and positions. The two
                tensors both have int64 data type and with 2D shape 
                `[batch_size, sequence_length]` where `sequence_length` is 1
                for inference.
            states(list): It caches the multi-head attention intermediate results
                of history decoding steps. It is a list of dict where the length
                of list is decoder layer number, and each dict has `k` and `v` as
                keys and values are cached results. Default None
            enc_output(Variable): The output of Transformer encoder. It is a tensor
                with shape `[batch_size, sequence_length, d_model]`. The data type
                should be float32 or float64.
            trg_slf_attn_bias(Variable, optional): A tensor used in decoder self
                attention to mask out attention on unwanted target positions. It
                is a tensor with shape `[batch_size, n_head, target_length, target_length]`,
                where the unwanted positions have `-INF` values and the others
                have 0 values. It can be None when nothing wanted or needed to
                be masked out. It can be None for inference. The data type should
                be float32 or float64. Default None
            trg_src_attn_bias(Variable, optional): A tensor used in decoder-encoder
                cross attention to mask out unwanted attention on source (encoder output).
                It is a tensor with shape `[batch_size, n_head, target_length, source_length]`,
                where the unwanted positions have `-INF` values and the others
                have 0 values. It can be None when nothing wanted or needed to
                be masked out. The data type should be float32 or float64. Default None
            static_caches(list): It stores projected results of encoder output
                to be used as keys and values in decoder-encoder cross attention
                It is a list of dict where the length of list is decoder layer
                number, and each dict has `static_k` and `static_v` as keys and
                values are stored results. Default empty list

        Returns:
            tuple: A tuple( :code:`(outputs, new_states)` ), where `outputs` \
                is a float32 or float64 3D tensor representing logits shaped \
                `[batch_size, sequence_length, vocab_size]`. `new_states has \
                the same structure and data type with `states` while the length \
                is one larger since the intermediate results of current step are \
                concatenated into it.
        """
        trg_word, trg_pos = inputs
        if states and static_caches:
            for cache, static_cache in zip(states, static_caches):
                cache.update(static_cache)
        if self.embedding_fn is not None:
            dec_input = self.embedding_fn(trg_word, trg_pos)
            outputs = self.decoder(dec_input, enc_output, None,
                                   trg_src_attn_bias, states)
        else:
            outputs = self.decoder(trg_word, trg_pos, enc_output, None,
                                   trg_src_attn_bias, states)
        if self.output_fn is not None:
            outputs = self.output_fn(outputs)

        new_states = [{
            "k": cache["k"],
            "v": cache["v"]
        } for cache in states] if states else states
        return outputs, new_states

    @property
    def state_shape(self):
        """
        States of TransformerCell cache the multi-head attention intermediate
        results of history decoding steps, and have a increasing length as
        decoding continued.
        
        `state_shape` of TransformerCell is used to initialize states. It is a
        list of dict where the length of list is decoder layer, and each dict
        has `k` and `v` as keys and values are `[n_head, 0, d_key]`, `[n_head, 0, d_value]`
        separately. (-1 for batch size would be automatically inserted into shape).

        Returns:
            list: It is a list of dict where the length of list is decoder layer \
                number, and each dict has `k` and `v` as keys and values are cached \
                results.
        """
        return [{
            "k": [self.decoder.n_head, 0, self.decoder.d_key],
            "v": [self.decoder.n_head, 0, self.decoder.d_value],
        } for i in range(self.decoder.n_layer)]


class TransformerBeamSearchDecoder(layers.BeamSearchDecoder):
    """
    Compared with a RNN step :code:`outputs, new_states = cell(inputs, states)`,
    Transformer decoder's `inputs` uses 2D tensor shaped `[batch_size * beam_size, 1]`
    and includes extra position data. And its `states` (caches) has increasing
    length. These are not consistent with `BeamSearchDecoder`, thus subclass
    `BeamSearchDecoder` to make beam search adapt to Transformer decoder.

    Parameters:
        cell(TransformerCell): An instance of `TransformerCell`.
        start_token(int): The start token id.
        end_token(int): The end token id.
        beam_size(int): The beam width used in beam search.
        var_dim_in_state(int): Indicate which dimension of states is variant.

    Examples:

        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            from paddle.fluid.dygraph import Embedding, Linear
            from paddle.incubate.hapi.text import TransformerDecoder
            from paddle.incubate.hapi.text import TransformerCell
            from paddle.incubate.hapi.text import TransformerBeamSearchDecoder
            from paddle.incubate.hapi.text import DynamicDecode

            paddle.disable_static()

            class Embedder(fluid.dygraph.Layer):
                def __init__(self):
                    super(Embedder, self).__init__()
                    self.word_embedder = Embedding(size=[1000, 128])
                    self.pos_embedder = Embedding(size=[500, 128])

                def forward(self, word, position):
                    return self.word_embedder(word) + self.pos_embedder(position)

            embedder = Embedder()
            output_layer = Linear(128, 1000)
            decoder = TransformerDecoder(2, 2, 64, 64, 128, 512)
            transformer_cell = TransformerCell(decoder, embedder, output_layer)
            dynamic_decoder = DynamicDecode(
                TransformerBeamSearchDecoder(
                    transformer_cell,
                    start_token=0,
                    end_token=1,
                    beam_size=4,
                    var_dim_in_state=2),
                max_step_num=10,
                is_test=True)
            
            enc_output = paddle.rand((2, 4, 128))
            # cross attention bias: [batch_size, n_head, tgt_len, src_len]
            trg_src_attn_bias = paddle.rand((2, 2, 1, 4))
            # inputs for beam search on Transformer
            caches = transformer_cell.get_initial_states(enc_output)
            enc_output = TransformerBeamSearchDecoder.tile_beam_merge_with_batch(
                enc_output, beam_size=4)
            trg_src_attn_bias = TransformerBeamSearchDecoder.tile_beam_merge_with_batch(
                trg_src_attn_bias, beam_size=4)
            static_caches = decoder.prepare_static_cache(enc_output)
            outputs = dynamic_decoder(
                inits=caches,
                enc_output=enc_output,
                trg_src_attn_bias=trg_src_attn_bias,
                static_caches=static_caches)
    """

    def __init__(self, cell, start_token, end_token, beam_size,
                 var_dim_in_state):
        super(TransformerBeamSearchDecoder,
              self).__init__(cell, start_token, end_token, beam_size)
        self.cell = cell
        self.var_dim_in_state = var_dim_in_state

    def _merge_batch_beams_with_var_dim(self, x):
        """
        Reshape a tensor with shape `[batch_size, beam_size, ...]` to a new
        tensor with shape `[batch_size * beam_size, ...]`. 

        Parameters:
            x(Variable): A tensor with shape `[batch_size, beam_size, ...]`. The
                data type should be float32, float64, int32, int64 or bool.

        Returns:
            Variable: A tensor with shape `[batch_size * beam_size, ...]`, whose \
                data type is same as `x`.
        """
        # init length of cache is 0, and it increases with decoding carrying on,
        # thus need to reshape elaborately
        var_dim_in_state = self.var_dim_in_state + 1  # count in beam dim
        x = layers.transpose(x,
                             list(range(var_dim_in_state, len(x.shape))) +
                             list(range(0, var_dim_in_state)))
        x = layers.reshape(
            x, [0] * (len(x.shape) - var_dim_in_state
                      ) + [self.batch_size * self.beam_size] +
            [int(size) for size in x.shape[-var_dim_in_state + 2:]])
        x = layers.transpose(
            x,
            list(range((len(x.shape) + 1 - var_dim_in_state), len(x.shape))) +
            list(range(0, (len(x.shape) + 1 - var_dim_in_state))))
        return x

    def _split_batch_beams_with_var_dim(self, x):
        """
        Reshape a tensor with shape `[batch_size * beam_size, ...]` to a new
        tensor with shape `[batch_size, beam_size, ...]`. 

        Parameters:
            x(Variable): A tensor with shape `[batch_size * beam_size, ...]`. The
                data type should be float32, float64, int32, int64 or bool.

        Returns:
            Variable: A tensor with shape `[batch_size, beam_size, ...]`, whose \
                data type is same as `x`.     
        """
        var_dim_size = layers.shape(x)[self.var_dim_in_state]
        x = layers.reshape(
            x, [-1, self.beam_size] +
            [int(size)
             for size in x.shape[1:self.var_dim_in_state]] + [var_dim_size] +
            [int(size) for size in x.shape[self.var_dim_in_state + 1:]])
        return x

    def step(self, time, inputs, states, **kwargs):
        """
        Perform a beam search decoding step, which uses `cell` to get probabilities,
        and follows a beam search step to calculate scores and select candidate
        token ids.

        Note: compared with `BeamSearchDecoder.step`, it feed 2D id tensor shaped
        `[batch_size * beam_size, 1]` rather than `[batch_size * beam_size]` combined
        position data as inputs to `cell`.

        Parameters:
            time(Variable): An `int64` tensor with shape `[1]` provided by the caller,
                representing the current time step number of decoding.
            inputs(Variable): A tensor variable. It is same as `initial_inputs`
                returned by `initialize()` for the first decoding step and
                `next_inputs` returned by `step()` for the others. It is a int64
                id tensor with shape `[batch_size * beam_size]`
            states(Variable): A structure of tensor variables.
                It is same as the `initial_states` returned by `initialize()` for
                the first decoding step and `beam_search_state` returned by
                `step()` for the others.
            **kwargs: Additional keyword arguments, provided by the caller. 
        
        Returns:
            tuple: A tuple( :code:`(beam_search_output, beam_search_state, next_inputs, finished)` ). \
                `beam_search_state` and `next_inputs` have the same structure, \
                shape and data type as the input arguments `states` and `inputs` separately. \
                `beam_search_output` is a namedtuple(including scores, predicted_ids, \
                parent_ids as fields) of tensor variables, where \
                `scores, predicted_ids, parent_ids` all has a tensor value shaped \
                `[batch_size, beam_size]` with data type `float32, int64, int64`. \
                `finished` is a `bool` tensor with shape `[batch_size, beam_size]`.
        """
        # compared to RNN, Transformer has 3D data at every decoding step
        inputs = layers.reshape(inputs, [-1, 1])  # token
        pos = layers.ones_like(inputs) * time  # pos
        cell_states = map_structure(self._merge_batch_beams_with_var_dim,
                                    states.cell_states)

        cell_outputs, next_cell_states = self.cell((inputs, pos), cell_states,
                                                   **kwargs)

        # squeeze to adapt to BeamSearchDecoder which use 2D logits
        cell_outputs = map_structure(
            lambda x: layers.squeeze(x, [1]) if len(x.shape) == 3 else x,
            cell_outputs)
        cell_outputs = map_structure(self._split_batch_beams, cell_outputs)
        next_cell_states = map_structure(self._split_batch_beams_with_var_dim,
                                         next_cell_states)

        beam_search_output, beam_search_state = self._beam_search_step(
            time=time,
            logits=cell_outputs,
            next_cell_states=next_cell_states,
            beam_state=states)
        next_inputs, finished = (beam_search_output.predicted_ids,
                                 beam_search_state.finished)

        return (beam_search_output, beam_search_state, next_inputs, finished)
