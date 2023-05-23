#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import collections

import paddle
from paddle import tensor
from paddle.nn.layer.transformer import (
    _convert_attention_mask,
    _convert_param_attr_to_list,
)

from . import sparse_mask

__all__ = []


class MultiHeadAttention(paddle.nn.MultiHeadAttention):
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
        weight_attr(ParamAttr, optional):  To specify the weight parameter property.
            Default: None, which means the default weight parameter property is used.
            See usage for details in :code:`ParamAttr` .
        bias_attr (ParamAttr|bool, optional): To specify the bias parameter property.
            Default: None, which means the default bias parameter property is used.
            If it is set to False, this layer will not have trainable bias parameter.
            See usage for details in :code:`ParamAttr` .

    Examples:

        .. code-block:: python

            import paddle

            # encoder input: [batch_size, sequence_length, d_model]
            query = paddle.rand((2, 4, 128))
            # self attention mask: [batch_size, num_heads, query_len, query_len]
            attn_mask = paddle.rand((2, 2, 4, 4))
            multi_head_attn = paddle.nn.MultiHeadAttention(128, 2)
            output = multi_head_attn(query, None, None, attn_mask=attn_mask)  # [2, 4, 128]
    """

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
        mask_name="erine",
    ):
        super().__init__(
            embed_dim,
            num_heads,
            dropout=dropout,
            kdim=kdim,
            vdim=vdim,
            need_weights=need_weights,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
        )

        self.sparse = sparse_mask.get_sparse_mask(mask_name)
        self.sparse_mask = None

    def forward(self, query, key=None, value=None, attn_mask=None, cache=None):
        r"""
        Applies multi-head attention to map queries and a set of key-value pairs
        to outputs.

        Parameters:
            query (Tensor): The queries for multi-head attention. It is a
                tensor with shape `[batch_size, query_length, embed_dim]`. The
                data type should be float32 or float64.
            key (Tensor, optional): The keys for multi-head attention. It is
                a tensor with shape `[batch_size, key_length, kdim]`. The
                data type should be float32 or float64. If None, use `query` as
                `key`. Default None.
            value (Tensor, optional): The values for multi-head attention. It
                is a tensor with shape `[batch_size, value_length, vdim]`.
                The data type should be float32 or float64. If None, use `query` as
                `value`. Default None.
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
            cache (MultiHeadAttention.Cache|MultiHeadAttention.StaticCache, optional):
                It is a namedtuple with `k` and `v` as fields, and stores tensors
                shaped `[batch_size, num_heads, length, embed_dim]` which are results
                of linear projection, reshape and transpose calculations in
                MultiHeadAttention. If it is an instance of `Cache`, `k` and `v`
                fields reserve intermediate results of previous positions, which
                mostly used for decoder self attention. If it is an instance of
                `StaticCache`, `key` and `value` args would be ignored, `k` and
                `v` fields would be used as calculated results on `key` and
                `value`, which mostly used for decoder-encoder cross attention.
                It is only used for inference and should be None for training.
                Default None.

        Returns:
            Tensor|tuple: It is a tensor that has the same shape and data type \
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

        if (
            self.sparse_mask is None
            or self.sparse_mask.shape[0] != q.shape[0] * q.shape[1]
            or self.sparse_mask.shape[1] != q.shape[2]
        ):
            mask = self.sparse(q.shape[0], self.num_heads, q.shape[2])
            self.sparse_mask = mask.astype('float32').to_sparse_csr()

        if attn_mask is not None:
            attn_mask = _convert_attention_mask(attn_mask, q.dtype)
            attn_mask = attn_mask.squeeze()
            if len(attn_mask.shape) == 1:
                attn_mask = attn_mask.reshape((1, attn_mask.shape[0]))

        out, weights = paddle.sparse.nn.functional.attention(
            q, k, v, self.sparse_mask, attn_mask=attn_mask
        )

        # combine heads
        out = tensor.transpose(out, perm=[0, 2, 1, 3])
        out = tensor.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])

        # project to output
        out = self.out_proj(out)

        outs = [out]
        if self.need_weights:
            outs.append(weights)
        if cache is not None:
            outs.append(cache)
        return out if len(outs) == 1 else tuple(outs)


class TransformerEncoderLayer(paddle.nn.TransformerEncoderLayer):
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
        weight_attr(ParamAttr|list|tuple, optional): To specify the weight parameter property.
            If it is a list/tuple, `weight_attr[0]` would be used as `weight_attr` for
            MHA, and `weight_attr[1]` would be used as `weight_attr` for linear in FFN.
            Otherwise, MHA and FFN both use it as `weight_attr` to create parameters.
            Default: None, which means the default weight parameter property is used.
            See usage for details in :code:`ParamAttr` .
        bias_attr (ParamAttr|list|tuple|bool, optional): To specify the bias parameter property.
            If it is a list/tuple, `bias_attr[0]` would be used as `bias_attr` for
            MHA, and `bias_attr[1]` would be used as `bias_attr` for linear in FFN.
            Otherwise, MHA and FFN both use it as `bias_attr` to create parameters.
            The `False` value means the corresponding layer would not have trainable
            bias parameter. See usage for details in :code:`ParamAttr` . Default: None,
            which means the default bias parameter property is used.


    Examples:

        .. code-block:: python

            import paddle
            from paddle.nn import TransformerEncoderLayer

            # encoder input: [batch_size, src_len, d_model]
            enc_input = paddle.rand((2, 4, 128))
            # self attention mask: [batch_size, n_head, src_len, src_len]
            attn_mask = paddle.rand((2, 2, 4, 4))
            encoder_layer = TransformerEncoderLayer(128, 2, 512)
            enc_output = encoder_layer(enc_input, attn_mask)  # [2, 4, 128]
    """

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
        mask_name="erine",
    ):
        super().__init__(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            attn_dropout,
            act_dropout,
            normalize_before,
            weight_attr,
            bias_attr,
        )
        weight_attrs = _convert_param_attr_to_list(weight_attr, 2)
        bias_attrs = _convert_param_attr_to_list(bias_attr, 2)

        self.self_attn = MultiHeadAttention(
            d_model,
            nhead,
            dropout=attn_dropout,
            weight_attr=weight_attrs[0],
            bias_attr=bias_attrs[0],
            mask_name=mask_name,
        )


class TransformerEncoder(paddle.nn.TransformerEncoder):
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
            from paddle.nn import TransformerEncoderLayer, TransformerEncoder

            # encoder input: [batch_size, src_len, d_model]
            enc_input = paddle.rand((2, 4, 128))
            # self attention mask: [batch_size, n_head, src_len, src_len]
            attn_mask = paddle.rand((2, 2, 4, 4))
            encoder_layer = TransformerEncoderLayer(128, 2, 512)
            encoder = TransformerEncoder(encoder_layer, 2)
            enc_output = encoder(enc_input, attn_mask)  # [2, 4, 128]
    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__(encoder_layer, num_layers, norm)


class TransformerDecoderLayer(paddle.nn.TransformerDecoderLayer):
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
        weight_attr(ParamAttr|list|tuple, optional): To specify the weight parameter property.
            If it is a list/tuple, `weight_attr[0]` would be used as `weight_attr` for
            self attention, `weight_attr[1]` would be used as `weight_attr` for
            cross attention, and `weight_attr[2]` would be used as `weight_attr`
            for linear in FFN. Otherwise, the three sub-layers all uses it as
            `weight_attr` to create parameters. Default: None, which means the
            default weight parameter property is used. See usage for details
            in :ref:`api_paddle_fluid_param_attr_ParamAttr` .
        bias_attr (ParamAttr|list|tuple|bool, optional): To specify the bias parameter property.
            If it is a list/tuple, `bias_attr[0]` would be used as `bias_attr` for
            self attention, `bias_attr[1]` would be used as `bias_attr` for
            cross attention, and `bias_attr[2]` would be used as `bias_attr`
            for linear in FFN. Otherwise, the three sub-layers all uses it as
            `bias_attr` to create parameters. The `False` value means the
            corresponding layer would not have trainable bias parameter. See
            usage for details in :code:`ParamAttr` . Default: None,which means
            the default bias parameter property is used.

    Examples:

        .. code-block:: python

            import paddle
            from paddle.nn import TransformerDecoderLayer

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
        mask_name="erine",
    ):
        self._config = locals()
        self._config.pop("self")
        self._config.pop("__class__", None)  # py3

        super().__init__(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            attn_dropout,
            act_dropout,
            normalize_before,
            weight_attr,
            bias_attr,
        )

        weight_attrs = _convert_param_attr_to_list(weight_attr, 3)
        bias_attrs = _convert_param_attr_to_list(bias_attr, 3)

        self.self_attn = MultiHeadAttention(
            d_model,
            nhead,
            dropout=attn_dropout,
            weight_attr=weight_attrs[0],
            bias_attr=bias_attrs[0],
            mask_name=mask_name,
        )


class TransformerDecoder:
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
            from paddle.nn import TransformerDecoderLayer, TransformerDecoder

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
        super().__init__(decoder_layer, num_layers, norm)


class Transformer(paddle.nn.Transformer):
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
    The default architecture here places layer normalization in post-process and
    applies another layer normalization on the output of last encoder/decoder layer.

    Parameters:
        d_model (int, optional): The expected feature size in the encoder/decoder input
            and output. Default 512
        nhead (int, optional): The number of heads in multi-head attention(MHA). Default 8
        num_encoder_layers (int, optional): The number of layers in encoder. Default 6
        num_decoder_layers (int, optional): The number of layers in decoder. Default 6
        dim_feedforward (int, optional): The hidden layer size in the feedforward network(FFN). Default 2048
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
        weight_attr(ParamAttr|list|tuple, optional): To specify the weight parameter property.
            If it is a list/tuple, the length of `weight_attr` could be 1, 2 or 3. If it is 3,
            `weight_attr[0]` would be used as `weight_attr` for self attention, `weight_attr[1]`
            would be used as `weight_attr` for cross attention of `TransformerDecoder`,
            and `weight_attr[2]` would be used as `weight_attr` for linear in FFN.
            If it is 2, `weight_attr[0]` would be used as `weight_attr` both for self attention
            and cross attntion and `weight_attr[1]` would be used as `weight_attr` for
            linear in FFN. If it is 1, `weight_attr[0]` would be used as `weight_attr`
            for self attention, cross attention and linear in FFN. Otherwise,
            the three sub-layers all uses it as `weight_attr` to create parameters.
            Default: None, which means the default weight parameter property is used.
            See usage for details
            in :code:`ParamAttr` .
        bias_attr (ParamAttr|list|tuple|bool, optional): To specify the bias parameter property.
            If it is a list/tuple, the length of `bias_attr` could be 1, 2 or 3. If it is 3,
            `bias_attr[0]` would be used as `bias_attr` for self attention, `bias_attr[1]`
            would be used as `bias_attr` for cross attention of `TransformerDecoder`,
            and `bias_attr[2]` would be used as `bias_attr` for linear in FFN.
            If it is 2, `bias_attr[0]` would be used as `bias_attr` both for self attention
            and cross attntion and `bias_attr[1]` would be used as `bias_attr` for
            linear in FFN. If it is 1, `bias_attr[0]` would be used as `bias_attr`
            for self attention, cross attention and linear in FFN. Otherwise,
            the three sub-layers all uses it as `bias_attr` to create parameters.
            The `False` value means the corresponding layer would not have trainable
            bias parameter. See usage for details in :code:`ParamAttr` .
            Default: None,which means the default bias parameter property is used.
        custom_encoder (Layer, optional): If custom encoder is provided, use it as the encoder.
            Default None
        custom_decoder (Layer, optional): If custom decoder is provided, use it as the decoder.
            Default None

    Examples:

        .. code-block:: python

            import paddle
            from paddle.nn import Transformer

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
            output = transformer(enc_input,
                                 dec_input,
                                 enc_self_attn_mask,
                                 dec_self_attn_mask,
                                 cross_attn_mask)  # [2, 6, 128]
    """

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
        super().__init__(
            d_model,
            nhead,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            dropout,
            activation,
            attn_dropout,
            act_dropout,
            normalize_before,
            weight_attr,
            bias_attr,
            custom_encoder,
            custom_decoder,
        )
