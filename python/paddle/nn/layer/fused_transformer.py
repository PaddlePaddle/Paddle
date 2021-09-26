# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from ...fluid.layers import core
from ...fluid.framework import in_dygraph_mode
from paddle import _C_ops
from paddle.fluid.layer_helper import LayerHelper
import copy
from .. import functional as F
from paddle.nn import Layer
from ...framework import ParamAttr
from ...framework import get_default_dtype, set_default_dtype
from paddle.nn.layer.transformer import _convert_attention_mask
from ...fluid.data_feeder import check_variable_and_dtype, check_dtype
from ..initializer import Constant

import collections


def fused_multihead_attention(x,
                              qkv_weight,
                              out_linear_weight,
                              pre_layer_norm=False,
                              ln_scale=None,
                              ln_bias=None,
                              ln_2_scale=None,
                              ln_2_bias=None,
                              epsilon=1e-05,
                              qkv_bias=None,
                              out_linear_bias=None,
                              src_mask=None,
                              dropout=0.,
                              attn_dropout=0.,
                              ln2_epsilon=1e-05,
                              name=None):
    r"""
    """
    if in_dygraph_mode():
        ln_mean, ln_variance, ln_out, qkv_out, qkv_bias_out, transpose_out_2, qk_out, qktv_out, softmax_out, attn_dropout_mask_out, attn_dropout_out, src_mask_out, fmha_out, out_linear_out, dropout_mask_out, ln2_mean_out, ln2_var_out, bias_dropout_residual_out, final_out = _C_ops.fused_attention(
            x, ln_scale, ln_bias, qkv_weight, qkv_bias, src_mask,
            out_linear_weight, out_linear_bias, ln_2_scale, ln_2_bias,
            'pre_layer_norm', pre_layer_norm, 'epsilon', epsilon,
            'dropout_prob', dropout, 'attn_dropout_prob', attn_dropout)
        return final_out
    else:
        helper = LayerHelper('FusedMultiHeadAttention', **locals())
        dtype = x.dtype
        # check dtypes
        check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'],
                                 'FusedMultiHeadAttention')
        check_dtype(dtype, 'dtype', ['float16', 'float32', 'float64'],
                    'FusedMultiHeadAttention')

        # set inputs
        inputs = dict()
        inputs['X'] = [x]
        if ln_scale:
            inputs['LnScale'] = [ln_scale]
        if ln_bias:
            inputs['LnBias'] = [ln_bias]
        inputs['QKVW'] = [qkv_weight]
        inputs['QKVBias'] = [qkv_bias]
        inputs['SrcMask'] = src_mask
        inputs['OutLinearW'] = [out_linear_weight]
        inputs['OutLinearBias'] = [out_linear_bias]
        if ln_2_scale:
            inputs['Ln2Scale'] = [ln_2_scale]
        if ln_2_bias:
            inputs['Ln2Bias'] = [ln_2_bias]

        # set attrs
        attrs = {
            'pre_layer_norm': pre_layer_norm,
            'epsilon': epsilon,
            'ln2_epsilon': ln2_epsilon,
            'dropout_prob': dropout,
            'attn_dropout_prob': attn_dropout
        }

        # set outputs
        ln_mean_out = helper.create_variable_for_type_inference(
            dtype=dtype, stop_gradient=True)
        ln_variance_out = helper.create_variable_for_type_inference(
            dtype=dtype, stop_gradient=True)
        ln_out = helper.create_variable_for_type_inference(dtype=dtype)

        qkv_out = helper.create_variable_for_type_inference(dtype=dtype)
        qkv_bias_out = helper.create_variable_for_type_inference(dtype=dtype)

        transpose_out_2 = helper.create_variable_for_type_inference(dtype=dtype)
        qk_out = helper.create_variable_for_type_inference(dtype=dtype)
        qktv_out = helper.create_variable_for_type_inference(dtype=dtype)
        softmax_out = helper.create_variable_for_type_inference(dtype=dtype)
        attn_dropout_mask_out = helper.create_variable_for_type_inference(
            dtype=core.VarDesc.VarType.UINT8, stop_gradient=True)
        attn_dropout_out = helper.create_variable_for_type_inference(
            dtype=dtype)
        src_mask_out = helper.create_variable_for_type_inference(dtype=dtype)
        fmha_out = helper.create_variable_for_type_inference(dtype=dtype)
        out_linear_out = helper.create_variable_for_type_inference(dtype=dtype)
        dropout_mask_out = helper.create_variable_for_type_inference(
            dtype=core.VarDesc.VarType.UINT8, stop_gradient=True)
        ln_2_mean_out = helper.create_variable_for_type_inference(
            dtype=dtype, stop_gradient=True)
        ln_2_variance_out = helper.create_variable_for_type_inference(
            dtype=dtype, stop_gradient=True)
        bias_dropout_residual_out = helper.create_variable_for_type_inference(
            dtype=dtype)
        final_out = helper.create_variable_for_type_inference(dtype=dtype)

        helper.append_op(
            type='fused_attention',
            inputs=inputs,
            outputs={
                "LnMean": ln_mean_out,
                "LnVariance": ln_variance_out,
                "LnOut": ln_out,
                "QKVOut": qkv_out,
                "QKVBiasOut": qkv_bias_out,
                "TransposeOut2": transpose_out_2,
                "QKOut": qk_out,
                "QKTVOut": qktv_out,
                "SoftmaxOut": softmax_out,
                "AttnDropoutMaskOut": attn_dropout_mask_out,
                "AttnDropoutOut": attn_dropout_out,
                "SrcMaskOut": src_mask_out,
                "FMHAOut": fmha_out,
                "OutLinearOut": out_linear_out,
                "DropoutMaskOut": dropout_mask_out,
                "Ln2Mean": ln_2_mean_out,
                "Ln2Variance": ln_2_variance_out,
                "BiasDropoutResidualOut": bias_dropout_residual_out,
                'Y': final_out
            },
            attrs=attrs)
        return final_out


class FusedMultiHeadAttention(Layer):
    """
    """

    # todo (@limin, do we need cache in FusedMultiHeadAttention layer?)
    # Cache = collections.namedtuple("Cache", ["k", "v"])
    # StaticCache = collections.namedtuple("StaticCache", ["k", "v"])

    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.,
                 attn_dropout=0.,
                 kdim=None,
                 vdim=None,
                 normalize_before=False,
                 need_weights=False,
                 weight_attr=None,
                 bias_attr=None,
                 name=None):
        super(FusedMultiHeadAttention, self).__init__()

        assert embed_dim > 0, ("Expected embed_dim to be greater than 0, "
                               "but recieved {}".format(embed_dim))
        assert num_heads > 0, ("Expected nhead to be greater than 0, "
                               "but recieved {}".format(num_heads))

        attn_dropout = dropout if attn_dropout is None else attn_dropout
        self.normalize_before = normalize_before
        self._dtype = self._helper.get_default_dtype()
        self._weight_attr = weight_attr
        self._bias_attr = bias_attr

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        ## linear parameters.
        self.qkv_weight = self.create_parameter(
            shape=[3, num_heads, self.head_dim, embed_dim],
            attr=self._weight_attr,
            dtype=self._dtype,
            is_bias=False)
        self.qkv_bias = self.create_parameter(
            shape=[3, num_heads, self.head_dim],
            attr=self._bias_attr,
            dtype=self._dtype,
            is_bias=True)
        self.out_linear_weight = self.create_parameter(
            shape=[embed_dim, embed_dim],
            attr=self._weight_attr,
            dtype=self._dtype,
            is_bias=False)
        self.out_linear_bias = self.create_parameter(
            shape=[embed_dim],
            attr=self._bias_attr,
            dtype=self._dtype,
            is_bias=True)

        if get_default_dtype() == 'float16':
            set_default_dtype('float32')
        ## layer_norm parameters.
        self.ln_scale = self.create_parameter(
            attr=self._weight_attr,
            shape=[embed_dim],
            default_initializer=Constant(value=1.0))
        self.ln_bias = self.create_parameter(
            attr=self._bias_attr, shape=[embed_dim], is_bias=True)
        self.ln_2_scale = self.create_parameter(
            attr=self._weight_attr,
            shape=[embed_dim],
            default_initializer=Constant(value=1.0))
        self.ln_2_bias = self.create_parameter(
            attr=self._bias_attr, shape=[embed_dim], is_bias=True)
        if get_default_dtype() == 'float16':
            set_default_dtype('float16')

        ## dropout parameters
        self.dropout = dropout
        self.attn_dropout = attn_dropout

        self.name = name

    def forward(self, query, key=None, value=None, attn_mask=None, cache=None):
        """
        """
        if attn_mask is not None:
            # Support bool or int mask
            attn_mask = _convert_attention_mask(attn_mask, query.dtype)
        out = fused_multihead_attention(
            x=query,
            qkv_weight=self.qkv_weight,
            out_linear_weight=self.out_linear_weight,
            pre_layer_norm=self.normalize_before,
            ln_scale=self.ln_scale,
            ln_bias=self.ln_bias,
            ln_2_scale=self.ln_2_scale,
            ln_2_bias=self.ln_2_bias,
            epsilon=1e-05,
            qkv_bias=self.qkv_bias,
            out_linear_bias=self.out_linear_bias,
            src_mask=attn_mask,
            dropout=self.dropout,
            attn_dropout=self.attn_dropout,
            ln2_epsilon=1e-05)
        return out


class FusedFeedForward(Layer):
    def __init__(self,
                 d_model,
                 dim_feedforward,
                 dropout=0.1,
                 activation="relu",
                 act_dropout=None,
                 normalize_before=False,
                 weight_attr=None,
                 bias_attr=None):

        super(FusedFeedForward, self).__init__()
        raise NotImplementedError()

    def forward(self, src, cache=None):
        raise NotImplementedError()


class FusedTransformerEncoderLayer(Layer):
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

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward,
                 dropout=0.1,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False,
                 weight_attr=None,
                 bias_attr=None):
        self._config = locals()
        self._config.pop("self")
        self._config.pop("__class__", None)  # py3

        super(FusedTransformerEncoderLayer, self).__init__()
        raise NotImplementedError()

    def forward(self, src, src_mask=None, cache=None):
        """
        Applies a Transformer encoder layer on the input.
        Parameters:
            src (Tensor): The input of Transformer encoder layer. It is
                a tensor with shape `[batch_size, sequence_length, d_model]`.
                The data type should be float32 or float64.
            src_mask (Tensor, optional): A tensor used in multi-head attention
                to prevents attention to some unwanted positions, usually the
                paddings or the subsequent positions. It is a tensor with shape
                broadcasted to `[batch_size, n_head, sequence_length, sequence_length]`.
                When the data type is bool, the unwanted positions have `False` 
                values and the others have `True` values. When the data type is 
                int, the unwanted positions have 0 values and the others have 1 
                values. When the data type is float, the unwanted positions have 
                `-INF` values and the others have 0 values. It can be None when 
                nothing wanted or needed to be prevented attention to. Default None.
            cache (Tensor, optional): It is an instance of `MultiHeadAttention.Cache`.
                See `TransformerEncoderLayer.gen_cache` for more details. It is
                only used for inference and should be None for training. Default
                None.
        Returns:
            Tensor|tuple: It is a tensor that has the same shape and data type \
                as `enc_input`, representing the output of Transformer encoder \
                layer. Or a tuple if `cache` is not None, except for encoder \
                layer output, the tuple includes the new cache which is same \
                as input `cache` argument but `incremental_cache` has an \
                incremental length. See `MultiHeadAttention.gen_cache` and \
                `MultiHeadAttention.forward` for more details.
        """
        raise NotImplementedError()


class FusedTransformer(Layer):
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
                 weight_attr=None,
                 bias_attr=None,
                 custom_encoder=None,
                 custom_decoder=None):
        super(fusedTransformer, self).__init__()
        raise NotImplementedError()

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        raise NotImplementedError()
