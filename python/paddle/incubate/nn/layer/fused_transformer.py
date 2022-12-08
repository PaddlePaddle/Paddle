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
import numpy as np

import paddle
from paddle.fluid import core
from paddle.fluid.core import VarDesc
from paddle.fluid.dygraph import no_grad
from paddle.fluid.framework import _non_static_mode, convert_np_dtype_to_dtype_
from paddle.incubate.nn import functional as incubate_f
from paddle.nn import Layer
from paddle.nn.initializer import Constant
from paddle.nn.layer.transformer import (
    _convert_attention_mask,
    _convert_param_attr_to_list,
)


# for distributed tensor model parallel
def _set_var_distributed(var):
    if var is None:
        return

    var.is_distributed = True

    if not _non_static_mode():
        # NOTE: use current_block and find_var_recursive to support while_loop
        startup_block = paddle.static.default_startup_program().current_block()
        main_block = paddle.static.default_main_program().current_block()
        startup_block._find_var_recursive(var.name).is_distributed = True
        main_block._find_var_recursive(var.name).is_distributed = True


def _to_dtype(t, dtype):
    # this function is a prune of Layer._transform function to fix fused op under amp.decorator(O2)
    if not paddle.is_floating_point(t):
        return t

    if type(dtype) is not VarDesc.VarType:
        dtype = convert_np_dtype_to_dtype_(dtype)

    if t.place.is_gpu_place():
        size_dtype = core.size_of_dtype(dtype)
        waiting_alloc_memory = (
            ((np.prod(t.shape) * size_dtype) / 256 + 1) * 256 * 1.2
        )
        gpu_memory_available = core.gpu_memory_available()
        if gpu_memory_available < waiting_alloc_memory:
            t_used = t._copy_to(paddle.CPUPlace(), False)
            t.value().get_tensor()._clear()
        else:
            t_used = t
    else:
        t_used = t

    if dtype is not None and dtype != t_used.dtype:
        with paddle.fluid.framework._dygraph_place_guard(place=t_used.place):
            t_casted = t_used.cast(dtype=dtype)
    else:
        t_casted = t_used

    new_t = t_casted

    dst_tensor = t.value().get_tensor()
    src_tensor = new_t.value().get_tensor()
    dst_tensor._share_data_with(src_tensor)

    return t


class FusedBiasDropoutResidualLayerNorm(Layer):
    """
    Applies fused_bias_dropout_residual_layer_norm operation.

    Parameters:
        embed_dim (int): The expected feature size in the input and output.
        dropout_rate (float, optional): The dropout probability used on attention
            weights to drop some attention targets for the dropout after attention.
            0 for no dropout. Default 0.5.
        bias_attr (ParamAttr|bool, optional): To specify the bias parameter property.
            Default: None, which means the default bias parameter property is used.
            If it is set to False, this layer will not have trainable bias parameter.
            See usage for details in :code:`ParamAttr`.
        epsilon (float, optional): The small value added to the variance to prevent
            division by zero. Default: 1e-05.

    Examples:

        .. code-block:: python

            # required: gpu
            import paddle
            # input: [batch_size, seq_len, embed_dim]
            x = paddle.rand((2, 4, 128))
            # residual: [batch_size, seq_len, embed_dim]
            residual = paddle.rand((2, 4, 128))
            fused_bias_dropout_residual_ln = paddle.incubate.nn.FusedBiasDropoutResidualLayerNorm(128)
            output = fused_bias_dropout_residual_ln(x, residual)  # [2, 4, 128]
    """

    def __init__(
        self,
        embed_dim,
        dropout_rate=0.5,
        weight_attr=None,
        bias_attr=None,
        epsilon=1e-5,
        name=None,
    ):
        super().__init__()
        assert embed_dim > 0, (
            "Expected embed_dim to be greater than 0, "
            "but recieved {}".format(embed_dim)
        )
        self._dtype = self._helper.get_default_dtype()
        self._bias_attr = bias_attr
        self._weight_attr = weight_attr
        self.embed_dim = embed_dim
        self.linear_bias = self.create_parameter(
            shape=[embed_dim],
            attr=self._bias_attr,
            dtype=self._dtype,
            is_bias=True,
        )
        self.ln_scale = self.create_parameter(
            attr=self._weight_attr,
            shape=[embed_dim],
            default_initializer=Constant(value=1.0),
        )
        self.ln_bias = self.create_parameter(
            attr=self._bias_attr, shape=[embed_dim], is_bias=True
        )
        self.dropout_rate = dropout_rate
        self._epsilon = epsilon

        self.name = name

    def forward(self, x, residual):
        """
        Applies fused_bias_dropout_residual_layer_norm operation.

        Parameters:
            x (Tensor): The input tensor. It is a tensor with shape
                `[batch_size, seq_len, embed_dim]`. The data type should be
                float32 or float64.
            residual (Tensor, optional): The residual tensor. It is a tensor
                with shape `[batch_size, value_length, vdim]`. The data type
                should be float32 or float64.

        Returns:
            Tensor|tuple: It is a tensor that has the same shape and data type \
                as `x`.
        """

        out = incubate_f.fused_bias_dropout_residual_layer_norm(
            x=x,
            residual=residual,
            bias=self.linear_bias,
            ln_scale=self.ln_scale,
            ln_bias=self.ln_bias,
            dropout_rate=self.dropout_rate,
            ln_epsilon=self._epsilon,
            training=self.training,
            mode='upscale_in_train',
            name=self.name,
        )
        return out

    def extra_repr(self):
        name_str = ', name={}'.format(self.name) if self.name else ''
        return 'embed_dim={}, seq_len={}, dropout_rate={}, epsilon={}, dtype={}{}'.format(
            self.embed_dim,
            self.seq_len,
            self.dropout_rate,
            self._epsilon,
            self._dtype,
            name_str,
        )


class FusedMultiHeadAttention(Layer):
    """
    Attention mapps queries and a set of key-value pairs to outputs, and
    Multi-Head Attention performs multiple parallel attention to jointly attending
    to information from different representation subspaces.
    Please refer to `Attention Is All You Need <https://arxiv.org/pdf/1706.03762.pdf>`_
    for more details.

    Parameters:
        embed_dim (int): The expected feature size in the input and output.
        num_heads (int): The number of heads in multi-head attention.
        dropout_rate (float, optional): The dropout probability used on attention
            weights to drop some attention targets for the dropout after attention.
            0 for no dropout. Default 0.5.
        attn_dropout_rate (float, optional): The dropout probability used on attention
            weights to drop some attention targets for the dropout in attention.
            0 for no dropout. Default 0.5.
        kdim (int, optional): The feature size in key. If None, assumed equal to
            `embed_dim`. Default None.
        vdim (int, optional): The feature size in value. If None, assumed equal to
            `embed_dim`. Default None.
        normalize_before (bool, optional): Indicate  whether it is pre_layer_norm
            (True) or post_layer_norm architecture (False). Default False.
        need_weights (bool, optional): Indicate whether to return the attention
            weights. Now, only False is supported. Default False.
        qkv_weight_attr(ParamAttr, optional): To specify the weight parameter property
            for QKV projection computation. Default: None, which means the default weight
            parameter property is used. See usage for details in :code:`ParamAttr`.
        qkv_bias_attr(ParamAttr|bool, optional): To specify the bias parameter property
            for QKV projection computation. The `False` value means the corresponding layer
            would not have trainable bias parameter. Default: None, which means the
            default bias parameter property is used. See usage for details in :code:`ParamAttr`.
        linear_weight_attr(ParamAttr, optional): To specify the weight parameter property
            for linear projection computation. Default: None, which means the default weight
            parameter property is used. See usage for details in :code:`ParamAttr`.
        linear_bias_attr(ParamAttr|bool, optional): To specify the bias parameter property
            for linear projection computation. The `False` value means the corresponding layer would
            not have trainable bias parameter. Default: None, which means the default bias
            parameter property is used. See usage for details in :code:`ParamAttr`.
        pre_ln_scale_attr(ParamAttr, optional): To specify the weight parameter property
            for pre_layer_norm computation. Otherwise, all layers both use it as
            `attr` to create parameters. Default: None, which means the default weight
            parameter property is used. See usage for details in :code:`ParamAttr`.
        pre_ln_bias_attr(ParamAttr|bool, optional): To specify the bias parameter property
            for pre_layer_norm computation. The `False` value means the corresponding layer would
            not have trainable bias parameter. Default: None, which means the default bias
            parameter property is used. See usage for details in :code:`ParamAttr`.
        ln_scale_attr(ParamAttr, optional): To specify the weight parameter property
            for post_layer_norm computation. Default: None, which means the default weight
            parameter property is used. See usage for details in :code:`ParamAttr`.
        ln_bias_attr(ParamAttr|bool, optional): To specify the bias parameter property
            for post_layer_norm computation. The `False` value means the corresponding layer would
            not have trainable bias parameter. Default: None, which means the default bias
            parameter property is used. See usage for details in :code:`ParamAttr`.
        epsilon (float, optional): The small value added to the variance to prevent
            division by zero. Default: 1e-05.
        nranks (int, optional): Distributed tensor model parallel nranks. Default is 1, means not using tensor parallel.
        ring_id (int, optional): For distributed tensor model parallel. Default is -1, means not using tensor parallel.

    Examples:

        .. code-block:: python

            # required: gpu
            import paddle
            # input: [batch_size, sequence_length, embed_dim]
            query = paddle.rand((2, 4, 128))
            # self attention mask: [batch_size, num_heads, query_len, query_len]
            attn_mask = paddle.rand((2, 2, 4, 4))
            multi_head_attn = paddle.incubate.nn.FusedMultiHeadAttention(128, 2)
            output = multi_head_attn(query, None, None, attn_mask=attn_mask)  # [2, 4, 128]
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout_rate=0.5,
        attn_dropout_rate=0.5,
        kdim=None,
        vdim=None,
        normalize_before=False,
        need_weights=False,
        qkv_weight_attr=None,
        qkv_bias_attr=None,
        linear_weight_attr=None,
        linear_bias_attr=None,
        pre_ln_scale_attr=None,
        pre_ln_bias_attr=None,
        ln_scale_attr=None,
        ln_bias_attr=None,
        epsilon=1e-5,
        nranks=1,
        ring_id=-1,
        name=None,
    ):
        super().__init__()

        assert embed_dim > 0, (
            "Expected embed_dim to be greater than 0, "
            "but received {}".format(embed_dim)
        )
        assert (
            num_heads > 0
        ), "Expected nhead to be greater than 0, " "but received {}".format(
            num_heads
        )

        self.normalize_before = normalize_before
        self._dtype = self._helper.get_default_dtype()
        self._epsilon = epsilon
        self._ring_id = ring_id

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.kdim = kdim
        self.vdim = vdim
        self.need_weights = need_weights
        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"
        assert need_weights is False, "Only support need_weight is False now."

        # tensor model parallel
        assert num_heads % nranks == 0
        num_heads = num_heads // nranks

        self.qkv_weight = self.create_parameter(
            shape=[3, num_heads, self.head_dim, embed_dim],
            attr=qkv_weight_attr,
            dtype=self._dtype,
            is_bias=False,
        )
        self.qkv_bias = self.create_parameter(
            shape=[3, num_heads, self.head_dim],
            attr=qkv_bias_attr,
            dtype=self._dtype,
            is_bias=True,
        )
        self.linear_weight = self.create_parameter(
            shape=[num_heads * self.head_dim, embed_dim],
            attr=linear_weight_attr,
            dtype=self._dtype,
            is_bias=False,
        )
        self.linear_bias = self.create_parameter(
            shape=[embed_dim],
            attr=linear_bias_attr,
            dtype=self._dtype,
            is_bias=True,
        )

        # tensor model parallel
        if nranks > 1:
            assert ring_id != -1
            # column parallel
            _set_var_distributed(self.qkv_weight)
            _set_var_distributed(self.qkv_bias)
            # row parallel
            _set_var_distributed(self.linear_weight)

        if normalize_before:
            self.pre_ln_scale = self.create_parameter(
                attr=pre_ln_scale_attr,
                shape=[embed_dim],
                default_initializer=Constant(value=1.0),
            )
            self.pre_ln_bias = self.create_parameter(
                attr=pre_ln_bias_attr, shape=[embed_dim], is_bias=True
            )
            self.ln_scale = None
            self.ln_bias = None
        else:
            self.pre_ln_scale = None
            self.pre_ln_bias = None
            self.ln_scale = self.create_parameter(
                attr=ln_scale_attr,
                shape=[embed_dim],
                default_initializer=Constant(value=1.0),
            )
            self.ln_bias = self.create_parameter(
                attr=ln_bias_attr, shape=[embed_dim], is_bias=True
            )

        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate

        self.name = name

    def forward(self, query, key=None, value=None, attn_mask=None, cache=None):
        """
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
                Now, only None is supported. Default None.

        Returns:
            Tensor|tuple: It is a tensor that has the same shape and data type \
                as `query`, representing attention output.
        """
        if attn_mask is not None:
            # Support bool or int mask
            attn_mask = _convert_attention_mask(attn_mask, query.dtype)

        out = incubate_f.fused_multi_head_attention(
            x=query,
            qkv_weight=self.qkv_weight,
            linear_weight=self.linear_weight,
            pre_layer_norm=self.normalize_before,
            pre_ln_scale=self.pre_ln_scale,
            pre_ln_bias=self.pre_ln_bias,
            ln_scale=self.ln_scale,
            ln_bias=self.ln_bias,
            pre_ln_epsilon=self._epsilon,
            qkv_bias=self.qkv_bias,
            linear_bias=self.linear_bias,
            cache_kv=cache,
            attn_mask=attn_mask,
            dropout_rate=self.dropout_rate,
            attn_dropout_rate=self.attn_dropout_rate,
            ln_epsilon=self._epsilon,
            training=self.training,
            ring_id=self._ring_id,
            name=self.name,
        )
        return out

    def extra_repr(self):
        name_str = ', name={}'.format(self.name) if self.name else ''
        return 'embed_dim={}, num_heads={}, dropout_rate={}, attn_dropout_rate={}, epsilon={}, kdim={}, vdim={}, normalize_before={}, need_weights={}, dtype={}{}'.format(
            self.embed_dim,
            self.num_heads,
            self.dropout_rate,
            self.attn_dropout_rate,
            self._epsilon,
            self.kdim,
            self.vdim,
            self.normalize_before,
            self.need_weights,
            self._dtype,
            name_str,
        )

    def _amp_decorate(self, dtype):
        # tmp fix for amp.decorator(O2)
        layer_norm_params_id = []
        if self.normalize_before:
            layer_norm_params_id.append(id(self.pre_ln_scale))
            layer_norm_params_id.append(id(self.pre_ln_bias))
        else:
            layer_norm_params_id.append(id(self.ln_scale))
            layer_norm_params_id.append(id(self.ln_bias))

        for key, param in self._parameters.items():
            if id(param) in layer_norm_params_id:
                continue
            if param is not None:
                with no_grad():
                    param_applied = _to_dtype(param, dtype)

        self._dtype = dtype


class FusedFeedForward(Layer):
    """
    Parameters:
        d_model (int): The expected feature size in the input and output.
        dim_feedforward (int): The hidden layer size.
        dropout_rate (float, optional): The dropout probability used in pre-process
            and post-precess. Default 0.1
        epsilon (float, optional): he small value added to the variance to prevent
            division by zero. Default: 1e-05.
        activation (str, optional): The activation function. Default relu.
        act_dropout_rate (float, optional): The dropout probability after activition.
            If None, use the value of `dropout_rate`. Default None
        normalize_before (bool, optional): Indicate whether to put layer normalization
            into, preprocessing or postprocessing. Default False
        linear1_weight_attr(ParamAttr, optional): To specify the weight parameter property
            for FFN first linear. Default: None, which means the default weight
            parameter property is used. See usage for details in :code:`ParamAttr`.
        linear1_bias_attr(ParamAttr|bool, optional): To specify the bias parameter property
            for FFN first linear. The `False` value means the corresponding layer would
            not have trainable bias parameter. Default: None, which means the default bias
            parameter property is used. See usage for details in :code:`ParamAttr`.
        linear2_weight_attr(ParamAttr, optional): To specify the weight parameter property
            for FFN second linear. Default: None, which means the default weight
            parameter property is used. See usage for details in :code:`ParamAttr`.
        linear2_bias_attr(ParamAttr|bool, optional): To specify the bias parameter property
            for FFN second linear. The `False` value means the corresponding layer would
            not have trainable bias parameter. Default: None, which means the default bias
            parameter property is used. See usage for details in :code:`ParamAttr`.
        ln1_scale_attr(ParamAttr, optional): To specify the weight parameter property
            for FFN pre_layer_norm. Default: None, which means the default weight
            parameter property is used. See usage for details in :code:`ParamAttr`.
        ln1_bias_attr(ParamAttr|bool, optional): To specify the bias parameter property
            for FFN pre_layer_norm. The `False` value means the corresponding layer would
            not have trainable bias parameter. Default: None, which means the default bias
            parameter property is used. See usage for details in :code:`ParamAttr`.
        ln2_scale_attr(ParamAttr, optional): To specify the weight parameter property
            for FFN post_layer_norm. Default: None, which means the default weight
            parameter property is used. See usage for details in :code:`ParamAttr`.
        ln2_bias_attr(ParamAttr|bool, optional): To specify the bias parameter property
            for FFN layer_norm. The `False` value means the corresponding layer would
            not have trainable bias parameter. Default: None, which means the default bias
            parameter property is used. See usage for details in :code:`ParamAttr`.
        nranks (int, optional): Distributed tensor model parallel nranks. Default is 1, means not using tensor parallel.
        ring_id (int, optional): For distributed tensor model parallel. Default is -1, means not using tensor parallel.
        name (str, optional): The default value is None.  Normally there is no need for user to set
            this property. For more information, please refer to :ref:`api_guide_Name`.

    Examples:
        .. code-block:: python

            # required: gpu
            import paddle
            from paddle.incubate.nn import FusedFeedForward

            fused_feedforward_layer = FusedFeedForward(8, 8)
            x = paddle.rand((1, 8, 8))
            out = fused_feedforward_layer(x)
            print(out.shape)
            # [1, 8, 8]
    """

    def __init__(
        self,
        d_model,
        dim_feedforward,
        dropout_rate=0.1,
        epsilon=1e-05,
        activation="relu",
        act_dropout_rate=None,
        normalize_before=False,
        linear1_weight_attr=None,
        linear1_bias_attr=None,
        linear2_weight_attr=None,
        linear2_bias_attr=None,
        ln1_scale_attr=None,
        ln1_bias_attr=None,
        ln2_scale_attr=None,
        ln2_bias_attr=None,
        nranks=1,
        ring_id=-1,
        name=None,
    ):

        super().__init__()
        assert (
            d_model > 0
        ), "Expected d_model to be greater than 0, but received {}".format(
            d_model
        )
        assert (
            dim_feedforward > 0
        ), "Expected dim_feedforward to be greater than 0, but received {}".format(
            dim_feedforward
        )

        self._dtype = self._helper.get_default_dtype()
        self._d_model = d_model

        assert dim_feedforward % nranks == 0
        dim_feedforward = dim_feedforward // nranks
        self._dim_feedforward = dim_feedforward
        self._dropout_rate = dropout_rate
        self._act_dropout_rate = (
            dropout_rate if act_dropout_rate is None else act_dropout_rate
        )
        self._act_method = activation
        self._normalize_before = normalize_before
        self._epsilon = epsilon
        self._ring_id = ring_id

        self._linear1_weight = self.create_parameter(
            shape=[d_model, dim_feedforward],
            attr=linear1_weight_attr,
            dtype=self._dtype,
            is_bias=False,
        )
        self._linear1_bias = self.create_parameter(
            shape=[dim_feedforward],
            attr=linear1_bias_attr,
            dtype=self._dtype,
            is_bias=True,
        )

        self._linear2_weight = self.create_parameter(
            shape=[dim_feedforward, d_model],
            attr=linear2_weight_attr,
            dtype=self._dtype,
            is_bias=False,
        )

        self._linear2_bias = self.create_parameter(
            shape=[d_model],
            attr=linear2_bias_attr,
            dtype=self._dtype,
            is_bias=True,
        )

        if nranks > 1:
            assert ring_id != -1
            # column parallel
            _set_var_distributed(self._linear1_weight)
            _set_var_distributed(self._linear1_bias)
            _set_var_distributed(self._linear2_weight)

        if normalize_before:
            self._ln1_scale = self.create_parameter(
                shape=[d_model],
                attr=ln1_scale_attr,
                is_bias=False,
                default_initializer=Constant(1.0),
            )
            self._ln1_bias = self.create_parameter(
                shape=[d_model], attr=ln1_bias_attr, is_bias=True
            )
            self._ln2_scale = None
            self._ln2_bias = None
        else:
            self._ln1_scale = None
            self._ln1_bias = None
            self._ln2_scale = self.create_parameter(
                shape=[d_model],
                attr=ln2_scale_attr,
                is_bias=False,
                default_initializer=Constant(1.0),
            )
            self._ln2_bias = self.create_parameter(
                shape=[d_model], attr=ln2_bias_attr, is_bias=True
            )

        self.name = name

    def forward(self, src, cache=None):
        out = incubate_f.fused_feedforward(
            src,
            self._linear1_weight,
            self._linear2_weight,
            self._linear1_bias,
            self._linear2_bias,
            self._ln1_scale,
            self._ln1_bias,
            self._ln2_scale,
            self._ln2_bias,
            dropout1_rate=self._act_dropout_rate,
            dropout2_rate=self._dropout_rate,
            activation=self._act_method,
            ln1_epsilon=self._epsilon,
            ln2_epsilon=self._epsilon,
            pre_layer_norm=self._normalize_before,
            training=self.training,
            ring_id=self._ring_id,
            name=self.name,
        )
        return out

    def extra_repr(self):
        name_str = ', name={}'.format(self.name) if self.name else ''
        return 'd_model={}, dim_feedforward={}, dropout_rate={}, epsilon={}, activation={}, act_dropout_rate={}, normalize_before={}, dtype={}{}'.format(
            self._d_model,
            self._dim_feedforward,
            self._dropout_rate,
            self._epsilon,
            self._act_method,
            self._act_dropout_rate,
            self._normalize_before,
            self._dtype,
            name_str,
        )

    def _amp_decorate(self, dtype):
        # tmp fix for amp.decorator(O2)
        layer_norm_params_id = []
        if self._normalize_before:
            layer_norm_params_id.append(id(self._ln1_scale))
            layer_norm_params_id.append(id(self._ln1_bias))
        else:
            layer_norm_params_id.append(id(self._ln2_scale))
            layer_norm_params_id.append(id(self._ln2_bias))

        for key, param in self._parameters.items():
            if id(param) in layer_norm_params_id:
                continue
            if param is not None:
                with no_grad():
                    param_applied = _to_dtype(param, dtype)

        self._dtype = dtype


class FusedTransformerEncoderLayer(Layer):
    """

    FusedTransformerEncoderLayer is composed of two sub-layers which are self (multi-head)
    attention and feedforward network. Before and after each sub-layer, pre-process
    and post-precess would be applied on the input and output accordingly. If
    `normalize_before` is True, pre-process is layer normalization and post-precess
    includes dropout, residual connection. Otherwise, no pre-process and post-precess
    includes dropout, residual connection, layer normalization.

    Parameters:
        d_model (int): The expected feature size in the input and output.
        nhead (int): The number of heads in multi-head attention(MHA).
        dim_feedforward (int): The hidden layer size in the feedforward network(FFN).
        dropout_rate (float, optional): The dropout probability used in pre-process
            and post-precess of MHA and FFN sub-layer. Default 0.1
        activation (str, optional): The activation function in the feedforward
            network. Default relu.
        attn_dropout_rate (float, optional): The dropout probability used
            in MHA to drop some attention target. If None, use the value of
            `dropout`. Default None
        act_dropout_rate (float, optional): The dropout probability used after FFN
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

            # required: gpu
            import paddle
            from paddle.incubate.nn import FusedTransformerEncoderLayer

            # encoder input: [batch_size, src_len, d_model]
            enc_input = paddle.rand((2, 4, 128))
            # self attention mask: [batch_size, n_head, src_len, src_len]
            attn_mask = paddle.rand((2, 2, 4, 4))
            encoder_layer = FusedTransformerEncoderLayer(128, 2, 512)
            enc_output = encoder_layer(enc_input, attn_mask)  # [2, 4, 128]

    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        dropout_rate=0.1,
        activation="relu",
        attn_dropout_rate=None,
        act_dropout_rate=None,
        normalize_before=False,
        weight_attr=None,
        bias_attr=None,
    ):
        self._config = locals()
        self._config.pop("self")
        self._config.pop("__class__", None)  # py3

        super().__init__()
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
        attn_dropout_rate = (
            dropout_rate if attn_dropout_rate is None else attn_dropout_rate
        )
        act_dropout_rate = (
            dropout_rate if act_dropout_rate is None else act_dropout_rate
        )
        self.normalize_before = normalize_before

        weight_attrs = _convert_param_attr_to_list(weight_attr, 2)
        bias_attrs = _convert_param_attr_to_list(bias_attr, 2)

        self.fused_attn = FusedMultiHeadAttention(
            d_model,
            nhead,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
            normalize_before=self.normalize_before,
            qkv_weight_attr=weight_attrs[0],
            qkv_bias_attr=bias_attrs[0],
            linear_weight_attr=weight_attrs[0],
            linear_bias_attr=bias_attrs[0],
            pre_ln_scale_attr=weight_attrs[0],
            pre_ln_bias_attr=bias_attrs[0],
            ln_scale_attr=weight_attrs[0],
            ln_bias_attr=bias_attrs[0],
        )

        self.ffn = FusedFeedForward(
            d_model,
            dim_feedforward,
            dropout_rate=dropout_rate,
            activation=activation,
            act_dropout_rate=act_dropout_rate,
            normalize_before=self.normalize_before,
            linear1_weight_attr=weight_attrs[1],
            linear1_bias_attr=bias_attrs[1],
            linear2_weight_attr=weight_attrs[1],
            linear2_bias_attr=bias_attrs[1],
        )

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
                See :ref:`api_paddle_nn_TransformerEncoderLayer`.gen_cache for more details. It is
                only used for inference and should be None for training. Default
                None.

        Returns:
            Tensor|tuple, It is a tensor that has the same shape and data type \
                as `enc_input`, representing the output of Transformer encoder \
                layer. Or a tuple if `cache` is not None, except for encoder \
                layer output, the tuple includes the new cache which is same \
                as input `cache` argument but `incremental_cache` has an \
                incremental length. See `MultiHeadAttention.gen_cache` and \
                `MultiHeadAttention.forward` for more details.

        """
        src_mask = _convert_attention_mask(src_mask, src.dtype)
        if cache is None:
            attn_out = self.fused_attn(src, attn_mask=src_mask)
        else:
            attn_out, incremental_cache = self.fused_attn(
                src, attn_mask=src_mask, cache=cache
            )

        ffn_out = self.ffn(attn_out)

        return ffn_out if cache is None else (ffn_out, incremental_cache)


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
        super().__init__()
        raise NotImplementedError()

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        raise NotImplementedError()


class FusedMultiTransformer(Layer):
    """
    FusedMultiTransformer is composed of multi transformer layers which contains two
    sub-layers which are self (multi-head) attention and feedforward network. The
    function of one transformer layer is consistent with the following pseudo code:

    .. code-block:: python

        if pre_layer_norm:
            out = layer_norm(x)
            out = qkv_linear(out) + qkv_bias
        else:
            out = qkv_linear(x) + qkv_bias
        out = transpose(out, perm=[2, 0, 3, 1, 4])
        # extract q, k and v from out.
        q = out[0:1, ::]
        k = out[1:2, ::]
        v = out[2:3, ::]
        out = q * k^t
        out = attn_mask + out
        out = softmax(out)
        out = dropout(out)
        out = out * v
        out = transpose(out, perm=[0, 2, 1, 3])
        out = linear(out)
        if pre_layer_norm:
            out = x + dropout(out + bias)
        else:
            out = layer_norm(x + dropout(out + bias))

        residual = out;
        if pre_layer_norm:
            out = ffn_layer_norm(out)
        out = ffn1_linear(out)
        out = dropout(activation(out + ffn1_bias))
        out = ffn2_linear(out)
        out = residual + dropout(out + ffn2_bias)
        if not pre_layer_norm:
            out = ffn_layer_norm(out)

    Parameters:
        embed_dim (int): The expected feature size in the input and output.
        num_heads (int): The number of heads in multi-head attention(MHA).
        dim_feedforward (int): The hidden layer size in the feedforward network(FFN).
        dropout_rate (float, optional): The dropout probability used in pre-process
            and post-precess of MHA and FFN sub-layer. Default 0.0
        activation (str, optional): The activation function in the feedforward
            network. Default "gelu".
        normalize_before (bool, optional): Indicate whether to put layer normalization
            into preprocessing of MHA and FFN sub-layers. If True, pre-process is layer
            normalization and post-precess includes dropout, residual connection.
            Otherwise, no pre-process and post-precess includes dropout, residual
            connection, layer normalization. Default True
        ln_scale_attrs(ParamAttr|list|tuple, optional): To specify the weight parameter property
            for Attention layer_norm. For Attention layer_norm weight, if it is a list/tuple, `attrs[0]`
            would be used as `attr` for transformer layer 0, and `attrs[1]` would be used as
            `attr` for transformer layer 1，etc. Otherwise, all layers both use it as
            `attr` to create parameters. Default: None, which means the default weight
            parameter property is used. See usage for details in :code:`ParamAttr`.
        ln_bias_attrs(ParamAttr|list|tuple|bool, optional): To specify the bias parameter property
            for Attention layer_norm. For Attention layer_norm bias, if it is a list/tuple, `attrs[0]`
            would be used as `attr` for transformer layer 0, and `attrs[1]` would be used as
            `attr` for transformer layer 1，etc. Otherwise, all layers both use it as
            `attr` to create parameters. The `False` value means the corresponding layer would
            not have trainable bias parameter. Default: None, which means the default bias
            parameter property is used. See usage for details in :code:`ParamAttr`.
        qkv_weight_attrs(ParamAttr|list|tuple, optional): To specify the weight parameter property
            for Attention qkv computation. For Attention qkv weight, if it is a list/tuple, `attrs[0]`
            would be used as `attr` for transformer layer 0, and `attrs[1]` would be used as
            `attr` for transformer layer 1，etc. Otherwise, all layers both use it as
            `attr` to create parameters. Default: None, which means the default weight
            parameter property is used. See usage for details in :code:`ParamAttr`.
        qkv_bias_attrs(ParamAttr|list|tuple|bool, optional): To specify the bias parameter property
            for Attention qkv computation. For Attention qkv bias, if it is a list/tuple, `attrs[0]`
            would be used as `attr` for transformer layer 0, and `attrs[1]` would be used as
            `attr` for transformer layer 1，etc. Otherwise, all layers both use it as
            `attr` to create parameters. The `False` value means the corresponding layer would
            not have trainable bias parameter. Default: None, which means the default bias
            parameter property is used. See usage for details in :code:`ParamAttr`.
        linear_weight_attrs(ParamAttr|list|tuple, optional): To specify the weight parameter property
            for Attention linear. For Attention linear weight, if it is a list/tuple, `attrs[0]`
            would be used as `attr` for transformer layer 0, and `attrs[1]` would be used as
            `attr` for transformer layer 1，etc. Otherwise, all layers both use it as
            `attr` to create parameters. Default: None, which means the default weight
            parameter property is used. See usage for details in :code:`ParamAttr`.
        linear_bias_attrs(ParamAttr|list|tuple|bool, optional): To specify the bias parameter property
            for Attention linear computation. For Attention linear bias, if it is a list/tuple, `attrs[0]`
            would be used as `attr` for transformer layer 0, and `attrs[1]` would be used as
            `attr` for transformer layer 1，etc. Otherwise, all layers both use it as
            `attr` to create parameters. The `False` value means the corresponding layer would
            not have trainable bias parameter. Default: None, which means the default bias
            parameter property is used. See usage for details in :code:`ParamAttr`.
        ffn_ln_scale_attrs(ParamAttr|list|tuple, optional): To specify the weight parameter property
            for FFN layer_norm. For FFN layer_norm weight, if it is a list/tuple, `attrs[0]`
            would be used as `attr` for transformer layer 0, and `attrs[1]` would be used as
            `attr` for transformer layer 1，etc. Otherwise, all layers both use it as
            `attr` to create parameters. Default: None, which means the default weight
            parameter property is used. See usage for details in :code:`ParamAttr`.
        ffn_ln_bias_attrs(ParamAttr|list|tuple|bool, optional): To specify the bias parameter property
            for FFN layer_norm. For FFN layer_norm bias, if it is a list/tuple, `attrs[0]`
            would be used as `attr` for transformer layer 0, and `attrs[1]` would be used as
            `attr` for transformer layer 1，etc. Otherwise, all layers both use it as
            `attr` to create parameters. The `False` value means the corresponding layer would
            not have trainable bias parameter. Default: None, which means the default bias
            parameter property is used. See usage for details in :code:`ParamAttr`.
        ffn1_weight_attrs(ParamAttr|list|tuple, optional): To specify the weight parameter property
            for FFN first linear. For FFN first linear weight, if it is a list/tuple, `attrs[0]`
            would be used as `attr` for transformer layer 0, and `attrs[1]` would be used as
            `attr` for transformer layer 1，etc. Otherwise, all layers both use it as
            `attr` to create parameters. Default: None, which means the default weight
            parameter property is used. See usage for details in :code:`ParamAttr`.
        ffn1_bias_attrs(ParamAttr|list|tuple|bool, optional): To specify the bias parameter property
            for FFN first linear. For FFN first linear bias, if it is a list/tuple, `attrs[0]`
            would be used as `attr` for transformer layer 0, and `attrs[1]` would be used as
            `attr` for transformer layer 1，etc. Otherwise, all layers both use it as
            `attr` to create parameters. The `False` value means the corresponding layer would
            not have trainable bias parameter. Default: None, which means the default bias
            parameter property is used. See usage for details in :code:`ParamAttr`.
        ffn2_weight_attrs(ParamAttr|list|tuple, optional): To specify the weight parameter property
            for FFN second linear. For FFN second linear weight, if it is a list/tuple, `attrs[0]`
            would be used as `attr` for transformer layer 0, and `attrs[1]` would be used as
            `attr` for transformer layer 1，etc. Otherwise, all layers both use it as
            `attr` to create parameters. Default: None, which means the default weight
            parameter property is used. See usage for details in :code:`ParamAttr`.
        ffn2_bias_attrs(ParamAttr|list|tuple|bool, optional): To specify the bias parameter property
            for FFN second linear. For FFN second linear bias, if it is a list/tuple, `attrs[0]`
            would be used as `attr` for transformer layer 0, and `attrs[1]` would be used as
            `attr` for transformer layer 1，etc. Otherwise, all layers both use it as
            `attr` to create parameters. The `False` value means the corresponding layer would
            not have trainable bias parameter. Default: None, which means the default bias
            parameter property is used. See usage for details in :code:`ParamAttr`.
        epsilon (float, optional): Small float value added to denominator of the layer_norm to
            avoid dividing by zero. Default: 1e-05.
        num_layers (int, optional): The number of layers of the transformer. If `qkv_weight_attrs`
            is a list or tuple, the number of layers is obtained from `qkv_weight_attrs`. num_layers
            only takes effect when `qkv_weight_attrs` is not a list or tuple. Default: -1.
        nranks (int, optional): Distributed tensor model parallel nranks. Default is 1, means not using mp.
        trans_qkvw (bool, optional): Whether to transpose for weights of qkv.
            If true, the shape eights of qkv should be [3, num_head, dim_head, dim_embed].
            Otherwise the shape of weights of qkv should be [dim_embed, 3, num_head, dim_head]. Default: True.
        ring_id (int, optional): For distributed tensor model parallel. Default is -1, means not using mp.
        name (str, optional): The default value is None.  Normally there is no need for user to set
            this property. For more information, please refer to :ref:`api_guide_Name`.

    Examples:

        .. code-block:: python

            # required: gpu
            import paddle
            from paddle.incubate.nn import FusedMultiTransformer

            # encoder input: [batch_size, src_len, d_model]
            enc_input = paddle.rand((2, 4, 128))
            # self attention mask: [batch_size, 1, src_len, src_len]
            attn_mask = paddle.rand((2, 1, 4, 4))
            encoder_layers = FusedMultiTransformer(128, 2, 512, num_layers=1)
            enc_output = encoder_layers(enc_input, attn_mask)  # [2, 4, 128]
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dim_feedforward,
        dropout_rate=0.0,
        activation="gelu",
        normalize_before=True,
        ln_scale_attrs=None,
        ln_bias_attrs=None,
        qkv_weight_attrs=None,
        qkv_bias_attrs=None,
        linear_weight_attrs=None,
        linear_bias_attrs=None,
        ffn_ln_scale_attrs=None,
        ffn_ln_bias_attrs=None,
        ffn1_weight_attrs=None,
        ffn1_bias_attrs=None,
        ffn2_weight_attrs=None,
        ffn2_bias_attrs=None,
        epsilon=1e-5,
        num_layers=-1,
        nranks=1,
        trans_qkvw=True,
        ring_id=-1,
        name=None,
    ):
        super().__init__()

        assert embed_dim > 0, (
            "Expected embed_dim to be greater than 0, "
            "but received {}".format(embed_dim)
        )
        assert (
            num_heads > 0
        ), "Expected nhead to be greater than 0, " "but received {}".format(
            num_heads
        )
        assert (
            dim_feedforward > 0
        ), "Expected dim_feedforward to be greater than 0, but received {}".format(
            dim_feedforward
        )

        self.normalize_before = normalize_before
        self._dtype = self._helper.get_default_dtype()
        self._epsilon = epsilon
        self._trans_qkvw = trans_qkvw
        self._ring_id = ring_id

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        # tensor model parallel
        if nranks > 1:
            assert ring_id != -1
        assert num_heads % nranks == 0
        assert dim_feedforward % nranks == 0
        num_heads = num_heads // nranks
        dim_feedforward = dim_feedforward // nranks
        self._dim_feedforward = dim_feedforward

        if isinstance(qkv_weight_attrs, (list, tuple)):
            num_layers = len(qkv_weight_attrs)
        assert num_layers > 0

        self.ln_scales, self.ln_biases = [], []
        self.qkv_weights, self.qkv_biases = [], []
        self.linear_weights, self.linear_biases = [], []
        self.ffn_ln_scales, self.ffn_ln_biases = [], []
        self.ffn1_weights, self.ffn1_biases = [], []
        self.ffn2_weights, self.ffn2_biases = [], []

        def get_attr(attrs, idx):
            if isinstance(attrs, (list, tuple)):
                assert len(attrs) == num_layers
                return attrs[idx]
            return attrs

        for i in range(num_layers):
            ln_scale_attr = get_attr(ln_scale_attrs, i)
            ln_bias_attr = get_attr(ln_bias_attrs, i)
            qkv_weight_attr = get_attr(qkv_weight_attrs, i)
            qkv_bias_attr = get_attr(qkv_bias_attrs, i)
            linear_weight_attr = get_attr(linear_weight_attrs, i)
            linear_bias_attr = get_attr(linear_bias_attrs, i)

            ffn_ln_scale_attr = get_attr(ffn_ln_scale_attrs, i)
            ffn_ln_bias_attr = get_attr(ffn_ln_bias_attrs, i)
            ffn1_weight_attr = get_attr(ffn1_weight_attrs, i)
            ffn1_bias_attr = get_attr(ffn1_bias_attrs, i)
            ffn2_weight_attr = get_attr(ffn2_weight_attrs, i)
            ffn2_bias_attr = get_attr(ffn2_bias_attrs, i)

            ln_scale = self.create_parameter(
                attr=ln_scale_attr,
                shape=[embed_dim],
                default_initializer=Constant(value=1.0),
            )
            ln_bias = self.create_parameter(
                attr=ln_bias_attr, shape=[embed_dim], is_bias=True
            )
            qkv_weight = self.create_parameter(
                shape=[3, num_heads, self.head_dim, embed_dim]
                if trans_qkvw
                else [embed_dim, 3, num_heads, self.head_dim],
                attr=qkv_weight_attr,
                dtype=self._dtype,
                is_bias=False,
            )
            qkv_bias = self.create_parameter(
                shape=[3, num_heads, self.head_dim],
                attr=qkv_bias_attr,
                dtype=self._dtype,
                is_bias=True,
            )
            linear_weight = self.create_parameter(
                shape=[num_heads * self.head_dim, embed_dim],
                attr=linear_weight_attr,
                dtype=self._dtype,
                is_bias=False,
            )
            linear_bias = self.create_parameter(
                shape=[embed_dim],
                attr=linear_bias_attr,
                dtype=self._dtype,
                is_bias=True,
            )

            ffn_ln_scale = self.create_parameter(
                shape=[embed_dim],
                attr=ffn_ln_scale_attr,
                is_bias=False,
                default_initializer=Constant(1.0),
            )
            ffn_ln_bias = self.create_parameter(
                shape=[embed_dim], attr=ffn_ln_bias_attr, is_bias=True
            )
            ffn1_weight = self.create_parameter(
                shape=[embed_dim, dim_feedforward],
                attr=ffn1_weight_attr,
                dtype=self._dtype,
                is_bias=False,
            )
            ffn1_bias = self.create_parameter(
                shape=[dim_feedforward],
                attr=ffn1_bias_attr,
                dtype=self._dtype,
                is_bias=True,
            )
            ffn2_weight = self.create_parameter(
                shape=[dim_feedforward, embed_dim],
                attr=ffn2_weight_attr,
                dtype=self._dtype,
                is_bias=False,
            )
            ffn2_bias = self.create_parameter(
                shape=[embed_dim],
                attr=ffn2_bias_attr,
                dtype=self._dtype,
                is_bias=True,
            )

            # tensor model parallel
            if nranks > 1:
                # column parallel
                _set_var_distributed(qkv_weight)
                _set_var_distributed(qkv_bias)
                _set_var_distributed(ffn1_weight)
                _set_var_distributed(ffn1_bias)
                # row parallel
                _set_var_distributed(linear_weight)
                _set_var_distributed(ffn2_weight)

            self.ln_scales.append(ln_scale)
            self.ln_biases.append(ln_bias)
            self.qkv_weights.append(qkv_weight)
            self.qkv_biases.append(qkv_bias)
            self.linear_weights.append(linear_weight)
            self.linear_biases.append(linear_bias)

            self.ffn_ln_scales.append(ffn_ln_scale)
            self.ffn_ln_biases.append(ffn_ln_bias)
            self.ffn1_weights.append(ffn1_weight)
            self.ffn1_biases.append(ffn1_bias)
            self.ffn2_weights.append(ffn2_weight)
            self.ffn2_biases.append(ffn2_bias)

        self.dropout_rate = dropout_rate
        self.activation = activation
        self.name = name

    def forward(
        self, src, attn_mask=None, caches=None, pre_caches=None, time_step=None
    ):
        r"""
        Applies multi transformer layers on the input.

        Parameters:
            src (Tensor): The input of Transformer layers. It is
                a tensor with shape `[batch_size, sequence_length, d_model]`.
                The data type should be float16 or float32.
            attn_mask (Tensor, optional): A tensor used in multi-head attention
                to prevents attention to some unwanted positions, usually the
                paddings or the subsequent positions. It is a tensor with shape
                `[batch_size, 1, sequence_length, sequence_length]`. It can be
                None when nothing wanted or needed to be prevented attention to.
                Default None.
            caches (list(Tensor)|tuple(Tensor), optional): The cache structure
                tensors for the inference generation model. It is only used for
                inference and should be None for training. The shape is
                `[2, batch_size, num_head, max_seq_len, head_dim]`. Default None.
            pre_caches (list(Tensor)|tuple(Tensor), optional): The prefix caches
                for the generation model. The shape is `[2, bsz, num\_head, cache\_len, head\_dim]`. Default None.
            time_step (Tensor, optional): The time step tensor for the generation
                model. Which used in decode stage, to represent the time step,
                that is, the real seq_len of CacheKV. The shape is `[1]`, must be
                in CPUPlace. Default None.

        Returns:
            Tensor|tuple: If `caches` is None, return a tensor that has
            the same shape and data type with `src`, representing the output
            of Transformer layers. If `caches` is not None, return the
            tuple (output, caches), which output is the output of
            Transformer layers, caches is inplace with input `caches`.
        """

        if caches is not None:
            assert len(caches) == len(self.qkv_weights)
        out = incubate_f.fused_multi_transformer(
            src,
            self.ln_scales,
            self.ln_biases,
            self.qkv_weights,
            self.qkv_biases,
            self.linear_weights,
            self.linear_biases,
            self.ffn_ln_scales,
            self.ffn_ln_biases,
            self.ffn1_weights,
            self.ffn1_biases,
            self.ffn2_weights,
            self.ffn2_biases,
            pre_layer_norm=self.normalize_before,
            epsilon=self._epsilon,
            cache_kvs=caches,
            pre_caches=pre_caches,
            time_step=time_step,
            attn_mask=attn_mask,
            dropout_rate=self.dropout_rate,
            activation=self.activation,
            training=self.training,
            mode='upscale_in_train',
            trans_qkvw=self._trans_qkvw,
            ring_id=self._ring_id,
            name=self.name,
        )
        return out
