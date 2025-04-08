# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved
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

import logging

import numpy as np
import tensorrt as trt

from paddle.base.log_helper import get_logger
from paddle.tensorrt.converter_utils import (
    WithFp16,
    add_1D_constant_layer,
    get_axes_for_reduce_op,
    get_dynamic_dims,
    get_trt_plugin,
    has_dynamic_shape,
    set_layer_name,
    trt_expand,
    trt_prod,
    trt_reshape,
    trt_sum,
)
from paddle.tensorrt.register import converter_registry
from paddle.tensorrt.util import (
    RefitManager,
    RefitRole,
    TensorRTConstantManager,
    support_fp32_mix_precision,
)

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s'
)


@converter_registry.register(
    "pd_op.layer_norm", trt_version="trt_version_ge=8.6"
)
def layernorm_converter(network, paddle_op, inputs):
    input_a, scale, bias = inputs

    begin_norm_axis = paddle_op.attrs().get("begin_norm_axis", 0)
    epsilon = paddle_op.attrs().get("epsilon", 1e-5)
    assert len(paddle_op.operands()) == 3
    scale_shape = paddle_op.operands()[1].source().shape

    if isinstance(scale, trt.Weights):
        scale_tensor = network.add_constant(scale_shape, scale)
        set_layer_name(scale_tensor, paddle_op)
        scale_tensor = scale_tensor.get_output(0)
        bias_shape = paddle_op.operands()[2].source().shape
        bias_tensor = network.add_constant(bias_shape, bias)
        set_layer_name(bias_tensor, paddle_op)
        bias_tensor = bias_tensor.get_output(0)
    else:
        scale_tensor = scale
        bias_tensor = bias

    dims = list(range(len(input_a.shape)))[begin_norm_axis:]
    axes = get_axes_for_reduce_op(dims)

    broadcast_shape = [1] * begin_norm_axis
    normalized_shape = list(input_a.shape)[begin_norm_axis:]
    broadcast_shape.extend(normalized_shape)

    scale_reshape = network.add_shuffle(scale_tensor)
    scale_reshape.reshape_dims = tuple(broadcast_shape)
    set_layer_name(scale_reshape, [paddle_op.name(), "scale_reshape"])
    scale_tensor = scale_reshape.get_output(0)

    bias_reshape = network.add_shuffle(bias_tensor)
    bias_reshape.reshape_dims = tuple(broadcast_shape)
    set_layer_name(bias_reshape, [paddle_op.name(), "bias_reshape"])
    bias_tensor = bias_reshape.get_output(0)

    layer_norm = network.add_normalization(
        input_a, scale_tensor, bias_tensor, axes
    )
    layer_norm.epsilon = epsilon
    set_layer_name(layer_norm, paddle_op)
    support_fp32_mix_precision(paddle_op.name(), layer_norm)
    layer_norm.compute_precision = trt.float32
    return layer_norm.get_output(0)


@converter_registry.register(
    "pd_op.batch_norm", trt_version="trt_version_ge=8.0"
)
@converter_registry.register(
    "pd_op.batch_norm_", trt_version="trt_version_ge=8.0"
)
def batch_norm_converter(network, paddle_op, inputs):
    constant_manager = TensorRTConstantManager()
    refit_manager = RefitManager()

    input_tensor, mean, variance, scale, bias = inputs

    scale_shape = paddle_op.operands()[3].source().shape
    eps = paddle_op.attrs().get("epsilon", 1e-8)

    scale_name = None
    bias_name = None
    if isinstance(mean, trt.ITensor):
        mean_name = (
            paddle_op.operands()[1]
            .source()
            .get_defining_op()
            .attrs()['parameter_name']
        )
        variance_name = (
            paddle_op.operands()[2]
            .source()
            .get_defining_op()
            .attrs()['parameter_name']
        )
        scale_name = (
            paddle_op.operands()[3]
            .source()
            .get_defining_op()
            .attrs()['parameter_name']
        )
        bias_name = (
            paddle_op.operands()[4]
            .source()
            .get_defining_op()
            .attrs()['parameter_name']
        )
        mean_np = constant_manager.get_constant_value(mean_name)
        variance_np = constant_manager.get_constant_value(variance_name)
        scale_np = constant_manager.get_constant_value(scale_name)
        bias_np = constant_manager.get_constant_value(bias_name)
    else:
        mean_np = mean.numpy()
        variance_np = variance.numpy()
        scale_np = scale.numpy()
        bias_np = bias.numpy()

    actual_scale_np = scale_np / np.sqrt(variance_np + eps)
    actual_bias_np = bias_np - mean_np * actual_scale_np

    bias = trt.Weights(actual_bias_np)
    scale = trt.Weights(actual_scale_np)
    power = trt.Weights(np.ones(scale_shape, dtype='float32'))

    input_tensor_shape = paddle_op.operands()[0].source().shape
    if has_dynamic_shape(input_tensor_shape):
        assert (
            input_tensor.shape[1] != -1
        ), "Channel dim can't be dynamic for batch norm."

    output_shape = input_tensor_shape

    if not network.has_implicit_batch_dimension and len(input_tensor_shape) < 4:
        assert (
            len(get_dynamic_dims(input_tensor.shape)) <= 1
        ), "BatchNorm1D with more than one dynamic dims is not currently supported."
        reshape_layer = network.add_shuffle(input_tensor)
        if len(input_tensor_shape) == 2:
            reshape_layer.reshape_dims = (
                input_tensor_shape[0],
                input_tensor_shape[1],
                1,
                1,
            )
        else:  # len(input_tensor_shape) ==3
            reshape_layer.reshape_dims = (
                input_tensor_shape[0],
                input_tensor_shape[1],
                input_tensor_shape[2],
                1,
            )
        set_layer_name(reshape_layer, paddle_op)
        input_tensor = reshape_layer.get_output(0)

    batch_norm_layer = network.add_scale(
        input_tensor, trt.ScaleMode.CHANNEL, bias, scale, power
    )
    support_fp32_mix_precision(paddle_op.name(), batch_norm_layer)
    set_layer_name(batch_norm_layer, paddle_op)
    if isinstance(mean, trt.ITensor):
        refit_manager.set_mapping(
            bias_name, batch_norm_layer.name, RefitRole.SHIFT
        )
        refit_manager.set_mapping(
            scale_name, batch_norm_layer.name, RefitRole.SCALE
        )

    if not network.has_implicit_batch_dimension and len(output_shape) < 4:
        reshape_output_layer = network.add_shuffle(
            batch_norm_layer.get_output(0)
        )
        reshape_output_layer.reshape_dims = tuple(output_shape)
        batch_norm_layer = reshape_output_layer
        set_layer_name(batch_norm_layer, paddle_op)

    return batch_norm_layer.get_output(0)


@converter_registry.register(
    "pd_op.instance_norm", trt_version="trt_version_ge=8.0"
)
def instance_norm_converter(network, paddle_op, inputs):
    eps = paddle_op.attrs().get("epsilon", 1e-8)
    instance_norm_inputs = [inputs[0], inputs[1], inputs[2]]
    plugin_fields = [
        trt.PluginField(
            "epsilon",
            np.array(eps, dtype=np.float32),
            trt.PluginFieldType.FLOAT32,
        ),
    ]
    plugin_field_collection = trt.PluginFieldCollection(plugin_fields)
    plugin_name = "pir_instance_norm"
    plugin_version = "1"
    plugin = get_trt_plugin(
        plugin_name, plugin_field_collection, plugin_version
    )
    instance_norm_layer = network.add_plugin_v2(instance_norm_inputs, plugin)
    set_layer_name(instance_norm_layer, paddle_op)
    return instance_norm_layer.get_output(0)


@converter_registry.register(
    "pd_op.fused_bias_dropout_residual_layer_norm",
    trt_version="trt_version_ge=8.0",
)
def fused_bias_dropout_residual_layer_norm_converter(
    network, paddle_op, inputs
):
    input1, input2, ele_bias, scale, bias = inputs
    if isinstance(ele_bias, trt.ITensor):
        refit_manager = RefitManager
        ele_bias = refit_manager.get_trt_weight_tensor(ele_bias.name)
        scale = refit_manager.get_trt_weight_tensor(scale.name)
        bias = refit_manager.get_trt_weight_tensor(bias.name)
    else:
        ele_bias = ele_bias
        scale = scale
        bias = bias
    has_bias = ele_bias is not None
    bias_size = bias.size
    scale_size = scale.size
    ele_bias_size = ele_bias.size if has_bias else 0
    epsilon = paddle_op.attrs().get("ln_epsilon", 1e-5)
    with_fp16 = int(WithFp16())
    # TODO: FusedBiasDropoutResidualLayerNorm will support FP16 UT in the future.
    if with_fp16 == 1:
        raise NotImplementedError(
            "FusedBiasDropoutResidualLayerNorm will support FP16 UT in the future."
        )
    ele_bias_data = (
        ele_bias.numpy().astype('float16') if with_fp16 else ele_bias.numpy()
    )
    plugin_fields = [
        trt.PluginField("bias", bias.numpy(), trt.PluginFieldType.FLOAT32),
        trt.PluginField("scale", scale.numpy(), trt.PluginFieldType.FLOAT32),
        trt.PluginField(
            "ele_bias",
            ele_bias_data,
            (
                trt.PluginFieldType.FLOAT16
                if with_fp16
                else trt.PluginFieldType.FLOAT32
            ),
        ),
        trt.PluginField(
            "bias_size",
            np.array([bias_size], dtype=np.int32),
            trt.PluginFieldType.INT32,
        ),
        trt.PluginField(
            "scale_size",
            np.array([scale_size], dtype=np.int32),
            trt.PluginFieldType.INT32,
        ),
        trt.PluginField(
            "ele_bias_size",
            np.array([ele_bias_size], dtype=np.int32),
            trt.PluginFieldType.INT32,
        ),
        trt.PluginField(
            "epsilon",
            np.array([epsilon], dtype=np.float32),
            trt.PluginFieldType.FLOAT32,
        ),
        trt.PluginField(
            "with_fp16",
            np.array([with_fp16], dtype=np.bool_),
            trt.PluginFieldType.INT32,
        ),
    ]
    plugin_field_collection = trt.PluginFieldCollection(plugin_fields)
    plugin_name = "pir_preln_residual_bias_plugin_dynamic"
    plugin_version = "1"
    plugin = get_trt_plugin(
        plugin_name, plugin_field_collection, plugin_version
    )
    plugin_inputs = [input1, input2]
    layer = network.add_plugin_v2(plugin_inputs, plugin)
    set_layer_name(layer, paddle_op)
    return layer.get_output(0)


@converter_registry.register(
    "pd_op.group_norm", trt_version="trt_version_ge=8.6"
)
def group_norm_converter(network, paddle_op, inputs):
    x, scale, bias = inputs
    groups = paddle_op.attrs().get("groups", 1)
    eps = paddle_op.attrs().get("epsilon", 1e-05)

    axes_mask = 0
    x_shape = paddle_op.operands()[0].source().shape
    rank_x = len(x_shape)

    fake_shape = [1, groups] + [1] * (rank_x - 2)
    broadcast_shape = [1, x_shape[1]] + [1] * (rank_x - 2)
    for d in range(2, rank_x):
        axes_mask |= 1 << d

    weight_one = add_1D_constant_layer(
        network, 1.0, np.float32, name=[paddle_op.name(), 'weight_one']
    )
    bias_zero = add_1D_constant_layer(
        network, 0.0, np.float32, name=[paddle_op.name(), 'bias_zero']
    )
    fake_shape = add_1D_constant_layer(
        network, fake_shape, np.int32, name=[paddle_op.name(), 'fake_shape']
    )
    weight_one = trt_expand(
        network,
        weight_one,
        1,
        fake_shape,
        rank_x,
        name=[paddle_op.name(), 'weight_one'],
    )
    bias_zero = trt_expand(
        network,
        bias_zero,
        1,
        fake_shape,
        rank_x,
        name=[paddle_op.name(), 'bias_zero'],
    )
    layer = network.add_normalization(x, weight_one, bias_zero, axes_mask)
    layer.num_groups = groups
    layer.epsilon = eps
    set_layer_name(layer, paddle_op)
    output = layer.get_output(0)
    if scale is not None:
        scale = trt_reshape(
            network, scale, broadcast_shape, name=[paddle_op.name(), 'scale']
        )
        output = trt_prod(
            network, output, scale, name=[paddle_op.name(), 'output']
        )
    if bias is not None:
        bias = trt_reshape(
            network, bias, broadcast_shape, name=[paddle_op.name(), 'bias']
        )
        output = trt_sum(
            network, output, bias, name=[paddle_op.name(), 'output']
        )

    return output
