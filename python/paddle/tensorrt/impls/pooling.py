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


import numpy as np
import tensorrt as trt

from paddle.tensorrt.converter_utils import (
    get_input_constant_value,
    get_trt_plugin,
    set_layer_name,
)
from paddle.tensorrt.register import converter_registry


@converter_registry.register("pd_op.pool2d", trt_version="trt_version_ge=8.0")
def pool2d_converter(network, paddle_op, inputs):
    input_tensor = inputs[0]

    input_shape = paddle_op.operands()[0].source().shape
    input_dims = len(input_shape)

    global_pooling = paddle_op.attrs().get("global_pooling", False)
    pool_type = paddle_op.attrs().get("pooling_type", "avg")
    strides = paddle_op.attrs().get("strides", [1, 1])
    paddings = paddle_op.attrs().get("paddings", [0, 0])
    exclusive = paddle_op.attrs().get("exclusive", True)
    ceil_mode = paddle_op.attrs().get("ceil_mode", False)
    adaptive = paddle_op.attrs().get("adaptive", False)
    padding_algorithm = paddle_op.attrs().get("padding_algorithm", "EXPLICIT")

    if not paddle_op.attrs().get("kernel_size") and len(inputs) == 2:
        kernel_size = get_input_constant_value(paddle_op, inputs, 1)
        if kernel_size is None:
            raise Exception(
                "The defining op of kernel size must be builtin.constant/pd_op.full_int_array"
            )
    else:
        kernel_size = paddle_op.attrs().get("kernel_size", [1, 1])

    def create_pool_plugin(
        network,
        input_tensor,
        ceil_mode,
        pool_type,
        adaptive,
        exclusive,
        kernel_size,
        strides,
        paddings,
        global_pooling,
    ):
        plugin_fields = [
            trt.PluginField(
                "ceil_mode",
                np.array([ceil_mode], dtype=np.bool_),
                trt.PluginFieldType.INT32,
            ),
            trt.PluginField(
                "pool_type",
                np.array(list(pool_type), dtype=np.bytes_),
                trt.PluginFieldType.CHAR,
            ),
            trt.PluginField(
                "adaptive",
                np.array([adaptive], dtype=np.bool_),
                trt.PluginFieldType.INT32,
            ),
            trt.PluginField(
                "exclusive",
                np.array([exclusive], dtype=np.bool_),
                trt.PluginFieldType.INT32,
            ),
            trt.PluginField(
                "ksize",
                np.array(kernel_size, dtype=np.int32),
                trt.PluginFieldType.INT32,
            ),
            trt.PluginField(
                "strides",
                np.array(strides, dtype=np.int32),
                trt.PluginFieldType.INT32,
            ),
            trt.PluginField(
                "paddings",
                np.array(paddings, dtype=np.int32),
                trt.PluginFieldType.INT32,
            ),
            trt.PluginField(
                "global_pooling",
                np.array([global_pooling], dtype=np.bool_),
                trt.PluginFieldType.INT32,
            ),
        ]
        plugin_field_collection = trt.PluginFieldCollection(plugin_fields)
        plugin_name = "pir_pool_plugin_dynamic"
        plugin_version = "1"
        plugin = get_trt_plugin(
            plugin_name, plugin_field_collection, plugin_version
        )
        layer = network.add_plugin_v2([input_tensor], plugin)
        set_layer_name(layer, paddle_op)
        return layer

    reduce_operation = trt.ReduceOperation.MAX
    nv_pool_type = trt.PoolingType.MAX
    if pool_type == "max":
        nv_pool_type = trt.PoolingType.MAX
        reduce_operation = trt.ReduceOperation.MAX
    elif pool_type == "avg":
        nv_pool_type = trt.PoolingType.AVERAGE
        reduce_operation = trt.ReduceOperation.AVG
    else:
        raise ValueError(f"Unsupported pooling type: {pool_type}")

    if global_pooling or adaptive:
        paddings = [0, 0, 0, 0]

    if padding_algorithm == "VALID":
        paddings = [0] * len(paddings)

    nv_paddings = trt.DimsHW(paddings[0], paddings[1])
    nv_ksize = trt.DimsHW(kernel_size[0], kernel_size[1])
    nv_strides = trt.DimsHW(strides[0], strides[1])

    layer = None
    g_pre_pad = trt.DimsHW(0, 0)
    g_post_pad = trt.DimsHW(0, 0)

    if input_shape[input_dims - 2] - kernel_size[0] + 2 * paddings[0] < 0:
        g_post_pad.h = strides[0] - 1
    if input_shape[input_dims - 1] - kernel_size[1] + 2 * paddings[1] < 0:
        g_post_pad.w = strides[1] - 1

    real_paddings = paddings.copy()
    for i in range(2):
        copy_pad = paddings[i]
        real_paddings.insert(2 * i + 1, copy_pad)

    if padding_algorithm == "SAME":
        for i in range(2):
            copy_pad = paddings[2 * i]
            paddings.insert(2 * i + 1, copy_pad)

        for i in range(2):
            out_size = (input_shape[2 + i] + strides[i] - 1) // strides[i]
            pad_sum = max(
                (out_size - 1) * strides[i]
                + kernel_size[i]
                - input_shape[2 + i],
                0,
            )
            pad_0 = pad_sum // 2
            pad_1 = pad_sum - pad_0
            paddings[2 * i] = pad_0
            paddings[2 * i + 1] = pad_1
        real_paddings = paddings.copy()

    paddings = [paddings[i] for i in range(len(paddings)) if i % 2 == 0]

    if padding_algorithm == "VALID":
        read_paddings = [0] * len(real_paddings)

    if adaptive and pool_type == "avg":
        output_h, output_w = kernel_size
        if output_h == 1 and output_w == 1:
            reduce_axes = (1 << (input_dims - 2)) | (1 << (input_dims - 1))
            reduce_layer = network.add_reduce(
                input=input_tensor,
                op=trt.ReduceOperation.AVG,
                axes=reduce_axes,
                keep_dims=True,
            )
            if reduce_layer is None:
                raise RuntimeError("Failed to add reduce layer in TensorRT.")
            layer = reduce_layer
            set_layer_name(layer, paddle_op)
        else:
            input_h = input_shape[input_dims - 2]
            input_w = input_shape[input_dims - 1]

            if input_h < 0 or input_w < 0:
                layer = create_pool_plugin(
                    network,
                    input_tensor,
                    ceil_mode,
                    pool_type,
                    adaptive,
                    exclusive,
                    kernel_size,
                    strides,
                    paddings,
                    global_pooling,
                )
            else:
                stride_h = input_h // output_h
                stride_w = input_w // output_w
                kernel_h = input_h - (output_h - 1) * stride_h
                kernel_w = input_w - (output_w - 1) * stride_w

                if stride_h <= 0 or stride_w <= 0:
                    raise ValueError(
                        "Calculated stride is non-positive, which is invalid."
                    )

                nv_ksize = trt.DimsHW(kernel_h, kernel_w)
                nv_strides = trt.DimsHW(stride_h, stride_w)
                nv_paddings = trt.DimsHW(0, 0)

                pooling_layer = network.add_pooling_nd(
                    input=input_tensor,
                    type=nv_pool_type,
                    window_size=nv_ksize,
                )
                if pooling_layer is None:
                    raise RuntimeError(
                        "Failed to add pooling layer in TensorRT."
                    )
                pooling_layer.stride_nd = nv_strides
                pooling_layer.padding_nd = nv_paddings
                pooling_layer.average_count_excludes_padding = exclusive
                layer = pooling_layer
                set_layer_name(layer, paddle_op)

    elif not adaptive and not global_pooling and not ceil_mode:
        if padding_algorithm != "SAME" and (
            (g_post_pad.h > 0 and input_shape[input_dims - 2] > 0)
            or (g_post_pad.w > 0 and input_shape[input_dims - 1] > 0)
        ):
            pad_layer = network.add_padding_nd(
                input=input_tensor,
                pre_padding=(g_pre_pad.h, g_pre_pad.w),
                post_padding=(g_post_pad.h, g_post_pad.w),
            )
            if pad_layer is None:
                raise RuntimeError("Failed to add padding layer in TensorRT.")
            set_layer_name(pad_layer, paddle_op)
            input_tensor = pad_layer.get_output(0)
        pooling_layer = network.add_pooling_nd(
            input=input_tensor, type=nv_pool_type, window_size=nv_ksize
        )
        if pooling_layer is None:
            raise RuntimeError("Failed to add pooling layer in TensorRT.")
        pooling_layer.stride_nd = nv_strides
        pooling_layer.padding_nd = nv_paddings
        pooling_layer.average_count_excludes_padding = exclusive
        if padding_algorithm == "SAME":
            pooling_layer.padding_mode = trt.PaddingMode.SAME_UPPER

        layer = pooling_layer
        set_layer_name(layer, paddle_op)
    elif not adaptive and not global_pooling and ceil_mode:
        pooling_layer = network.add_pooling_nd(
            input=input_tensor, type=nv_pool_type, window_size=nv_ksize
        )
        if pooling_layer is None:
            raise RuntimeError("Failed to add pooling layer in TensorRT.")
        pooling_layer.stride_nd = nv_strides
        pooling_layer.padding_nd = nv_paddings
        pooling_layer.average_count_excludes_padding = exclusive
        if padding_algorithm == "SAME":
            pooling_layer.padding_mode = trt.PaddingMode.SAME_UPPER
        else:
            pooling_layer.padding_mode = trt.PaddingMode.EXPLICIT_ROUND_UP
        layer = pooling_layer
        set_layer_name(layer, paddle_op)
    elif global_pooling and not adaptive:
        reduce_layer = network.add_reduce(
            input_tensor, reduce_operation, 12, True
        )
        layer = reduce_layer
        set_layer_name(layer, paddle_op)
    else:
        layer = create_pool_plugin(
            network,
            input_tensor,
            ceil_mode,
            pool_type,
            adaptive,
            exclusive,
            kernel_size,
            strides,
            paddings,
            global_pooling,
        )

    if layer is None:
        raise RuntimeError("Failed to create pooling layer in TensorRT.")

    return layer.get_output(0)


@converter_registry.register("pd_op.pool3d", trt_version="8.x")
def pool3d_converter(network, paddle_op, inputs):
    input_tensor = inputs[0]
    global_pooling = paddle_op.attrs()["global_pooling"]
    pooling_type = paddle_op.attrs()["pooling_type"]
    ksize = paddle_op.attrs()["kernel_size"]
    strides = paddle_op.attrs()["strides"]
    paddings = paddle_op.attrs()["paddings"]
    exclusive = paddle_op.attrs().get("exclusive", True)
    ceil_mode = paddle_op.attrs()["ceil_mode"]
    adaptive = paddle_op.attrs().get("adaptive", False)
    padding_algorithm = paddle_op.attrs().get("padding_algorithm", "EXPLICIT")

    if padding_algorithm == "VALID" or padding_algorithm == "SAME":
        paddings = [0] * len(paddings)

    nv_pool_type = trt.PoolingType.MAX
    reduce_operation = trt.ReduceOperation.MAX

    if pooling_type == "max":
        nv_pool_type = trt.PoolingType.MAX
        reduce_operation = trt.ReduceOperation.MAX
    elif pooling_type == "avg":
        nv_pool_type = trt.PoolingType.AVERAGE
        reduce_operation = trt.ReduceOperation.AVG

    nv_ksize = trt.Dims3(ksize[0], ksize[1], ksize[2])
    nv_strides = trt.Dims3(strides[0], strides[1], strides[2])
    nv_paddings = trt.Dims3(paddings[0], paddings[1], paddings[2])

    layer = None
    if not adaptive and not global_pooling and not ceil_mode:
        pool_layer = network.add_pooling_nd(
            input_tensor, nv_pool_type, nv_ksize
        )
        pool_layer.stride_nd = nv_strides
        pool_layer.padding_nd = nv_paddings
        pool_layer.average_count_excludes_padding = exclusive
        set_layer_name(pool_layer, paddle_op)
        layer = pool_layer
    elif global_pooling:
        reduce_layer = network.add_reduce(
            input_tensor, reduce_operation, 28, True
        )
        set_layer_name(reduce_layer, paddle_op)
        layer = reduce_layer
    else:
        plugin_fields = [
            trt.PluginField(
                "ceil_mode",
                np.array([ceil_mode], dtype=np.bool_),
                trt.PluginFieldType.INT32,
            ),
            trt.PluginField(
                "pool3d_type",
                np.array(list(pooling_type), dtype=np.bytes_),
                trt.PluginFieldType.CHAR,
            ),
            trt.PluginField(
                "adaptive",
                np.array([adaptive], dtype=np.bool_),
                trt.PluginFieldType.INT32,
            ),
            trt.PluginField(
                "ksize",
                np.array(ksize, dtype=np.int32),
                trt.PluginFieldType.INT32,
            ),
            trt.PluginField(
                "strides",
                np.array(strides, dtype=np.int32),
                trt.PluginFieldType.INT32,
            ),
            trt.PluginField(
                "paddings",
                np.array(paddings, dtype=np.int32),
                trt.PluginFieldType.INT32,
            ),
            trt.PluginField(
                "is_global",
                np.array([global_pooling], dtype=np.bool_),
                trt.PluginFieldType.INT32,
            ),
        ]
        plugin_field_collection = trt.PluginFieldCollection(plugin_fields)
        plugin_name = "pir_pool3d_plugin_dynamic"
        plugin_version = "1"
        plugin = get_trt_plugin(
            plugin_name, plugin_field_collection, plugin_version
        )
        layer = network.add_plugin_v2([input_tensor], plugin)
        set_layer_name(layer, paddle_op)
    return layer.get_output(0)
