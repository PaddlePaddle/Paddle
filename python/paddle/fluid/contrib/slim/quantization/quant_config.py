#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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


class BaseQuantizer:
    """
    Basic quantization configuration class, configure some quantized hyperparameters,
    including the op that needs to be quantized, the number of quantized bits,
    the value range and the rounding method, etc.
    """

    def __init__(
        self,
        quant_operation_types=None,
        quant_bits=8,
    ):
        self._quantize_op = quant_operation_types
        self.quant_bits = quant_bits
        self._quant_min = -128
        self._quant_max = 127

    @property
    def weight_quant_operation_types(self):
        """
        Operation type list which should support weight quantization.
        And before these ops, quant dequant nodes will be inserted.
        """
        base_weight_op_type_list = [
            'conv2d',
            'depthwise_conv2d',
            'conv2d_transpose',
            'mul',
            'matmul',
            'matmul_v2',
        ]
        if self._quantize_op:
            weight_list = []
            for _op_type in self._quantize_op:
                if _op_type in base_weight_op_type_list:
                    weight_list.append(_op_type)
            return weight_list
        else:
            return base_weight_op_type_list

    @property
    def activation_quant_operation_types(self):
        """
        Operation type list which should support activation quantization.
        And before these ops, quant dequant nodes will be inserted.
        """
        act_quant_op_list = []
        if self._quantize_op:
            for _op_type in self._quantize_op:
                if _op_type in self.observer_operation_types:
                    act_quant_op_list.append(_op_type)
        else:
            act_quant_op_list = [
                'mul',
                'matmul',
                'matmul_v2',
            ]
        return act_quant_op_list

    @property
    def observer_operation_types(self):
        """
        Operation type list for observer in quantization. These nodes only count the
        calibration boundary scale and do not participate in the fake quantization.
        In order to facilitate the prediction of the prediction engine, quant
        and dequant nodes will be inserted after these ops when exporting the model.
        """
        return [
            "pool2d",
            "elementwise_add",
            "concat",
            "softmax",
            "argmax",
            "transpose",
            "equal",
            "gather",
            "greater_equal",
            "greater_than",
            "less_equal",
            "less_than",
            "mean",
            "not_equal",
            "reshape",
            "reshape2",
            "dropout",
            "bilinear_interp",
            "nearest_interp",
            "trilinear_interp",
            "slice",
            "squeeze",
            "elementwise_sub",
            "mul",
            "matmul",
            "relu",
            "relu6",
            "leaky_relu",
            "tanh",
            "swish",
            "transpose",
            "transpose2",
            "sigmoid",
            "pad2d",
            "flatten",
            "flatten2",
            "batch_norm",
            "layer_norm",
            "matmul_v2",
            "split",
            "flatten_contiguous_range",
            "squeeze2",
            "nearest_interp_v2",
            "bilinear_interp",
            "bilinear_interp_v2",
            "fill_constant_batch_size_like",
            "arg_max",
            "abs",
            "assign",
            "cast",
            "clip",
            "box_coder",
            "crop",
            "cumsum",
            "elementwise_mul",
            "elementwise_pow",
            "expand_v2",
            "fill_any_like",
            "fill_constant",
            "gelu",
            "hard_sigmoid",
            "hard_swish",
            "instance_norm",
            "lookup_table",
            "lookup_table_v2",
            "norm",
            "p_norm",
            "pad3d",
            "pow",
            "prelu",
            "reduce_mean",
            "unsqueeze",
            "unsqueeze2",
            "logical_and",
            "logical_not",
            "meshgrid",
            "roi_align",
            "strided_slice",
            "where",
            "grid_sampler",
            "tile",
            "group_norm",
            "reduce_sum",
            "square",
            "softplus",
            "shuffle_channel",
            "reduce_max",
            "scale",
        ]


class TensorRTQuantizer(BaseQuantizer):
    """
    TensorRT quantization configuration class.
    """

    def __init__(
        self,
        quant_operation_types=None,
    ):
        super().__init__()

    @property
    def activation_quant_operation_types(self):
        """
        Operation type list which should support activation quantization.
        And before these ops, quant dequant nodes will be inserted.
        """
        return [
            "pool2d",
            "elementwise_add",
            "elementwise_sub",
            "elementwise_mul",
            "elementwise_pow",
            "concat",
            "softmax",
            "argmax",
            "mean",
            "relu",
            "relu6",
            "leaky_relu",
            "tanh",
            "swish",
            "softplus",
            "gelu",
            "hard_sigmoid",
            "hard_swish",
            "sigmoid",
            "layer_norm",
            "matmul_v2",
            "split",
            "bilinear_interp",
            "nearest_interp",
            "trilinear_interp",
            "nearest_interp_v2",
            "bilinear_interp",
            "bilinear_interp_v2",
            "clip",
            "pow",
            "reduce_mean",
            "reduce_sum",
            "reduce_max",
        ]


class MKLDNNQuantizer(BaseQuantizer):
    """
    MKLDNN quantization configuration class.
    """

    def __init__(
        self,
        quant_operation_types=None,
    ):
        super().__init__()

    @property
    def activation_quant_operation_types(self):
        """
        Operation type list which should support activation quantization.
        And before these ops, quant dequant nodes will be inserted.
        """
        return [
            "pool2d",
            "elementwise_add",
            "elementwise_mul",
            "concat",
            "nearest_interp",
            "nearest_interp_v2",
            "split",
        ]
