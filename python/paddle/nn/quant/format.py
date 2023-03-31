# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
"""Define some layers used to export quantization model with ONNX style."""
import abc
from typing import List, Tuple

import paddle
from paddle import _legacy_C_ops as _C_ops
from paddle.framework import in_dygraph_mode
from paddle.nn import Layer


class LinearQuanterDequanter(Layer):
    def __init__(self, quanter, dequanter):
        super().__init__()
        self._quanter = quanter
        self._dequanter = dequanter

    def forward(self, input):
        out = input
        if self._quanter is not None:
            out = self._quanter(out)
        if self._dequanter is not None:
            out = self._dequanter(out)
        return out

    @staticmethod
    def from_quanter(quanter):
        assert quanter is not None
        return LinearQuanterDequanter(
            LinearQuanter.from_quanter(quanter),
            LinearDequanter.from_quanter(quanter),
        )


class LinearQuanter(Layer):
    def __init__(self, scales, zero_point=None, quant_axis=None, bit_length=8):
        super().__init__()
        self._scales = paddle.to_tensor(scales, dtype="float32")
        self._zero_point = (
            paddle.zeros([1], dtype="float32")
            if zero_point is None
            else paddle.to_tensor(zero_point)
        )
        self._quant_axis = -1 if quant_axis is None else quant_axis
        self._bit_length = bit_length

    def forward(self, input):
        if in_dygraph_mode():
            return _C_ops.quantize_linear(
                input,
                self._scales,
                self._zero_point,
                "quant_axis",
                self._quant_axis,
                "bit_length",
                self._bit_length,
            )
        else:
            out = self._helper.create_variable_for_type_inference(input.dtype)
            self._helper.append_op(
                type='quantize_linear',
                inputs={
                    'X': input,
                    'Scale': self._scales,
                    'ZeroPoint': self._zero_point,
                },
                outputs={'Y': out},
                attrs={
                    'quant_axis': self._quant_axis,
                    'bit_length': self._bit_length,
                },
            )
            return out

    @staticmethod
    def from_quanter(quanter):

        return LinearQuanter(
            quanter.scales(),
            zero_point=quanter.zero_points(),
            quant_axis=quanter.quant_axis(),
            bit_length=quanter.bit_length(),
        )


class LinearDequanter(Layer):
    def __init__(self, scales, zero_point=None, quant_axis=None, bit_length=8):
        super().__init__()
        self._scales = paddle.to_tensor(scales, dtype="float32")
        self._zero_point = (
            paddle.zeros([1], dtype="float32")
            if zero_point is None
            else paddle.to_tensor(zero_point)
        )
        self._quant_axis = -1 if quant_axis is None else quant_axis
        self._bit_length = bit_length

    def forward(self, input):
        if in_dygraph_mode():
            return _C_ops.dequantize_linear(
                input,
                self._scales,
                self._zero_point,
                "quant_axis",
                self._quant_axis,
                "bit_length",
                self._bit_length,
            )
        else:
            out = self._helper.create_variable_for_type_inference(input.dtype)
            self._helper.append_op(
                type='dequantize_linear',
                inputs={
                    'X': input,
                    'Scale': self._scales,
                    'ZeroPoint': self._zero_point,
                },
                outputs={'Y': out},
                attrs={
                    'quant_axis': self._quant_axis,
                    'bit_length': self._bit_length,
                },
            )
            return out

    @staticmethod
    def from_quanter(quanter):
        return LinearDequanter(
            quanter.scales(),
            zero_point=quanter.zero_points(),
            quant_axis=quanter.quant_axis(),
            bit_length=quanter.bit_length(),
        )


class ConvertibleQuantedLayer(Layer, metaclass=abc.ABCMeta):
    r"""Abstract class to help convert quantized layer to inference model.
    It defines some functions to convert quantizers and observers to quantize
    or dequantize operators that maintain the quantization parameters used
    during inference.
    Examples:
       .. code-block:: python

            # Given codes in ./customized_quanter.py
            class CustomizedQuantedLayer(ConvertibleQuantedLayer):
                def __init__(self):
                    super().__init__()
                    self.weight_a = paddle.create_parameter(shape=[1], dtype='float32')
                    self.weight_b = paddle.create_parameter(shape=[1], dtype='float32')
                    self.quanter_for_weight_a = None
                    self.activation_weight = None
                def forward(self, input):
                    qweight_a = self.quanter_for_weight_a(self.weight_a)
                    weight_b = self.weight_b
                    qinput = self.activation_weight(input)
                    // compute with qweight_a, weight_b and qinput.
                    return qweight * qinput + weight_b

                def weights_to_quanters(self):
                    return [('weight_a', 'quanter_for_weight_a')]

                def activation_quanters(self):
                    return ['activation_weight']
    """

    def __init__(self):
        super().__init__()
        self.converted = False

    @abc.abstractmethod
    def weights_to_quanters(self) -> List[Tuple[str, str]]:
        r"""Get the name pairs of weights to be quantized and their corresponding
        quantizers. In the convert function of this abstract class, it will call
        the ‘weights_to_quanters’ function and do something as follows:
        For each pair, the quantizer will be converted to a quantize operator and
        a dequantize operator. Then, the weight will be quantized by the quantize
        operator. Finally, the quantize operator will be removed and the weights
        will be stored in integer data type.

        Returns: A list of name pairs. Each pair contains two names. The first is name of weight
        to be quantized and the second is name of corresponding quanter.
        """
        pass

    @abc.abstractmethod
    def activation_quanters(self) -> List[str]:
        r"""Get the names of quanters used to quantize activations.
        All the quanters or observers returned by this function will be converted to quantize
        and dequantize operators for deployment.
        Returns: A list of quanter names.
        """
        pass

    def _convert_quanter_to_qdq(self, quanter_name) -> LinearQuanterDequanter:
        r"""Convert quanter to an instance of LinearQuanterDequanter."""
        assert hasattr(
            self, quanter_name
        ), f"{quanter_name} is not attribute of current layer."
        quanter = getattr(self, quanter_name)
        if quanter is None:
            return None
        quanter = LinearQuanterDequanter.from_quanter(quanter)
        setattr(self, quanter_name, quanter)
        self._sub_layers[quanter_name] = quanter
        return quanter

    def _quant_weights(self, weight_name, quanter):
        r"""Quantize the weight by given quanter."""
        weight = getattr(self, weight_name)
        qweight = quanter(weight)
        weight.set_value(qweight)

    def _convert(self):
        r"""Convert current layer to onnx style for inference."""
        assert not self.converted, "The model should be converted only once."
        for weight_name, quanter_name in self.weights_to_quanters():
            qdq = self._convert_quanter_to_qdq(quanter_name)
            if qdq is not None:
                self._quant_weights(weight_name, qdq._quanter)
                qdq._quanter = None
                qdq._sub_layers['_quanter'] = None

        for quanter_name in self.activation_quanters():
            self._convert_quanter_to_qdq(quanter_name)

        self.converted = True
