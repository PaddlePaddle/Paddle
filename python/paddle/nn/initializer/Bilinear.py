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

import numpy as np

from paddle import _C_ops

from ...fluid import core, framework, unique_name
from ...fluid.framework import _current_expected_place, in_dygraph_mode
from .initializer import Initializer

__all__ = []


class Bilinear(Initializer):
    """
    This initializer can be used in transposed convolution operator to
    act as upsampling. Users can upsample a feature map with shape of
    (B, C, H, W) by any integer factor.

    Returns:
        Bilinear initializer instance objects.

    Examples:

        .. code-block:: python

            import math

            import paddle
            import paddle.nn as nn
            from paddle.regularizer import L2Decay

            factor = 2
            C = 2
            B = 8
            H = W = 32
            w_attr = paddle.ParamAttr(learning_rate=0.,
                                      regularizer=L2Decay(0.),
                                      initializer=nn.initializer.Bilinear())
            data = paddle.rand([B, 3, H, W], dtype='float32')
            conv_up = nn.Conv2DTranspose(3,
                                         out_channels=C,
                                         kernel_size=2 * factor - factor % 2,
                                         padding=int(
                                             math.ceil((factor - 1) / 2.)),
                                         stride=factor,
                                         weight_attr=w_attr,
                                         bias_attr=False)
            x = conv_up(data)

    Where, `out_channels=C` and `groups=C` means this is channel-wise transposed
    convolution. The filter shape will be (C, 1, K, K) where K is `kernel_size`,
    This initializer will set a (K, K) interpolation kernel for every channel
    of the filter identically. The resulting shape of the output feature map
    will be (B, C, factor * H, factor * W). Note that the learning rate and the
    weight decay are set to 0 in order to keep coefficient values of bilinear
    interpolation unchanged during training.

    """

    def __init__(self):
        """Constructor for BilinearInitializer."""
        super().__init__()

    def forward(self, var, block=None):
        """Initialize the input tensor with Bilinear initialization.

        Args:
            var(Tensor): Tensor that needs to be initialized.
            block(Block, optional): The block in which initialization ops
                   should be added. Used in static graph only, default None.

        Returns:
            The initialization op
        """
        block = self._check_block(block)

        if not isinstance(var, framework.Variable):
            raise ValueError("var must be framework.Variable.")

        if not isinstance(block, framework.Block):
            raise ValueError("block must be framework.Block.")

        shape = var.shape
        if len(shape) != 4:
            raise ValueError("the length of shape must be 4.")
        if shape[2] != shape[3]:
            raise ValueError("shape[2] must be equal to shape[3].")

        weight = np.zeros(np.prod(var.shape), dtype='float32')
        size = shape[3]
        # factor
        f = np.ceil(size / 2.0)
        # center
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        for i in range(np.prod(shape)):
            x = i % size
            y = (i / size) % size
            weight[i] = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
        weight = np.reshape(weight, shape)

        # to be compatible of fp16 initalizers
        if var.dtype in [
            core.VarDesc.VarType.FP16,
            core.VarDesc.VarType.BF16,
            core.VarDesc.VarType.FP64,
        ]:
            out_dtype = core.VarDesc.VarType.FP32
            out_var = block.create_var(
                name=unique_name.generate(
                    ".".join(['bilinear_init', var.name, 'tmp'])
                ),
                shape=var.shape,
                dtype=out_dtype,
                type=core.VarDesc.VarType.LOD_TENSOR,
                persistable=False,
            )
        else:
            out_dtype = var.dtype
            out_var = var

        if out_dtype == core.VarDesc.VarType.FP32:
            value_name = "fp32_values"
            values = [float(v) for v in weight.flat]
        else:
            raise TypeError("Unsupported dtype %s", var.dtype)

        if np.prod(shape) > 1024 * 1024:
            raise ValueError("The size of input is too big. ")

        if in_dygraph_mode():
            _C_ops.assign_value_(
                out_var,
                list(shape),
                out_dtype,
                values,
                _current_expected_place(),
            )
            if var.dtype in [
                core.VarDesc.VarType.FP16,
                core.VarDesc.VarType.BF16,
                core.VarDesc.VarType.FP64,
            ]:
                var_tmp = _C_ops.cast(out_var, var.dtype)
                var_tmp._share_underline_tensor_to(var)
            else:
                out_var._share_underline_tensor_to(var)
            return None
        else:
            op = block.append_op(
                type='assign_value',
                outputs={'Out': [out_var]},
                attrs={
                    'dtype': out_dtype,
                    'shape': list(shape),
                    value_name: values,
                },
            )

            if var.dtype in [
                core.VarDesc.VarType.FP16,
                core.VarDesc.VarType.BF16,
                core.VarDesc.VarType.FP64,
            ]:
                block.append_op(
                    type="cast",
                    inputs={"X": out_var},
                    outputs={"Out": var},
                    attrs={"in_dtype": out_var.dtype, "out_dtype": var.dtype},
                )

            var.op = op
            return op
