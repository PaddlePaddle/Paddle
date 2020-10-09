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

# TODO: define the initializers of Constant in neural network
from ...fluid.initializer import Initializer
from ...fluid import framework
from ...fluid import unique_name
from ...fluid.core import VarDesc

__all__ = ['Constant']


class Constant(Initializer):
    """Implements the constant initializer

    Args:
        value (float32): constant value to initialize the variable 

    Examples:
        .. code-block:: python

    	    import paddle
            import paddle.nn as nn

            data = paddle.rand([30, 10, 32], dtype='float32')
            linear = nn.Linear(32,
                               64,
                               weight_attr=nn.initializer.Constant(value=2.0))
            res = linear(data)

    """

    def __init__(self, value=0.0):
        assert value is not None
        super(Constant, self).__init__()
        self._value = value
        self._force_cpu = False

    def __call__(self, var, block=None):
        """Initialize the input tensor with constant.

        Args:
            var(Tensor): Tensor that needs to be initialized.
            block(Block, optional): The block in which initialization ops
                   should be added. Used in static graph only, default None.

        Returns:
            The initialization op
        """
        block = self._check_block(block)

        assert isinstance(var, framework.Variable)
        assert isinstance(block, framework.Block)

        # to be compatible of fp16 initializers
        if var.dtype == VarDesc.VarType.FP16:
            out_dtype = VarDesc.VarType.FP32
            out_var = block.create_var(
                name=unique_name.generate(".".join(
                    ['constant_init', var.name, 'tmp'])),
                shape=var.shape,
                dtype=out_dtype,
                type=VarDesc.VarType.LOD_TENSOR,
                persistable=False)
        else:
            out_dtype = var.dtype
            out_var = var

        # Initialization Ops should be prepended and not appended
        op = block._prepend_op(
            type="fill_constant",
            outputs={"Out": out_var},
            attrs={
                "shape": var.shape,
                "dtype": int(out_dtype),
                "value": float(self._value),
                'force_cpu': self._force_cpu
            },
            stop_gradient=True)

        if var.dtype == VarDesc.VarType.FP16:
            block.append_op(
                type="cast",
                inputs={"X": out_var},
                outputs={"Out": var},
                attrs={"in_dtype": out_var.dtype,
                       "out_dtype": var.dtype})

        if not framework.in_dygraph_mode():
            var.op = op
        return op
