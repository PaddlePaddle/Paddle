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


from paddle import _C_ops

from ...fluid import core, framework, unique_name
from ...fluid.data_feeder import check_variable_and_dtype
from ...fluid.framework import _current_expected_place, in_dygraph_mode
from ...fluid.initializer import Initializer

__all__ = []


class UniformInitializer(Initializer):
    """Implements the random uniform distribution initializer

    Args:
        low (float): lower boundary of the uniform distribution
        high (float): upper boundary of the uniform distribution
        seed (int): random seed
        diag_num (int): the number of diagonal elements to initialize.
            If set to 0, diagonal initialization will be not performed.
        diag_step (int): Step size between two diagonal elements,
            which is generally the width of the square matrix.
        diag_val (float): the value of the diagonal element to be initialized,
            default 1.0. It takes effect only if the diag_num is greater than 0.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            x = fluid.data(name='x', shape=[None, 1], dtype='float32')
            fc = fluid.layers.fc(input=x, size=10,
                param_attr=fluid.initializer.Uniform(low=-0.5, high=0.5))
    """

    def __init__(
        self, low=-1.0, high=1.0, seed=0, diag_num=0, diag_step=0, diag_val=1.0
    ):
        assert low is not None
        assert high is not None
        assert high >= low
        assert seed is not None
        assert diag_num is not None
        assert diag_step is not None
        assert diag_val is not None
        if diag_num > 0 or diag_step > 0:
            assert diag_num > 0 and diag_step > 0
        super().__init__()
        self._low = low
        self._high = high
        self._seed = seed
        self._diag_num = diag_num
        self._diag_step = diag_step
        self._diag_val = diag_val

    def forward(self, var, block=None):
        """Initialize the input tensor with Uniform distribution.

        Args:
            var(Tensor): Tensor that needs to be initialized.
            block(Block, optional): The block in which initialization ops
                   should be added. Used in static graph only, default None.

        Returns:
            The initialization op
        """
        block = self._check_block(block)

        assert isinstance(block, framework.Block)
        check_variable_and_dtype(
            var,
            "Out",
            ["uint16", "float16", "float32", "float64"],
            "uniform_random",
        )

        if self._seed == 0:
            self._seed = block.program.random_seed

        # to be compatible of fp16 initializers
        if var.dtype == core.VarDesc.VarType.FP16:
            out_dtype = core.VarDesc.VarType.FP32
            out_var = block.create_var(
                name=unique_name.generate(
                    ".".join(['uniform_random', var.name, 'tmp'])
                ),
                shape=var.shape,
                dtype=out_dtype,
                type=core.VarDesc.VarType.LOD_TENSOR,
                persistable=False,
            )
        else:
            out_dtype = var.dtype
            out_var = var

        if in_dygraph_mode():
            out_var = _C_ops.uniform(
                var.shape,
                out_dtype,
                self._low,
                self._high,
                self._seed,
                _current_expected_place(),
            )
            if var.dtype == core.VarDesc.VarType.FP16:
                var_tmp = _C_ops.cast(out_var, var.dtype)
                var_tmp._share_underline_tensor_to(var)
            else:
                out_var._share_underline_tensor_to(var)
            return None
        else:
            op = block.append_op(
                type="uniform_random",
                inputs={},
                outputs={"Out": out_var},
                attrs={
                    "shape": var.shape,
                    "dtype": out_dtype,
                    "min": self._low,
                    "max": self._high,
                    "seed": self._seed,
                    "diag_num": self._diag_num,
                    "diag_step": self._diag_step,
                    "diag_val": self._diag_val,
                },
                stop_gradient=True,
            )

            if var.dtype == core.VarDesc.VarType.FP16:
                block.append_op(
                    type="cast",
                    inputs={"X": out_var},
                    outputs={"Out": var},
                    attrs={"in_dtype": out_var.dtype, "out_dtype": var.dtype},
                )

            var.op = op
            return op


class ConstantInitializer(Initializer):
    """Implements the constant initializer

    Args:
        value (float32): constant value to initialize the variable

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            paddle.enable_static()
            x = fluid.data(name="data", shape=[8, 32, 32], dtype="float32")
            fc = fluid.layers.fc(
                input=x,
                size=10,
                param_attr=paddle.nn.initializer.ConstantInitializer(value=2.0))

    """

    def __init__(self, value=0.0, force_cpu=False):
        assert value is not None
        super().__init__()
        self._value = value
        self._force_cpu = force_cpu

    def forward(self, var, block=None):
        """Initialize the input tensor with constant.

        Args:
            var(Tensor): Tensor that needs to be initialized.
            block(Block, optional): The block in which initialization ops
                   should be added. Used in static graph only, default None.

        Returns:
            The initialization op
        """
        block = self._check_block(block)

        assert isinstance(var, framework.Variable) or isinstance(
            var, framework.EagerParamBase
        )
        assert isinstance(block, framework.Block)

        if in_dygraph_mode():
            place = _current_expected_place()
            if self._force_cpu:
                place = core.CPUPlace()
            _C_ops.full_(
                var, var.shape, str(float(self._value)), var.dtype, place
            )
            return None
        else:
            op = block.append_op(
                type="fill_constant",
                outputs={"Out": var},
                attrs={
                    "shape": var.shape,
                    "dtype": int(var.dtype),
                    "value": float(self._value),
                    'str_value': str(float(self._value)),
                    'force_cpu': self._force_cpu,
                },
                stop_gradient=True,
            )

            var.op = op
            return op
