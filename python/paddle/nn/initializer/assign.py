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

from ...fluid import framework
from ...fluid import core
from ...fluid.core import VarDesc
from ...fluid import unique_name
from ...fluid.initializer import Initializer

__all__ = ['Assign']


class Assign(Initializer):
    """Init an parameter with an numpy array
    This op initialize the variable by numpy array.

    Args:
        value (numpy): numpy array, list, or paddle tensor to initialize the variable
        name(str, optional): The default value is None. Normally there is no need for user to set this
            property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A Tensor variable initialized by numpy.

    Examples:
        .. code-block:: python

            import paddle
            import numpy as np

            # numpy array
            data = paddle.ones(shape=[1, 2], dtype='float32')
            weight_attr=paddle.framework.ParamAttr(name="linear_weight", learning_rate=1.0,
                trainable=False, regularizer=None, initializer=paddle.nn.initializer.Assign(np.array([2,2])))
            bias_attr=paddle.framework.ParamAttr(name="linear_bias", learning_rate=1.0,
                trainable=False, regularizer=None, initializer=paddle.nn.initializer.Assign(np.array([2])))
            linear = paddle.nn.Linear(2, 2, weight_attr=weight_attr, bias_attr=bias_attr)
            res = linear(data)

            # python list
            data = paddle.ones(shape=[1, 2], dtype='float32')
            weight_attr=paddle.framework.ParamAttr(name="linear_weight", learning_rate=1.0,
                trainable=False, regularizer=None, initializer=paddle.nn.initializer.Assign([2,2]))
            bias_attr=paddle.framework.ParamAttr(name="linear_bias", learning_rate=1.0,
                trainable=False, regularizer=None, initializer=paddle.nn.initializer.Assign([2]))
            linear = paddle.nn.Linear(2, 2, weight_attr=weight_attr, bias_attr=bias_attr)
            res = linear(data)

            # paddle tensor
            data = paddle.ones(shape=[1, 2], dtype='float32')
            weight_attr=paddle.framework.ParamAttr(name="linear_weight", learning_rate=1.0,
                trainable=False, regularizer=None, initializer=paddle.nn.initializer.Assign(paddle.full([2], 2)))
            bias_attr=paddle.framework.ParamAttr(name="linear_bias", learning_rate=1.0,
                trainable=False, regularizer=None, initializer=paddle.nn.initializer.Assign(paddle.full([1], 2)))
            linear = paddle.nn.Linear(2, 2, weight_attr=weight_attr, bias_attr=bias_attr)
            res = linear(data)
    """

    def __init__(self, value):
        import numpy
        assert isinstance(value, (numpy.ndarray, list, core.VarBase))
        if (isinstance(value, list)):
            value = numpy.array(value)
        if (isinstance(value, core.VarBase)):
            value = value.numpy()
        super(Assign, self).__init__()
        self._value = value

    def __call__(self, var, block=None):
        """Initialize the input tensor with Numpy array.

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

        # to be compatible of fp16 initalizers
        if var.dtype == VarDesc.VarType.FP16:
            out_dtype = VarDesc.VarType.FP32
            np_value = self._value.astype("float32")
            out_var = block.create_var(
                name=unique_name.generate(".".join(
                    ['numpy_array_init', var.name, 'tmp'])),
                shape=var.shape,
                dtype=out_dtype,
                type=VarDesc.VarType.LOD_TENSOR,
                persistable=False)
        else:
            out_var = var
            out_dtype = var.dtype
            np_value = self._value

        # Initialization Ops should be prepended and not appended
        if out_dtype == VarDesc.VarType.FP32:
            value_name = "fp32_values"
            values = [float(v) for v in np_value.flat]
        elif out_dtype == VarDesc.VarType.INT32:
            value_name = "int32_values"
            values = [int(v) for v in np_value.flat]
        else:
            raise ValueError("Unsupported dtype %s", self._value.dtype)
        if self._value.size > 1024 * 1024 * 1024:
            raise ValueError("The size of input is too big. Please consider "
                             "saving it to file and 'load_op' to load it")
        op = block._prepend_op(
            type='assign_value',
            outputs={'Out': out_var},
            attrs={
                'dtype': out_dtype,
                'shape': list(self._value.shape),
                value_name: values
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
