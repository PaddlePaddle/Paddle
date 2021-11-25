#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from ...fluid.initializer import Initializer
from ...fluid.data_feeder import check_variable_and_dtype
from ...fluid.core import VarDesc
from ...fluid import unique_name, framework

__all__ = []


class Dirac(Initializer):
    """Initialize the 3D/4D/5D Tensor with Dirac delta function.
    
    It can reserve the feature of convolution layer input, which means that
    as many channels are reserved as possible.

    In this initialize method, elements in the middle of convolution kernels will
    be set to 1 . The formula can be described as:

    $ Assuming:  N=min(in\_channels, out\_channels)$

    $ X[d, d, shape[2]//2, shape[3]//2, ...]=1,  \   d=0,1...N$

    Args:
        groups(int): 0-dimension of the Tensor will be divided by groups, each group has the same value.
        name(str, optional): The default value is None. Normally there is no need for user to set this
            property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        Dirac initializer instance objects.

    Examples:
        .. code-block:: python

            import paddle
            
            #1.For kernel_size is uneven number:
            
            attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Dirac())
            conv = paddle.nn.Conv1D(3, 2, 3, weight_attr=attr)
            conv.weight
            # Tensor(shape=[2, 3, 3], dtype=float32, place=CPUPlace, stop_gradient=False,
            #       [[[0., 1., 0.],
            #         [0., 0., 0.],
            #         [0., 0., 0.]],
            # 
            #        [[0., 0., 0.],
            #         [0., 1., 0.],
            #         [0., 0., 0.]]])

            input = paddle.rand([8, 3, 10])
            output = conv(input)
            output == input[:, 0:2, 1:9]  
            # output.shape is [8, 2, 8], It means output is almost the same with input, 2 channels are reserved


            #2. For kernel_size is even number:
            attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Dirac())
            conv = paddle.nn.Conv1D(3, 2, 4, weight_attr=attr)
            conv.weight
            # Tensor(shape=[2, 3, 4], dtype=float32, place=CPUPlace, stop_gradient=False,
            #       [[[0., 0., 1., 0.],
            #         [0., 0., 0., 0.],
            #         [0., 0., 0., 0.]],
            # 
            #        [[0., 0., 0., 0.],
            #         [0., 0., 1., 0.],
            #         [0., 0., 0., 0.]]])
    """

    def __init__(self, groups=1, name=None):
        assert groups > 0 and isinstance(
            groups, int), " 'groups' must be a positive integer. "
        super(Dirac, self).__init__()
        self._groups = groups

    def __call__(self, var, block=None):
        """Initialize the input tensor with dirac initializer.

        Args:
            var(Tensor): Tensor that needs to be initialized.
            block(Block, optional): The block in which initialization ops
                   should be added. Used in static graph only, default None.

        Returns:
            The most critical OP(scatter) in this initializer, which contains 7~8 ops in total.
        """
        block = self._check_block(block)
        assert isinstance(var, framework.Parameter)
        assert isinstance(block, framework.Block)
        check_variable_and_dtype(
            var, "Out", ['float16', 'bfloat16', 'float32', 'float64'], 'Dirac')

        assert len(var.shape) in [
            3, 4, 5
        ], "Only Tensor with 3/4/5 dimensions can be initialized by Dirac"
        assert (var.shape[0] % self._groups
                ) == 0, "Tensor 0-dimension must be divisible by groups"

        if var.dtype != VarDesc.VarType.FP32:
            out_var = block.create_var(
                name=unique_name.generate(".".join(['dirac', var.name, 'tmp'])),
                shape=var.shape,
                dtype=VarDesc.VarType.FP32,
                type=VarDesc.VarType.LOD_TENSOR,
                persistable=False)
        else:
            out_var = var

        block.append_op(
            type='fill_constant',
            inputs={},
            outputs={'Out': out_var},
            attrs={
                'value': float(0),
                'dtype': out_var.dtype,
                'shape': out_var.shape,
            },
            stop_gradient=True)

        origin_shape = var.shape
        num_per_group = origin_shape[0] // self._groups
        min_shape = min(num_per_group, origin_shape[1])

        idx_list = []
        value_list = []
        strides = []
        prod = 1
        for dim in reversed(origin_shape):
            strides.insert(0, prod)
            prod *= dim
        for i in range(self._groups):
            for j in range(min_shape):
                value_list.append(1.0)
                offset = 0
                for (k, stride) in enumerate(strides):
                    if (k == 0):
                        offset += (j + i * num_per_group) * stride
                    elif (k == 1):
                        offset += j * stride
                    else:
                        offset += origin_shape[k] // 2 * stride
                idx_list.append(offset)

        block.append_op(
            type="reshape",
            inputs={"X": out_var},
            attrs={'shape': [-1]},
            outputs={"Out": out_var},
            stop_gradient=True)

        index_tensor = block.create_var(
            name=unique_name.generate('scatter_index'),
            persistable=False,
            stop_gradient=True)

        block.append_op(
            type='assign_value',
            outputs={'Out': index_tensor},
            attrs={
                'dtype': VarDesc.VarType.INT64,
                'shape': [len(idx_list)],
                'int64_values': idx_list
            },
            stop_gradient=True)

        value_tensor = block.create_var(
            name=unique_name.generate('scatter_value'),
            persistable=False,
            stop_gradient=True)

        block.append_op(
            type='assign_value',
            outputs={'Out': value_tensor},
            attrs={
                'dtype': VarDesc.VarType.FP32,
                'shape': [len(value_list)],
                'fp32_values': value_list
            },
            stop_gradient=True)

        op = block.append_op(
            type="scatter",
            inputs={
                "X": out_var,
                "Ids": index_tensor,
                "Updates": value_tensor
            },
            attrs={'overwrite': True},
            outputs={"Out": out_var},
            stop_gradient=True)

        block.append_op(
            type="reshape",
            inputs={"X": out_var},
            attrs={'shape': origin_shape},
            outputs={"Out": out_var},
            stop_gradient=True)

        if var.dtype != VarDesc.VarType.FP32:
            block.append_op(
                type="cast",
                inputs={"X": out_var},
                outputs={"Out": var},
                attrs={"in_dtype": out_var.dtype,
                       "out_dtype": var.dtype},
                stop_gradient=True)

        if not framework.in_dygraph_mode():
            var.op = op
        return op
