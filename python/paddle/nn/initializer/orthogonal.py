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
from ...tensor import diag, transpose, sign, qr, reshape

__all__ = []


class Orthogonal(Initializer):
    """The orthogonal initializer. The initialized tensor is (semi) orthogonal.

    Assuming that 'weight' will be initialized, its shape is [M, N].

    .. code-block:: text

        if M < N:
            The rows are orthogonal vectors
        elif M > N:
            The columns are orthogonal vectors
        else M = N:
            Both rows and columns are orthogonal vectors
    
    Only Tensor with 2 or more dimensions can initialized by Orthogonal.

    Args:
        gain(float, optional): The multiplication coefficient for initialized tensor. Default: 1.0.
        name(str, optional): The default value is None. Normally there is no need for user to set this
            property. For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A parameter initialized by orthogonal initialized.

    Examples:
        .. code-block:: python

            import paddle

            weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Orthogonal())
            linear = paddle.nn.Linear(10, 15, weight_attr=weight_attr)
            # linear.weight: X * X' = I

            linear = paddle.nn.Linear(15, 10, weight_attr=weight_attr)
            # linear.weight: X' * X = I
    """

    def __init__(self, gain=1.0, name=None):
        assert gain is not None, 'gain should not be None'
        super(Orthogonal, self).__init__()
        self._gain = gain

    def __call__(self, var, block=None):
        """Initialize the input tensor with orthogonal initializer.

        Args:
            var(Tensor): Tensor that needs to be initialized.
            block(Block, optional): The block in which initialization ops
                   should be added. Used in static graph only, default None.

        Returns:
            The last initialization op, it contain 8 ops in orthogonal initializer.
        """
        block = self._check_block(block)
        assert isinstance(var, framework.Parameter)
        assert isinstance(block, framework.Block)
        # 'qr' op only support float32/float64 now
        check_variable_and_dtype(var, "Out", ["float32", "float64"],
                                 "Orthogonal")

        self._seed = block.program.random_seed

        shape = var.shape
        assert len(
            shape
        ) >= 2, "Only Tensor with 2 or more dimensions can be initialized by Orthogonal"

        row = shape[0]
        col = 1
        for i in shape[1:]:
            col *= i

        flatten_shape = [max(row, col), min(row, col)]

        normal_var = block.create_var(
            name=unique_name.generate('.'.join(['gaussian_random', 'tmp'])),
            dtype=var.dtype,
            persistable=False,
            stop_gradient=True)
        block.append_op(
            type='gaussian_random',
            inputs={},
            outputs={'Out': normal_var},
            attrs={
                'mean': 0.0,
                'std': 1.0,
                'shape': flatten_shape,
                'seed': self._seed,
                'dtype': var.dtype
            },
            stop_gradient=True)

        q = block.create_var(
            name=unique_name.generate('.'.join(['qr', 'q', 'tmp'])),
            dtype=normal_var.dtype,
            persistable=False,
            stop_gradient=True)
        r = block.create_var(
            name=unique_name.generate('.'.join(['qr', 'r', 'tmp'])),
            dtype=normal_var.dtype,
            persistable=False,
            stop_gradient=True)
        block.append_op(
            type='qr',
            inputs={'X': [normal_var]},
            outputs={
                'Q': q,
                'R': r,
            },
            attrs={'mode': 'reduced'},
            stop_gradient=True)

        r_diag = block.create_var(
            name=unique_name.generate('.'.join(['diag', 'tmp'])),
            dtype=r.dtype,
            persistable=False,
            stop_gradient=True)
        block.append_op(
            type='diag_v2',
            inputs={'X': r},
            outputs={'Out': r_diag},
            attrs={'offset': 0,
                   'padding_value': 0},
            stop_gradient=True)

        r_sign = r_diag
        block.append_op(
            type='sign',
            inputs={'X': [r_diag]},
            outputs={'Out': r_sign},
            stop_gradient=True)

        block.append_op(
            type='elementwise_mul',
            inputs={'X': q,
                    'Y': r_sign},
            outputs={'Out': q},
            attrs={},
            stop_gradient=True)

        x_shape = block.create_var(
            name=unique_name.generate('.'.join(['transpose', 'shape', 'tmp'])),
            dtype=q.dtype,
            persistable=False,
            stop_gradient=True)
        if row < col:
            q_transpose = block.create_var(
                name=unique_name.generate('.'.join(['transpose', 'tmp'])),
                dtype=q.dtype,
                persistable=False,
                stop_gradient=True)
            block.append_op(
                type='transpose2',
                inputs={'X': q},
                outputs={'Out': q_transpose,
                         'XShape': x_shape},
                attrs={'axis': [1, 0]},
                stop_gradient=True)
            q = q_transpose

        block.append_op(
            type='reshape2',
            inputs={'X': q},
            outputs={'Out': q,
                     "XShape": x_shape},
            attrs={'shape': var.shape},
            stop_gradient=True)

        op = block.append_op(
            type='scale',
            inputs={'X': q},
            outputs={'Out': var},
            attrs={'scale': self._gain,
                   'bias': 0.0})

        return op
