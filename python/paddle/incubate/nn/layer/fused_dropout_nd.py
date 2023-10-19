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

import paddle
from paddle import _legacy_C_ops
from paddle.framework import in_dynamic_mode


class FusedDropout(paddle.nn.Layer):
    r"""
    Dropout is a regularization technique for reducing overfitting by preventing
    neuron co-adaption during training as described in the paper:
    `Improving neural networks by preventing co-adaptation of feature detectors <https://arxiv.org/abs/1207.0580>`_
    The dropout operator randomly sets the outputs of some units to zero, while upscale others
    according to the given dropout probability.

    It is an optimized implementation for ``paddle.nn.Dropout``.

    In dygraph mode, please use ``eval()`` to switch to evaluation mode, where dropout is disabled.

    Parameters:
        p (float|int, optional): Probability of setting units to zero. Default: 0.5
        axis (int|list|tuple, optional): The axis along which the dropout is performed. Default: None.
        mode(str, optional): ['upscale_in_train'(default) | 'downscale_in_infer']

                               1. upscale_in_train (default), upscale the output at training time

                                  - train: :math:`out = input \times \frac{mask}{(1.0 - p)}`
                                  - inference: :math:`out = input`

                               2. downscale_in_infer, downscale the output at inference

                                  - train: :math:`out = input \times mask`
                                  - inference: :math:`out = input \times (1.0 - p)`
        name (str, optional): Name for the operation, Default: None. For more information, please refer to :ref:`api_guide_Name`.

    Shape:
        - input: N-D tensor.
        - output: N-D tensor, the same shape as input.


    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(2023)

            >>> x = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype="float32")
            >>> m = paddle.incubate.nn.FusedDropout(p=0.5)

            >>> y_train = m(x)
            >>> print(y_train)
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[0., 0., 6.],
             [0., 0., 0.]])

            >>> m.eval()  # switch the model to test phase
            >>> y_test = m(x)
            >>> print(y_test)
            Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[1., 2., 3.],
             [4., 5., 6.]])
    """

    def __init__(self, p=0.5, axis=None, mode="upscale_in_train", name=None):
        super().__init__()

        if not isinstance(p, (float, int)):
            raise TypeError("p argument should be a number")
        if p < 0 or p > 1:
            raise ValueError("p argument should between 0 and 1")

        mode = (
            'downgrade_in_infer' if mode == 'downscale_in_infer' else mode
        )  # semantic transfer
        if mode not in ('downscale_in_infer', 'upscale_in_train'):
            raise ValueError(
                "mode argument should be 'downscale_in_infer' or 'upscale_in_train'"
            )

        if axis and not isinstance(axis, (int, list, tuple)):
            raise TypeError("datatype of axis argument should be int or list")

        self.p = p
        self.mode = mode
        self.name = name

        self.axis = None
        if axis is not None:
            self.axis = [axis] if isinstance(axis, int) else list(axis)

    def forward(self, input):
        # fast return for p == 0
        if self.p == 0:
            return input

        if self.axis is not None and in_dynamic_mode():
            seed = None
            if paddle.static.default_main_program().random_seed != 0:
                seed = paddle.static.default_main_program().random_seed

            out, mask = _legacy_C_ops.dropout_nd(
                input,
                'dropout_prob',
                self.p,
                'is_test',
                not self.training,
                'fix_seed',
                seed is not None,
                'seed',
                seed if seed is not None else 0,
                'dropout_implementation',
                self.mode,
                'axis',
                self.axis,
            )
        else:
            out = paddle.nn.functional.dropout(
                input,
                p=self.p,
                axis=self.axis,
                training=self.training,
                mode=self.mode,
                name=self.name,
            )
        return out

    def extra_repr(self):
        name_str = f', name={self.name}' if self.name else ''
        return f'p={self.p}, axis={self.axis}, mode={self.mode}{name_str}'
