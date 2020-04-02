# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from ..fluid import core
from ..fluid.layer_helper import LayerHelper
from ..fluid.framework import in_dygraph_mode
from ..fluid.data_feeder import check_variable_and_dtype
from ..fluid.dygraph.layers import Layer

__all__ = ['MSELoss']


class MSELoss(Layer):
    """
    **Mean Square Error Loss**
    Computes the mean square error (squared L2 norm) of given input and label.

    If :attr:`reduction` is set to ``'none'``, loss is calculated as:

    .. math::
        Out = (input - label)^2

    If :attr:`reduction` is set to ``'mean'``, loss is calculated as:

    .. math::
        Out = \operatorname{mean}((input - label)^2)

    If :attr:`reduction` is set to ``'sum'``, loss is calculated as:

    .. math::
        Out = \operatorname{sum}((input - label)^2)

    where `input` and `label` are `float32` tensors of arbitrary shapes.

    Parameters:
        reduction (string, optional): The reduction method for the output,
            could be 'none' | 'mean' | 'sum'.
            'none': no reduction will be applied
            'mean': the output will be averaged
            'sum': the output will be summed

    Examples:
        .. code-block:: python
        import numpy as np
        from paddle import fluid
        import paddle.fluid.dygraph as dg
        from paddle.nn.loss import MSELoss

        mse_loss = MSELoss()
        input = fluid.data(name="input", shape=[1])
        label = fluid.data(name="label", shape=[1])
        place = fluid.CPUPlace()
        input_data = np.array([1.5]).astype("float32")
        label_data = np.array([1.7]).astype("float32")

        # declarative mode
        output = mse_loss(input,label)
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        output_data = exe.run(
            fluid.default_main_program(),
            feed={"input":input_data, "label":label_data},
            fetch_list=[output],
            return_numpy=True)
        print(output_data)
        # [array([0.04000002], dtype=float32)]

        # imperative mode
        with dg.guard(place) as g:
            input = dg.to_variable(input_data)
            label = dg.to_variable(label_data)
            output = mse_loss(input, label)
            print(output.numpy())
            # [0.04000002]
    """

    def __init__(self, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['sum', 'mean', 'none']:
            raise ValueError(
                "'reduction' in 'MSELoss' should be 'sum', 'mean' or 'none', "
                "but received {}.".format(reduction))
        self.reduction = reduction

    def forward(self, input, label):
        if not in_dygraph_mode():
            check_variable_and_dtype(input, 'input', ['float32'], 'MSELoss')
            check_variable_and_dtype(label, 'label', ['float32'], 'MSELoss')

        helper = LayerHelper('MSELoss')
        minus_out = helper.create_variable_for_type_inference(input.dtype)
        helper.append_op(
            type='elementwise_sub',
            inputs={'X': [input],
                    'Y': [label]},
            outputs={'Out': [minus_out]})

        square_out = helper.create_variable_for_type_inference(input.dtype)
        helper.append_op(
            type='square',
            inputs={'X': [minus_out]},
            outputs={'Out': [square_out]})

        if self.reduction == 'none':
            return square_out

        reduce_op = 'reduce_mean'
        if self.reduction == 'sum':
            reduce_op = 'reduce_sum'

        attrs = {'dim': [0], 'keep_dim': False, 'reduce_all': True}

        if in_dygraph_mode():
            inputs = {'X': [square_out]}
            outs = getattr(core.ops, reduce_op)(inputs, attrs)
            return outs['Out'][0]

        out = helper.create_variable_for_type_inference(input.dtype)
        helper.append_op(
            type=reduce_op,
            inputs={'X': square_out},
            outputs={'Out': out},
            attrs=attrs)
        return out
