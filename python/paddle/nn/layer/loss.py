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

# TODO: define loss functions of neural network  
import paddle.fluid as fluid
__all__ = [
    #'NCELoss',
    #    'CrossEntropyLoss',
    #    'MSELoss',
    'L1Loss',
    #    'NLLLoss',
    #   'BCELoss'
]


class L1Loss(fluid.dygraph.Layer):
    """
    This interface is used to construct a callable object of the ``L1Loss`` class.
    The L1Loss layer calculates the L1 Loss of input predictions and target 
    labels as follows.

    If :attr:`reduction` set to ``'none'``, the unreduced loss is:
    .. math::
        Out = |input - label|
    If :attr:`reduction` set to ``'mean'``, the reduced mean loss is:
    .. math::
        Out = MEAN(|input - label|)
    If :attr:`reduction` set to ``'sum'``, the reduced sum loss is:
    .. math::
        Out = SUM(|input - label|)

    The shape of input predictions and target labels are [N, *], where N is batch_size and `*` 
    means any number of additional dimensions.
    If :attr:`reduction` is ``'none'``, the shape of output loss is [N, *], the same as input.
    If :attr:`reduction` is ``'mean'`` or ``'sum'``, the shape of output loss is [1], which means the output is a scalar.
    
    Parameters:
        reduction (str, optional): Indicate the reduction to apply to the loss, 
            the candicates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned; 
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned. 
            If :attr:`reduction` is ``'sum'``, the reduced sum loss is returned. 
            Default is ``'mean'``.
    Returns:
        A callable object of L1Loss.
    Examples:
        .. code-block:: python
            # declarative mode
            import paddle.fluid as fluid
            import numpy as np
            import paddle
            input = fluid.data(name="input", shape=[1])
            label = fluid.data(name="label", shape=[1])
            l1_loss = paddle.nn.loss.L1Loss(reduction='mean')
            output = l1_loss(input,label)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
    
            input_data = np.array([1.5]).astype("float32")
            label_data = np.array([1.7]).astype("float32")
            output_data = exe.run(fluid.default_main_program(),
                    feed={"input":input_data, "label":label_data},
                    fetch_list=[output],
                    return_numpy=True)
    
            print(output_data)  # [array([0.2], dtype=float32)]
            
            # imperative mode
            import paddle.fluid.dygraph as dg
            with dg.guard(place) as g:
                input = dg.to_variable(input_data)
                label = dg.to_variable(label_data)
                l1_loss = paddle.nn.loss.L1Loss(reduction='mean')
                output = l1_loss(input,label)
                print(output.numpy())  # [0.2]
    """

    def __init__(self, reduction='mean'):
        if reduction not in ['sum', 'mean', 'none']:
            raise ValueError(
                "The value of 'reduction' in L1Loss should be 'sum', 'mean' or 'none', but "
                "received %s, which is not allowed." % reduction)
        super(L1Loss, self).__init__()
        self.reduction = reduction

    def forward(self, input, label):
        fluid.data_feeder.check_variable_and_dtype(
            input, 'input', ['float32', 'float64', 'int32', 'int64'], 'l1_loss')
        fluid.data_feeder.check_variable_and_dtype(
            label, 'label', ['float32', 'float64', 'int32', 'int64'], 'l1_loss')

        unreduced = fluid.layers.elementwise_sub(input, label, act='abs')

        if self.reduction == 'sum':
            return fluid.layers.reduce_sum(unreduced)
        elif self.reduction == 'mean':
            return fluid.layers.reduce_mean(unreduced)
        else:
            return unreduced
