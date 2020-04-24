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
    'CrossEntropyLoss',
    'MSELoss',
    'L1Loss',
    #    'NLLLoss',
    'BCELoss'
]


class CrossEntropyLoss(fluid.dygraph.Layer):
    """
    This operator implements the cross entropy loss function. This OP combines `softmax`,
    `cross_entropy`, and `reduce_sum`/`reduce_mean` together.

    It is useful when training a classification problem with `C` classes.
    If provided, the optional argument `weight` should be a 1D Variable assigning
    weight to each of the classes.

    For predictions label, and target label, the loss is calculated as follows.
    .. math::

        loss_j =  -\\text{input[class]} +
        \\log\\left(\\sum_{i=0}^{K}\\exp(\\text{input}_i)\\right), j = 1,..., K

    If weight is not `None`:
    .. math::

        loss_j =  \\text{weight[class]}(-\\text{input[class]} +
        \\log\\left(\\sum_{i=0}^{K}\\exp(\\text{input}_i)\\right)), j = 1,..., K

    Parameters:
        input (Variable): Input tensor, the data type is float32,
            float64, int32, int64.
        label (Variable): Label tensor, the data type is float32,
            float64, int32, int64.
        weight (Variable, optional): Weight tensor, a manual rescaling weight given
            to each class. It has the same dimensions as class number and the data type
            is float32, float64, int32, int64. Default is ``'None'``.
        reduction (str, optional): Indicate how to average the loss by batch_size,
            the candicates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned;
            If :attr:`size_average` is ``'sum'``, the reduced sum loss is returned.
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned.
            Default is ``'mean'``.
    Returns:
        The tensor variable storing the cross_entropy_loss of input and label.
    Return type: Variable.
    Examples:
        .. code-block:: python

            # declarative mode
            import paddle
            import paddle.fluid as fluid
            import numpy as np

            input = fluid.layers.data(name='input', shape=[5, 100], dtype='float32')
            label = fluid.layers.data(name='label', shape=[5, 1], dtype='int64')
            weight = fluid.layers.data(name='weight', shape=[100], dtype='float32')
            ce_loss = paddle.nn.loss.CrossEntropyLoss(weight=weight, reduction='mean')
            output = ce_loss(input,label)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            input_data = np.random.random([5, 100]).astype("float32")
            label_data = np.array([[1], [9], [40], [50], [90]]).astype("int64")
            weight_data = np.random.random([100]).astype("float32")
            output = exe.run(fluid.default_main_program(),
                        feed={"input": input_data, "label": label_data,"weight": weight_data},
                        fetch_list=[output],
                        return_numpy=True)
            print(output)

            # imperative mode
            import paddle.fluid.dygraph as dg
            with dg.guard(place) as g:
                input = dg.to_variable(input_data)
                label = dg.to_variable(label_data)
                weight = dg.to_variable(weight_data)
                ce_loss = paddle.nn.loss.CrossEntropyLoss(weight=weight, reduction='mean')
                output = ce_loss(input, label)
                print(output.numpy())
    """

    def __init__(self, weight=None, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, label):
        fluid.data_feeder.check_variable_and_dtype(
            input, 'input', ['float32', 'float64', 'int32', 'int64'],
            'cross_entropy_loss')
        fluid.data_feeder.check_variable_and_dtype(
            label, 'label', ['float32', 'float64', 'int32', 'int64'],
            'cross_entropy_loss')

        if self.reduction not in ['sum', 'mean', 'none']:
            raise ValueError(
                "The value of 'reduction' in cross_entropy_loss should be 'sum', 'mean' or 'none',"
                " but received %s, which is not allowed." % self.reduction)

        softmax_out = fluid.layers.softmax(input)
        if self.weight is not None:
            if isinstance(self.weight, fluid.framework.Variable):
                softmax_out = fluid.layers.elementwise_pow(
                    softmax_out, self.weight, axis=-1)
            else:
                raise ValueError(
                    "The weight' is not a Variable, please convert to Variable.")

        out = fluid.layers.cross_entropy(softmax_out, label)

        if self.reduction == 'sum':
            return fluid.layers.reduce_sum(out)
        elif self.reduction == 'mean':
            return fluid.layers.reduce_mean(out)
        else:
            return out


class MSELoss(fluid.dygraph.layers.Layer):
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

    where `input` and `label` are `float32` tensors of same shape.

    Parameters:
        input (Variable): Input tensor, the data type is float32,
        label (Variable): Label tensor, the data type is float32,
        reduction (string, optional): The reduction method for the output,
            could be 'none' | 'mean' | 'sum'.
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned. 
            If :attr:`size_average` is ``'sum'``, the reduced sum loss is returned. 
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned. 
            Default is ``'mean'``.

    Returns:
        The tensor variable storing the MSE loss of input and label.

    Return type:
        Variable.

    Examples:
        .. code-block:: python

            import numpy as np
            import paddle
            from paddle import fluid
            import paddle.fluid.dygraph as dg

            mse_loss = paddle.nn.loss.MSELoss()
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
        if not fluid.framework.in_dygraph_mode():
            fluid.data_feeder.check_variable_and_dtype(input, 'input',
                                                       ['float32'], 'MSELoss')
            fluid.data_feeder.check_variable_and_dtype(label, 'label',
                                                       ['float32'], 'MSELoss')

        square_out = fluid.layers.square(
            fluid.layers.elementwise_sub(input, label))
        if self.reduction == 'none':
            return square_out

        reduce_op = 'reduce_mean'
        if self.reduction == 'sum':
            reduce_op = 'reduce_sum'

        return getattr(fluid.layers, reduce_op)(square_out)


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


class BCELoss(fluid.dygraph.Layer):
    """
    This interface is used to construct a callable object of the ``BCELoss`` class.
    The BCELoss layer measures the binary_cross_entropy loss between input predictions 
    and target labels. The binary_cross_entropy loss can be described as:

    If :attr:`weight` is set, the loss is:

    .. math::
        Out = -1 * weight * (label * log(input) + (1 - label) * log(1 - input))
    If :attr:`weight` is None, the loss is:

    .. math::
        Out = -1 * (label * log(input) + (1 - label) * log(1 - input))

    If :attr:`reduction` set to ``'none'``, the unreduced loss is:

    .. math::
        Out = Out
    If :attr:`reduction` set to ``'mean'``, the reduced mean loss is:

    .. math::
        Out = MEAN(Out)
    If :attr:`reduction` set to ``'sum'``, the reduced sum loss is:

    .. math::
        Out = SUM(Out)

    Note that the input predictions always be the output of sigmoid, and the target labels 
    should be numbers between 0 and 1.

    The shape of input predictions and target labels are [N, *], where N is batch_size and `*` 
    means any number of additional dimensions. If ``reduction`` is ``'none'``, the shape of 
    output is scalar, else the shape of output is same as input.

    Parameters:
        weight (Variable, optional): A manual rescaling weight given to the loss of each 
            batch element. If given, has to be a Variable of size nbatch and the data type
            is float32, float64. Default is ``'None'``.
        reduction (str, optional): Indicate how to average the loss by batch_size, 
            the candicates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned;
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned; 
            If :attr:`reduction` is ``'sum'``, the summed loss is returned.
            Default is ``'mean'``.

    Returns: 
        A callable object of BCELoss.

    Examples:
        .. code-block:: python

            # declarative mode
            import paddle.fluid as fluid
            import numpy as np
            import paddle
            input = fluid.data(name="input", shape=[3, 1], dtype='float32')
            label = fluid.data(name="label", shape=[3, 1], dtype='float32')
            bce_loss = paddle.nn.loss.BCELoss()
            output = bce_loss(input, label)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
    
            input_data = np.array([0.5, 0.6, 0.7]).astype("float32")
            label_data = np.array([1.0, 0.0, 1.0]).astype("float32")
            output_data = exe.run(fluid.default_main_program(),
                    feed={"input":input_data, "label":label_data},
                    fetch_list=[output],
                    return_numpy=True)
    
            print(output_data)  # [array([0.65537095], dtype=float32)]
            
            # imperative mode
            import paddle.fluid.dygraph as dg
            with dg.guard(place) as g:
                input = dg.to_variable(input_data)
                label = dg.to_variable(label_data)
                output = bce_loss(input, label)
                print(output.numpy())  # [0.65537095]
    """

    def __init__(self, weight=None, reduction='mean'):
        if reduction not in ['sum', 'mean', 'none']:
            raise ValueError(
                "The value of 'reduction' in bce_loss should be 'sum', 'mean' or 'none', but "
                "received %s, which is not allowed." % reduction)

        super(BCELoss, self).__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, label):
        dtype = self._helper.input_dtype(input)

        fluid.data_feeder.check_variable_and_dtype(
            input, 'input', ['float32', 'float64'], 'bce_loss')
        fluid.data_feeder.check_variable_and_dtype(
            label, 'label', ['float32', 'float64'], 'bce_loss')

        out = self._helper.create_variable_for_type_inference(dtype=input.dtype)
        self._helper.append_op(
            type='bce_loss',
            inputs={
                'X': [input],
                'Label': [label],
            },
            outputs={'Out': [out]})

        if self.weight is not None:
            if isinstance(self.weight, fluid.framework.Variable):
                w = self.weight
                out = fluid.layers.elementwise_mul(out, w, axis=-1)
            else:
                raise ValueError(
                    "The weight is not a Variable, please convert to Variable.")

        if self.reduction == 'sum':
            return fluid.layers.reduce_sum(out)
        elif self.reduction == 'mean':
            return fluid.layers.reduce_mean(out)
        else:
            return out
