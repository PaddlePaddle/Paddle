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


from paddle import _C_ops, pir
from paddle.base import framework
from paddle.base.framework import in_dynamic_or_pir_mode

__all__ = ['L1Decay', 'L2Decay']


class WeightDecayRegularizer:
    """Base class for weight decay regularizers

    Defines the common interface of weight-decay regularizers.
    Weight-decay regularizers are added only during the backward
    pass for faster regularization. They add operations to the network
    that correspond to gradient of the regularization function.
    Users should not use this class directly, but need to use one
    of its implementations
    """

    def __init__(self):
        pass

    def __call__(self, param, grad, block):
        """Add corresponding weight decay operations to the network"""
        raise NotImplementedError()

    def __str__(self):
        """Debug string"""
        raise NotImplementedError()


class L1Decay(WeightDecayRegularizer):
    r"""
    Implement the L1 Weight Decay Regularization, which encourages the weights to be sparse.

    It can be set in :ref:`api_paddle_ParamAttr` or ``optimizer`` (such as :ref:`api_paddle_optimizer_Momentum` ).
    When set in ``ParamAttr`` , it only takes effect for trainable parameters in this layer. When set in
    ``optimizer`` , it takes effect for all trainable parameters. When set together, ``ParamAttr`` has
    higher priority than ``optimizer`` , which means that for a trainable parameter, if regularizer is defined
    in its ParamAttr, then the regularizer in Optimizer will be ignored. Otherwise the  regularizer
    in Optimizer will be used.

    In the implementation, the loss function of L1 Weight Decay Regularization is as follows:

    .. math::

        loss = coeff * reduce\_sum(abs(x))

    Args:
        coeff(float, optional): regularization coeff. Default:0.0.

    Examples:
        .. code-block:: python
            :name: code-example1

            >>> # Example1: set Regularizer in optimizer
            >>> import paddle
            >>> from paddle.regularizer import L1Decay

            >>> linear = paddle.nn.Linear(10, 10)
            >>> inp = paddle.rand(shape=[10, 10], dtype="float32")
            >>> out = linear(inp)
            >>> loss = paddle.mean(out)
            >>> beta1 = paddle.to_tensor([0.9], dtype="float32")
            >>> beta2 = paddle.to_tensor([0.99], dtype="float32")
            >>> momentum = paddle.optimizer.Momentum(
            ...     learning_rate=0.1,
            ...     parameters=linear.parameters(),
            ...     weight_decay=L1Decay(0.0001))
            >>> back = out.backward()
            >>> momentum.step()
            >>> momentum.clear_grad()

        .. code-block:: python
            :name: code-example2

            >>> # Example2: set Regularizer in parameters
            >>> # Set L1 regularization in parameters.
            >>> # Global regularizer does not take effect on my_conv2d for this case.
            >>> from paddle.nn import Conv2D
            >>> from paddle import ParamAttr
            >>> from paddle.regularizer import L1Decay

            >>> my_conv2d = Conv2D(
            ...         in_channels=10,
            ...         out_channels=10,
            ...         kernel_size=1,
            ...         stride=1,
            ...         padding=0,
            ...         weight_attr=ParamAttr(regularizer=L1Decay(coeff=0.01)),
            ...         bias_attr=False)
    """

    def __init__(self, coeff=0.0):
        assert coeff is not None
        super().__init__()
        self._coeff = coeff

    def __call__(self, param, grad, block):
        """Add L1 weight decay ops to network

        Adds L1 weight decay ops.
        L1WeightDecay = reg_coeff * sign(parameter)

        Args:
            param: parameter variable for which regularization is applied
            block: block in which variable is to be created

        Returns:
            new variable for weight decay
        """
        assert isinstance(
            param, (framework.Variable, pir.Value, pir.core.ParameterMeta)
        )
        assert isinstance(block, (framework.Block, pir.Block))

        if in_dynamic_or_pir_mode():
            sign = _C_ops.sign(param)
            return _C_ops.scale(sign, self._coeff, 0.0, True)
        else:
            sign = block.create_var(
                dtype=param.dtype, shape=param.shape, lod_level=param.lod_level
            )
            decay = block.create_var(
                dtype=param.dtype, shape=param.shape, lod_level=param.lod_level
            )
            # Append sign op
            block.append_op(
                type='sign', inputs={"X": param}, outputs={"Out": sign}
            )

            # Append scale op to the output of sign op
            block.append_op(
                type='scale',
                inputs={"X": sign},
                outputs={"Out": decay},
                attrs={"scale": self._coeff},
            )
            return decay

    def __str__(self):
        return "L1Decay, coeff=%f" % self._coeff


class L2Decay(WeightDecayRegularizer):
    r"""
    Implement the L2 Weight Decay Regularization, which helps to prevent the model over-fitting.

    It can be set in :ref:`api_paddle_ParamAttr` or ``optimizer`` (such as :ref:`api_paddle_optimizer_Momentum` ).
    When set in ``ParamAttr`` , it only takes effect for trainable parameters in this layer. When set in
    ``optimizer`` , it takes effect for all trainable parameters. When set together, ``ParamAttr`` has
    higher priority than ``optimizer`` , which means that for a trainable parameter, if regularizer is defined
    in its ParamAttr, then the regularizer in Optimizer will be ignored. Otherwise the  regularizer
    in Optimizer will be used.

    In the implementation, the loss function of L2 Weight Decay Regularization is as follows:

    .. math::

        loss = 0.5 * coeff * reduce\_sum(square(x))

    Args:
        coeff(float, optional): regularization coeff. Default:0.0

    Examples:
        .. code-block:: python
            :name: code-example1

            >>> # Example1: set Regularizer in optimizer
            >>> import paddle
            >>> from paddle.regularizer import L2Decay
            >>> linear = paddle.nn.Linear(10, 10)
            >>> inp = paddle.rand(shape=[10, 10], dtype="float32")
            >>> out = linear(inp)
            >>> loss = paddle.mean(out)
            >>> beta1 = paddle.to_tensor([0.9], dtype="float32")
            >>> beta2 = paddle.to_tensor([0.99], dtype="float32")
            >>> momentum = paddle.optimizer.Momentum(
            ...     learning_rate=0.1,
            ...     parameters=linear.parameters(),
            ...     weight_decay=L2Decay(0.0001))
            >>> back = out.backward()
            >>> momentum.step()
            >>> momentum.clear_grad()

        .. code-block:: python
            :name: code-example2

            >>> # Example2: set Regularizer in parameters
            >>> # Set L2 regularization in parameters.
            >>> # Global regularizer does not take effect on my_conv2d for this case.
            >>> from paddle.nn import Conv2D
            >>> from paddle import ParamAttr
            >>> from paddle.regularizer import L2Decay

            >>> my_conv2d = Conv2D(
            ...         in_channels=10,
            ...         out_channels=10,
            ...         kernel_size=1,
            ...         stride=1,
            ...         padding=0,
            ...         weight_attr=ParamAttr(regularizer=L2Decay(coeff=0.01)),
            ...         bias_attr=False)
    """

    def __init__(self, coeff=0.0):
        assert coeff is not None
        super().__init__()
        self._coeff = coeff

    def __call__(self, param, grad, block):
        """Add L2 weight decay ops to network

        Adds L2 weight decay ops.
        L2WeightDecay = reg_coeff * parameter

        Args:
            param: parameter variable for which regularization is applied
            block: block in which variable is to be created

        Returns:
            new variable for weight decay
        """
        assert isinstance(
            param, (framework.Variable, pir.Value, pir.core.ParameterMeta)
        )
        assert isinstance(block, (framework.Block, pir.Block))

        if in_dynamic_or_pir_mode():
            return _C_ops.scale(param, self._coeff, 0.0, True)
        else:
            decay = block.create_var(
                dtype=param.dtype, shape=param.shape, lod_level=param.lod_level
            )

            # Append Op to calculate decay
            block.append_op(
                type='scale',
                inputs={"X": param},
                outputs={"Out": decay},
                attrs={"scale": self._coeff},
            )

            return decay

    def __str__(self):
        return "L2Decay, coeff=%f" % self._coeff
