#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function
import logging

from . import framework
from .framework import in_dygraph_mode, _varbase_creator
from . import core

__all__ = ['L1Decay', 'L2Decay', 'L1DecayRegularizer', 'L2DecayRegularizer']


def _create_regularization_of_grad(param, grad, regularization=None):
    """ Create and add backward regularization Operators

    Function helper of append_regularization_ops.
    """
    # If no gradient or no regularization is specified,  then we don't need to do anything
    if grad is None or (param.regularizer is None and regularization is None):
        return grad
    regularization_term = None
    if param.regularizer is not None:
        # Add variable for regularization term in grad block
        regularization_term = param.regularizer(param, grad, grad.block)
    elif regularization is not None:
        regularization_term = regularization(param, grad, grad.block)

    assert regularization_term is not None

    new_grad = grad
    if grad.type == core.VarDesc.VarType.SELECTED_ROWS:
        # FIXME(zcd): If the grad is SELECTED_ROWS, after regularization,
        # the grad's type and name will be changed. But the gradient's name
        # is used in ParallelExecutor Reduce mode, so I add a flag for
        # the new_grad here.
        new_grad = grad.block.create_var(
            name=grad.name + core.kNewGradSuffix(),
            dtype=param.dtype,
            shape=param.shape,
            lod_level=param.lod_level,
            type=core.VarDesc.VarType.LOD_TENSOR)

    inputs = {"X": [grad, regularization_term]}
    outputs = {"Out": [new_grad]}
    if in_dygraph_mode():
        new_grad = core.ops.sum([grad, regularization_term])
    else:
        grad.block.append_op(type='sum', inputs=inputs, outputs=outputs)

    return new_grad


def append_regularization_ops(parameters_and_grads, regularization=None):
    r"""Create and add backward regularization Operators

    Creates and adds backward regularization operators in the BlockDesc.
    This will add gradients of the regularizer function to the gradients
    of the parameters and return these modified gradients. This is the
    same as implementing weight decay in optimizers for regularization.

    Args:
        parameters_and_grads: A list of (parameters, gradients) pairs
                              that need to be regularized.
        regularization: A global regularizer. If the parameter is not
                        set. It will be applied with regularizer.

    Returns:
        list[(Variable, Variable)]: list of (parameters, gradients) \
        pair with the regularized gradient

    Raises:
        Exception: Unknown regularization type
    """
    params_and_grads = []
    if in_dygraph_mode():
        for param, grad in parameters_and_grads:
            new_grad = _create_regularization_of_grad(param, grad,
                                                      regularization)
            params_and_grads.append((param, new_grad))
    else:
        repeate_regularizer = False
        with framework.name_scope('regularization'):
            for param, grad in parameters_and_grads:
                if not repeate_regularizer and param.regularizer is not None and regularization is not None:
                    repeate_regularizer = True
                    logging.info(
                        "If regularizer of a Parameter has been set by 'fluid.ParamAttr' or 'fluid.WeightNormParamAttr' already. "
                        "The Regularization[%s] in Optimizer will not take effect, and it will only be applied to other Parameters!"
                        % regularization.__str__())
                with param.block.program._optimized_guard([param, grad]):
                    new_grad = _create_regularization_of_grad(param, grad,
                                                              regularization)
                    params_and_grads.append((param, new_grad))
    return params_and_grads


class WeightDecayRegularizer(object):
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
        """Add corresponding weight decay operations to the network
        """
        raise NotImplementedError()

    def __str__(self):
        """Debug string
        """
        raise NotImplementedError()


class L2DecayRegularizer(WeightDecayRegularizer):
    r""" 
    Implement the L2 Weight Decay Regularization, which helps to prevent the model over-fitting.

    It can be set in :ref:`api_fluid_ParamAttr` or ``optimizer`` (such as :ref:`api_fluid_optimizer_SGDOptimizer` ). 
    When set in ``ParamAttr`` , it only takes effect for trainable parameters in this layer. When set in 
    ``optimizer`` , it takes effect for all trainable parameters. When set together, ``ParamAttr`` has 
    higher priority than ``optimizer`` .
    
    In the implementation, the formula of L2 Weight Decay Regularization is as follows:

    .. math::

        L2WeightDecay = reg\_coeff * parameter

    Args:
        regularization_coeff(float, optional): regularization coeff. Default:0.0

    Examples:
        .. code-block:: python

            # Example1: set Regularizer in optimizer
            import paddle.fluid as fluid

            main_prog = fluid.Program()
            startup_prog = fluid.Program()
            with fluid.program_guard(main_prog, startup_prog):
                data = fluid.layers.data(name='image', shape=[3, 28, 28], dtype='float32')
                label = fluid.layers.data(name='label', shape=[1], dtype='int64')
                hidden = fluid.layers.fc(input=data, size=128, act='relu')
                prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
                loss = fluid.layers.cross_entropy(input=prediction, label=label)
                avg_loss = fluid.layers.mean(loss)
            optimizer = fluid.optimizer.Adagrad(
                learning_rate=1e-4,
                regularization=fluid.regularizer.L2Decay(
                    regularization_coeff=0.1))
            optimizer.minimize(avg_loss)


            # Example2: set Regularizer both in ParamAttr and optimizer
            import paddle.fluid as fluid

            l1 = fluid.regularizer.L1Decay(regularization_coeff=0.1)
            l2 = fluid.regularizer.L2Decay(regularization_coeff=0.1)
            x = fluid.layers.uniform_random([3,4])
            
            # set L1 regularization in fluid.ParamAttr
            w_param = fluid.ParamAttr(regularizer=l1)
            hidden1 = fluid.layers.fc(x, 8, param_attr=w_param)  # fc_0.w_0(L1), fc_0.b_0
            hidden2 = fluid.layers.fc(hidden1, 16, param_attr=w_param)   # fc_1.w_0(L1), fc_1.b_0
            predict = fluid.layers.fc(hidden2, 32)    # fc_3.w_0, fc_3.b_0
            avg_loss = fluid.layers.mean(predict)

            # set L2 regularization in optimizer
            optimizer = fluid.optimizer.SGD(learning_rate=1e-4, regularization=l2)
            optimizer.minimize(avg_loss)
            
            # it will Print Message:
            # Regularization of [fc_0.w_0, fc_1.w_0] have been set by ParamAttr or WeightNormParamAttr already. 
            # So, the Regularization of Optimizer will not take effect for these parameters!

    """

    def __init__(self, regularization_coeff=0.0):
        assert regularization_coeff is not None
        super(L2DecayRegularizer, self).__init__()
        self._regularization_coeff = regularization_coeff

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
        assert isinstance(param, framework.Parameter)
        assert isinstance(block, framework.Block)

        inputs = {"X": [param]}
        attrs = {"scale": self._regularization_coeff}

        if framework.in_dygraph_mode():
            return core.ops.scale(param, "scale", self._regularization_coeff)
        else:
            decay = block.create_var(
                dtype=param.dtype, shape=param.shape, lod_level=param.lod_level)

            # Append Op to calculate decay
            block.append_op(
                type='scale',
                inputs={"X": param},
                outputs={"Out": decay},
                attrs={"scale": self._regularization_coeff})

            return decay

    def __str__(self):
        return "L2Decay, regularization_coeff=%f" % self._regularization_coeff


class L1DecayRegularizer(WeightDecayRegularizer):
    r"""
    Implement the L1 Weight Decay Regularization, which encourages the weights to be sparse.
    
    It can be set in :ref:`api_fluid_ParamAttr` or ``optimizer`` (such as :ref:`api_fluid_optimizer_SGDOptimizer` ). 
    When set in ``ParamAttr`` , it only takes effect for trainable parameters in this layer. When set in 
    ``optimizer`` , it takes effect for all trainable parameters. When set together, ``ParamAttr`` has 
    higher priority than ``optimizer`` .
    
    In the implementation, the formula of L1 Weight Decay Regularization is as follows:
	
    .. math::

        L1WeightDecay = reg\_coeff * sign(parameter)

    Args:
        regularization_coeff(float, optional): regularization coeff. Default:0.0.
	
    Examples:
        .. code-block:: python

            # Example1: set Regularizer in optimizer
            import paddle.fluid as fluid

            main_prog = fluid.Program()
            startup_prog = fluid.Program()
            with fluid.program_guard(main_prog, startup_prog):
                data = fluid.layers.data(name='image', shape=[3, 28, 28], dtype='float32')
                label = fluid.layers.data(name='label', shape=[1], dtype='int64')
                hidden = fluid.layers.fc(input=data, size=128, act='relu')
                prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
                loss = fluid.layers.cross_entropy(input=prediction, label=label)
                avg_loss = fluid.layers.mean(loss)
            optimizer = fluid.optimizer.Adagrad(
                learning_rate=1e-4,
                regularization=fluid.regularizer.L1DecayRegularizer(
                    regularization_coeff=0.1))
            optimizer.minimize(avg_loss)
 

            # Example2: set Regularizer both in ParamAttr and optimizer
            import paddle.fluid as fluid

            l1 = fluid.regularizer.L1Decay(regularization_coeff=0.1)
            l2 = fluid.regularizer.L2Decay(regularization_coeff=0.1)
            x = fluid.layers.uniform_random([3,4])
            
            # set L1 regularization in fluid.ParamAttr
            w_param = fluid.ParamAttr(regularizer=l1)
            hidden1 = fluid.layers.fc(x, 8, param_attr=w_param)  # fc_0.w_0(L1), fc_0.b_0
            hidden2 = fluid.layers.fc(hidden1, 16, param_attr=w_param)  # fc_1.w_0(L1), fc_1.b_0
            predict = fluid.layers.fc(hidden2, 32)   # fc_3.w_0, fc_3.b_0
            avg_loss = fluid.layers.mean(predict)

            # set L2 regularization in optimizer
            optimizer = fluid.optimizer.SGD(learning_rate=1e-4, regularization=l2)
            optimizer.minimize(avg_loss)
            
            # it will Print Message:
            # Regularization of [fc_0.w_0, fc_1.w_0] have been set by ParamAttr or WeightNormParamAttr already. 
            # So, the Regularization of Optimizer will not take effect for these parameters!

    """

    def __init__(self, regularization_coeff=0.0):
        assert regularization_coeff is not None
        super(L1DecayRegularizer, self).__init__()
        self._regularization_coeff = regularization_coeff

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
        assert isinstance(param, framework.Parameter)
        assert isinstance(block, framework.Block)

        if framework.in_dygraph_mode():
            decay = block.create_var(dtype=param.dtype, shape=param.shape)
        else:
            decay = block.create_var(
                dtype=param.dtype, shape=param.shape, lod_level=param.lod_level)

        # Append sign op
        block.append_op(
            type='sign', inputs={"X": param}, outputs={"Out": decay})

        # Append scale op to the output of sign op
        block.append_op(
            type='scale',
            inputs={"X": decay},
            outputs={"Out": decay},
            attrs={"scale": self._regularization_coeff})

        return decay

    def __str__(self):
        return "L1Decay, regularization_coeff=%f" % self._regularization_coeff


# We short the class name, since users will use the regulaizer with the package
# name. The sample code:
#
# import paddle.fluid as fluid
#
# hidden = fluid.layers.fc(...,
#                          param_attr=fluid.regularizer.Xavier())
#
# It is no need to add a `Regularizer` as the class suffix
L1Decay = L1DecayRegularizer
L2Decay = L2DecayRegularizer
