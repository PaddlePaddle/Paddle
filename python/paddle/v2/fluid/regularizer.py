#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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

import framework

__all__ = [
    'append_regularization_ops',
    'L1Decay',
    'L2Decay',
]


def append_regularization_ops(parameters_and_grads, regularization=None):
    """Create and add backward regularization Operators

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
        list of (parameters, gradients) pair with the regularized gradient

    Raises:
        Exception: Unknown regularization type
    """
    params_and_grads = []
    for param, grad in parameters_and_grads:
        regularization_term = None
        if param.regularizer is not None:
            # Add variable for regularization term in grad block
            regularization_term = param.regularizer(param, grad.block)
        elif regularization is not None:
            regularization_term = regularization(param, grad.block)

        # If no gradient or no regularization specified,
        # then we don't need to do anything
        if grad is None or regularization_term is None:
            params_and_grads.append((param, grad))
            continue

        assert grad.shape == regularization_term.shape

        grad.block.append_op(
            type='elementwise_add',
            inputs={"X": grad,
                    "Y": regularization_term},
            outputs={"Out": grad})
        params_and_grads.append((param, grad))

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

    def __call__(self, param, block):
        """Add corresponding weight decay operations to the network
        """
        raise NotImplementedError()

    def __str__(self):
        """Debug string
        """
        raise NotImplementedError()


class L2DecayRegularizer(WeightDecayRegularizer):
    """Implements the L2 Weight Decay Regularization
    """

    def __init__(self, regularization_coeff=0.0):
        assert regularization_coeff is not None
        super(L2DecayRegularizer, self).__init__()
        self._regularization_coeff = regularization_coeff

    def __call__(self, param, block):
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
        decay = block.create_var(
            dtype="float32", shape=param.shape, lod_level=param.lod_level)
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
    """Implements the L1 Weight Decay Regularization
    """

    def __init__(self, regularization_coeff=0.0):
        assert regularization_coeff is not None
        super(L1DecayRegularizer, self).__init__()
        self._regularization_coeff = regularization_coeff

    def __call__(self, param, block):
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
        decay = block.create_var(
            dtype="float32", shape=param.shape, lod_level=param.lod_level)
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
