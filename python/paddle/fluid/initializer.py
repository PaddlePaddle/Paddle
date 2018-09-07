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

import framework
import numpy as np
import contextlib

__all__ = [
    'Constant', 'Uniform', 'Normal', 'Xavier', 'force_init_on_cpu',
    'init_on_cpu', 'ConstantInitializer', 'UniformInitializer',
    'NormalInitializer', 'XavierInitializer'
]

_force_init_on_cpu_ = False


def force_init_on_cpu():
    return _force_init_on_cpu_


@contextlib.contextmanager
def init_on_cpu():
    """
    Switch program with `with` statement

    Examples:
        >>> with init_on_cpu():
        >>>   step = layers.create_global_var()

    """
    global _force_init_on_cpu_

    pre_state = force_init_on_cpu()
    _force_init_on_cpu_ = True
    yield
    _force_init_on_cpu_ = pre_state


class Initializer(object):
    """Base class for variable initializers

    Defines the common interface of variable initializers.
    They add operations to the init program that are used
    to initialize variables. Users should not use this class
    directly, but need to use one of its implementations.
    """

    def __init_(self):
        pass

    def __call__(self, param, block):
        """Add corresponding initialization operations to the network
        """
        raise NotImplementedError()

    def _compute_fans(self, var):
        """Compute the fan_in and the fan_out for layers

        This method computes the fan_in and the fan_out
        for neural network layers, if not specified. It is
        not possible to perfectly estimate fan_in and fan_out.
        This method will estimate it correctly for matrix multiply and
        convolutions.

        Args:
            var: variable for which fan_in and fan_out have to be computed

        Returns:
            tuple of two integers (fan_in, fan_out)
        """
        shape = var.shape
        if not shape or len(shape) == 0:
            fan_in = fan_out = 1
        elif len(shape) == 1:
            fan_in = fan_out = shape[0]
        elif len(shape) == 2:
            # This is the case for simple matrix multiply
            fan_in = shape[0]
            fan_out = shape[1]
        else:
            # Assume this to be a convolutional kernel
            # In PaddlePaddle, the shape of the kernel is like:
            # [num_filters, num_filter_channels, ...] where the remaining
            # dimensions are the filter_size
            receptive_field_size = np.prod(shape[2:])
            fan_in = shape[1] * receptive_field_size
            fan_out = shape[0] * receptive_field_size

        return (fan_in, fan_out)


class ConstantInitializer(Initializer):
    """Implements the constant initializer
    """

    def __init__(self, value=0.0, force_cpu=False):
        """Constructor for ConstantInitializer

        Args:
            value: constant value to initialize the variable
        """
        assert value is not None
        super(ConstantInitializer, self).__init__()
        self._value = value
        self._force_cpu = force_cpu

    def __call__(self, var, block):
        """Add constant initialization ops for a variable

        Args:
            var: Variable that needs to be initialized
            block: The block in which initialization ops
                   should be added

        Returns:
            the initialization op
        """
        assert isinstance(var, framework.Variable)
        assert isinstance(block, framework.Block)
        # Initialization Ops should be prepended and not appended
        op = block.prepend_op(
            type="fill_constant",
            outputs={"Out": var},
            attrs={
                "shape": var.shape,
                "dtype": int(var.dtype),
                "value": float(self._value),
                'force_cpu': self._force_cpu or force_init_on_cpu()
            })
        var.op = op
        return op


class UniformInitializer(Initializer):
    """Implements the random uniform distribution initializer
    """

    def __init__(self, low=-1.0, high=1.0, seed=0):
        """Constructor for UniformInitializer

        Args:
            low: lower boundary of the uniform distribution
            high: upper boundary of the uniform distribution
            seed: random seed
        """
        assert low is not None
        assert high is not None
        assert high >= low
        assert seed is not None
        super(UniformInitializer, self).__init__()
        self._low = low
        self._high = high
        self._seed = seed

    def __call__(self, var, block):
        """Add uniform distribution initialization ops for a variable

        Args:
            var: Variable that needs to be initialized
            block: The block in which initialization ops
                   should be added

        Returns:
            the initialization op
        """
        assert isinstance(var, framework.Variable)
        assert isinstance(block, framework.Block)
        # Initialization Ops should be prepended and not appended
        if self._seed == 0:
            self._seed = block.program.random_seed
        op = block.prepend_op(
            type="uniform_random",
            outputs={"Out": var},
            attrs={
                "shape": var.shape,
                "dtype": int(var.dtype),
                "min": self._low,
                "max": self._high,
                "seed": self._seed
            })
        var.op = op
        return op


class NormalInitializer(Initializer):
    """Implements the  random Normal(Gaussian) distribution initializer
    """

    def __init__(self, loc=0.0, scale=1.0, seed=0):
        """Constructor for NormalInitializer

        Args:
            loc: mean of the normal distribution
            scale: standard deviation of the normal distribution
            seed: random seed
        """
        assert loc is not None
        assert scale is not None
        assert seed is not None
        super(NormalInitializer, self).__init__()
        self._mean = loc
        self._std_dev = scale
        self._seed = seed

    def __call__(self, var, block):
        """Add normal distribution initialization ops for a variable

        Args:
            var: Variable that needs to be initialized
            block: The block in which initialization ops
                   should be added

        Returns:
            the initialization op
        """
        assert isinstance(var, framework.Variable)
        assert isinstance(block, framework.Block)
        # Initialization Ops should be prepended and not appended
        if self._seed == 0:
            self._seed = block.program.random_seed
        op = block.prepend_op(
            type="gaussian_random",
            outputs={"Out": var},
            attrs={
                "shape": var.shape,
                "dtype": int(var.dtype),
                "mean": self._mean,
                "std": self._std_dev,
                "seed": self._seed
            })
        var.op = op
        return op


class XavierInitializer(Initializer):
    """Implements the Xavier initializer

    This class implements the Xavier weight initializer from the paper
    Understanding the difficulty of training deep feedforward neural
    networks[1] by Xavier Glorot and Yoshua Bengio.

    This initializer is designed to keep the scale of the gradients
    approximately same in all the layers. In case of Uniform distribution,
    the range is [-x, x], where x = sqrt(6 / (fan_in + fan_out)).
    In case of Normal distribution, the mean is 0 and the standard deviation
    is sqrt(2/ (fan_in + fan_out)).

    References:
        [1] Understanding the difficulty of training deep feedforward neural
            networks. International conference on artificial intelligence and
            statistics.
            (http://proceedings.mlr.press/v9/glorot10a.html)
    """

    def __init__(self, uniform=True, fan_in=None, fan_out=None, seed=0):
        """Constructor for XavierInitializer

        Args:
            uniform: whether to use uniform or normal distribution
            fan_in: fan_in for Xavier initialization. If None, it is
                    inferred from the variable.
            fan_out: fan_out for Xavier initialization. If None, it is
                     inferred from the variable.
            seed: random seed

        Note: It is recommended to set fan_in and fan_out to None for
              most cases.
        """
        assert uniform is not None
        assert seed is not None
        super(XavierInitializer, self).__init__()
        self._uniform = uniform
        self._fan_in = fan_in
        self._fan_out = fan_out
        self._seed = seed

    def __call__(self, var, block):
        """Add xavier initialization ops for a variable

        Args:
            var: Variable that needs to be initialized
            block: The block in which initialization ops
                   should be added

        Returns:
            the initialization op
        """
        assert isinstance(var, framework.Variable)
        assert isinstance(block, framework.Block)
        f_in, f_out = self._compute_fans(var)

        # If fan_in and fan_out are passed, use them
        fan_in = f_in if self._fan_in is None else self._fan_in
        fan_out = f_out if self._fan_out is None else self._fan_out

        if self._seed == 0:
            self._seed = block.program.random_seed

        if self._uniform:
            limit = np.sqrt(6.0 / float(fan_in + fan_out))
            op = block.prepend_op(
                type="uniform_random",
                outputs={"Out": var},
                attrs={
                    "shape": var.shape,
                    "dtype": int(var.dtype),
                    "min": -limit,
                    "max": limit,
                    "seed": self._seed
                })

        else:
            std = np.sqrt(2.0 / float(fan_in + fan_out))
            op = block.prepend_op(
                type="gaussian_random",
                outputs={"Out": var},
                attrs={
                    "shape": var.shape,
                    "dtype": int(var.dtype),
                    "mean": 0.0,
                    "std": std,
                    "seed": self._seed
                })
        var.op = op
        return op


class MSRAInitializer(Initializer):
    """Implements the MSRA initializer a.k.a. Kaiming Initializer

    This class implements the weight initialization from the paper
    Delving Deep into Rectifiers: Surpassing Human-Level Performance on
    ImageNet Classification[1] by Kaiming He, Xiangyu Zhang, Shaoqing Ren
    and Jian Sun. This is a robust initialization method that particularly
    considers the rectifier nonlinearities. In case of Uniform distribution,
    the range is [-x, x], where x = sqrt(6 / fan_in). In case of Normal
    distribution, the mean is 0 and the standard deviation
    is sqrt(2/ fan_in).

    References:
        [1] Delving Deep into Rectifiers: Surpassing Human-Level Performance
            on ImageNet Classification
            (https://arxiv.org/abs/1502.01852)
    """

    def __init__(self, uniform=True, fan_in=None, seed=0):
        """Constructor for MSRAInitializer

        Args:
            uniform: whether to use uniform or normal distribution
            fan_in: fan_in for MSRAInitializer. If None, it is
                    inferred from the variable.
            seed: random seed

        Note: It is recommended to set fan_in to None for most cases.
        """
        assert uniform is not None
        assert seed is not None
        super(MSRAInitializer, self).__init__()
        self._uniform = uniform
        self._fan_in = fan_in
        self._seed = seed

    def __call__(self, var, block):
        """Add MSRA initialization ops for a variable

        Args:
            var: Variable that needs to be initialized
            block: The block in which initialization ops
                   should be added

        Returns:
            the initialization op
        """
        assert isinstance(var, framework.Variable)
        assert isinstance(block, framework.Block)
        f_in, f_out = self._compute_fans(var)

        # If fan_in is passed, use it
        fan_in = f_in if self._fan_in is None else self._fan_in

        if self._seed == 0:
            self._seed = block.program.random_seed

        if self._uniform:
            limit = np.sqrt(6.0 / float(fan_in))
            op = block.prepend_op(
                type="uniform_random",
                outputs={"Out": var},
                attrs={
                    "shape": var.shape,
                    "dtype": int(var.dtype),
                    "min": -limit,
                    "max": limit,
                    "seed": self._seed
                })

        else:
            std = np.sqrt(2.0 / float(fan_in))
            op = block.prepend_op(
                type="gaussian_random",
                outputs={"Out": var},
                attrs={
                    "shape": var.shape,
                    "dtype": int(var.dtype),
                    "mean": 0.0,
                    "std": std,
                    "seed": self._seed
                })
        var.op = op
        return op


# We short the class name, since users will use the initializer with the package
# name. The sample code:
#
# import paddle.fluid as fluid
#
# hidden = fluid.layers.fc(...,
#                          param_attr=ParamAttr(fluid.initializer.Xavier()))
#
# It is no need to add an `Initializer` as the class suffix
Constant = ConstantInitializer
Uniform = UniformInitializer
Normal = NormalInitializer
Xavier = XavierInitializer
MSRA = MSRAInitializer
