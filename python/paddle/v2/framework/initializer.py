import paddle.v2.framework.framework as framework
import numpy as np

__all__ = [
    'ConstantInitializer', 'UniformInitializer', 'NormalInitializer',
    'XavierInitializer'
]


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

    def __init__(self, value=0.0):
        """Constructor for ConstantInitializer

        Args:
            value: constant value to initialize the variable
        """
        assert value is not None
        super(ConstantInitializer, self).__init__()
        self._value = value

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
                "data_type": int(var.data_type),
                "value": self._value
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
        op = block.prepend_op(
            type="uniform_random",
            outputs={"Out": var},
            attrs={
                "shape": var.shape,
                "data_type": int(var.data_type),
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
        op = block.prepend_op(
            type="gaussian_random",
            outputs={"Out": var},
            attrs={
                "shape": var.shape,
                "data_type": int(var.data_type),
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

        if self._uniform:
            limit = np.sqrt(6.0 / float(fan_in + fan_out))
            op = block.prepend_op(
                type="uniform_random",
                outputs={"Out": var},
                attrs={
                    "shape": var.shape,
                    "data_type": int(var.data_type),
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
                    "data_type": int(var.data_type),
                    "mean": 0.0,
                    "std": std,
                    "seed": self._seed
                })
        var.op = op
        return op
