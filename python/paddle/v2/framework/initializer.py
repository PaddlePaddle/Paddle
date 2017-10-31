import paddle.v2.framework.framework as framework

__all__ = ['ConstantInitializer', 'UniformInitializer']


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
