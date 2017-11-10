import paddle.v2.framework.framework as framework

__all__ = [
    'append_regularization_ops', 'L2DecayRegularizer', 'L1DecayRegularizer'
]


def append_regularization_ops(parameters_and_grads):
    """Create and add backward regularization Operators

    Creates and adds backward regularization operators in the BlockDesc.
    This will add gradients of the regularizer function to the gradients
    of the parameters and return these modified gradients. This is the
    same as implementing weight decay in optimizers for regularization.

    Args:
        parameters_and_grads: A list of (parameters, gradients) pairs
                              that need to be regularized.

    Returns:
        list of (parameters, gradients) pair with the regularized gradient

    Raises:
        Exception: Unknown regularization type
    """
    params_and_grads = []
    for param, grad in parameters_and_grads:
        # If no gradient or no regularization specified,
        # then we don't need to do anything
        if grad is None or param.regularizer is None:
            params_and_grads.append((param, grad))
            continue

        # Add variable for regularization term in grad block
        regularization_term = param.regularizer(param, grad.block)
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
