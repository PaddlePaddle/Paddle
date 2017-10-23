from collections import defaultdict

import paddle.v2.framework.framework as framework
from paddle.v2.framework.backward import append_backward_ops

__all__ = ['SGDOptimizer', 'MomentumOptimizer', 'AdagradOptimizer']


class Optimizer(object):
    """Optimizer Base class.

    Define the common interface of an optimizer.
    User should not use this class directly,
    but need to use one of it's implementation.
    """

    def __init__(self):
        # Dictionary of accumulators. Some optimizer subclasses need to
        # allocate and manage extra variables associated with the parameters
        # to train. These variables are called accumulators.
        # {accum_name : { paramter_name : accumulator_for_parameter, ...}, ...}
        self._accumulators = defaultdict(lambda: dict())

    def _append_optimize_op(self, block, param_and_grad):
        """ append optimize operator to block and return all the added optimize_op
        """
        raise NotImplementedError()

    def _initialize_tensors(self, block):
        """Create all necessary tensors, that will be shared for all parameter updates.

        Tensors like learning rate should be initialized here.

        Args:
            block: the block in which the loss variable is present
        """
        pass

    def _create_accumulators(self, block, parameters):
        """Create all accumulators needed by the parameters

        Args:
            block: the block in which the loss variable is present
            parameters: list of parameter variables for the optimizer
        """
        pass

    def _add_accumulator(self, block, name, param, dtype=None, fill_value=0.0):
        """Utility function to add an accumulator for a parameter

        Args:
            block: the block in which the loss variable is present
            name: name of the accumulator
            param: parameter variable for which accumulator is to be added
            dtype: data type of the accumulator variable
            fill_value: value to initialize the accumulator variable
        """
        if (name in self._accumulators and
                param.name in self._accumulators[name]):
            raise Exception("Accumulator {} already exists for parmeter {}".
                            format(name, param.name))
        global_block = block.program.global_block()
        param_shape = list(param.shape)
        param_acc = global_block.create_var(
            dtype=dtype, shape=param_shape, lod_level=0)

        # Initialize the accumulator with fill_value
        # FIXME: Fix when Initialization design has been implemented
        # https://github.com/PaddlePaddle/Paddle/pull/4852
        global_block.append_op(
            type="fill_constant",
            outputs={"Out": param_acc},
            attrs={"shape": param_shape,
                   "value": fill_value})

        # Add to accumulators dict
        self._accumulators[name][param.name] = param_acc

    def _get_accumulator(self, name, param):
        """Utility function to fetch an accumulator for a parameter

        Args:
            name: name of the accumulator
            param: parameter variable for which accumulator is to be fetched

        Returns:
            accumulator variable for the parameter
        """
        if (name not in self._accumulators or
                param.name not in self._accumulators[name]):
            raise Exception("Accumulator {} does not exist for parameter {}".
                            format(name, param.name))
        return self._accumulators[name][param.name]

    def create_optimization_pass(self, parameters_and_grads, loss):
        """Add optimization operators to update gradients to variables.

        Args:
          loss: the target that this optimization is for.
          parameters_and_grads: a list of (variable, gradient) pair to update.

        Returns:
          optmization_op_list: a list of optimization operator that will update
          parameter using gradient.
        """
        # This is a default implementation of create_optimization_pass that
        # can be shared by most optimizers. This implementation assumes that
        # the subclass will implement the _append_optimize_op method and the
        #  _initialize_tensors method. The subclass can extend the
        # _create_accumulators method if it needs to create accumulators
        # for parameters.

        # Create any accumulators
        self._create_accumulators(loss.block,
                                  [p[0] for p in parameters_and_grads])
        # Create any necessary tensors
        self._initialize_tensors(loss.block)

        optimize_ops = []
        for param_and_grad in parameters_and_grads:
            if param_and_grad[1] is not None:
                optimize_op = self._append_optimize_op(loss.block,
                                                       param_and_grad)
                optimize_ops.append(optimize_op)

        return optimize_ops

    def minimize(self, loss, parameter_list=None, no_grad_set=None):
        """Add operations to minimize `loss` by updating `parameter_list`.

        This method combines interface `create_backward_pass()` and
        `create_optimization_pass()` into one.
        """
        params_grads = append_backward_ops(loss, parameter_list, no_grad_set or
                                           set())
        optimize_ops = self.create_optimization_pass(params_grads, loss)
        return optimize_ops


class SGDOptimizer(Optimizer):
    """ Simple SGD optimizer without any state.
    """

    def __init__(self, learning_rate):
        assert learning_rate is not None
        super(SGDOptimizer, self).__init__()
        self.type = "sgd"
        self._learning_rate = learning_rate

    def _initialize_tensors(self, block):
        assert isinstance(block, framework.Block)
        lr_shape = [1]
        # create a variable for learning_rate
        self._lr = block.create_var(
            dtype="float32", shape=lr_shape, lod_level=0)

        # create an op to init the learning_rate
        # FIXME: Fix when Initialization design has been implemented
        # https://github.com/PaddlePaddle/Paddle/pull/4852
        block.append_op(
            type="fill_constant",
            outputs={"Out": self._lr},
            attrs={"shape": lr_shape,
                   "value": self._learning_rate})

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, framework.Block)

        # create the optimize op
        sgd_op = block.append_op(
            type=self.type,
            inputs={
                "Param": param_and_grad[0],
                "Grad": param_and_grad[1],
                "LearningRate": self._lr
            },
            outputs={"ParamOut": param_and_grad[0]})

        return sgd_op


class MomentumOptimizer(Optimizer):
    """Simple Momentum optimizer with velocity state
    """
    _velocity_acc_str = "velocity"

    def __init__(self, learning_rate, momentum):
        assert learning_rate is not None
        assert momentum is not None
        super(MomentumOptimizer, self).__init__()
        self.type = "momentum"
        self._learning_rate = learning_rate
        self._momentum = momentum

    def _initialize_tensors(self, block):
        assert isinstance(block, framework.Block)
        lr_shape = [1]
        # create a variable for learning_rate
        self._lr = block.create_var(
            dtype="float32", shape=lr_shape, lod_level=0)

        # create an op to init the learning_rate
        # FIXME: Fix when Initialization design has been implemented
        # https://github.com/PaddlePaddle/Paddle/pull/4852
        block.append_op(
            type="fill_constant",
            outputs={"Out": self._lr},
            attrs={"shape": lr_shape,
                   "value": self._learning_rate})

    def _create_accumulators(self, block, parameters):
        assert isinstance(block, framework.Block)

        for p in parameters:
            self._add_accumulator(block, self._velocity_acc_str, p, 'float32')

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, framework.Block)

        velocity_acc = self._get_accumulator(self._velocity_acc_str,
                                             param_and_grad[0])
        # create the momentum optimize op
        momentum_op = block.append_op(
            type=self.type,
            inputs={
                "Param": param_and_grad[0],
                "Grad": param_and_grad[1],
                "Velocity": velocity_acc,
                "LearningRate": self._lr
            },
            outputs={
                "ParamOut": param_and_grad[0],
                "VelocityOut": velocity_acc
            },
            attrs={"mu": self._momentum})

        return momentum_op


class AdagradOptimizer(Optimizer):
    """Simple Adagrad optimizer with moment state
    """
    _moment_acc_str = "moment"

    def __init__(self, learning_rate, epsilon=1.0e-6):
        assert learning_rate is not None
        assert epsilon is not None
        super(AdagradOptimizer, self).__init__()
        self.type = "adagrad"
        self._learning_rate = learning_rate
        self._epsilon = epsilon

    def _initialize_tensors(self, block):
        assert isinstance(block, framework.Block)
        lr_shape = [1]
        # create a variable for learning_rate
        self._lr = block.create_var(
            dtype="float32", shape=lr_shape, lod_level=0)

        # create an op to init the learning_rate
        # FIXME: Fix when Initialization design has been implemented
        # https://github.com/PaddlePaddle/Paddle/pull/4852
        block.append_op(
            type="fill_constant",
            outputs={"Out": self._lr},
            attrs={"shape": lr_shape,
                   "value": self._learning_rate})

    def _create_accumulators(self, block, parameters):
        assert isinstance(block, framework.Block)

        for p in parameters:
            self._add_accumulator(block, self._moment_acc_str, p, 'float32')

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, framework.Block)

        moment_acc = self._get_accumulator(self._moment_acc_str,
                                           param_and_grad[0])

        # create the adagrad optimizer op
        adagrad_op = block.append_op(
            type=self.type,
            inputs={
                "Param": param_and_grad[0],
                "Grad": param_and_grad[1],
                "Moment": moment_acc,
                "LearningRate": self._lr
            },
            outputs={"ParamOut": param_and_grad[0],
                     "MomentOut": moment_acc},
            attrs={"epsilon": self._epsilon})

        return adagrad_op
