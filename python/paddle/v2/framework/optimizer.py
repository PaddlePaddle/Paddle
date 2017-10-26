from collections import defaultdict

import paddle.v2.framework.framework as framework
from paddle.v2.framework.backward import append_backward_ops

__all__ = [
    'SGDOptimizer', 'MomentumOptimizer', 'AdagradOptimizer', 'AdamOptimizer'
]


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

    def _finish_update(self, block):
        """Finish any custom updates needed
           before completing an optimization step

        Args:
            block: the block in which the loss variable is present
            parameters: list of parameter variables for the optimizer

        Returns:
            list of finish ops or None
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
          return_op_list: a list of operators that will complete one step of
          optimization. This will include parameter update ops, global step
          update ops and any other custom ops required by subclasses to manage
          their internal state.
        """
        # This is a default implementation of create_optimization_pass that
        # can be shared by most optimizers. This implementation assumes that
        # the subclass will implement the _append_optimize_op method and the
        #  _initialize_tensors method. The subclass can extend the
        # _create_accumulators method if it needs to create accumulators
        # for parameters and extend _finish_update method to add custom ops.

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

        # Returned list of ops can include more ops in addition
        # to optimization ops
        return_ops = optimize_ops

        # Get custom finish ops for subclasses
        # FIXME: Need to fix this once we figure out how to handle dependencies
        finish_ops = self._finish_update(loss.block)
        if finish_ops is not None:
            return_ops += finish_ops

        return return_ops

    def minimize(self, loss, parameter_list=None, no_grad_set=None):
        """Add operations to minimize `loss` by updating `parameter_list`.

        This method combines interface `append_backward_ops()` and
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

    def __init__(self, learning_rate, momentum, use_nesterov=False):
        assert learning_rate is not None
        assert momentum is not None
        super(MomentumOptimizer, self).__init__()
        self.type = "momentum"
        self._learning_rate = learning_rate
        self._momentum = momentum
        self._use_nesterov = bool(use_nesterov)

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
            attrs={"mu": self._momentum,
                   "useNesterov": self._use_nesterov})

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


class AdamOptimizer(Optimizer):
    """Implements the Adam Optimizer
    """
    _moment1_acc_str = "moment1"
    _moment2_acc_str = "moment2"

    def __init__(self,
                 learning_rate=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8):
        assert learning_rate is not None
        assert beta1 is not None
        assert beta2 is not None
        assert epsilon is not None
        super(AdamOptimizer, self).__init__()
        self.type = "adam"
        self._learning_rate = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
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

        global_block = block.program.global_block()
        # Create beta1 and beta2 power tensors
        beta_shape = [1]
        # Create variables for beta1 and beta2 powers
        self._beta1_pow_acc = global_block.create_var(
            dtype="float32", shape=beta_shape, lod_level=0)
        self._beta2_pow_acc = global_block.create_var(
            dtype="float32", shape=beta_shape, lod_level=0)

        # Initialize beta1 and beta2 power accumulators
        # FIXME: Fix when Initialization design has been implemented
        # https://github.com/PaddlePaddle/Paddle/pull/4852
        global_block.append_op(
            type="fill_constant",
            outputs={"Out": self._beta1_pow_acc},
            attrs={"shape": beta_shape,
                   "value": self._beta1})
        global_block.append_op(
            type="fill_constant",
            outputs={"Out": self._beta2_pow_acc},
            attrs={"shape": beta_shape,
                   "value": self._beta2})

        # Create accumulator tensors for first and second moments
        for p in parameters:
            self._add_accumulator(block, self._moment1_acc_str, p, 'float32')
            self._add_accumulator(block, self._moment2_acc_str, p, 'float32')

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, framework.Block)

        moment1 = self._get_accumulator(self._moment1_acc_str,
                                        param_and_grad[0])
        moment2 = self._get_accumulator(self._moment2_acc_str,
                                        param_and_grad[0])
        # create the momentum optimize op
        adam_op = block.append_op(
            type=self.type,
            inputs={
                "Param": param_and_grad[0],
                "Grad": param_and_grad[1],
                "LearningRate": self._lr,
                "Moment1": moment1,
                "Moment2": moment2,
                "Beta1Pow": self._beta1_pow_acc,
                "Beta2Pow": self._beta2_pow_acc
            },
            outputs={
                "ParamOut": param_and_grad[0],
                "Moment1Out": moment1,
                "Moment2Out": moment2
            },
            attrs={
                "beta1": self._beta1,
                "beta2": self._beta2,
                "epsilon": self._epsilon
            })

        return adam_op

    def _finish_update(self, block):
        """Update Beta1 and Beta2 Power accumulators
        """
        assert isinstance(block, framework.Block)
        global_block = block.program.global_block()
        scale_beta1 = global_block.append_op(
            type="scale",
            inputs={"X": self._beta1_pow_acc},
            outputs={"Out": self._beta1_pow_acc},
            attrs={"scale": self._beta1})

        scale_beta2 = global_block.append_op(
            type="scale",
            inputs={"X": self._beta2_pow_acc},
            outputs={"Out": self._beta2_pow_acc},
            attrs={"scale": self._beta2})

        return [scale_beta1, scale_beta2]
