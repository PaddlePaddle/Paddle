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

from collections import defaultdict

import framework
import layers
from backward import append_backward
from framework import unique_name, program_guard
from initializer import Constant
from layer_helper import LayerHelper
from regularizer import append_regularization_ops
from clip import append_gradient_clip_ops, error_clip_callback

__all__ = ['SGD', 'Momentum', 'Adagrad', 'Adam', 'Adamax', 'DecayedAdagrad']


class Optimizer(object):
    """Optimizer Base class.

    Define the common interface of an optimizer.
    User should not use this class directly,
    but need to use one of it's implementation.
    """

    def __init__(self, learning_rate, global_step=None, regularization=None):
        assert learning_rate is not None
        self._global_step = global_step
        self.regularization = regularization
        self._global_learning_rate = learning_rate
        # Dictionary of accumulators. Some optimizer subclasses need to
        # allocate and manage extra variables associated with the parameters
        # to train. These variables are called accumulators.
        # {accum_name : { paramter_name : accumulator_for_parameter, ...}, ...}
        self._accumulators = defaultdict(lambda: dict())
        self.helper = None

    def _create_global_learning_rate(self):
        if isinstance(self._global_learning_rate, float):
            self._global_learning_rate = layers.create_global_var(
                name=unique_name("learning_rate"),
                shape=[1],
                value=float(self._global_learning_rate),
                dtype='float32',
                persistable=True)

        if not isinstance(self._global_learning_rate, framework.Variable):
            raise ValueError("learning rate should be a Variable, "
                             "actual type is %s",
                             type(self._global_learning_rate))

    @property
    def global_learning_rate(self):
        """
        get global decayed learning rate
        :return:
        """
        return self._global_learning_rate

    def _append_optimize_op(self, block, param_and_grad):
        """ append optimize operator to block and return all the added optimize_op
        """
        raise NotImplementedError()

    def _create_param_lr(self, param_and_grad):
        # create learning rate variable for every parameter
        param = param_and_grad[0]
        param_lr = param.optimize_attr['learning_rate']
        return self._global_learning_rate * param_lr

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

    def _add_accumulator(self, name, param, dtype=None, fill_value=0.0):
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
            raise Exception("Accumulator {} already exists for parameter {}".
                            format(name, param.name))

        assert isinstance(self.helper, LayerHelper)
        var = self.helper.create_global_variable(
            name=unique_name(name),
            persistable=True,
            dtype=dtype or param.dtype,
            type=param.type,
            shape=param.shape)
        self.helper.set_variable_initializer(
            var, initializer=Constant(value=float(fill_value)))
        self._accumulators[name][param.name] = var

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

    def _increment_global_step(self, block):
        """Increment the global step by 1 after every iteration

        Args:
            block: the block in which the loss variable is present

        Returns:
            list with global_step increment op as its only element
        """
        assert isinstance(block, framework.Block)
        assert self._global_step is not None
        # create the increment op
        increment_op = block.append_op(
            type="increment",
            inputs={"X": self._global_step},
            outputs={"Out": self._global_step},
            attrs={"step": 1.0})

        return increment_op

    def create_optimization_pass(self,
                                 parameters_and_grads,
                                 loss,
                                 startup_program=None):
        """Add optimization operators to update gradients to variables.

        Args:
          loss: the target that this optimization is for.
          parameters_and_grads: a list of (variable, gradient) pair to update.

        Returns:
          return_op_list: a list of operators that will complete one step of
          optimization. This will include parameter update ops, global step
          update ops and any other custom ops required by subclasses to manage
          their internal state.
          :param startup_program:
        """
        # This is a default implementation of create_optimization_pass that
        # can be shared by most optimizers. This implementation assumes that
        # the subclass will implement the _append_optimize_op method and the
        #  _initialize_tensors method. The subclass can extend the
        # _create_accumulators method if it needs to create accumulators
        # for parameters and extend _finish_update method to add custom ops.

        # Create any accumulators
        program = loss.block.program
        with program_guard(program, startup_program):
            self.helper = LayerHelper(self.__class__.__name__)
            self._create_accumulators(loss.block,
                                      [p[0] for p in parameters_and_grads])
            self._create_global_learning_rate()

            optimize_ops = []
            for param_and_grad in parameters_and_grads:
                if param_and_grad[0].trainable is True and param_and_grad[
                        1] is not None:
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

            if self._global_step is not None:
                return_ops.append(self._increment_global_step(loss.block))
            return return_ops

    def minimize(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None):
        """Add operations to minimize `loss` by updating `parameter_list`.

        This method combines interface `append_backward()` and
        `create_optimization_pass()` into one.
        """
        params_grads = append_backward(loss, parameter_list, no_grad_set,
                                       error_clip_callback)

        params_grads = append_gradient_clip_ops(params_grads)

        # Add regularization if any
        params_grads = append_regularization_ops(params_grads,
                                                 self.regularization)

        optimize_ops = self.create_optimization_pass(params_grads, loss,
                                                     startup_program)
        return optimize_ops, params_grads


class SGDOptimizer(Optimizer):
    """ Simple SGD optimizer without any state.
    """

    def __init__(self, learning_rate, **kwargs):
        assert learning_rate is not None
        super(SGDOptimizer, self).__init__(
            learning_rate=learning_rate, **kwargs)
        self.type = "sgd"

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, framework.Block)

        # create the optimize op
        sgd_op = block.append_op(
            type=self.type,
            inputs={
                "Param": param_and_grad[0],
                "Grad": param_and_grad[1],
                "LearningRate": self._create_param_lr(param_and_grad)
            },
            outputs={"ParamOut": param_and_grad[0]})

        return sgd_op


class MomentumOptimizer(Optimizer):
    """Simple Momentum optimizer with velocity state
    """
    _velocity_acc_str = "velocity"

    def __init__(self, learning_rate, momentum, use_nesterov=False, **kwargs):
        assert learning_rate is not None
        assert momentum is not None
        super(MomentumOptimizer, self).__init__(
            learning_rate=learning_rate, **kwargs)
        self.type = "momentum"
        self._momentum = momentum
        self._use_nesterov = bool(use_nesterov)

    def _create_accumulators(self, block, parameters):
        assert isinstance(block, framework.Block)

        for p in parameters:
            self._add_accumulator(self._velocity_acc_str, p)

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
                "LearningRate": self._create_param_lr(param_and_grad)
            },
            outputs={
                "ParamOut": param_and_grad[0],
                "VelocityOut": velocity_acc
            },
            attrs={"mu": self._momentum,
                   "use_nesterov": self._use_nesterov})

        return momentum_op


class AdagradOptimizer(Optimizer):
    """Simple Adagrad optimizer with moment state
    """
    _moment_acc_str = "moment"

    def __init__(self, learning_rate, epsilon=1.0e-6, **kwargs):
        assert learning_rate is not None
        assert epsilon is not None
        super(AdagradOptimizer, self).__init__(
            learning_rate=learning_rate, **kwargs)
        self.type = "adagrad"
        self._epsilon = epsilon

    def _create_accumulators(self, block, parameters):
        assert isinstance(block, framework.Block)

        for p in parameters:
            self._add_accumulator(self._moment_acc_str, p)

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, framework.Block)

        moment_acc = self._get_accumulator(self._moment_acc_str,
                                           param_and_grad[0])

        # Create the adagrad optimizer op
        adagrad_op = block.append_op(
            type=self.type,
            inputs={
                "Param": param_and_grad[0],
                "Grad": param_and_grad[1],
                "Moment": moment_acc,
                "LearningRate": self._create_param_lr(param_and_grad)
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
                 epsilon=1e-8,
                 **kwargs):
        assert learning_rate is not None
        assert beta1 is not None
        assert beta2 is not None
        assert epsilon is not None
        super(AdamOptimizer, self).__init__(
            learning_rate=learning_rate, **kwargs)
        self.type = "adam"
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

    def _create_accumulators(self, block, parameters):
        assert isinstance(block, framework.Block)

        main_block = block.program.global_block()
        # Create beta1 and beta2 power tensors
        beta_shape = [1]
        self._beta1_pow_acc = self.helper.create_global_variable(
            name=unique_name('beta1_pow_acc'),
            dtype='float32',
            shape=beta_shape,
            lod_level=0,
            persistable=True)
        self.helper.set_variable_initializer(
            self._beta1_pow_acc, initializer=Constant(self._beta1))

        self._beta2_pow_acc = self.helper.create_global_variable(
            name=unique_name('beta2_pow_acc'),
            dtype='float32',
            shape=beta_shape,
            lod_level=0,
            persistable=True)

        self.helper.set_variable_initializer(
            self._beta2_pow_acc, initializer=Constant(self._beta2))

        # Create accumulator tensors for first and second moments
        for p in parameters:
            self._add_accumulator(self._moment1_acc_str, p)
            self._add_accumulator(self._moment2_acc_str, p)

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, framework.Block)

        moment1 = self._get_accumulator(self._moment1_acc_str,
                                        param_and_grad[0])
        moment2 = self._get_accumulator(self._moment2_acc_str,
                                        param_and_grad[0])
        # create the adam optimize op
        adam_op = block.append_op(
            type=self.type,
            inputs={
                "Param": param_and_grad[0],
                "Grad": param_and_grad[1],
                "LearningRate": self._create_param_lr(param_and_grad),
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
        main_block = block.program.global_block()
        scale_beta1 = main_block.append_op(
            type="scale",
            inputs={"X": self._beta1_pow_acc},
            outputs={"Out": self._beta1_pow_acc},
            attrs={"scale": self._beta1})

        scale_beta2 = main_block.append_op(
            type="scale",
            inputs={"X": self._beta2_pow_acc},
            outputs={"Out": self._beta2_pow_acc},
            attrs={"scale": self._beta2})

        return [scale_beta1, scale_beta2]


class AdamaxOptimizer(Optimizer):
    """Implements the Adamax Optimizer
    """
    _moment_acc_str = "moment"
    _inf_norm_acc_str = "inf_norm"

    def __init__(self,
                 learning_rate=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 **kwargs):
        assert learning_rate is not None
        assert beta1 is not None
        assert beta2 is not None
        assert epsilon is not None
        super(AdamaxOptimizer, self).__init__(
            learning_rate=learning_rate, **kwargs)
        self.type = "adamax"
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

    def _create_accumulators(self, block, parameters):
        # Create beta1 power accumulator tensor
        beta_shape = [1]
        self._beta1_pow_acc = self.helper.create_global_variable(
            name=unique_name('beta1_pow_acc'),
            dtype='float32',
            shape=beta_shape,
            lod_level=0,
            persistable=True)
        self.helper.set_variable_initializer(
            self._beta1_pow_acc, initializer=Constant(self._beta1))

        # Create accumulator tensors for first moment and infinity norm
        for p in parameters:
            self._add_accumulator(self._moment_acc_str, p)
            self._add_accumulator(self._inf_norm_acc_str, p)

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, framework.Block)

        moment = self._get_accumulator(self._moment_acc_str, param_and_grad[0])
        inf_norm = self._get_accumulator(self._inf_norm_acc_str,
                                         param_and_grad[0])
        # create the adamax optimize op
        adamax_op = block.append_op(
            type=self.type,
            inputs={
                "Param": param_and_grad[0],
                "Grad": param_and_grad[1],
                "LearningRate": self._create_param_lr(param_and_grad),
                "Moment": moment,
                "InfNorm": inf_norm,
                "Beta1Pow": self._beta1_pow_acc
            },
            outputs={
                "ParamOut": param_and_grad[0],
                "MomentOut": moment,
                "InfNormOut": inf_norm
            },
            attrs={
                "beta1": self._beta1,
                "beta2": self._beta2,
                "epsilon": self._epsilon
            })

        return adamax_op

    def _finish_update(self, block):
        """Update Beta1 Power accumulator
        """
        assert isinstance(block, framework.Block)
        main_block = block.program.global_block()
        scale_beta1 = main_block.append_op(
            type="scale",
            inputs={"X": self._beta1_pow_acc},
            outputs={"Out": self._beta1_pow_acc},
            attrs={"scale": self._beta1})

        return [scale_beta1]


class DecayedAdagradOptimizer(Optimizer):
    """Simple Decayed Adagrad optimizer with moment state
    """
    _moment_acc_str = "moment"

    def __init__(self, learning_rate, decay=0.95, epsilon=1.0e-6, **kwargs):
        assert learning_rate is not None
        assert decay is not None
        assert epsilon is not None

        super(DecayedAdagradOptimizer, self).__init__(
            learning_rate=learning_rate, **kwargs)
        self.type = "decayed_adagrad"
        self._decay = decay
        self._epsilon = epsilon

    def _create_accumulators(self, block, parameters):
        assert isinstance(block, framework.Block)

        for p in parameters:
            self._add_accumulator(self._moment_acc_str, p)

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, framework.Block)

        moment_acc = self._get_accumulator(self._moment_acc_str,
                                           param_and_grad[0])

        # Create the decayed adagrad optimizer op
        decayed_adagrad_op = block.append_op(
            type=self.type,
            inputs={
                "Param": param_and_grad[0],
                "Grad": param_and_grad[1],
                "Moment": moment_acc,
                "LearningRate": self._create_param_lr(param_and_grad)
            },
            outputs={"ParamOut": param_and_grad[0],
                     "MomentOut": moment_acc},
            attrs={"epsilon": self._epsilon})

        return decayed_adagrad_op


# We short the class name, since users will use the optimizer with the package
# name. The sample code:
#
# import paddle.fluid as fluid
#
# sgd = fluid.optimizer.SGD(...)
#
# It is no need to add an `Optimizer` as the class suffix
SGD = SGDOptimizer
Momentum = MomentumOptimizer
Adagrad = AdagradOptimizer
Adam = AdamOptimizer
Adamax = AdamaxOptimizer
DecayedAdagrad = DecayedAdagradOptimizer
