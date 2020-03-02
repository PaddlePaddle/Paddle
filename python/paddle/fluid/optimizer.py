# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
from collections import defaultdict

from paddle.fluid.distribute_lookup_table import find_distributed_lookup_table
from paddle.fluid.framework import Program, Variable, name_scope, default_main_program, default_startup_program

from . import framework
from . import layers
from . import unique_name
from .backward import append_backward, _some_in_set_, _append_grad_suffix_, _get_no_grad_set_name
from .clip import append_gradient_clip_ops, error_clip_callback
from .framework import program_guard
from .initializer import Constant
from .layer_helper import LayerHelper
from .layers import ops
from .regularizer import append_regularization_ops
from .dygraph import base as imperative_base
from .dygraph import no_grad
from .dygraph.learning_rate_scheduler import LearningRateDecay
from paddle.fluid import core
from paddle.fluid.layers import tensor
from functools import reduce
from .wrapped_decorator import signature_safe_contextmanager
from .. import compat as cpt

__all__ = [
    'SGD', 'Momentum', 'Adagrad', 'Adam', 'Adamax', 'Dpsgd', 'DecayedAdagrad',
    'Ftrl', 'SGDOptimizer', 'MomentumOptimizer', 'AdagradOptimizer',
    'AdamOptimizer', 'AdamaxOptimizer', 'DpsgdOptimizer',
    'DecayedAdagradOptimizer', 'RMSPropOptimizer', 'FtrlOptimizer', 'Adadelta',
    'AdadeltaOptimizer', 'ModelAverage', 'LarsMomentum',
    'LarsMomentumOptimizer', 'DGCMomentumOptimizer', 'LambOptimizer',
    'ExponentialMovingAverage', 'PipelineOptimizer', 'LookaheadOptimizer',
    'RecomputeOptimizer'
]


class Optimizer(object):
    """Optimizer Base class.

    Define the common interface of an optimizer.
    User should not use this class directly,
    but need to use one of it's implementation.
    """

    @imperative_base.no_grad
    def __init__(self,
                 learning_rate,
                 parameter_list=None,
                 regularization=None,
                 name=None):
        self._parameter_list = None
        if framework.in_dygraph_mode():
            if not isinstance(learning_rate, float) and \
                    not isinstance(learning_rate, LearningRateDecay):
                raise TypeError(
                    "learning rate should be float or LearningRateDecay, got %s here"
                    % type(learning_rate))
            if name is not None:
                self._name = unique_name.generate(name)
            else:
                self._name = unique_name.generate(self.__class__.__name__)
            if parameter_list is not None:
                self._parameter_list = parameter_list
            else:
                raise AttributeError(
                    "parameter_list argument given to the Optimizer should not be None in dygraph mode."
                )
        else:
            if not isinstance(learning_rate, float) and \
                    not isinstance(learning_rate, framework.Variable):
                raise TypeError(
                    "learning rate should be float or Variable, got %s here" %
                    type(learning_rate))
            self._name = name

        self.regularization = regularization
        self._learning_rate = learning_rate
        # the learning rate type should be inferenced from loss
        self._dtype = None
        # each program should have a independent learning rate
        # program -> Variable(learning_rate)
        self._learning_rate_map = dict()
        if isinstance(self._learning_rate, framework.Variable):
            self._learning_rate_map[framework.default_main_program(
            )] = self._learning_rate
        # Dictionary of accumulators. Some optimizer subclasses need to
        # allocate and manage extra variables associated with the parameters
        # to train. These variables are called accumulators.
        # {accum_name : { paramter_name : accumulator_for_parameter, ...}, ...}
        self._accumulators = defaultdict(lambda: dict())
        self.helper = None
        self._opti_name_list = []
        self._accumulators_holder = {}

    @framework.dygraph_only
    def state_dict(self):
        '''
        Get state dict information from optimizer. It contain all the variable used by optimizer. For Adam optimizer, contains beta1, beta2, momentum etc. If LearningRateDecay have been used, global_step will be include in state dict.
        If the optimizer never be called(minimize function), the state_dict is empty.

        Args: None
        Return:
            state_dict(dict) : dict contains all the variable used by optimizer
        
        Examples:
            .. code-block:: python

                import paddle.fluid as fluid

                with fluid.dygraph.guard():
                    emb = fluid.dygraph.Embedding([10, 10])

                    adam = fluid.optimizer.Adam(0.001, parameter_list=emb.parameters())
                    state_dict = adam.state_dict()

        '''
        state_dict = {}
        for k, v in self._accumulators.items():
            for para_name, var_tmp in v.items():
                state_dict[var_tmp.name] = var_tmp
        # global step if use lr decay
        if isinstance(self._learning_rate, LearningRateDecay):
            var_tmp = None
            if framework.in_dygraph_mode():
                var_temp = framework._varbase_creator(
                    None, name='global_step', dtype='int32')
            else:
                var_temp = Variable(None, name='global_step', dtype='int32')

            tensor.fill_constant(
                [1], "int32", self._learning_rate.step_num, out=var_temp)

            state_dict['global_step'] = var_temp
        return state_dict

    @framework.dygraph_only
    def set_dict(self, state_dict):
        '''
        Load optimizer state dict. For Adam optimizer, contains beta1, beta2, momentum etc. If LearningRateDecay have been used, global_step will be changed.

        Args: 
            state_dict(dict) : Dict contains all the Variable needed by optimizer
        Return:
            None
        
        Examples:
            .. code-block:: python

                with fluid.dygraph.guard():
                    emb = fluid.dygraph.Embedding([10, 10])

                    state_dict = emb.state_dict()
                    fluid.save_dygraph(state_dict, "paddle_dy")

                    adam = fluid.optimizer.Adam(learning_rate=fluid.layers.noam_decay( 100, 10000), 
                                                parameter_list=emb.parameters())
                    state_dict = adam.state_dict()
                    fluid.save_dygraph(state_dict, "paddle_dy")

                    para_state_dict, opti_state_dict = fluid.load_dygraph( "paddle_dy")

                    adam.set_dict(opti_state_dict)

        '''

        if isinstance(self._learning_rate, LearningRateDecay):
            assert 'global_step' in state_dict, \
                    'Global step not in state dict, Dygraph use LearningRateDecay, global_step must in state_dict'
            global_step = state_dict['global_step']

            if isinstance(global_step, core.VarBase):
                step_np = global_step
                step_np = np.array(step_np.value().get_tensor())
                assert step_np.shape == (1,),  \
                        "global step shape is (1,), the shape is {}".format( step_np.shape )

                self._learning_rate.step_num = int(step_np[0])
            elif isinstance(global_step, Variable):
                step_np = global_step.numpy()
                assert step_np.shape == (1,),  \
                        "global step shape is (1,), the shape is {}".format( step_np.shape )
                self._learning_rate.step_num = step_np[0]
            elif isinstance(global_step, np.ndarray):
                assert global_step.shape == (1,),  \
                        "global step shape is (1,), the shape is {}".format( global_step.shape )
                self._learning_rate.step_num = global_step[0]
            else:
                raise RuntimeError(
                    "Type not supprt, value in state dict must be [VarBase, Variable, numpy], the type is ",
                    type(global_step))

        self._accumulators_holder = state_dict
        for k, v in self._accumulators.items():
            for para_name, var_tmp in v.items():
                assert var_tmp.name in state_dict, \
                        "optimizer variable {} not found".format( var_tmp.name )
                var = var_tmp.value()
                tensor = var.get_tensor()
                model_np = np.array(tensor)

                load_para = state_dict[var_tmp.name]

                if isinstance(load_para, Variable):
                    load_para_np = load_para.numpy()
                elif isinstance(load_para, core.VarBase):
                    load_para_np = load_para.numpy()
                elif isinstance(load_para, np.ndarray):
                    load_para_np = load_para
                else:
                    raise RuntimeError("State dict type {} not supprt".format(
                        str(type(load_para))))

                assert model_np.shape == load_para_np.shape,  \
                                          "Parameter shape not match, Dygraph Parameter [ {} ] need tensor with shape {} but load tensor with shape {}".format(
                                                 item.name, model_np.shape, load_para_np.shape)

                assert model_np.dtype == load_para_np.dtype, \
                                          "Parameter dtype not match, Dygraph Parameter [ {} ] need tensor with dtype {}  but load tensor with dtype {}".format(
                                                item.name, model_np.dtype, load_para_np.dtype)

                tensor.set(load_para_np, framework._current_expected_place())

    def get_opti_var_name_list(self):
        return self._opti_name_list

    def _create_global_learning_rate(self):
        if imperative_base.enabled():
            # create learning rate Variable
            if isinstance(self._learning_rate, float):
                lr = self._global_learning_rate()

                if isinstance(lr, framework.Variable):
                    return
                else:
                    self._learning_rate_map[framework.default_main_program(
                    )] = layers.create_global_var(
                        name=unique_name.generate("learning_rate"),
                        shape=[1],
                        value=float(self._learning_rate),
                        dtype='float32' if self._dtype is None else self._dtype,
                        persistable=True)
            # get learning rate Variable from LearningRateDecay
            elif isinstance(self._learning_rate, LearningRateDecay):
                self._learning_rate_map[framework.default_main_program(
                )] = self._learning_rate()
            else:
                raise TypeError(
                    "optimizer's learning rate must be float or LearningRateDecay"
                )
        else:
            lr = self._global_learning_rate()

            if isinstance(lr, framework.Variable):
                return
            else:
                if not isinstance(self._learning_rate, float):
                    raise TypeError(
                        "learning rate variable is create outside optimizer,"
                        "can not create new learning rate variable for new program"
                    )

            # create learning rate in the current main program
            self._learning_rate_map[framework.default_main_program(
            )] = layers.create_global_var(
                name=unique_name.generate("learning_rate"),
                shape=[1],
                value=float(self._learning_rate),
                dtype='float32' if self._dtype is None else self._dtype,
                persistable=True)

    @framework.dygraph_only
    def current_step_lr(self):
        """
        .. note::
          **This API is ONLY available in Dygraph mode**
        
        Get current step learning rate. The return value is all the same When LearningRateDecay is not used,
        otherwise return the step learning rate.

        Returns:
            float: The learning rate of the current step.

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                import numpy as np

                # example1: LearningRateDecay is not used, return value is all the same
                with fluid.dygraph.guard():
                    emb = fluid.dygraph.Embedding([10, 10])
                    adam = fluid.optimizer.Adam(0.001, parameter_list = emb.parameters())
                    lr = adam.current_step_lr()
                    print(lr) # 0.001

                # example2: PiecewiseDecay is used, return the step learning rate
                with fluid.dygraph.guard():
                    inp = np.random.uniform(-0.1, 0.1, [10, 10]).astype("float32")
                    linear = fluid.dygraph.nn.Linear(10, 10)
                    inp = fluid.dygraph.to_variable(inp)
                    out = linear(inp)
                    loss = fluid.layers.reduce_mean(out)
                    
                    bd = [2, 4, 6, 8]
                    value = [0.2, 0.4, 0.6, 0.8, 1.0]
                    adam = fluid.optimizer.Adam(fluid.dygraph.PiecewiseDecay(bd, value, 0),
                                           parameter_list=linear.parameters())

                    # first step: learning rate is 0.2
                    np.allclose(adam.current_step_lr(), 0.2, rtol=1e-06, atol=0.0) # True

                    # learning rate for different steps
                    ret = [0.2, 0.2, 0.4, 0.4, 0.6, 0.6, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0]
                    for i in range(12):
                        adam.minimize(loss)
                        lr = adam.current_step_lr()
                        np.allclose(lr, ret[i], rtol=1e-06, atol=0.0) # True

        """
        current_lr = self._global_learning_rate()
        if current_lr:
            return self._global_learning_rate().numpy()[0]

        if isinstance(self._learning_rate, float):
            return self._learning_rate
        else:
            step_lr = self._learning_rate.step()
            if isinstance(step_lr, (float, int)):
                return step_lr
            else:
                return step_lr.numpy()[0]

    def _global_learning_rate(self, program=None):
        """
        get global decayed learning rate
        :return:
        """
        if program is None:
            program = framework.default_main_program()
        return self._learning_rate_map.get(program, None)

    def _append_optimize_op(self, block, param_and_grad):
        """ append optimize operator to block and return all the added optimize_op
        """
        raise NotImplementedError()

    def _create_param_lr(self, param_and_grad):
        # create learning rate variable for every parameter
        param = param_and_grad[0]
        param_lr = param.optimize_attr['learning_rate']
        if type(param_lr) == Variable:
            return param_lr
        else:
            if param_lr == 1.0:
                return self._global_learning_rate()
            else:
                with default_main_program()._lr_schedule_guard(
                        is_with_opt=True), framework.name_scope(
                            'scale_with_param_lr'):
                    return self._global_learning_rate() * param_lr

    def _create_accumulators(self, block, parameters):
        """Create all accumulators needed by the parameters

        Args:
            block: the block in which the loss variable is present
            parameters: list of parameter variables for the optimizer
        """
        pass

    def _finish_update(self, block, parameters_and_grads):
        """Finish any custom updates needed
           before completing an optimization step

        Args:
            block: the block in which the loss variable is present
            parameters: list of parameter variables for the optimizer

        Returns:
            None
        """
        pass

    def _add_accumulator(self,
                         name,
                         param,
                         dtype=None,
                         fill_value=0.0,
                         shape=None,
                         type=None,
                         force_cpu=False):
        """Utility function to add an accumulator for a parameter

        Args:
            block: the block in which the loss variable is present
            name: name of the accumulator
            param: parameter variable for which accumulator is to be added
            dtype: data type of the accumulator variable
            fill_value: value to initialize the accumulator variable
        """
        if self._name is not None:
            name = self._name + "_" + name
        if (name in self._accumulators and
                param.name in self._accumulators[name]):
            if framework.in_dygraph_mode():
                return self._accumulators[name][param.name]
            raise Exception("Accumulator {} already exists for parameter {}".
                            format(name, param.name))
        if shape == None:
            shape = param.shape
        assert isinstance(self.helper, LayerHelper)

        var_name = param.name + "_" + name
        var_name = unique_name.generate(var_name)
        self._opti_name_list.append(var_name)

        var = self.helper.create_global_variable(
            name=var_name,
            persistable=True,
            dtype=dtype or param.dtype,
            type=param.type if type is None else type,
            shape=shape,
            belong_to_optimizer=True)
        self.helper.set_variable_initializer(
            var,
            initializer=Constant(
                value=float(fill_value), force_cpu=force_cpu))

        if framework.in_dygraph_mode():
            if len(self._accumulators_holder) > 0:
                assert var_name in self._accumulators_holder, \
                        "Optimizer set error, {} should in state dict".format( var_name )
                var.set_value(self._accumulators_holder[var_name])

        self._accumulators[name][param.name] = var
        return var

    def _get_accumulator(self, name, param):
        """Utility function to fetch an accumulator for a parameter

        Args:
            name: name of the accumulator
            param: parameter variable for which accumulator is to be fetched

        Returns:
            accumulator variable for the parameter
        """
        if self._name is not None:
            name = self._name + "_" + name
        if (name not in self._accumulators or
                param.name not in self._accumulators[name]):
            raise Exception("Accumulator {} does not exist for parameter {}".
                            format(name, param.name))
        return self._accumulators[name][param.name]

    def _create_optimization_pass(self, parameters_and_grads):
        """Add optimization operators to update gradients to variables.

        Args:
          parameters_and_grads(list(tuple(Variable, Variable))):
            a list of (variable, gradient) pair to update.

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

        # Allways called under program_guard use global block as loss block
        # But if current block is in control flow, append optimize op in the
        # grad block of current block

        global_block = framework.default_main_program().global_block()
        target_block = global_block
        current_block = framework.default_main_program().current_block()
        if current_block.idx != global_block.idx:
            assert current_block.backward_block_idx != -1, \
                "current block is not global_block, but it doesn't have backward block."
            target_block = framework.default_main_program().blocks[
                current_block.backward_block_idx]

        start = len(target_block.ops)
        self.helper = LayerHelper(self.__class__.__name__)
        self._create_accumulators(
            target_block,
            [p[0] for p in parameters_and_grads if p[0].trainable])
        self._create_global_learning_rate()

        if framework.in_dygraph_mode():
            for param_and_grad in parameters_and_grads:
                if param_and_grad[1] is None:
                    continue
                if param_and_grad[0].trainable is True:
                    self._append_optimize_op(target_block, param_and_grad)
        else:
            for param_and_grad in parameters_and_grads:
                if param_and_grad[1] is None:
                    continue
                with param_and_grad[0].block.program._optimized_guard(
                        param_and_grad), name_scope("optimizer"):
                    if param_and_grad[0].trainable is True:
                        self._append_optimize_op(target_block, param_and_grad)

        # Get custom finish ops for subclasses
        # FIXME: Need to fix this once we figure out how to handle dependencies
        self._finish_update(target_block, parameters_and_grads)

        end = len(target_block.ops)
        return target_block._slice_ops(start, end)

    def _process_distribute_lookuptable(self, param_grads):
        """
        Because distribute lookup table only support SGD optimizer for now, not support
        other optimizer and regularization, so we should find the table parameter out,
        and avoid to add regularization and other op for it, and add sgd optimize op
        for it independently.
        :param param_grads(list((Var, Var))): list of (param, grad) pair.
        :param loss: the loss variable.
        :param startup_program: the startup program
        """
        program = framework.default_main_program()
        global_block = framework.default_main_program().global_block()
        table_name = find_distributed_lookup_table(program)
        table_param = None
        table_grad = None
        new_param_grads = []
        for p, g in param_grads:
            if p.name == table_name:
                if table_param is not None:
                    raise RuntimeError(
                        "multi dist table var found, only support one now!")
                table_param = p
                table_grad = g
            else:
                new_param_grads.append((p, g))
        sgd_op = None
        if table_param is not None:
            param_and_grad = [table_param, table_grad]
            with table_param.block.program._optimized_guard(param_and_grad), \
                    framework.name_scope("optimizer"):
                self._create_global_learning_rate()
                # create the optimize op
                sgd_op = global_block.append_op(
                    type='sgd',
                    inputs={
                        "Param": table_param,
                        "Grad": table_grad,
                        "LearningRate": self._create_param_lr(param_and_grad)
                    },
                    outputs={"ParamOut": param_and_grad[0]})
        return new_param_grads, (table_param, table_grad), sgd_op

    def _append_dgc_ops(self, param_and_grad):
        pass

    def backward(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None,
                 callbacks=None):
        """
        The first part of ``minimize``, do auto-diff to append backward operations for
        the current program.

        Args:
            loss (Variable): ``loss`` variable to run optimizations.
            startup_program (Program, optional): :ref:`api_fluid_Program` for
                initializing parameters in ``parameter_list``. The default value
                is None, at this time :ref:`api_fluid_default_startup_program` will be used.
            parameter_list (list, optional): List of ``Variable`` or ``Variable.name`` to update
                to minimize ``loss``. The default value is None, at this time all parameters
                will be updated.
            no_grad_set (set, optional): Set of ``Variable``  or ``Variable.name`` that don't need
                to be updated. The default value is None.
            callbacks (list, optional): list of callable objects to run when appending backward
                operator for one parameter. The default value is None.

        Return:
            list: list of (param, grad) variable pairs, param is ``Parameter``,
                grad is the gradient value corresponding to the parameter.

        Examples:
            See examples in ``apply_gradients``.
        """
        act_no_grad_set = None
        if framework.in_dygraph_mode():
            pass
        else:
            act_no_grad_set = self._get_no_grad_set(loss, no_grad_set)

        self._dtype = loss.dtype
        if framework.in_dygraph_mode():
            params_grads = []
            for param in self._parameter_list:
                if not param.trainable:
                    continue
                if param._grad_ivar() is not None:
                    # create gradient variable
                    grad_var = param._grad_ivar()
                    params_grads.append((param, grad_var))
        else:
            if callbacks is None:
                callbacks = [error_clip_callback]
            else:
                assert (isinstance(callbacks, list))
            program = loss.block.program
            assert len(loss.shape) == 1 and loss.shape[0] == 1, \
                "The loss.shape should be (1L,), but the current loss.shape is {}. " \
                "Maybe that you should call fluid.layers.mean to process the current loss.".format(
                    loss.shape)
            with program_guard(program, startup_program):
                params_grads = append_backward(loss, parameter_list,
                                               act_no_grad_set, callbacks)
                # Note: since we can't use all_reduce_op now,
                #  dgc_op should be the last op of one grad.
                self._append_dgc_ops(params_grads)
        return params_grads

    def apply_gradients(self, params_grads):
        """
        Second part of `minimize`, appending optimization operators for
        given `params_grads` pairs.

        Args:
            params_grads (list): list of (param, grad) pair to do optimization.

        Returns:
            list: A list of operators appended to the current program.

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                loss = network()
                optimizer = fluid.optimizer.SGD(learning_rate=0.1)
                params_grads = optimizer.backward(loss)
                # you may append operations for params_grads here
                # ...
                optimizer.apply_gradients(params_grads)
        """
        params_grads = sorted(params_grads, key=lambda x: x[0].name)

        params_grads, table_param_and_grad, table_optimize_op = \
            self._process_distribute_lookuptable(params_grads)

        params_grads = append_gradient_clip_ops(params_grads)

        # Add regularization if any
        params_grads = append_regularization_ops(params_grads,
                                                 self.regularization)

        optimize_ops = self._create_optimization_pass(params_grads)
        if table_optimize_op is not None:
            optimize_ops.append(table_optimize_op)
            params_grads.append(table_param_and_grad)

        return optimize_ops

    def apply_optimize(self, loss, startup_program, params_grads):
        """
        Second part of `minimize`, appending optimization operators for
        given `params_grads` pairs.

        Args:
            loss (Variable): loss variable to run optimizations.
            startup_program (Program): startup_program for initializing parameters
                in `parameter_list`.
            params_grads (list): list of (param, grad) pair to do optimization.

        Returns:
            list: A list of operators appended to the current program.
        """
        if framework.in_dygraph_mode():
            with program_guard(framework.default_main_program(),
                               framework.default_startup_program()):
                params_grads = append_regularization_ops(params_grads,
                                                         self.regularization)
                optimize_ops = self._create_optimization_pass(params_grads)
        else:
            program = loss.block.program
            with program_guard(program, startup_program):
                optimize_ops = self.apply_gradients(params_grads)
        return optimize_ops

    def _get_no_grad_set(self, loss, no_grad_set=None):
        no_grad_set = _get_no_grad_set_name(no_grad_set)
        parameters = loss.block.program.global_block().all_parameters()
        param_no_trainable = set(
            [param.name for param in parameters if param.trainable is False])
        # If the parameter is no trainable, it should not have a gradient.
        no_grad_set.update(param_no_trainable)

        return no_grad_set

    @framework.dygraph_only
    def clear_gradients(self):
        """
        Clear the gradients of all optimized parameters for model.
        
        Returns:
            None
        
        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                import numpy as np

                with fluid.dygraph.guard():
                    value = np.arange(26).reshape(2, 13).astype("float32")
                    a = fluid.dygraph.to_variable(value)
                    linear = fluid.Linear(13, 5, dtype="float32")
                    # This can be any optimizer supported by dygraph.
                    adam = fluid.optimizer.Adam(learning_rate = 0.01, 
                                                parameter_list = linear.parameters())
                    out = linear(a)
                    out.backward()
                    adam.minimize(out)
                    adam.clear_gradients()

        """
        for p in self._parameter_list:
            if p.trainable:
                p.clear_gradient()

    @imperative_base.no_grad
    def minimize(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None,
                 grad_clip=None):
        """
        Add operations to minimize ``loss`` by updating ``parameter_list``.

        Args:
            loss (Variable): A ``Variable`` containing the value to minimize.
            startup_program (Program, optional): :ref:`api_fluid_Program` for
                initializing parameters in ``parameter_list``. The default value
                is None, at this time :ref:`api_fluid_default_startup_program` will be used.
            parameter_list (list, optional): List of ``Variable`` or ``Variable.name`` to update
                to minimize ``loss``. The default value is None, at this time all parameters
                will be updated.
            no_grad_set (set, optional): Set of ``Variable``  or ``Variable.name`` that don't need
                to be updated. The default value is None.
            grad_clip (GradClipBase, optional) : Gradient clipping strategy, static
                graph mode does not need to use this argument. Currently, this argument
                only supports gradient clipping in dygraph mode. In the future, this
                argument my be adjusted. The default value is None.

        Returns:
            tuple: tuple (optimize_ops, params_grads), A list of operators appended
            by minimize and a list of (param, grad) variable pairs, param is
            ``Parameter``, grad is the gradient value corresponding to the parameter.

        Examples:
            Please refer to the example of current Optimizer.
        """
        assert isinstance(loss, Variable), "The loss should be an Variable."
        params_grads = self.backward(
            loss,
            startup_program=startup_program,
            parameter_list=parameter_list,
            no_grad_set=no_grad_set)

        if grad_clip is not None and framework.in_dygraph_mode():
            # TODO(hongyu): FIX later, this is only for dygraph, should be work for static mode
            params_grads = grad_clip(params_grads)

        optimize_ops = self.apply_optimize(
            loss, startup_program=startup_program, params_grads=params_grads)

        return optimize_ops, params_grads


class SGDOptimizer(Optimizer):
    """
    Optimizer of the stochastic gradient descent algorithm.

    .. math::

        param\_out = param - learning\_rate * grad

    Parameters:
        learning_rate (float|Variable): The learning rate used to update parameters. \
            Can be a float value or a Variable with one float value as data element.
        parameter_list (list, optional):  List of ``Variable`` names to update to minimize ``loss``. \
            This parameter is required in dygraph mode. \
            The default value is None in static mode, at this time all parameters will be updated.
        regularization: A Regularizer, such as :ref:`api_fluid_regularizer_L2DecayRegularizer`. \
            Optional, default is None.
        name (str, optional): This parameter is used by developers to print debugging information. \
            For details, please refer to :ref:`api_guide_Name`. Default is None.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            import numpy as np

            place = fluid.CPUPlace()
            main = fluid.Program()
            with fluid.program_guard(main):
                x = fluid.layers.data(name='x', shape=[13], dtype='float32')
                y = fluid.layers.data(name='y', shape=[1], dtype='float32')
                y_predict = fluid.layers.fc(input=x, size=1, act=None)
                cost = fluid.layers.square_error_cost(input=y_predict, label=y)
                avg_cost = fluid.layers.mean(cost)

                sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
                sgd_optimizer.minimize(avg_cost)

                fetch_list = [avg_cost]
                train_reader = paddle.batch(
                    paddle.dataset.uci_housing.train(), batch_size=1)
                feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
                exe = fluid.Executor(place)
                exe.run(fluid.default_startup_program())
                for data in train_reader():
                    exe.run(main, feed=feeder.feed(data), fetch_list=fetch_list)

    """

    def __init__(self,
                 learning_rate,
                 parameter_list=None,
                 regularization=None,
                 name=None):
        assert learning_rate is not None
        super(SGDOptimizer, self).__init__(
            learning_rate=learning_rate,
            parameter_list=parameter_list,
            regularization=regularization,
            name=name)
        self.type = "sgd"

    @no_grad
    def _append_optimize_op(self, block, param_and_grad):
        if framework.in_dygraph_mode():
            inputs = {
                "Param": [param_and_grad[0]],
                "Grad": [param_and_grad[1]],
                "LearningRate": [self._create_param_lr(param_and_grad)]
            }
            attrs = {}
            outputs = {'ParamOut': [param_and_grad[0]]}
            outs = core.ops.sgd(inputs, attrs, outputs)
            return outs['ParamOut'][0]

        assert isinstance(block, framework.Block)
        # create the optimize op
        sgd_op = block.append_op(
            type=self.type,
            inputs={
                "Param": param_and_grad[0],
                "Grad": param_and_grad[1],
                "LearningRate": self._create_param_lr(param_and_grad)
            },
            outputs={"ParamOut": param_and_grad[0]},
            stop_gradient=True)

        return sgd_op


class MomentumOptimizer(Optimizer):
    """

    Simple Momentum optimizer with velocity state

    This optimizer has a flag for Nestrov Momentum.

    The update equations are as follows:

    .. math::

        & velocity = mu * velocity + gradient

        & if (use\_nesterov):

        &\quad   param = param - (gradient + mu * velocity) * learning\_rate

        & else:

        &\quad   param = param - learning\_rate * velocity

    Parameters:
        learning_rate (float|Variable): The learning rate used to update parameters. \
            Can be a float value or a Variable with one float value as data element.
        momentum (float): Momentum factor
        parameter_list (list, optional):  List of ``Variable`` names to update to minimize ``loss``. \
            This parameter is required in dygraph mode. \
            The default value is None in static mode, at this time all parameters will be updated.
        use_nesterov (bool, optional): Enables Nesterov momentum, default is false.
        regularization: A Regularizer, such as :ref:`api_fluid_regularizer_L2DecayRegularizer`. \
            Optional, default is None.
        name (str, optional): This parameter is used by developers to print debugging information. \
            For details, please refer to :ref:`api_guide_Name`. Default is None.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            import numpy as np

            place = fluid.CPUPlace()
            main = fluid.Program()
            with fluid.program_guard(main):
                x = fluid.layers.data(name='x', shape=[13], dtype='float32')
                y = fluid.layers.data(name='y', shape=[1], dtype='float32')
                y_predict = fluid.layers.fc(input=x, size=1, act=None)
                cost = fluid.layers.square_error_cost(input=y_predict, label=y)
                avg_cost = fluid.layers.mean(cost)

                moment_optimizer = fluid.optimizer.MomentumOptimizer(learning_rate=0.001, momentum=0.9)
                moment_optimizer.minimize(avg_cost)

                fetch_list = [avg_cost]
                train_reader = paddle.batch(
                    paddle.dataset.uci_housing.train(), batch_size=1)
                feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
                exe = fluid.Executor(place)
                exe.run(fluid.default_startup_program())
                for data in train_reader():
                    exe.run(main, feed=feeder.feed(data), fetch_list=fetch_list)

    """
    _velocity_acc_str = "velocity"

    def __init__(self,
                 learning_rate,
                 momentum,
                 parameter_list=None,
                 use_nesterov=False,
                 regularization=None,
                 name=None):
        assert learning_rate is not None
        assert momentum is not None
        super(MomentumOptimizer, self).__init__(
            learning_rate=learning_rate,
            parameter_list=parameter_list,
            regularization=regularization,
            name=name)
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
        attrs = {"mu": self._momentum, "use_nesterov": self._use_nesterov}

        inputs = {
            "Param": [param_and_grad[0]],
            "Grad": [param_and_grad[1]],
            "Velocity": [velocity_acc],
            "LearningRate": [self._create_param_lr(param_and_grad)]
        }

        outputs = {
            "ParamOut": [param_and_grad[0]],
            "VelocityOut": [velocity_acc]
        }

        if framework.in_dygraph_mode():
            core.ops.momentum(inputs, attrs, outputs)
            return None

        # create the momentum optimize op
        momentum_op = block.append_op(
            type=self.type,
            inputs=inputs,
            outputs=outputs,
            attrs=attrs,
            stop_gradient=True)

        return momentum_op


class DGCMomentumOptimizer(Optimizer):
    """
    DGC (Deep Gradient Compression) Momentum Optimizer. Original paper is https://arxiv.org/abs/1712.01887

    DGC reduces the communication bandwidth by sending only the important gradients (sparse update):\
        only gradients larger than a threshold are transmitted.

    To avoid losing information, DGC accumulates the rest of the gradients locally.

    Eventually, these gradients become large enough to be transmitted.

    Thus, DGC sends the large gradients immediately but eventually sends all of the gradients over time.

    To ensure no loss of accuracy, DGC employs momentum correction and local gradient clipping on top of the gradient sparsification to maintain model performance.

    DGC also uses momentum factor masking and warmup training to overcome the staleness problem caused by reduced communication.

    This optimizer will do two things:

        1. Compress the gradient by get TopK import value from tensor \
            and use it for allreduce to reduce network bandwidth.

        2. Call momentum to optimize the cost.

    Args:
        learning_rate (float|Variable): The learning rate used to update parameters. \
            It can be a float value or a Variable with one float value as a data element.
        momentum (float): Momentum factor.
        rampup_begin_step (int): The beginning step from which gradient compression is implemented.
        rampup_step (int): Time steps used in sparsity warm-up periods. Default is 1.
            For example, if the sparsity is [0.75, 0.9375, 0.984375, 0.996, 0.999], and the rampup_step is 100, \
                it will use 0.75 at 0~19 steps, and 0.9375 at 20~39 steps, and so on. \
                And when reach sparsity array ends, it will use 0.999 then and after.
        sparsity (list[float]): Get top important element from gradient tensor, the ratio is (1 - current sparsity). \
            Default is [0.999]. For example, if the sparsity is [0.99, 0.999], \
                the top [1%, 0.1%] important element will be transmitted.
        parameter_list (list, optional):  List of ``Variable`` names to update to minimize ``loss``. \
            This parameter is required in dygraph mode. \
            The default value is None in static mode, at this time all parameters will be updated.
        use_nesterov (bool): Enables Nesterov momentum. True means use Nesterov. Default is False.
        local_grad_clip_norm (float, optional): Local gradient clip norm value. Optional, default is None, represent no need clip.
        num_trainers (int, optional): The number of training nodes. Optional, default is None.
        regularization (WeightDecayRegularizer, optional): A Regularizer, such as \
            :ref:`api_fluid_regularizer_L2DecayRegularizer`. Optional, default is None.
        name (str, optional): This parameter is used by developers to print debugging information. \
            For details, please refer to :ref:`api_guide_Name`. Default is None.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            optimizer = fluid.optimizer.DGCMomentumOptimizer(
                        learning_rate=0.0001,
                        momentum=0.9,
                        rampup_step=1000,
                        rampup_begin_step=1252,
                        sparsity=[0.999, 0.999])

    """
    _u_velocity_acc_str = "_dgc_u_"
    _v_velocity_acc_str = "_dgc_v_"

    def __init__(self,
                 learning_rate,
                 momentum,
                 rampup_begin_step,
                 rampup_step=1,
                 sparsity=[0.999],
                 parameter_list=None,
                 use_nesterov=False,
                 local_grad_clip_norm=None,
                 num_trainers=None,
                 regularization=None,
                 name=None):
        if framework.in_dygraph_mode():
            raise Exception("In dygraph, don't support DGCMomentumOptimizer.")

        assert core.is_compiled_with_cuda(), \
            "Paddle is not compiled with CUDA. DGC is only support GPU for now."

        assert learning_rate is not None
        assert momentum is not None
        super(DGCMomentumOptimizer, self).__init__(
            learning_rate=learning_rate,
            parameter_list=parameter_list,
            regularization=regularization,
            name=name)
        self.type = "dgc_momentum"
        self._momentum = momentum
        self._use_nesterov = bool(use_nesterov)

        assert rampup_begin_step >= 0, "rampup_begin_step must >= 0"
        self._rampup_begin_step = rampup_begin_step
        self._rampup_step = rampup_step
        self._sparsity = sparsity

        self._rampup_begin_step_var = None
        self._global_step_var = None

        self._local_grad_clip_norm = None
        self._clip_norm = None
        if local_grad_clip_norm is not None:
            assert isinstance(num_trainers, int)
            assert isinstance(local_grad_clip_norm, float)
            assert num_trainers > 0

            self._local_grad_clip_norm = local_grad_clip_norm
            self._num_trainers = num_trainers
            self._clip_norm = local_grad_clip_norm * (num_trainers**-0.5)

        self._get_dgc_regularization_param()

    def _get_dgc_regularization_param(self):
        self.regular_coeff = 0.0
        self.regular_type = 0

        if self.regularization is not None:
            self.regular_coeff = self.regularization._regularization_coeff
            from .regularizer import L1Decay, L2Decay
            if isinstance(self.regularization, L1Decay):
                self.regular_type = 1
            elif isinstance(self.regularization, L2Decay):
                self.regular_type = 2
            else:
                assert False, 'regularization must be None|L1Decay|L2Deacy'

    def _is_use_dgc(self, param_var, grad_var):
        var_numel = abs(reduce(lambda x, y: x * y, param_var.shape))
        if var_numel < 16384 or \
           param_var.type == core.VarDesc.VarType.SELECTED_ROWS  or \
           grad_var.type == core.VarDesc.VarType.SELECTED_ROWS  or  \
               param_var.dtype != core.VarDesc.VarType.FP32 :
            return False
        return True

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, framework.Block)
        velocity_acc = self._get_accumulator(self._u_velocity_acc_str,
                                             param_and_grad[0])
        assert velocity_acc is not None

        inputs = {
            "Param": param_and_grad[0],
            "Grad": param_and_grad[1],
            "Velocity": velocity_acc,
            "LearningRate": self._create_param_lr(param_and_grad),
        }
        outputs = {
            "ParamOut": param_and_grad[0],
            "VelocityOut": velocity_acc,
        }
        attrs = {"mu": self._momentum, "use_nesterov": self._use_nesterov}

        if not self._is_use_dgc(param_and_grad[0], param_and_grad[1]):
            type = "momentum"
        else:
            type = "dgc_momentum"
            inputs.update({
                "current_step": self._global_step_var,
                "nranks": self._nranks_var
            })
            outputs.update({'Grad_out': param_and_grad[1]})
            attrs.update({"rampup_begin_step": float(self._rampup_begin_step)})

        # create the dgc momentum optimize op
        dgc_momentum_op = block.append_op(
            type=type,
            inputs=inputs,
            outputs=outputs,
            attrs=attrs,
            stop_gradient=True)
        return dgc_momentum_op

    def _add_auto_increment_var(self, counter_name, begin, step=1):
        helper = LayerHelper('global_step_counter')
        counter, is_new_var = helper.create_or_get_global_variable(
            name=counter_name, dtype='float32', shape=[1], persistable=True)
        if is_new_var:
            helper.set_variable_initializer(
                counter,
                initializer=Constant(
                    value=float(begin - 1), force_cpu=True))
            helper.main_program.global_block()._prepend_op(
                type='increment',
                inputs={'X': [counter]},
                outputs={'Out': [counter]},
                attrs={'step': float(step)},
                stop_gradient=True)
            counter.stop_gradient = True

        return counter

    def _add_nranks_var(self, name, value=-1):
        helper = LayerHelper('global_step_counter')
        counter, is_new_var = helper.create_or_get_global_variable(
            name=name, dtype='float32', shape=[1], persistable=True)
        if is_new_var:
            helper.set_variable_initializer(
                counter,
                initializer=Constant(
                    value=float(value), force_cpu=True))
            counter.stop_gradient = True

        return counter

    def _append_dgc_ops(self, param_and_grads):
        main_program = default_main_program()
        main_program._enable_dgc = True

        # step counter
        self._global_step_var = self._add_auto_increment_var(
            counter_name=core.dgc.kDGCCounterName(), begin=0)

        self._nranks_var = self._add_nranks_var(
            name=core.dgc.kDGCNRanksName(), value=-1)

        # rampup begin step var for all_reduce_op_handle
        self._rampup_begin_step_var = tensor.create_global_var(
            shape=[1],
            dtype=core.VarDesc.VarType.FP32,
            persistable=True,
            name=core.dgc.kDGCRampUpBeginStepName(),
            value=self._rampup_begin_step * 1.0,
            force_cpu=True)

        self.helper = LayerHelper(self.__class__.__name__)

        for param_var, grad_var in param_and_grads:
            # reuse velocity in dgc_op and dgc_momentum_op
            u_var = self._add_accumulator(self._u_velocity_acc_str, param_var)

            if not self._is_use_dgc(param_var, grad_var):
                continue

            v_var = self._add_accumulator(self._v_velocity_acc_str, param_var)

            k_var = tensor.create_global_var(
                shape=[1],
                dtype=param_var.dtype,
                persistable=True,
                name=param_var.name + core.dgc.kDGCKName(),
                value=0.0,
                force_cpu=True)

            encoded_var = tensor.create_global_var(
                shape=[1],
                dtype=param_var.dtype,
                persistable=True,
                name=param_var.name + core.dgc.kDGCEncodedName(),
                value=0.0,
                force_cpu=False)

            gather_var = tensor.create_global_var(
                shape=[1],
                dtype=param_var.dtype,
                persistable=True,
                name=param_var.name + core.dgc.kDGCGatherName(),
                value=0.0,
                force_cpu=False)

            # del back oprolevarname
            op_maker = core.op_proto_and_checker_maker
            backward = core.op_proto_and_checker_maker.OpRole.Backward
            for op in main_program.global_block().ops:
                if not self._is_the_backward_op(op):
                    continue

                var_attr = op.all_attrs()[op_maker.kOpRoleVarAttrName()]
                if param_var.name not in var_attr:
                    continue

                var_attr.remove(param_var.name)
                var_attr.remove(grad_var.name)
                if len(var_attr) > 1:
                    op._set_attr(op_maker.kOpRoleVarAttrName(), var_attr)
                else:
                    op._remove_attr(op_maker.kOpRoleVarAttrName())

            clip_var = grad_var
            if self._local_grad_clip_norm is not None:
                clip_var = self._append_clip_norm(grad_var, self._clip_norm)
            self._dgc_op(param_var, clip_var, grad_var, u_var, v_var, k_var,
                         encoded_var, gather_var)

    def _is_the_backward_op(self, op):
        op_maker = core.op_proto_and_checker_maker
        backward = core.op_proto_and_checker_maker.OpRole.Backward
        if op_maker.kOpRoleVarAttrName() in op.attr_names and \
                int(op.all_attrs()[op_maker.kOpRoleAttrName()]) == int(backward):
            return True
        return False

    def _clip_by_norm(self, x, max_norm, name=None):
        args = {'x': x, 'max_norm': max_norm, 'name': name}

        helper = LayerHelper("dgc_clip_by_norm_op", **args)

        if name is None:
            name = unique_name.generate_with_ignorable_key(".".join(
                [helper.name, 'tmp']))

        out = helper.create_variable(
            type=x.type, name=name, dtype=x.dtype, persistable=False)

        helper.append_op(
            type="dgc_clip_by_norm",
            inputs={"X": x,
                    "current_step": self._global_step_var},
            attrs={
                "max_norm": max_norm,
                "rampup_begin_step": float(self._rampup_begin_step)
            },
            outputs={"Out": out})
        return out

    def _append_clip_norm(self, grad_var, clip_norm):
        with grad_var.block.program._backward_role_guard():
            return self._clip_by_norm(
                x=grad_var, max_norm=clip_norm, name=grad_var.name)

    def _dgc_op(self, param_var, clip_var, grad_var, u_var, v_var, k_var,
                encoded_var, gather_var):
        block = framework.default_main_program().global_block()
        op_maker = core.op_proto_and_checker_maker

        dgc_op = block.append_op(
            type="dgc",
            inputs={
                "U": u_var,
                "V": v_var,
                "Grad": clip_var,
                "Param": param_var,
                "current_step": self._global_step_var,
                "nranks": self._nranks_var,
            },
            outputs={
                "U_out": u_var,
                "V_out": v_var,
                "EncodeGrad": encoded_var,
                "k": k_var,
                "Grad_out": grad_var,
                "GatherBuff": gather_var,
            },
            attrs={
                "m": self._momentum,
                "sparsity": self._sparsity,
                "use_nesterov": self._use_nesterov,
                "rampup_begin_step": float(self._rampup_begin_step),
                "rampup_step": float(self._rampup_step),
                "regular_coeff": float(self.regular_coeff),
                "regular_type": int(self.regular_type),
            },
            stop_gradient=True)

        backward = op_maker.OpRole.Backward
        dgc_op._set_attr(op_maker.kOpRoleAttrName(), backward)
        dgc_op._set_attr(op_maker.kOpRoleVarAttrName(),
                         [param_var.name, grad_var.name])

    def apply_gradients(self, params_grads):
        params_grads = sorted(params_grads, key=lambda x: x[0].name)

        params_grads, table_param_and_grad, table_optimize_op = \
            self._process_distribute_lookuptable(params_grads)

        not_dgc_params_grads = []
        dgc_params_grads = []
        for param, grad in params_grads:
            if not self._is_use_dgc(param, grad):
                not_dgc_params_grads.append((param, grad))
            else:
                dgc_params_grads.append((param, grad))

        # DGC clip and regularization in local
        not_dgc_params_grads = append_gradient_clip_ops(not_dgc_params_grads)

        # Add regularization if any
        not_dgc_params_grads = append_regularization_ops(not_dgc_params_grads,
                                                         self.regularization)

        params_grads = not_dgc_params_grads + dgc_params_grads
        params_grads = sorted(params_grads, key=lambda x: x[0].name)

        optimize_ops = self._create_optimization_pass(params_grads)
        if table_optimize_op is not None:
            optimize_ops.append(table_optimize_op)
            params_grads.append(table_param_and_grad)

        return optimize_ops


class LarsMomentumOptimizer(Optimizer):
    """
    Momentum optimizer with LARS support

    The update equations are as follows:

    .. math::

        & local\_learning\_rate = learning\_rate * lars\_coeff * \\
          \\frac{||param||}{||gradient|| + lars\_weight\_decay * ||param||}

        & velocity = mu * velocity + local\_learning\_rate * (gradient + lars\_weight\_decay * param)

        & param = param - velocity

    Parameters:
        learning_rate (float|Variable): The learning rate used to update parameters. \
            Can be a float value or a Variable with one float value as data element. \
            momentum (float): momentum factor
        lars_coeff (float): Defines how much we trust the layer to change its weights.
        lars_weight_decay (float): Weight decay coefficient for decaying using LARS.
        parameter_list (list, optional):  List of ``Variable`` names to update to minimize ``loss``. \
            This parameter is required in dygraph mode. \
            The default value is None in static mode, at this time all parameters will be updated.
        regularization: A Regularizer, such as :ref:`api_fluid_regularizer_L2DecayRegularizer`.
            Optional, default is None.
        name (str, optional): This parameter is used by developers to print debugging information. \
            For details, please refer to :ref:`api_guide_Name`. Default is None.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np

            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            out = fluid.layers.fc(inp, size=3)
            out = fluid.layers.reduce_sum(out)
            optimizer = fluid.optimizer.LarsMomentumOptimizer(learning_rate=0.001, momentum=0.9)
            optimizer.minimize(out)

            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(fluid.default_startup_program())
            exe.run(
                feed={"inp": np_inp},
                fetch_list=[out.name])
    """
    _velocity_acc_str = "velocity"

    def __init__(self,
                 learning_rate,
                 momentum,
                 lars_coeff=0.001,
                 lars_weight_decay=0.0005,
                 parameter_list=None,
                 regularization=None,
                 name=None):
        assert learning_rate is not None
        assert momentum is not None
        super(LarsMomentumOptimizer, self).__init__(
            learning_rate=learning_rate,
            parameter_list=parameter_list,
            regularization=regularization,
            name=name)
        self.type = "lars_momentum"
        self._momentum = momentum
        self._lars_coeff = float(lars_coeff)
        self._lars_weight_decay = float(lars_weight_decay)

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
            attrs={
                "mu": self._momentum,
                "lars_coeff": self._lars_coeff,
                "lars_weight_decay": self._lars_weight_decay
            },
            stop_gradient=True)

        return momentum_op


class AdagradOptimizer(Optimizer):
    """
    The Adaptive Gradient optimizer (Adagrad for short) can adaptively assign
    different learning rates to individual parameters.

    The parameter ``param_out`` update rule with gradient ``grad``:

    .. math::

        moment\_out &= moment + grad * grad

        param\_out &= param - \\frac{learning\_rate * grad}{\sqrt{moment\_out} + \epsilon}

    Related paper: `Adaptive Subgradient Methods for Online Learning and
    Stochastic Optimization <http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf>`_.

    The original paper does not have the ``epsilon`` attribute. It is added here
    in our implementation as also proposed `Per-parameter adaptive learning rate
    methods <http://cs231n.github.io/neural-networks-3/#ada>`_
    for numerical stability to avoid the division by zero error.

    Args:
        learning_rate (float|Variable): The learning rate used to update ``Parameter``.
            It can be a float value or a ``Variable`` with a float type.
        epsilon (float, optional): A small float value for numerical stability.
            The default value is 1e-06.
        parameter_list (list, optional):  List of ``Variable`` names to update to minimize ``loss``. \
            This parameter is required in dygraph mode. \
            The default value is None in static mode, at this time all parameters will be updated.
        regularization (WeightDecayRegularizer, optional): A ``Regularizer``, such as
             :ref:`api_fluid_regularizer_L2DecayRegularizer`. The default value is None.
        name (str, optional): Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name`.
            The default value is None.
        initial_accumulator_value (float, optional): Initial value for moment accumulator.
            The default value is 0.0.

    Examples:
        .. code-block:: python

            import numpy as np
            import paddle.fluid as fluid

            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = fluid.data(name="inp", shape=[2, 2])
            out = fluid.layers.fc(inp, size=3)
            out = fluid.layers.reduce_sum(out)
            optimizer = fluid.optimizer.AdagradOptimizer(learning_rate=0.2)
            optimizer.minimize(out)

            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(fluid.default_startup_program())
            exe.run(
                feed={"inp": np_inp},
                fetch_list=[out.name])
    """
    _moment_acc_str = "moment"

    def __init__(self,
                 learning_rate,
                 epsilon=1.0e-6,
                 parameter_list=None,
                 regularization=None,
                 name=None,
                 initial_accumulator_value=0.0):
        assert learning_rate is not None
        assert epsilon is not None
        super(AdagradOptimizer, self).__init__(
            learning_rate=learning_rate,
            parameter_list=parameter_list,
            regularization=regularization,
            name=name)
        self.type = "adagrad"
        self._epsilon = epsilon
        self.initial_accumulator_value = initial_accumulator_value

    def _create_accumulators(self, block, parameters):
        assert isinstance(block, framework.Block)

        for p in parameters:
            self._add_accumulator(
                self._moment_acc_str,
                p,
                fill_value=self.initial_accumulator_value)

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
            attrs={"epsilon": self._epsilon},
            stop_gradient=True)

        return adagrad_op


class AdamOptimizer(Optimizer):
    """
    The Adam optimizer uses an optimization described at the end
    of section 2 of `Adam paper <https://arxiv.org/abs/1412.6980>`_ ,
    it can dynamically adjusts the learning rate of each parameter using
    the 1st moment estimates and the 2nd moment estimates of the gradient.
    
    The parameter ``param_out`` update rule with gradient ``grad``:

    .. math::

        t & = t + 1

        moment\_1\_out & = {\\beta}_1 * moment\_1 + (1 - {\\beta}_1) * grad

        moment\_2\_out & = {\\beta}_2 * moment\_2 + (1 - {\\beta}_2) * grad * grad

        learning\_rate & = learning\_rate * \\
                          \\frac{\sqrt{1 - {\\beta}_2^t}}{1 - {\\beta}_1^t}

        param\_out & = param - learning\_rate * \\frac{moment\_1}{\sqrt{moment\_2} + \epsilon}

    Related paper: `Adam: A Method for Stochastic Optimization <https://arxiv.org/abs/1412.6980>`_

    Args:
        learning_rate (float|Variable, optional): The learning rate used to update ``Parameter``.
            It can be a float value or a ``Variable`` with a float type. The default value is 0.001.
        beta1 (float|Variable, optional): The exponential decay rate for the 1st moment estimates.
            It should be a float number or a Variable with shape [1] and data type as float32.
            The default value is 0.9.
        beta2 (float|Variable, optional): The exponential decay rate for the 2nd moment estimates.
            It should be a float number or a Variable with shape [1] and data type as float32.
            The default value is 0.999.
        epsilon (float, optional): A small float value for numerical stability.
            The default value is 1e-08.
        parameter_list (list, optional):  List of ``Variable`` names to update to minimize ``loss``. \
            This parameter is required in dygraph mode. \
            The default value is None in static mode, at this time all parameters will be updated.
        regularization (WeightDecayRegularizer, optional): A ``Regularizer``, such as
             :ref:`api_fluid_regularizer_L2DecayRegularizer`. The default value is None.
        name (str, optional): Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name`.
            The default value is None.
        lazy_mode (bool, optional): The official Adam algorithm has two moving-average accumulators.
            The accumulators are updated at every step. Every element of the two moving-average
            is updated in both dense mode and sparse mode. If the size of parameter is very large,
            then the update may be very slow. The lazy mode only update the element that has
            gradient in current mini-batch, so it will be much more faster. But this mode has
            different semantics with the original Adam algorithm and may lead to different result.
            The default value is False.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid

            place = fluid.CPUPlace()
            main = fluid.Program()
            with fluid.program_guard(main):
                x = fluid.data(name='x', shape=[None, 13], dtype='float32')
                y = fluid.data(name='y', shape=[None, 1], dtype='float32')
                y_predict = fluid.layers.fc(input=x, size=1, act=None)
                cost = fluid.layers.square_error_cost(input=y_predict, label=y)
                avg_cost = fluid.layers.mean(cost)

                adam_optimizer = fluid.optimizer.AdamOptimizer(0.01)
                adam_optimizer.minimize(avg_cost)

                fetch_list = [avg_cost]
                train_reader = paddle.batch(
                    paddle.dataset.uci_housing.train(), batch_size=1)
                feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
                exe = fluid.Executor(place)
                exe.run(fluid.default_startup_program())
                for data in train_reader():
                    exe.run(main, feed=feeder.feed(data), fetch_list=fetch_list)

        .. code-block:: python

            # Adam with beta1/beta2 as Variable
            import paddle
            import paddle.fluid as fluid
            import paddle.fluid.layers.learning_rate_scheduler as lr_scheduler

            place = fluid.CPUPlace()
            main = fluid.Program()
            with fluid.program_guard(main):
                x = fluid.data(name='x', shape=[None, 13], dtype='float32')
                y = fluid.data(name='y', shape=[None, 1], dtype='float32')
                y_predict = fluid.layers.fc(input=x, size=1, act=None)
                cost = fluid.layers.square_error_cost(input=y_predict, label=y)
                avg_cost = fluid.layers.mean(cost)

                # define beta decay variable
                def get_decayed_betas(beta1_init, beta2_init, decay_steps, decay_rate):
                    global_step = lr_scheduler._decay_step_counter()

                    beta1 = fluid.layers.create_global_var(
                        shape=[1],
                        value=float(beta1_init),
                        dtype='float32',
                        # set persistable for save checkpoints and resume
                        persistable=True,
                        name="beta1")
                    beta2 = fluid.layers.create_global_var(
                        shape=[1],
                        value=float(beta2_init),
                        dtype='float32',
                        # set persistable for save checkpoints and resume
                        persistable=True,
                        name="beta2")

                    div_res = global_step / decay_steps
                    decayed_beta1 = beta1_init * (decay_rate**div_res)
                    decayed_beta2 = beta2_init * (decay_rate**div_res)
                    fluid.layers.assign(decayed_beta1, beta1)
                    fluid.layers.assign(decayed_beta2, beta2)

                    return beta1, beta2

                beta1, beta2 = get_decayed_betas(0.9, 0.99, 1e5, 0.9)
                adam_optimizer = fluid.optimizer.AdamOptimizer(
                                                    learning_rate=0.01,
                                                    beta1=beta1,
                                                    beta2=beta2)
                adam_optimizer.minimize(avg_cost)

                fetch_list = [avg_cost]
                train_reader = paddle.batch(
                    paddle.dataset.uci_housing.train(), batch_size=1)
                feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
                exe = fluid.Executor(place)
                exe.run(fluid.default_startup_program())
                for data in train_reader():
                    exe.run(main, feed=feeder.feed(data), fetch_list=fetch_list)
    """
    _moment1_acc_str = "moment1"
    _moment2_acc_str = "moment2"
    _beta1_pow_acc_str = "beta1_pow_acc"
    _beta2_pow_acc_str = "beta2_pow_acc"

    def __init__(self,
                 learning_rate=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 parameter_list=None,
                 regularization=None,
                 name=None,
                 lazy_mode=False):
        assert learning_rate is not None
        assert beta1 is not None
        assert beta2 is not None
        assert epsilon is not None
        super(AdamOptimizer, self).__init__(
            learning_rate=learning_rate,
            parameter_list=parameter_list,
            regularization=regularization,
            name=name)
        self.type = "adam"
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._lazy_mode = lazy_mode

    def _create_accumulators(self, block, parameters):
        assert isinstance(block, framework.Block)

        # Create accumulator tensors for first and second moments
        for p in parameters:
            self._add_accumulator(self._moment1_acc_str, p)
            self._add_accumulator(self._moment2_acc_str, p)
            self._add_accumulator(
                name=self._beta1_pow_acc_str,
                param=p,
                fill_value=0.9 if isinstance(self._beta1, Variable) \
                        else self._beta1,
                shape=[1],
                type=core.VarDesc.VarType.LOD_TENSOR, force_cpu=True)
            self._add_accumulator(
                name=self._beta2_pow_acc_str,
                param=p,
                fill_value=0.999 if isinstance(self._beta2, Variable) \
                        else self._beta2,
                shape=[1],
                type=core.VarDesc.VarType.LOD_TENSOR, force_cpu=True)

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, framework.Block)

        moment1 = self._get_accumulator(self._moment1_acc_str,
                                        param_and_grad[0])
        moment2 = self._get_accumulator(self._moment2_acc_str,
                                        param_and_grad[0])
        beta1_pow_acc = self._get_accumulator(self._beta1_pow_acc_str,
                                              param_and_grad[0])
        beta2_pow_acc = self._get_accumulator(self._beta2_pow_acc_str,
                                              param_and_grad[0])

        # create the adam optimize op
        inputs = {
            "Param": [param_and_grad[0]],
            "Grad": [param_and_grad[1]],
            "LearningRate": [self._create_param_lr(param_and_grad)],
            "Moment1": [moment1],
            "Moment2": [moment2],
            "Beta1Pow": [beta1_pow_acc],
            "Beta2Pow": [beta2_pow_acc]
        }
        outputs = {
            "ParamOut": [param_and_grad[0]],
            "Moment1Out": [moment1],
            "Moment2Out": [moment2],
            "Beta1PowOut": [beta1_pow_acc],
            "Beta2PowOut": [beta2_pow_acc],
        }
        attrs = {
            "epsilon": self._epsilon,
            "lazy_mode": self._lazy_mode,
            "min_row_size_to_use_multithread": 1000
        }

        if isinstance(self._beta1, Variable):
            inputs['Beta1Tensor'] = self._beta1
        else:
            attrs['beta1'] = self._beta1
        if isinstance(self._beta2, Variable):
            inputs['Beta2Tensor'] = self._beta2
        else:
            attrs['beta2'] = self._beta2

        if framework.in_dygraph_mode():
            core.ops.adam(inputs, attrs, outputs)
            return None

        adam_op = block.append_op(
            type=self.type,
            inputs=inputs,
            outputs=outputs,
            attrs=attrs,
            stop_gradient=True)

        return adam_op


class AdamaxOptimizer(Optimizer):
    """
    The Adamax optimizer is implemented based on the Adamax Optimization 
    in Section 7 of `Adam paper <https://arxiv.org/abs/1412.6980>`_.
    The Adamax algorithm is a variant of the Adam algorithm based on the infinite norm,
    which makes the learning rate update algorithm more stable and simple.

    The parameter ``param_out`` update rule with gradient ``grad``:

    .. math::

        t & = t + 1

        moment\_out & = {\\beta}_1 * moment + (1 - {\\beta}_1) * grad

        inf\_norm\_out & = max({\\beta}_2 * inf\_norm + \epsilon, |grad|)

        learning\_rate & = \\frac{learning\_rate}{1 - {\\beta}_1^t}

        param\_out & = param - learning\_rate * \\frac{moment\_out}{inf\_norm\_out}

    Related paper: `Adam: A Method for Stochastic Optimization <https://arxiv.org/abs/1412.6980>`_

    The original paper does not have an ``epsilon`` attribute,
    it is added here for numerical stability to prevent the division by 0 error.

    Args:
        learning_rate (float|Variable, optional): The learning rate used to update ``Parameter``.
            It can be a float value or a ``Variable`` with a float type. The default value is 0.001.
        beta1 (float, optional): The exponential decay rate for the 1st moment estimates.
            The default value is 0.9.
        beta2 (float, optional): The exponential decay rate for the 2nd moment estimates.
            The default value is 0.999.
        epsilon (float, optional): A small float value for numerical stability.
            The default value is 1e-08.
        parameter_list (list, optional):  List of ``Variable`` names to update to minimize ``loss``. \
            This parameter is required in dygraph mode. \
            The default value is None in static mode, at this time all parameters will be updated.
        regularization (WeightDecayRegularizer, optional): A ``Regularizer``, such as
             :ref:`api_fluid_regularizer_L2DecayRegularizer`. The default value is None.
        name (str, optional): Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name`.
            The default value is None.

    **Notes**:
        **Currently, AdamaxOptimizer doesn't support sparse parameter optimization.**

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import numpy

          # First create the Executor.
          place = fluid.CPUPlace() # fluid.CUDAPlace(0)
          exe = fluid.Executor(place)

          train_program = fluid.Program()
          startup_program = fluid.Program()
          with fluid.program_guard(train_program, startup_program):
              data = fluid.data(name='X', shape=[None, 1], dtype='float32')
              hidden = fluid.layers.fc(input=data, size=10)
              loss = fluid.layers.mean(hidden)
              adam = fluid.optimizer.AdamaxOptimizer(learning_rate=0.2)
              adam.minimize(loss)

          # Run the startup program once and only once.
          exe.run(startup_program)

          x = numpy.random.random(size=(10, 1)).astype('float32')
          outs = exe.run(program=train_program,
                        feed={'X': x},
                         fetch_list=[loss.name])
    """
    _moment_acc_str = "moment"
    _inf_norm_acc_str = "inf_norm"
    _beta1_pow_acc_str = "beta1_pow_acc"

    def __init__(self,
                 learning_rate=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 parameter_list=None,
                 regularization=None,
                 name=None):
        assert learning_rate is not None
        assert beta1 is not None
        assert beta2 is not None
        assert epsilon is not None
        super(AdamaxOptimizer, self).__init__(
            learning_rate=learning_rate,
            parameter_list=parameter_list,
            regularization=regularization,
            name=name)
        self.type = "adamax"
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

    def _create_accumulators(self, block, parameters):
        # Create accumulator tensors for first moment and infinity norm
        for p in parameters:
            self._add_accumulator(self._moment_acc_str, p)
            self._add_accumulator(self._inf_norm_acc_str, p)
            self._add_accumulator(
                name=self._beta1_pow_acc_str,
                param=p,
                fill_value=self._beta1,
                shape=[1])

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, framework.Block)

        moment = self._get_accumulator(self._moment_acc_str, param_and_grad[0])
        inf_norm = self._get_accumulator(self._inf_norm_acc_str,
                                         param_and_grad[0])
        beta1_pow_acc = self._get_accumulator(self._beta1_pow_acc_str,
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
                "Beta1Pow": beta1_pow_acc
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
            },
            stop_gradient=True)

        return adamax_op

    def _finish_update(self, block, parameters_and_grads):
        """Update Beta1 Power accumulator
        """
        assert isinstance(block, framework.Block)
        for param, grad in parameters_and_grads:
            if grad is None or param.trainable is False:
                continue
            with param.block.program._optimized_guard(
                [param, grad]), name_scope('adamx'):
                beta1_pow_acc = self._get_accumulator(self._beta1_pow_acc_str,
                                                      param)
                block.append_op(
                    type="scale",
                    inputs={"X": beta1_pow_acc},
                    outputs={"Out": beta1_pow_acc},
                    attrs={"scale": self._beta1},
                    stop_gradient=True)


class DpsgdOptimizer(Optimizer):
    """
    We implement the Dpsgd optimizer according to CCS16 paper -
    Deep Learning with Differential Privacy.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import numpy

          # First create the Executor.
          place = fluid.CPUPlace() # fluid.CUDAPlace(0)
          exe = fluid.Executor(place)

          train_program = fluid.Program()
          startup_program = fluid.Program()
          with fluid.program_guard(train_program, startup_program):
              data = fluid.layers.data(name='X', shape=[1], dtype='float32')
              hidden = fluid.layers.fc(input=data, size=10)
              loss = fluid.layers.mean(hidden)
              optimizer = fluid.optimizer.Dpsgd(learning_rate=0.01, clip=10.0, batch_size=16.0, sigma=1.0)
              optimizer.minimize(loss)

          # Run the startup program once and only once.
          exe.run(startup_program)

          x = numpy.random.random(size=(10, 1)).astype('float32')
          outs = exe.run(program=train_program,
                        feed={'X': x},
                         fetch_list=[loss.name])

    Args:
        learning_rate (float|Variable): the learning rate used to update parameters. \
        Can be a float value or a Variable with one float value as data element.
        clip (float): clipping threshold
        batch_size (float): batch size.
        sigma (float): for gaussian noise.
        parameter_list (list, optional):  List of ``Variable`` names to update to minimize ``loss``. \
            This parameter is required in dygraph mode. \
            The default value is None in static mode, at this time all parameters will be updated.
    Notes:
       Currently, DpsgdOptimizer doesn't support sparse parameter optimization.
    """

    def __init__(self,
                 learning_rate=0.001,
                 clip=0.9,
                 batch_size=0.999,
                 sigma=1e-8,
                 parameter_list=None):
        assert learning_rate is not None
        assert clip is not None
        assert batch_size is not None
        assert sigma is not None
        super(DpsgdOptimizer, self).__init__(
            learning_rate=learning_rate, parameter_list=parameter_list)
        self.type = "dpsgd"
        self._clip = clip
        self._batch_size = batch_size
        self._sigma = sigma
        '''
        Note(wangzhongpu):
        This property is only used for debugging, do not need to set it!
        Dpsgd operator use time(NULL) as random seed to generate random number.
        However, during debugging, we need determinated result, so we will set self._seed to a fixed number.
        '''
        self._seed = None

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, framework.Block)

        # create the dpsgd optimize op
        if self._seed == None:
            self._seed = 0

        dpsgd_op = block.append_op(
            type=self.type,
            inputs={
                "Param": param_and_grad[0],
                "Grad": param_and_grad[1],
                "LearningRate": self._create_param_lr(param_and_grad)
            },
            outputs={"ParamOut": param_and_grad[0]},
            attrs={
                "clip": self._clip,
                "batch_size": self._batch_size,
                "sigma": self._sigma,
                "seed": self._seed
            },
            stop_gradient=True)

        return dpsgd_op


class DecayedAdagradOptimizer(Optimizer):
    """
    The Decayed Adagrad optimizer can be seen as an Adagrad algorithm that introduces
    the decay rate to solve the problem of a sharp drop in the learning rate
    during model training when using the AdagradOptimizer.

    The parameter ``param_out`` update rule with gradient ``grad``:

    .. math::

        moment\_out & = decay * moment + (1 - decay) * grad * grad

        param\_out & = param - \\frac{learning\_rate * grad}{\sqrt{moment\_out} + \epsilon}

    Related paper: `Adaptive Subgradient Methods for Online Learning and Stochastic
    Optimization <http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf>`_.

    The original paper does not have an ``epsilon`` attribute. It is added here for numerical
    stability to avoid the division by zero error.

    Args:
        learning_rate (float|Variable): The learning rate used to update ``Parameter``.
            It can be a float value or a ``Variable`` with a float type.
        decay (float, optional): The decay rate. The default value is 0.95.
        epsilon (float, optional): A small float value for numerical stability.
            The default value is 1e-06.
        parameter_list (list, optional):  List of ``Variable`` names to update to minimize ``loss``. \
            This parameter is required in dygraph mode. \
            The default value is None in static mode, at this time all parameters will be updated.
        regularization (WeightDecayRegularizer, optional): A ``Regularizer``, such as
             :ref:`api_fluid_regularizer_L2DecayRegularizer`. The default value is None.
        name (str, optional): Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name`.
            The default value is None.

    **Notes**:
        **Currently, DecayedAdagradOptimizer doesn't support sparse parameter optimization.**

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            x = fluid.data( name='x', shape=[None, 10], dtype='float32' )
            trans = fluid.layers.fc( x, 100 )
            cost = fluid.layers.reduce_mean( trans )
            optimizer = fluid.optimizer.DecayedAdagradOptimizer(learning_rate=0.2)
            optimizer.minimize(cost)
    """
    _moment_acc_str = "moment"

    def __init__(self,
                 learning_rate,
                 decay=0.95,
                 epsilon=1.0e-6,
                 parameter_list=None,
                 regularization=None,
                 name=None):
        assert learning_rate is not None
        assert decay is not None
        assert epsilon is not None

        super(DecayedAdagradOptimizer, self).__init__(
            learning_rate=learning_rate,
            parameter_list=parameter_list,
            regularization=regularization,
            name=name)
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
            attrs={"epsilon": self._epsilon,
                   "decay": self._decay},
            stop_gradient=True)

        return decayed_adagrad_op


class AdadeltaOptimizer(Optimizer):
    """
    **Notes: This API does not support sparse parameter optimization.**

    Adadelta Optimizer. Please refer to this for details:
    `ADADELTA: AN ADAPTIVE LEARNING RATE METHOD <https://arxiv.org/abs/1212.5701>`_.

    The update is done as follows:

    .. math::

        E(g_t^2) &= \\rho * E(g_{t-1}^2) + (1-\\rho) * g^2

        learning\_rate &= \sqrt{ ( E(dx_{t-1}^2) + \\epsilon ) / ( E(g_t^2) + \\epsilon ) }

        E(dx_t^2) &= \\rho * E(dx_{t-1}^2) + (1-\\rho) * (-g*learning\_rate)^2

    Args:
        learning_rate (float|Variable): global learning rate.
        epsilon (float): a small float number for numeric stability. Default 1.0e-6.
        rho (float): a floating point value indicating the decay rate. Default 0.95.
        parameter_list (list, optional):  List of ``Variable`` names to update to minimize ``loss``. \
            This parameter is required in dygraph mode. \
            The default value is None in static mode, at this time all parameters will be updated.
        regularization (WeightDecayRegularizer, optional): A Regularizer, such as
                fluid.regularizer.L2DecayRegularizer. Default None, meaning that there is no
                regularization.
        name (str, optional): The default value is None. Normally there is no need for user
                to set this property. For more information, please refer to
                :ref:`api_guide_Name` .

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            image = fluid.data(name='image', shape=[None, 28], dtype='float32')
            fc = fluid.layers.fc(image, size=10)
            cost = fluid.layers.reduce_mean(fc)
            optimizer = fluid.optimizer.Adadelta(
                learning_rate=0.0003, epsilon=1.0e-6, rho=0.95)

            # optimizer_ops is a list of optimizer operators to update parameters
            # params_grads is a list of (param, param_grad), where param is each
            # parameter and param_grad is the gradient variable of param.
            optimizer_ops, params_grads = optimizer.minimize(cost)
    """

    _avg_squared_grad_acc_str = "_avg_squared_grad"
    _avg_squared_update_acc_str = "_avg_squared_update"

    def __init__(self,
                 learning_rate,
                 epsilon=1.0e-6,
                 rho=0.95,
                 parameter_list=None,
                 regularization=None,
                 name=None):
        if learning_rate is None:
            raise ValueError("learning_rate is not set.")
        if epsilon is None:
            raise ValueError("epsilon is not set.")
        if rho is None:
            raise ValueError("rho is not set.")
        super(AdadeltaOptimizer, self).__init__(
            learning_rate=learning_rate,
            parameter_list=parameter_list,
            regularization=regularization,
            name=name)
        self.type = "adadelta"
        self._epsilon = epsilon
        self._rho = rho

    def _create_accumulators(self, block, parameters):
        if not isinstance(block, framework.Block):
            raise TypeError("block is not instance of framework.Block.")

        for p in parameters:
            self._add_accumulator(self._avg_squared_grad_acc_str, p)
            self._add_accumulator(self._avg_squared_update_acc_str, p)

    def _append_optimize_op(self, block, param_and_grad):
        if not isinstance(block, framework.Block):
            raise TypeError("block is not instance of framework.Block.")

        avg_squared_grad_acc = self._get_accumulator(
            self._avg_squared_grad_acc_str, param_and_grad[0])
        avg_squared_update_acc = self._get_accumulator(
            self._avg_squared_update_acc_str, param_and_grad[0])

        # Create the adadelta optimizer op
        adadelta_op = block.append_op(
            type=self.type,
            inputs={
                "Param": param_and_grad[0],
                "Grad": param_and_grad[1],
                "AvgSquaredGrad": avg_squared_grad_acc,
                "AvgSquaredUpdate": avg_squared_update_acc
            },
            outputs={
                "ParamOut": param_and_grad[0],
                "AvgSquaredGradOut": avg_squared_grad_acc,
                "AvgSquaredUpdateOut": avg_squared_update_acc
            },
            attrs={"epsilon": self._epsilon,
                   "rho": self._rho},
            stop_gradient=True)

        return adadelta_op


class RMSPropOptimizer(Optimizer):
    """
    Root Mean Squared Propagation (RMSProp) is an unpublished, adaptive learning
    rate method. The original slides proposed RMSProp: Slide 29 of
    http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf .

    The original equation is as follows:

    ..  math::

        r(w, t) & = \\rho r(w, t-1) + (1 - \\rho)(\\nabla Q_{i}(w))^2

        w & = w - \\frac{\\eta} {\\sqrt{r(w,t) + \\epsilon}} \\nabla Q_{i}(w)

    The first equation calculates moving average of the squared gradient for
    each weight. Then dividing the gradient by :math:`sqrt{v(w,t)}`.

    In some cases, adding a momentum term :math: `\\beta` is beneficial.
    In our implementation, Nesterov momentum is used:

    ..  math::

        r(w, t) & = \\rho r(w, t-1) + (1 - \\rho)(\\nabla Q_{i}(w))^2

        v(w, t) & = \\beta v(w, t-1) + \\frac{\\eta} {\\sqrt{r(w,t) +
            \\epsilon}} \\nabla Q_{i}(w)

        w & = w - v(w, t)

    if centered is True:

    ..  math::

        r(w, t) & = \\rho r(w, t-1) + (1 - \\rho)(\\nabla Q_{i}(w))^2

        g(w, t) & = \\rho g(w, t-1) + (1 - \\rho)\\nabla Q_{i}(w)

        v(w, t) & = \\beta v(w, t-1) + \\frac{\\eta} {\\sqrt{r(w,t) - (g(w, t))^2 +
            \\epsilon}} \\nabla Q_{i}(w)

        w & = w - v(w, t)

    where, :math:`\\rho` is a hyperparameter and typical values are 0.9, 0.95
    and so on. :math: `beta` is the momentum term. :math: `\\epsilon` is a
    smoothing term to avoid division by zero, usually set somewhere in range
    from 1e-4 to 1e-8.


    Parameters:
        learning_rate(float): Global learning rate.
        rho(float): rho is :math: `\\rho` in equation, default is 0.95.
        epsilon(float): :math: `\\epsilon` in equation is smoothing term to
            avoid division by zero, default is 1e-6.
        momentum(float): :math:`\\beta` in equation is the momentum term,
            default is 0.0.
        centered(bool): If True, gradients are normalized by the estimated variance of
            the gradient; if False, by the uncentered second moment. Setting this to
            True may help with training, but is slightly more expensive in terms of
            computation and memory. Defaults to False.
        parameter_list (list, optional):  List of ``Variable`` names to update to minimize ``loss``. \
            This parameter is required in dygraph mode. \
            The default value is None in static mode, at this time all parameters will be updated.
        regularization: A Regularizer, such as :ref:`api_fluid_regularizer_L2DecayRegularizer`. \
            Optional, default is None.
        name (str, optional): This parameter is used by developers to print debugging information. \
            For details, please refer to :ref:`api_guide_Name`. Default is None.

    Raises:
        ValueError: If learning_rate, rho, epsilon, momentum are None.

    Examples:
          .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            import numpy as np

            place = fluid.CPUPlace()
            main = fluid.Program()
            with fluid.program_guard(main):
                x = fluid.layers.data(name='x', shape=[13], dtype='float32')
                y = fluid.layers.data(name='y', shape=[1], dtype='float32')
                y_predict = fluid.layers.fc(input=x, size=1, act=None)
                cost = fluid.layers.square_error_cost(input=y_predict, label=y)
                avg_cost = fluid.layers.mean(cost)

                rms_optimizer = fluid.optimizer.RMSProp(learning_rate=0.1)
                rms_optimizer.minimize(avg_cost)

                fetch_list = [avg_cost]
                train_reader = paddle.batch(
                    paddle.dataset.uci_housing.train(), batch_size=1)
                feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
                exe = fluid.Executor(place)
                exe.run(fluid.default_startup_program())
                for data in train_reader():
                    exe.run(main, feed=feeder.feed(data), fetch_list=fetch_list)

    """

    _momentum_acc_str = "momentum"
    _mean_square_acc_str = "mean_square"
    _mean_grad_acc_str = "mean_grad"

    def __init__(self,
                 learning_rate,
                 rho=0.95,
                 epsilon=1.0e-6,
                 momentum=0.0,
                 centered=False,
                 parameter_list=None,
                 regularization=None,
                 name=None):
        super(RMSPropOptimizer, self).__init__(
            learning_rate=learning_rate,
            parameter_list=parameter_list,
            regularization=regularization,
            name=name)
        if learning_rate is None:
            raise ValueError("learning_rate is not set.")
        if rho is None:
            raise ValueError("rho is not set.")
        if epsilon is None:
            raise ValueError("epsilon is not set.")
        if momentum is None:
            raise ValueError("momentum is not set.")

        self.type = "rmsprop"
        self._rho = rho
        self._epsilon = epsilon
        self._momentum = momentum
        self._centered = centered

    def _create_accumulators(self, block, parameters):
        if not isinstance(block, framework.Block):
            raise TypeError("block is not instance of framework.Block.")

        for p in parameters:
            self._add_accumulator(self._momentum_acc_str, p)
            self._add_accumulator(self._mean_square_acc_str, p)
            self._add_accumulator(self._mean_grad_acc_str, p)

    def _append_optimize_op(self, block, param_and_grad):
        if not isinstance(block, framework.Block):
            raise TypeError("block is not instance of framework.Block.")

        momentum_acc = self._get_accumulator(self._momentum_acc_str,
                                             param_and_grad[0])
        mean_square_acc = self._get_accumulator(self._mean_square_acc_str,
                                                param_and_grad[0])
        mean_grad_acc = self._get_accumulator(self._mean_grad_acc_str,
                                              param_and_grad[0])
        rmsprop_op = block.append_op(
            type=self.type,
            inputs={
                "Param": param_and_grad[0],
                "Grad": param_and_grad[1],
                "Moment": momentum_acc,
                "MeanSquare": mean_square_acc,
                "MeanGrad": mean_grad_acc,
                "LearningRate": self._create_param_lr(param_and_grad),
            },
            outputs={
                "ParamOut": param_and_grad[0],
                "MomentOut": momentum_acc,
                "MeanSquareOut": mean_square_acc,
                "MeanGradOut": mean_grad_acc
            },
            attrs={
                "epsilon": self._epsilon,
                "decay": self._rho,
                "momentum": self._momentum,
                "centered": self._centered
            },
            stop_gradient=True)

        return rmsprop_op


class FtrlOptimizer(Optimizer):
    """
    FTRL (Follow The Regularized Leader) Optimizer.

    The paper that proposed Follow The Regularized Leader (FTRL):
    (https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf)

    ..  math::

        &new\_accum = squared\_accum + grad^2

        &if (lr\_power == -0.5):

        &\quad  linear\_accum += grad - \\frac{\\sqrt{new\_accum} - \\sqrt{squared\_accum}}{learning\_rate * param}

        &else:

        &\quad   linear\_accum += grad - \\frac{new\_accum^{-lr\_power} - accum^{-lr\_power}}{learning\_rate * param}


        &x = l1 * sign(linear\_accum) - linear\_accum

        &if (lr\_power == -0.5):

        &\quad   y = \\frac{\\sqrt{new\_accum}}{learning\_rate} + (2 * l2)

        &\quad   pre\_shrink = \\frac{x}{y}

        &\quad   param = (abs(linear\_accum) > l1).select(pre\_shrink, 0.0)

        &else:

        &\quad   y = \\frac{new\_accum^{-lr\_power}}{learning\_rate} + (2 * l2)

        &\quad   pre\_shrink = \\frac{x}{y}

        &\quad   param = (abs(linear\_accum) > l1).select(pre\_shrink, 0.0)

        &squared\_accum += grad^2

    Parameters:
        learning_rate (float|Variable): Global learning rate.
        l1 (float): L1 regularization strength, default is 0.0.
        l2 (float): L2 regularization strength, default is 0.0.
        lr_power (float): Learning Rate Power, default is -0.5.
        parameter_list (list, optional):  List of ``Variable`` names to update to minimize ``loss``. \
            This parameter is required in dygraph mode. \
            The default value is None in static mode, at this time all parameters will be updated.
        regularization: A Regularizer, such as :ref:`api_fluid_regularizer_L2DecayRegularizer`. \
            Optional, default is None.
        name (str, optional): This parameter is used by developers to print debugging information. \
            For details, please refer to :ref:`api_guide_Name`. Default is None.

    Raises:
        ValueError: If learning_rate, rho, epsilon, momentum are None.

    Examples:
          .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            import numpy as np

            place = fluid.CPUPlace()
            main = fluid.Program()
            with fluid.program_guard(main):
                x = fluid.layers.data(name='x', shape=[13], dtype='float32')
                y = fluid.layers.data(name='y', shape=[1], dtype='float32')
                y_predict = fluid.layers.fc(input=x, size=1, act=None)
                cost = fluid.layers.square_error_cost(input=y_predict, label=y)
                avg_cost = fluid.layers.mean(cost)

                ftrl_optimizer = fluid.optimizer.Ftrl(learning_rate=0.1)
                ftrl_optimizer.minimize(avg_cost)

                fetch_list = [avg_cost]
                train_reader = paddle.batch(
                    paddle.dataset.uci_housing.train(), batch_size=1)
                feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
                exe = fluid.Executor(place)
                exe.run(fluid.default_startup_program())
                for data in train_reader():
                    exe.run(main, feed=feeder.feed(data), fetch_list=fetch_list)

    NOTE:
       Currently, FtrlOptimizer doesn't support sparse parameter optimization.
    """

    _squared_acc_str = "squared"
    _linear_acc_str = "linear"

    def __init__(self,
                 learning_rate,
                 l1=0.0,
                 l2=0.0,
                 lr_power=-0.5,
                 parameter_list=None,
                 regularization=None,
                 name=None):
        super(FtrlOptimizer, self).__init__(
            learning_rate=learning_rate,
            parameter_list=parameter_list,
            regularization=regularization,
            name=name)
        if learning_rate is None:
            raise ValueError("learning_rate is not set.")

        self.type = "ftrl"
        self._l1 = l1
        self._l2 = l2
        self._lr_power = lr_power

    def _create_accumulators(self, block, parameters):
        if not isinstance(block, framework.Block):
            raise TypeError("block is not instance of framework.Block.")

        for p in parameters:
            self._add_accumulator(self._squared_acc_str, p)
            self._add_accumulator(self._linear_acc_str, p)

    def _append_optimize_op(self, block, param_and_grad):
        if not isinstance(block, framework.Block):
            raise TypeError("block is not instance of framework.Block.")

        squared_acc = self._get_accumulator(self._squared_acc_str,
                                            param_and_grad[0])
        linear_acc = self._get_accumulator(self._linear_acc_str,
                                           param_and_grad[0])
        ftrl_op = block.append_op(
            type=self.type,
            inputs={
                "Param": param_and_grad[0],
                "Grad": param_and_grad[1],
                "SquaredAccumulator": squared_acc,
                "LinearAccumulator": linear_acc,
                "LearningRate": self._create_param_lr(param_and_grad),
            },
            outputs={
                "ParamOut": param_and_grad[0],
                "SquaredAccumOut": squared_acc,
                "LinearAccumOut": linear_acc
            },
            attrs={"l1": self._l1,
                   "l2": self._l1,
                   "lr_power": self._lr_power},
            stop_gradient=True)

        return ftrl_op


class LambOptimizer(AdamOptimizer):
    """
    LAMB (Layer-wise Adaptive Moments optimizer for Batching training) Optimizer.

    LAMB Optimizer is designed to scale up the batch size of training without losing 
    accuracy, which supports adaptive element-wise updating and accurate layer-wise 
    correction. For more information, please refer to `Large Batch Optimization for 
    Deep Learning: Training BERT in 76 minutes <https://arxiv.org/abs/1904.00962>`_ .

    The updating of parameters follows:

    ..  math::

        m_t &= \\beta_1 m_{t - 1}+ (1 - \\beta_1)g_t 

        v_t &= \\beta_2 v_{t - 1}  + (1 - \\beta_2)g_t^2

        r_t &= \\frac{m_t}{\\sqrt{v_t}+\\epsilon}

        w_t &= w_{t-1} -\\eta_t \\frac{\\left \| w_{t-1}\\right \|}{\\left \| r_t + \\lambda w_{t-1}\\right \|} (r_t + \\lambda w_{t-1})


    where :math:`m` is the 1st moment, and :math:`v` the 2nd moment, :math:`\\eta` the 
    learning rate, :math:`\\lambda` the LAMB weight decay rate.

    Args:
        learning_rate (float|Variable, optional): the learning rate used to update parameters. \
            Can be a float value or a Variable with data type float32. Default 0.001.
        lamb_weight_decay (float, optional): The LAMB weight decay rate. Default 0.01.
        beta1 (float, optional): The exponential decay rate for the 1st moment estimates.
            Default 0.9.
        beta2 (float, optional): The exponential decay rate for the 2nd moment estimates.
            Default 0.999.
        epsilon (float, optional): A small float value for numerical stability. Default 1e-6.
        parameter_list (list, optional):  List of ``Variable`` names to update to minimize ``loss``. \
            This parameter is required in dygraph mode. \
            The default value is None in static mode, at this time all parameters will be updated.
        regularization (Regularizer|None): A Regularizer, such as
           fluid.regularizer.L1DecayRegularizer. Default None.
        exclude_from_weight_decay_fn (function|None): Exclude a parameter from weight 
            decay when **exclude_from_weight_decay_fn(parameter)** returns true. 
            Default None.
        name(str|None): For detailed information, please refer to 
            :ref:`api_guide_Name` . Usually name is no need to set and None by default.

    Examples:
        .. code-block:: python
            
            import paddle.fluid as fluid 

            data = fluid.data(name='x', shape=[-1, 5], dtype='float32')
            hidden = fluid.layers.fc(input=data, size=10)
            cost = fluid.layers.mean(hidden)

            def exclude_fn(param):
                return param.name.endswith('.b_0')

            optimizer = fluid.optimizer.Lamb(learning_rate=0.002,
                                             exclude_from_weight_decay_fn=exclude_fn)
            optimizer.minimize(cost)
    """
    _moment1_acc_str = "moment1"
    _moment2_acc_str = "moment2"
    # these two not used in op temporarily
    _beta1_pow_acc_str = "beta1_pow_acc"
    _beta2_pow_acc_str = "beta2_pow_acc"

    def __init__(self,
                 learning_rate=0.001,
                 lamb_weight_decay=0.01,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-6,
                 parameter_list=None,
                 regularization=None,
                 exclude_from_weight_decay_fn=None,
                 name=None):
        assert learning_rate is not None
        assert lamb_weight_decay is not None
        assert beta1 is not None
        assert beta2 is not None
        assert epsilon is not None
        super(LambOptimizer, self).__init__(
            learning_rate=learning_rate,
            parameter_list=parameter_list,
            regularization=regularization,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            name=name)
        self.type = "lamb"
        self._weight_decay = lamb_weight_decay
        self._exclude_from_weight_decay_fn = exclude_from_weight_decay_fn

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, framework.Block)
        block.program._use_lamb = True

        moment1 = self._get_accumulator(self._moment1_acc_str,
                                        param_and_grad[0])
        moment2 = self._get_accumulator(self._moment2_acc_str,
                                        param_and_grad[0])
        beta1_pow_acc = self._get_accumulator(self._beta1_pow_acc_str,
                                              param_and_grad[0])
        beta2_pow_acc = self._get_accumulator(self._beta2_pow_acc_str,
                                              param_and_grad[0])

        if self._exclude_from_weight_decay_fn is not None \
            and self._exclude_from_weight_decay_fn(param_and_grad[0]):
            weight_decay = 0.0
        else:
            weight_decay = self._weight_decay

        # create the lamb optimize op
        lamb_op = block.append_op(
            type=self.type,
            inputs={
                "Param": param_and_grad[0],
                "Grad": param_and_grad[1],
                "LearningRate": self._create_param_lr(param_and_grad),
                "Moment1": moment1,
                "Moment2": moment2,
                "Beta1Pow": beta1_pow_acc,
                "Beta2Pow": beta2_pow_acc
            },
            outputs={
                "ParamOut": param_and_grad[0],
                "Moment1Out": moment1,
                "Moment2Out": moment2
            },
            attrs={
                "beta1": self._beta1,
                "beta2": self._beta2,
                "epsilon": self._epsilon,
                "weight_decay": weight_decay
            },
            stop_gradient=True)

        return lamb_op


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
Dpsgd = DpsgdOptimizer
DecayedAdagrad = DecayedAdagradOptimizer
Adadelta = AdadeltaOptimizer
RMSProp = RMSPropOptimizer
Ftrl = FtrlOptimizer
LarsMomentum = LarsMomentumOptimizer
Lamb = LambOptimizer


class ModelAverage(Optimizer):
    """
    The ModelAverage optimizer accumulates specific continuous historical parameters
    during training. The accumulated historical range can be controlled by the passed
    ``average_window_rate`` argument. The averaged ``Parameter`` are used in the prediction,
    which usually can improve the accuracy of the prediction.

    Accumulate the average of the ``Parameter`` in the sliding window, the result will be saved
    in a temporary variable, can be applied to the current model's ``Parameter`` by calling
    the ``apply()`` method, and the current model ``Parameter`` can be restored by calling
    the ``restore()`` method.

    The window size for calculating the average is determined by ``average_window_rate``,
    ``min_average_window``, ``max_average_window`` and the current ``Parameter`` update times (num_updates).

    When the cumulative times (num_accumulates) is greater than the specific window
    threshold (average_window), the accumulated ``Parameter`` temporary variable is set to 0.0.
    The following example will help to understand the role of these arguments:

    ::

        if num_accumulates >= min_average_window and num_accumulates >= min(max_average_window, num_updates * average_window_rate):
            num_accumulates = 0

    In the above conditional judgment statement, ``num_accumulates`` indicates the current
    accumulated number, which can be abstractly understood as the length of the cumulative window.
    The length of the window must be at least the length set by the ``min_average_window`` argument,
    and cannot exceed the length specified by the ``max_average_window`` argument or
    ``num_updates * average_window_rate``, where ``num_updates`` indicates the current ``Parameter``
    update times, ``average_window_rate`` is a coefficient that calculates the length of the window.

    Args:
        average_window_rate (float): The calculate ratio of the window length relative to ``Parameter`` update times.
        min_average_window (int, optional): the minimum size of average window length. The default value is 10000.
        max_average_window (int, optional): The maximum size of average window length. The default value is 10000.
        regularization (WeightDecayRegularizer, optional): A ``Regularizer``, such as
             :ref:`api_fluid_regularizer_L2DecayRegularizer`. The default value is None.
        name (str, optional): Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name`.
            The default value is None.

    Examples:

      .. code-block:: python

        import paddle.fluid as fluid
        import numpy

        # First create the Executor.
        place = fluid.CPUPlace()  # fluid.CUDAPlace(0)
        exe = fluid.Executor(place)

        train_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(train_program, startup_program):
            # build net
            data = fluid.data(name='X', shape=[None, 1], dtype='float32')
            hidden = fluid.layers.fc(input=data, size=10)
            loss = fluid.layers.mean(hidden)
            optimizer = fluid.optimizer.Momentum(learning_rate=0.2, momentum=0.1)
            optimizer.minimize(loss)

            # build ModelAverage optimizer
            model_average = fluid.optimizer.ModelAverage(0.15,
                                                         min_average_window=10000,
                                                         max_average_window=12500)

            exe.run(startup_program)
            for i in range(12500):
                x = numpy.random.random(size=(10, 1)).astype('float32')
                outs = exe.run(program=train_program,
                               feed={'X': x},
                               fetch_list=[loss.name])

            # apply ModelAverage
            with model_average.apply(exe):
                x = numpy.random.random(size=(10, 1)).astype('float32')
                exe.run(program=train_program,
                        feed={'X': x},
                        fetch_list=[loss.name])
    """

    def __init__(self,
                 average_window_rate,
                 min_average_window=10000,
                 max_average_window=10000,
                 regularization=None,
                 name=None):
        if framework.in_dygraph_mode():
            raise Exception("In dygraph, don't support ModelAverage.")
        super(ModelAverage, self).__init__(
            0.0, regularization=regularization, name=name)
        self.average_window = average_window_rate
        self.min_average_window = min_average_window
        self.max_average_window = max_average_window

        self.params_grads = []
        for param in framework.default_main_program().global_block(
        ).all_parameters():
            if param.do_model_average != False:
                grad = param.block.create_var(
                    name=unique_name.generate_with_ignorable_key(".".join(
                        [param.name, 'tmp'])),
                    dtype=param.dtype,
                    persistable=False,
                    stop_gradient=True)
                self.params_grads.append((param, grad))

        for param, grad in self.params_grads:
            if grad is None:
                continue
            with param.block.program._optimized_guard(
                [param, grad]), name_scope('move_average'):
                self._append_average_accumulate_op(param)

        self.apply_program = Program()
        block = self.apply_program.global_block()
        with program_guard(main_program=self.apply_program):
            for param_grad in self.params_grads:
                self._add_average_apply_op(block, param_grad)

        self.restore_program = Program()
        block = self.restore_program.global_block()
        with program_guard(main_program=self.restore_program):
            for param_grad in self.params_grads:
                self._add_average_restore_op(block, param_grad)

    def _add_average_apply_op(self, block, param_grad):
        param = block._clone_variable(param_grad[0])
        grad = block._clone_variable(param_grad[1])
        sum_1 = block._clone_variable(self._get_accumulator('sum_1', param))
        sum_2 = block._clone_variable(self._get_accumulator('sum_2', param))
        sum_3 = block._clone_variable(self._get_accumulator('sum_3', param))
        num_accumulates = block._clone_variable(
            self._get_accumulator('num_accumulates', param))
        old_num_accumulates = block._clone_variable(
            self._get_accumulator('old_num_accumulates', param))
        num_updates = block._clone_variable(
            self._get_accumulator('num_updates', param))
        # backup param value to grad
        layers.assign(input=param, output=grad)
        # param = (sum_1 + sum_2 + sum_3) / (num_accumulates + old_num_accumulates)
        tmp = layers.sum(x=[num_accumulates, old_num_accumulates])
        sum = layers.sum(x=[sum_1, sum_2, sum_3])
        tmp = layers.cast(
            x=tmp, dtype='float32' if self._dtype == None else self._dtype)
        sum = layers.cast(
            x=sum, dtype='float32' if self._dtype == None else self._dtype)
        ops._elementwise_div(x=sum, y=tmp, out=param)

    def _add_average_restore_op(self, block, param_grad):
        param = block._clone_variable(param_grad[0])
        grad = block._clone_variable(param_grad[1])
        layers.assign(input=grad, output=param)

    def _append_average_accumulate_op(self, param):
        self.helper = LayerHelper("average_accumulate")
        sum_1 = self._add_accumulator('sum_1', param)
        sum_2 = self._add_accumulator('sum_2', param)
        sum_3 = self._add_accumulator('sum_3', param)
        num_accumulates = self._add_accumulator(
            'num_accumulates', param, dtype='int64', shape=[1])
        old_num_accumulates = self._add_accumulator(
            'old_num_accumulates', param, dtype='int64', shape=[1])
        num_updates = self._add_accumulator(
            'num_updates', param, dtype='int64', shape=[1])

        self.helper.append_op(
            type='average_accumulates',
            inputs={
                "param": param,
                "in_sum_1": sum_1,
                "in_sum_2": sum_2,
                "in_sum_3": sum_3,
                "in_num_accumulates": num_accumulates,
                "in_old_num_accumulates": old_num_accumulates,
                "in_num_updates": num_updates
            },
            outputs={
                "out_sum_1": sum_1,
                "out_sum_2": sum_2,
                "out_sum_3": sum_3,
                "out_num_accumulates": num_accumulates,
                "out_old_num_accumulates": old_num_accumulates,
                "out_num_updates": num_updates,
            },
            attrs={
                "average_window": self.average_window,
                "min_average_window": self.min_average_window,
                "max_average_window": self.max_average_window,
            },
            stop_gradient=True)

    @signature_safe_contextmanager
    def apply(self, executor, need_restore=True):
        """
        Apply the average of the cumulative ``Parameter`` to the parameters of the current model.

        Args:
            executor(fluid.Executor): The current network executor.
            need_restore(bool): Restore flag variable, if set to True, the network will restore
                the parameters of the network to the default value, if set to False,
                it will not be restored. The default value is True.

        Examples:

          .. code-block:: python

            import paddle.fluid as fluid
            import numpy

            # First create the Executor.
            place = fluid.CPUPlace()  # fluid.CUDAPlace(0)
            exe = fluid.Executor(place)

            train_program = fluid.Program()
            startup_program = fluid.Program()
            with fluid.program_guard(train_program, startup_program):
                # build net
                data = fluid.data(name='X', shape=[None, 1], dtype='float32')
                hidden = fluid.layers.fc(input=data, size=10)
                loss = fluid.layers.mean(hidden)
                optimizer = fluid.optimizer.Momentum(learning_rate=0.2, momentum=0.1)
                optimizer.minimize(loss)

                # build ModelAverage optimizer
                model_average = fluid.optimizer.ModelAverage(0.15,
                                                            min_average_window=10000,
                                                            max_average_window=12500)

                exe.run(startup_program)
                for i in range(12500):
                    x = numpy.random.random(size=(10, 1)).astype('float32')
                    outs = exe.run(program=train_program,
                                feed={'X': x},
                                fetch_list=[loss.name])

                # apply ModelAverage
                with model_average.apply(exe):
                    x = numpy.random.random(size=(10, 1)).astype('float32')
                    exe.run(program=train_program,
                            feed={'X': x},
                            fetch_list=[loss.name])
        """
        executor.run(self.apply_program)
        try:
            yield
        finally:
            if need_restore:
                self.restore(executor)

    def restore(self, executor):
        """
        Restore ``Parameter`` values of current model.
        
        Args:
            executor(fluid.Executor): The current network executor.

        Examples:

          .. code-block:: python

            import paddle.fluid as fluid
            import numpy

            # First create the Executor.
            place = fluid.CPUPlace()  # fluid.CUDAPlace(0)
            exe = fluid.Executor(place)

            train_program = fluid.Program()
            startup_program = fluid.Program()
            with fluid.program_guard(train_program, startup_program):
                # build net
                data = fluid.data(name='X', shape=[None, 1], dtype='float32')
                hidden = fluid.layers.fc(input=data, size=10)
                loss = fluid.layers.mean(hidden)
                optimizer = fluid.optimizer.Momentum(learning_rate=0.2, momentum=0.1)
                optimizer.minimize(loss)

                # build ModelAverage optimizer
                model_average = fluid.optimizer.ModelAverage(0.15,
                                                            min_average_window=10000,
                                                            max_average_window=12500)

                exe.run(startup_program)
                for i in range(12500):
                    x = numpy.random.random(size=(10, 1)).astype('float32')
                    outs = exe.run(program=train_program,
                                feed={'X': x},
                                fetch_list=[loss.name])

                # apply ModelAverage
                with model_average.apply(exe, False):
                    x = numpy.random.random(size=(10, 1)).astype('float32')
                    exe.run(program=train_program,
                            feed={'X': x},
                            fetch_list=[loss.name])

                # restore Parameters
                model_average.restore(exe)
        """
        executor.run(self.restore_program)


class ExponentialMovingAverage(object):
    """
    Compute the moving average of parameters with exponential decay.
    Given a parameter :math:`\\theta`, its exponential moving average (EMA)
    will be

    ..  math::

        \\text{EMA}_0 & = 0

	\\text{EMA}_t & = \\text{decay} * \\text{EMA}_{t-1} + (1 - \\text{decay}) * \\theta_t

    The average results calculated by **update()** method will be saved in 
    temporary variables which are created and maintained by the object, and can 
    be applied to parameters of current model by calling **apply()** method. And 
    the **restore()** method is used to restore the parameters.

    **Bias correction**. All EMAs are initialized to :math:`0` and hence they will be 
    zero biased, which can be corrected by divided by a factor 
    :math:`(1 - \\text{decay}^t)` , i.e., the actual EMAs applied to parameters 
    when calling **apply()** method would be 

    ..  math::
    
        \\widehat{\\text{EMA}}_t = \\frac{\\text{EMA}_t}{1 - \\text{decay}^t}

    **Decay rate scheduling**. A large decay rate very close to 1 would result 
    in that the averages move very slowly. And a better strategy is to set a 
    relative smaller decay rate in the very beginning. The argument **thres_steps**
    allows users to pass a Variable to schedule the decay rate, in this case, 
    the actual decay rate becomes
     
    ..  math::
    
        \\min(\\text{decay}, \\frac{1 + \\text{thres_steps}}{10 + \\text{thres_steps}})

    Usually **thres_steps** can be the global training steps.


    Args:
	decay (float, optional): The exponential decay rate, usually close to 1, such as 
            0.999, 0.9999, ... . Default 0.999.
        thres_steps (Variable|None): If not `None`, schedule the decay rate. 
            Default None.
        name (str|None): For detailed information, please refer to 
            :ref:`api_guide_Name`. Usually name is no need to set and None by 
            default.


    Examples:

	.. code-block:: python

	    import numpy
	    import paddle
	    import paddle.fluid as fluid

	    data = fluid.data(name='x', shape=[-1, 5], dtype='float32')
	    hidden = fluid.layers.fc(input=data, size=10)
	    cost = fluid.layers.mean(hidden)

	    test_program = fluid.default_main_program().clone(for_test=True)

	    optimizer = fluid.optimizer.Adam(learning_rate=0.001)
	    optimizer.minimize(cost)

	    global_steps = fluid.layers.autoincreased_step_counter()
	    ema = fluid.optimizer.ExponentialMovingAverage(0.999, thres_steps=global_steps)
	    ema.update()

	    place = fluid.CPUPlace()
	    exe = fluid.Executor(place)
	    exe.run(fluid.default_startup_program())

	    for pass_id in range(3):
		for batch_id in range(6):
		    data = numpy.random.random(size=(10, 5)).astype('float32')
		    exe.run(program=fluid.default_main_program(),
			feed={'x': data}, 
			fetch_list=[cost.name])

		# usage 1
		with ema.apply(exe):
		    data = numpy.random.random(size=(10, 5)).astype('float32')
		    exe.run(program=test_program,
			    feed={'x': data}, 
			    fetch_list=[hidden.name])
			    

		 # usage 2
		with ema.apply(exe, need_restore=False):
		    data = numpy.random.random(size=(10, 5)).astype('float32')
		    exe.run(program=test_program,
			    feed={'x': data}, 
			    fetch_list=[hidden.name])
		ema.restore(exe)
    """

    def __init__(self, decay=0.999, thres_steps=None, name=None):
        if framework.in_dygraph_mode():
            raise Exception(
                "In dygraph, don't support ExponentialMovingAverage.")
        self._decay = decay
        self._thres_steps = thres_steps
        self._name = name if name is not None else ''
        self._decay_var = self._get_ema_decay()

        self._step_counter_name = "@EMA_STEP_COUNTER@"
        self._params_tmps = []
        for param in default_main_program().global_block().all_parameters():
            if param.do_model_average != False:
                tmp = param.block.create_var(
                    name=unique_name.generate(".".join(
                        [self._name + param.name, 'ema_tmp'])),
                    dtype=param.dtype,
                    persistable=False,
                    stop_gradient=True)
                self._params_tmps.append((param, tmp))

        self._ema_vars = {}
        for param, tmp in self._params_tmps:
            with param.block.program._optimized_guard(
                [param, tmp]), name_scope('moving_average'):
                self._ema_vars[param.name] = self._create_ema_vars(param)

        self.apply_program = Program()
        block = self.apply_program.global_block()
        with program_guard(main_program=self.apply_program):
            decay_pow, global_step = self._get_decay_pow(block)
            for param, tmp in self._params_tmps:
                param = block._clone_variable(param)
                tmp = block._clone_variable(tmp)
                ema = block._clone_variable(self._ema_vars[param.name])
                layers.assign(input=param, output=tmp)
                # bias correction
                with layers.control_flow.Switch() as switch:
                    with switch.case(global_step > 0):
                        layers.assign(output=ema, input=ema / (1.0 - decay_pow))
                layers.assign(input=ema, output=param)

        self.restore_program = Program()
        block = self.restore_program.global_block()
        with program_guard(main_program=self.restore_program):
            for param, tmp in self._params_tmps:
                tmp = block._clone_variable(tmp)
                param = block._clone_variable(param)
                layers.assign(input=tmp, output=param)

    def _get_ema_decay(self):
        with default_main_program()._lr_schedule_guard():
            decay_var = layers.tensor.create_global_var(
                shape=[1],
                value=self._decay,
                dtype='float32',
                persistable=True,
                name="scheduled_ema_decay_rate")

            if self._thres_steps is not None:
                decay_t = (self._thres_steps + 1.0) / (self._thres_steps + 10.0)
                with layers.control_flow.Switch() as switch:
                    with switch.case(decay_t < self._decay):
                        layers.tensor.assign(decay_t, decay_var)
                    with switch.default():
                        layers.tensor.assign(
                            np.array(
                                [self._decay], dtype=np.float32),
                            decay_var)
        return decay_var

    def _get_decay_pow(self, block):
        global_step = layers.create_global_var(
            name=self._step_counter_name,
            shape=[1],
            value=0,
            dtype='int64',
            persistable=True)
        global_step = layers.cast(global_step, "float32")
        decay_var = block._clone_variable(self._decay_var)
        decay_pow_acc = layers.elementwise_pow(decay_var, global_step)
        return decay_pow_acc, global_step

    def _create_ema_vars(self, param):
        param_ema = layers.create_global_var(
            name=unique_name.generate(self._name + param.name + '_ema'),
            shape=param.shape,
            value=0.0,
            dtype=param.dtype,
            persistable=True)

        return param_ema

    def update(self):
        """ 
        Update Exponential Moving Average. Should only call this method in 
        train program.
        """
        global_step = layers.autoincreased_step_counter(
            counter_name=self._step_counter_name)
        param_master_emas = []
        for param, tmp in self._params_tmps:
            with param.block.program._optimized_guard(
                [param, tmp]), name_scope('moving_average'):
                param_ema = self._ema_vars[param.name]
                if param.name + '.master' in self._ema_vars:
                    master_ema = self._ema_vars[param.name + '.master']
                    param_master_emas.append([param_ema, master_ema])
                else:
                    ema_t = param_ema * self._decay_var + param * (
                        1 - self._decay_var)
                    layers.assign(input=ema_t, output=param_ema)

        # for fp16 params
        for param_ema, master_ema in param_master_emas:
            default_main_program().global_block().append_op(
                type="cast",
                inputs={"X": master_ema},
                outputs={"Out": param_ema},
                attrs={
                    "in_dtype": master_ema.dtype,
                    "out_dtype": param_ema.dtype
                })

    @signature_safe_contextmanager
    def apply(self, executor, need_restore=True):
        """
        Apply moving average to parameters for evaluation.
        
        Args:
            executor (Executor): The Executor to execute applying.
            need_restore (bool, optional): Whether to restore parameters after 
                applying. Default True.
        """
        executor.run(self.apply_program)
        try:
            yield
        finally:
            if need_restore:
                self.restore(executor)

    def restore(self, executor):
        """Restore parameters.
        
        Args:
            executor (Executor): The Executor to execute restoring.
        """
        executor.run(self.restore_program)


class PipelineOptimizer(object):
    """
    Pipeline Optimizer

    Train with pipeline mode. The program will be split by cut_list. 

    If the len of cut_list is k, then the whole program (including \
    backward part) will be split to 2*k-1 sections. 
    
    So the length of place_list and concurrency_list must be also 2*k-1.

    Note: Though the asynchronous mode is applied in pipeline training to speed up, \
    the final performance depends on the training progress of each pipeline heavily.

    And we will try the synchronous mode in the future.

    Args:
        optimizer (Optimizer): The based optimizer, such as SGD.
        cut_list (list of Variable list): The cut variable of the main_program.
        place_list (list of Place): The place where the section will run on.
        concurrency_list (list of int): The concurrency degree.
        queue_size (int): Each section will consume scopes from its in-scope queue 
                        and produce scopes to out-scope queue. And this parameter 
                        specify the scope queue size. [Optional. Default: 30].
        sync_steps (int): The synchronization steps between different cards. [Optional. Default: 1].
        start_cpu_core_id (int): specify the first cpu core id. [Optional. Default:0].

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle.fluid.layers as layers

            x = fluid.layers.data(name='x', shape=[1], dtype='int64', lod_level=0)
            y = fluid.layers.data(name='y', shape=[1], dtype='int64', lod_level=0)
            emb_x = layers.embedding(input=x, param_attr=fluid.ParamAttr(name="embx"), size=[10,2], is_sparse=False)
            emb_y = layers.embedding(input=y, param_attr=fluid.ParamAttr(name="emby",learning_rate=0.9), size=[10,2], is_sparse=False)
            concat = layers.concat([emb_x, emb_y], axis=1)
            fc = layers.fc(input=concat, name="fc", size=1, num_flatten_dims=1, bias_attr=False)
            loss = layers.reduce_mean(fc)
            optimizer = fluid.optimizer.SGD(learning_rate=0.5)
            optimizer = fluid.optimizer.PipelineOptimizer(optimizer,
                    cut_list=[[emb_x, emb_y], [loss]],
                    place_list=[fluid.CPUPlace(), fluid.CUDAPlace(0), fluid.CPUPlace()],
                    concurrency_list=[1, 1, 4],
                    queue_size=2,
                    sync_steps=1,
                    )
            optimizer.minimize(loss)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            filelist = [] # you should set your own filelist, e.g. filelist = ["dataA.txt"]
            dataset = fluid.DatasetFactory().create_dataset("FileInstantDataset")
            dataset.set_use_var([x,y])
            dataset.set_batch_size(batch_size)
            dataset.set_filelist(filelist)
            exe.train_from_dataset(
                        fluid.default_main_program(),
                        dataset,
                        thread=2,
                        debug=False,
                        fetch_list=[],
                        fetch_info=[],
                        print_period=1)
    """

    def __init__(self,
                 optimizer,
                 cut_list=None,
                 place_list=None,
                 concurrency_list=None,
                 queue_size=30,
                 sync_steps=1,
                 start_cpu_core_id=0):
        if framework.in_dygraph_mode():
            raise Exception("In dygraph, don't support PipelineOptimizer.")
        # TODO: check properties
        self._optimizer = optimizer
        self._cut_list = cut_list
        self._place_list = place_list
        self._concurrency_list = concurrency_list
        self._queue_size = queue_size
        self._sync_steps = sync_steps
        self._start_cpu_core_id = start_cpu_core_id

    def _create_vars(self, block, main_program):
        used_var_set = set()
        for op_idx in range(block.desc.op_size()):
            op_desc = block.desc.op(op_idx)
            vars = op_desc.input_arg_names() + op_desc.output_arg_names()
            for var in vars:
                if var in used_var_set:
                    continue
                used_var_set.add(var)
                source_var = main_program.block(0).var(str(var))
                block._clone_variable(source_var, False)

    def _extract_section_opt_ops(self, ops, cut_point_name):
        """
        Extract opt ops in the given section
        """
        output_names = set(cut_point_name)
        relevant_op_flags = [True] * len(ops)
        for i, op in reversed(list(enumerate(ops))):
            if _some_in_set_(op.desc.output_arg_names(), output_names):
                for name in op.desc.input_arg_names():
                    output_names.add(name)
            else:
                relevant_op_flags[i] = False

        op_path = [ops[i] for i in range(len(ops)) if relevant_op_flags[i]]
        return op_path

    def _find_input_output(self, ops, name, is_forward=True):
        """
        Find the inputs or outputs of a section
        """
        all_set = set()
        part_set = set()
        for op in ops:
            if is_forward:
                part_set.update(op.desc.output_arg_names())
            else:
                part_set.update(op.desc.input_arg_names())
            all_set.update(op.desc.output_arg_names())
            all_set.update(op.desc.input_arg_names())
        return all_set - part_set

    def _find_persistable_vars(self, ops, whole_parameters):
        """
        find the persistable input vars in current section
        """
        res = set()
        for op in ops:
            vars = op.desc.input_arg_names()
            for var in vars:
                if var in whole_parameters:
                    res.add(var)
        return res

    def _is_opt_role_op(self, op):
        op_maker = core.op_proto_and_checker_maker
        optimize_role = core.op_proto_and_checker_maker.OpRole.Optimize
        if op_maker.kOpRoleAttrName() in op.attr_names and \
                int(op.all_attrs()[op_maker.kOpRoleAttrName()]) & int(optimize_role) != 0:
            return True
        return False

    def _is_lr_role_op(self, op):
        op_maker = core.op_proto_and_checker_maker
        optimize_role = core.op_proto_and_checker_maker.OpRole.LRSched
        if op_maker.kOpRoleAttrName() in op.attr_names and \
                int(op.all_attrs()[op_maker.kOpRoleAttrName()]) == int(optimize_role):
            return True
        return False

    def _extract_section_ops(self, ops, cut_point_name):
        """
        Extract ops in the given section 
        """
        output_names = set(cut_point_name)
        relevant_op_flags = [True] * len(ops)
        for i, op in reversed(list(enumerate(ops))):
            if not self._is_opt_role_op(op) and _some_in_set_(
                    op.desc.output_arg_names(), output_names):
                for name in op.desc.input_arg_names():
                    output_names.add(name)
            elif op.desc.type() == "print" and op.desc.input_arg_names()[
                    0] in output_names:
                continue
            else:
                relevant_op_flags[i] = False

        op_path = [ops[i] for i in range(len(ops)) if relevant_op_flags[i]]
        return op_path

    def _find_section_opt(self, ops, params):
        res = self._extract_section_opt_ops(ops, params)
        return res

    def _split_program(self, main_program, cut_list):
        programs = []
        block = main_program.block(0)
        whole_parameters = [e.name for e in block.all_parameters()]
        cut_var_names = []
        cut_len = len(cut_list)
        sec_params = []
        for i, cut_vars in enumerate(cut_list[:-1]):
            cut_var_names.append([cut_var.name for cut_var in cut_vars])
        for i, cut_vars in reversed(list(enumerate(cut_list[:-1]))):
            cut_var_names.append(
                [_append_grad_suffix_(cut_var.name) for cut_var in cut_vars])
            if i == 0:
                cut_var_names[-1] += [var.name for var in cut_list[-1]]
        ops = block.ops[:]
        for i, cut_vars in enumerate(cut_var_names):
            program = {
                "program": Program(),
                "input_set": set(),
                "output_set": set()
            }
            cur_ops = self._extract_section_ops(ops, cut_vars)
            if i == 0:
                for op in ops:
                    if self._is_lr_role_op(op):
                        cur_ops.append(op)
            #prevent inplace in/out
            program["input_set"].update(
                self._find_input_output(
                    cur_ops, [], is_forward=True))
            for e in cur_ops:
                ops.remove(e)

            if i < cut_len:
                sec_params.append(
                    self._find_persistable_vars(cur_ops, whole_parameters))
            if i >= cut_len - 1:
                opt_ops = self._find_section_opt(
                    ops, sec_params[2 * cut_len - 2 - i])

                for e in opt_ops:
                    ops.remove(e)
                cur_ops += opt_ops

            op_descs = [op.desc for op in cur_ops]
            for op_desc in op_descs:
                ap_op = program["program"].block(0).desc.append_op()
                ap_op.copy_from(op_desc)
            program["input_set"].update(
                self._find_input_output(
                    cur_ops, cut_vars, is_forward=True))
            program["input_set"].update(sec_params[min(i, 2 * cut_len - 2 - i)])
            program["output_set"].update(
                self._find_input_output(
                    cur_ops, cut_vars, is_forward=False))
            programs.append(program)
        program = {
            "program": Program(),
            "input_set": set(),
            "output_set": set()
        }
        op_descs = [op.desc for op in ops]
        for op_desc in op_descs:
            ap_op = program["program"].block(0).desc.append_op()
            ap_op.copy_from(op_desc)
        program["input_set"].update(
            [cut_var.name + "@GRAD" for cut_var in cut_list[0]])
        program["input_set"].update(
            self._find_input_output(
                ops, [], is_forward=True))
        program["input_set"].update(sec_params[0])
        programs.append(program)
        inputs = set()
        for program in reversed(list(programs)):
            output_list = list(program["output_set"])
            for output in output_list:
                if output not in inputs:
                    program["output_set"].remove(output)
            inputs.update(program["input_set"])
        return programs

    def minimize(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None):
        self._optimizer.minimize(loss, startup_program, parameter_list,
                                 no_grad_set)
        program = loss.block.program
        if len(self._cut_list) == 0:
            program_list = []
            ptmp = {"program": program, "input_set": set(), "output_set": set()}
            program_list.append(ptmp)
        else:
            program_list = self._split_program(program, self._cut_list)
            for p in program_list:
                self._create_vars(p["program"].block(0), program)
        whole_parameters = [e.name for e in program.block(0).all_parameters()]
        param_need_sync = []
        for i, section_p in enumerate(program_list):
            if not isinstance(self._place_list[i], core.CUDAPlace):
                continue
            section_var = [e for e in section_p["program"].block(0).vars]
            for p in section_var:
                if p in whole_parameters:
                    param_need_sync.append(p)
        program._pipeline_opt = {
            "trainer": "PipelineTrainer",
            "device_worker": "Section",
            "section_program_list": program_list,
            "place_list": self._place_list,
            "concurrency_list": self._concurrency_list,
            "queue_size": self._queue_size,
            "start_cpu_core_id": self._start_cpu_core_id,
            "sync_steps": self._sync_steps,
            "param_need_sync": param_need_sync
        }


class RecomputeOptimizer(Optimizer):
    """
    Recompute Optimizer Wrapper

    Normally, a training step contains three sub-steps: first, run forward
    Operators to calculate the loss; second, run backward Operators to 
    calculate gradient of the parameters; third, apply optimization method
    to update the value of the parameters.

    In the forward computation process, all variables that are needed by 
    backward computation process will be kept in memory, which occupy a great
    amount of memory when the network becomes very deep.

    Recompute split the network to k segments. In each segment, It will 
    recompute the forward Operators, before running backward operators. It is
    very helpful for saving memory.
 
    The Variables that separate a network to segments are called as checkpoints,
    and users should set it manually. The usage is very simple:

    Args:
        optimizer (Optimizer): The optimizer that is applied to parameters.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np
            def gen_data():
                return {"x": np.random.random(size=(32, 32)).astype('float32'),
                "y": np.random.randint(2, size=(32, 1)).astype('int64')}
            def mlp(input_x, input_y, hid_dim=128, label_dim=2):
                print(input_x)
                fc_1 = fluid.layers.fc(input=input_x, size=hid_dim)
                prediction = fluid.layers.fc(input=[fc_1], size=label_dim, act='softmax')
                cost = fluid.layers.cross_entropy(input=prediction, label=input_y)
                sum_cost = fluid.layers.reduce_mean(cost)
                return sum_cost, fc_1, prediction
            input_x = fluid.layers.data(name="x", shape=[32], dtype='float32')
            input_y = fluid.layers.data(name="y", shape=[1], dtype='int64')
            cost, fc_1, pred = mlp(input_x, input_y)

            sgd = fluid.optimizer.Adam(learning_rate=0.01)
            sgd = fluid.optimizer.RecomputeOptimizer(sgd)
            sgd._set_checkpoints([fc_1, pred])
            sgd.minimize(cost)

            print("Finished optimize")
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            step = 10

            for i in range(step):
                cost_val = exe.run(feed=gen_data(),
                       program=fluid.default_main_program(),
                       fetch_list=[cost.name])
                print("step=%d cost=%f" % (i, cost_val[0]))

    """

    def __init__(self, optimizer):
        if framework.in_dygraph_mode():
            raise Exception("In dygraph, don't support RecomputeOptimizer.")
        self._optimizer = optimizer
        self._checkpoints = None

    def _set_checkpoints(self, checkpoints):
        self._checkpoints = checkpoints

    def load(self, stat_dict):
        """
        load function is not supported by Recompute Optimizer for now.
        :return: None

        Args:
            stat_dict: the dict load by load_persistable method

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                import paddle.compat as cpt
                
                def mlp(input_x, input_y, hid_dim=128, label_dim=2):
                    fc_1 = fluid.layers.fc(input=input_x, size=hid_dim)
                    prediction = fluid.layers.fc(input=[fc_1], size=label_dim, act='softmax')
                    cost = fluid.layers.cross_entropy(input=prediction, label=input_y)
                    sum_cost = fluid.layers.reduce_mean(cost)
                    return sum_cost, fc_1, prediction
                
                input_x = fluid.layers.data(name="x", shape=[32], dtype='float32')
                input_y = fluid.layers.data(name="y", shape=[1], dtype='int64')
                cost, fc_1, pred = mlp(input_x, input_y)
                print("Finished FF")
                
                sgd = fluid.optimizer.Adam(learning_rate=0.01)
                sgd = fluid.optimizer.RecomputeOptimizer(sgd)
                sgd._set_checkpoints([fc_1, pred])
                try:
                    stat_dict = {}
                    sgd.load(stat_dict)
                except NotImplementedError as e:
                    print(cpt.get_exception_message(e))
        """
        raise NotImplementedError(
            "load function is not supported by Recompute Optimizer for now")

    def apply_gradients(self, params_grads):
        """
        call apply_gradients function of self._optimizer.

        Args:
            params_grads (list): list of (param, grad) pair to do optimization.

        Returns:
            list: A list of operators appended to the current program.

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                import paddle.fluid.framework as framework

                def mlp(input_x, input_y, hid_dim=128, label_dim=2):
                    fc_1 = fluid.layers.fc(input=input_x, size=hid_dim)
                    prediction = fluid.layers.fc(input=[fc_1], size=label_dim, act='softmax')
                    cost = fluid.layers.cross_entropy(input=prediction, label=input_y)
                    sum_cost = fluid.layers.reduce_mean(cost)
                    return sum_cost, fc_1, prediction


                input_x = fluid.layers.data(name="x", shape=[32], dtype='float32')
                input_y = fluid.layers.data(name="y", shape=[1], dtype='int64')
                cost, fc_1, pred = mlp(input_x, input_y)
                print("Finished FF")

                sgd = fluid.optimizer.Adam(learning_rate=0.01)
                sgd = fluid.optimizer.RecomputeOptimizer(sgd)
                params_grads = sgd.backward(
                    cost,
                    startup_program=None,
                    parameter_list=None,
                    no_grad_set=None,
                    checkpoints=[fc_1, pred])

                program = cost.block.program
                with framework.program_guard(program, None):
                    optimize_ops = sgd.apply_gradients(params_grads)

                print("Finished apply gradients")
        """

        return self._optimizer.apply_gradients(params_grads=params_grads)

    def backward(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None,
                 callbacks=None,
                 checkpoints=None):
        """
        call append_backward with checkpoints.

        Args:
            loss (Variable): loss variable to run optimizations.
            startup_program (Program): startup_program for initializing parameters
                in `parameter_list`.
            parameter_list (list): list of Variables or Variable.names to update.
            no_grad_set (set|None): set of Variables or Variables.names should be ignored.
            callbacks (list|None): list of callables to run when appending backward
                operator for one parameter.
            checkpoints (list): list of Variables as checkpoints

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
    
                def mlp(input_x, input_y, hid_dim=128, label_dim=2):
                    fc_1 = fluid.layers.fc(input=input_x, size=hid_dim)
                    prediction = fluid.layers.fc(input=[fc_1], size=label_dim, act='softmax')
                    cost = fluid.layers.cross_entropy(input=prediction, label=input_y)
                    sum_cost = fluid.layers.reduce_mean(cost)
                    return sum_cost, fc_1, prediction
    
    
                input_x = fluid.layers.data(name="x", shape=[32], dtype='float32')
                input_y = fluid.layers.data(name="y", shape=[1], dtype='int64')
                cost, fc_1, pred = mlp(input_x, input_y)
                print("Finished FF")
    
                sgd = fluid.optimizer.Adam(learning_rate=0.01)
                sgd = fluid.optimizer.RecomputeOptimizer(sgd)
                params_grads = sgd.backward(
                    cost,
                    startup_program=None,
                    parameter_list=None,
                    no_grad_set=None,
                    checkpoints=[fc_1, pred])
                print("Finished backward")
        """

        if framework.in_dygraph_mode():
            raise NotImplementedError(
                "DyGraph current does not support recompute")

        self._dtype = loss.dtype
        program = loss.block.program
        with program_guard(program, startup_program):
            params_grads = append_backward(
                loss,
                parameter_list,
                no_grad_set,
                checkpoints=self._checkpoints)
        return params_grads

    def apply_optimize(self, loss, startup_program, params_grads):
        """
        call the apply_optimize function of self._optimizer

        Args:
            loss (Variable): loss variable to run optimizations.
            startup_program (Program): startup_program for initializing parameters
                in `parameter_list`.
            params_grads (list): list of (param, grad) pair to do optimization.

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                
                def mlp(input_x, input_y, hid_dim=128, label_dim=2):
                    fc_1 = fluid.layers.fc(input=input_x, size=hid_dim)
                    prediction = fluid.layers.fc(input=[fc_1], size=label_dim, act='softmax')
                    cost = fluid.layers.cross_entropy(input=prediction, label=input_y)
                    sum_cost = fluid.layers.reduce_mean(cost)
                    return sum_cost, fc_1, prediction                
                
                input_x = fluid.layers.data(name="x", shape=[32], dtype='float32')
                input_y = fluid.layers.data(name="y", shape=[1], dtype='int64')
                cost, fc_1, pred = mlp(input_x, input_y)
                print("Finished FF")
                
                sgd = fluid.optimizer.Adam(learning_rate=0.01)
                sgd = fluid.optimizer.RecomputeOptimizer(sgd)
                params_grads = sgd.backward(
                    cost,
                    startup_program=None,
                    parameter_list=None,
                    no_grad_set=None,
                    checkpoints=[fc_1, pred])
                
                optimize_ops = sgd.apply_optimize(
                    cost, startup_program=None, params_grads=params_grads)
                
                print("Finished apply_optimize")

        """

        return self._optimizer.apply_optimize(
            loss, startup_program=startup_program, params_grads=params_grads)

    def minimize(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None,
                 grad_clip=None):

        assert (isinstance(loss, Variable)), "The loss should be an Variable."
        assert (self._checkpoints is not None
                ), "You should call _set_checkpoints first"
        if framework.in_dygraph_mode():
            raise NotImplementedError(
                "DyGraph current does not support recompute")

        params_grads = self.backward(
            loss,
            startup_program=startup_program,
            parameter_list=parameter_list,
            no_grad_set=no_grad_set,
            checkpoints=self._checkpoints)

        if grad_clip:
            # TODO(guru4elephant): should add grad_clip for static graph
            pass

        optimize_ops = self.apply_optimize(
            loss, startup_program=startup_program, params_grads=params_grads)

        return optimize_ops, params_grads


class LookaheadOptimizer(object):
    """
    This implements the Lookahead optimizer of the
    paper : https://arxiv.org/abs/1907.08610.

    Lookahead keeps two sets of params: the fast_params and
    the slow_params. inner_optimizer update fast_params every 
    training step. Lookahead updates the slow_params and fast_params 
    every k training steps as follows:

    .. math::
        
        slow\_param_t &= slow\_param_{t-1} + \\alpha * (fast\_param_{t-1} - slow\_param_{t-1})
	
	fast\_param_t &=  slow\_param_t

    Args:
        inner_optimizer (Optimizer): The optimizer that update fast params step by step. 
        alpha (float): The learning rate of Lookahead.
        k (int): The slow params is updated every k steps.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            import numpy as np

	    x = fluid.layers.data(name='x', shape=[2], dtype='float32')
	    label = fluid.layers.data(name="label", shape=[1], dtype="int64")
	    y = fluid.layers.fc(input=[x], size=2, act="softmax")
	    loss = fluid.layers.cross_entropy(input=y, label=label)
	    loss = fluid.layers.mean(x=loss)
	    sgd = fluid.optimizer.SGD(learning_rate=0.01)
	    optimizer = fluid.optimizer.LookaheadOptimizer(sgd,
                                            alpha=0.5,
                                            k=5)
	    optimizer.minimize(loss)
	    main_program = fluid.default_main_program()
	    place = fluid.CPUPlace()
	    exe = fluid.Executor(place)
	    exe.run(fluid.default_startup_program())

	    feeder = fluid.DataFeeder(feed_list=[x, label], place=place)

	    step = 0
            while(step < 10):
                step += 1
		exe.run(fluid.default_main_program(),
            	feed=feeder.feed(batch_data))

    """

    def __init__(self, inner_optimizer, alpha=0.5, k=5):

        if framework.in_dygraph_mode():
            raise Exception("In dygraph, don't support LookaheadOptimizer.")
        assert (inner_optimizer is not None), "inner optimizer can not be None"
        assert (
            0.0 <= alpha <= 1.0
        ), "alpha should be larger or equal to 0.0, and less or equal than 1.0"
        assert (isinstance(k, int) and k > 0), "k should be a positive integer"

        self.inner_optimizer = inner_optimizer
        self.alpha = alpha
        self.k = k
        self.type = "lookahead"

    def minimize(self, loss, startup_program=None):

        # Apply inner optimizer to the main_program
        mini_out = self.inner_optimizer.minimize(
            loss, startup_program=startup_program)

        # Get startup_program and main_program
        if startup_program is None:
            startup_program = default_startup_program()
        main_block = loss.block

        # add some vars to the main_program
        params = [param.name for param in main_block.all_parameters()]
        param_to_slow = {}
        for param in params:
            fast_var = main_block.var(param)
            assert (fast_var is not None)
            slow_var = main_block.create_var(
                name=param + "@SLOW",
                shape=fast_var.shape,
                dtype=fast_var.dtype,
                persistable=True)
            param_to_slow[param] = slow_var

        # add some vars to the startup_program
        startup_block = startup_program.global_block()
        for param in params:
            fast_var = startup_block.var(param)
            assert (fast_var is not None)
            slow_var = startup_block.create_var(
                name=param + "@SLOW",
                shape=fast_var.shape,
                dtype=fast_var.dtype,
                persistable=True)

            startup_block.append_op(
                type="assign",
                inputs={"X": fast_var},
                outputs={"Out": slow_var})

        # Add Var k to main prog and startup prog
        k = layers.create_global_var(
            name="lookahead_k",
            shape=[1],
            value=int(self.k),
            dtype='int32',
            persistable=True)

        # Add Var alpha to main prog and startup prog
        alpha = layers.create_global_var(
            name="lookahead_alpha",
            shape=[1],
            value=float(self.alpha),
            dtype='float32',
            persistable=True)

        # Add Var step
        step = layers.create_global_var(
            name="lookahead_step",
            shape=[1],
            value=int(0),
            dtype='int32',
            persistable=True)
        layers.increment(x=step, value=1.0, in_place=True)

        # lookahead
        zero_var = layers.fill_constant(shape=[1], dtype='float32', value=0.0)

        one_var = layers.fill_constant(shape=[1], dtype='float32', value=1.0)

        mod = layers.elementwise_mod(step, k)
        with layers.control_flow.Switch() as switch:
            with switch.case(mod == zero_var):
                for param_name in params:
                    fast_var = main_block.var(param_name)
                    slow_var = param_to_slow[param_name]
                    tmp_var = layers.elementwise_add(
                        layers.elementwise_mul(fast_var, alpha),
                        layers.elementwise_mul(
                            slow_var, layers.elementwise_sub(one_var, alpha)))
                    layers.assign(input=tmp_var, output=slow_var)
                    layers.assign(input=tmp_var, output=fast_var)
            with switch.default():
                pass
        return mini_out
