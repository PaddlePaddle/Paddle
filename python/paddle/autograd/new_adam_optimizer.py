# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import six
import logging
from collections import defaultdict

import paddle
from paddle.fluid.distribute_lookup_table import find_distributed_lookup_table
from paddle.fluid.framework import Program, Variable, Parameter, name_scope, default_main_program, default_startup_program, device_guard

from paddle.fluid import framework
from paddle.fluid import layers
from paddle.fluid import unique_name
from paddle.fluid.backward import append_backward, _some_in_set_, _append_grad_suffix_, _get_no_grad_set_name
from paddle.fluid.clip import GradientClipBase, GradientClipByNorm, error_clip_callback, append_gradient_clip_ops, ClipGradByGlobalNorm
from paddle.fluid.framework import program_guard
from paddle.fluid.initializer import Constant
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.layers import ops
from paddle.fluid.dygraph import base as imperative_base
from paddle.fluid.dygraph import no_grad
from paddle.fluid.dygraph.learning_rate_scheduler import LearningRateDecay, _LearningRateEpochDecay
from paddle.fluid import core
from paddle.fluid.layers import tensor
from functools import reduce
from functools import cmp_to_key
from paddle.fluid.wrapped_decorator import signature_safe_contextmanager
from paddle import compat as cpt
import warnings
from paddle import _C_ops
from paddle.fluid.framework import _in_legacy_dygraph, in_dygraph_mode
from .primx import orig2prim, prim2orig, Transform


@framework.static_only
def append_backward_new(loss_list,
                        parameter_list=None,
                        no_grad_set=None,
                        callbacks=None,
                        checkpoints=None,
                        distop_context=None):
    program = default_main_program()
    assert program.num_blocks == 1, "The append_backward_new interface is designed to process only one block."
    block = program.current_block()

    orig2prim(block, update_var_list=loss_list)
    ad = Transform(block)
    if parameter_list is None:
        parameter_list = program.global_block().all_parameters()
    param_dot, loss_dot = ad.linearize(parameter_list, loss_list)
    loss_bar, param_bar = ad.transpose(loss_dot, param_dot)

    # remove param_dot and their constructor ops
    op_indexes = []
    for var in param_dot:
        if var is not None:
            op_index = block.ops.index(var.op)
            assert op_index >= 0
            op_indexes.append(op_index)

    ad.erase_ops(sorted(op_indexes))
    ad.erase_dots(param_dot)

    if len(parameter_list) == 1:
        params_and_grads = [(paramteter_list, param_bar)]
    else:
        params_and_grads = zip(parameter_list, param_bar)
    return params_and_grads


class Optimizer(object):
    @imperative_base.no_grad
    def __init__(self,
                 learning_rate,
                 parameter_list=None,
                 regularization=None,
                 grad_clip=None,
                 flatten_param_grads=False,
                 align_size=-1,
                 name=None):
        """
        Args:
            flatten_param_grads (bool, optional): Whether to flatten all the parameters and grads. 
                If true, the parameters and gradients will be coalesce to contiguous mempry, 
                and the grad_clip ops / optimizer ops will be fuse to one operator.
        """
        # Because of the loop import, so place it in the function body
        from paddle.optimizer.lr import LRScheduler
        self._parameter_list = list(
            parameter_list) if parameter_list is not None else None
        self._name = name
        if framework._non_static_mode():
            if not isinstance(learning_rate,
                              (float, LearningRateDecay, LRScheduler)):
                raise TypeError(
                    "learning rate should be float or LRScheduler, got %s here"
                    % type(learning_rate))
            if self._parameter_list is None:
                raise AttributeError(
                    "parameter_list argument given to the Optimizer should not be None in dygraph mode."
                )
            if regularization is not None:
                for param in self._parameter_list:
                    if param.regularizer is not None:
                        logging.info(
                            "If regularizer of a Parameter has been set by 'fluid.ParamAttr' or 'fluid.WeightNormParamAttr' already. "
                            "The Regularization[%s] in Optimizer will not take effect, and it will only be applied to other Parameters!"
                            % regularization.__str__())
                        break
        else:
            if not isinstance(learning_rate,
                              (float, framework.Variable, LRScheduler)):
                raise TypeError(
                    "learning rate should be float or LRScheduler, got %s here"
                    % type(learning_rate))

        if grad_clip is not None:
            if not isinstance(grad_clip, GradientClipBase):
                raise TypeError(
                    "'grad_clip' should be an instance of GradientClipBase's derived class"
                )
        self.regularization = regularization
        self._grad_clip = grad_clip
        self._learning_rate = learning_rate
        self._flatten_param_grads = flatten_param_grads
        self._align_size = align_size

        self._dtype = None
        # Infer the dtype form parameter
        if self._parameter_list:
            self._dtype = self._parameter_list[0].dtype

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
        # global_accumulator dict, {accum_name : acc_variable, ...}
        self._global_accumulators = {}
        self.helper = LayerHelper(self.__class__.__name__)
        self._opti_name_list = []
        self._accumulators_holder = {}
        self._param_device_map = dict()
        # NOTE(zhiqiu): sometimes we want to add some variables(Tenosr) to the optimizer for a specific optimization,
        # for example, we want to pass 'found_inf' to adam optimizer so it can skip update when found_inf is True.
        # And these variables should not be the parameters of Optimizer's construnctor (because not commonly used). 
        # Use _auxiliary_vars together with _set_auxiliary_var/_get_auxiliary_var to achieve that.
        self._auxiliary_vars = dict()

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
        from paddle.optimizer.lr import LRScheduler
        state_dict = {}
        for k, v in self._accumulators.items():
            for para_name, var_tmp in v.items():
                state_dict[var_tmp.name] = var_tmp
        for k, v in self._global_accumulators.items():
            state_dict[v.name] = v
        # global step if use lr decay
        if isinstance(self._learning_rate, LRScheduler):
            state_dict["LR_Scheduler"] = self._learning_rate.state_dict()
            return state_dict
        if isinstance(self._learning_rate, LearningRateDecay):
            state_dict["LR_Scheduler"] = self._learning_rate.state_dict()

            if not isinstance(self._learning_rate, _LearningRateEpochDecay):
                var_tmp = None
                var_temp = framework._varbase_creator(
                    None, name='global_step', dtype='int32')

                tensor.fill_constant(
                    [1], "int32", self._learning_rate.step_num, out=var_temp)

                state_dict['global_step'] = var_temp
        return state_dict

    @framework.dygraph_only
    def set_state_dict(self, state_dict):
        '''
        Load optimizer state dict. For Adam optimizer, contains beta1, beta2, momentum etc. If LearningRateDecay have been used, global_step will be changed.

        Args: 
            state_dict(dict) : Dict contains all the Variable needed by optimizer
        Return:
            None
        
        Examples:
            .. code-block:: python

                import paddle
                import paddle.fluid as fluid

                paddle.disable_static()

                emb = paddle.nn.Embedding(10, 10)

                state_dict = emb.state_dict()
                fluid.save_dygraph(state_dict, "paddle_dy")

                scheduler = paddle.optimizer.lr.NoamDecay(	
                    d_model=0.01, warmup_steps=100, verbose=True)
                adam = paddle.optimizer.Adam(
                    learning_rate=scheduler,
                    parameters=emb.parameters())
                state_dict = adam.state_dict()
                fluid.save_dygraph(state_dict, "paddle_dy")

                para_state_dict, opti_state_dict = fluid.load_dygraph("paddle_dy")
        '''
        from paddle.optimizer.lr import LRScheduler
        if isinstance(self._learning_rate, LRScheduler):
            self._learning_rate.set_dict(state_dict["LR_Scheduler"])

        if isinstance(self._learning_rate, LearningRateDecay):
            self._learning_rate.set_dict(state_dict["LR_Scheduler"])

            if not isinstance(self._learning_rate, _LearningRateEpochDecay):
                assert 'global_step' in state_dict, \
                        'Global step not in state dict, Dygraph use LearningRateDecay, global_step must in state_dict'
                global_step = state_dict['global_step']

                if isinstance(global_step, Variable):
                    step_np = global_step
                    step_np = np.array(step_np.value().get_tensor())
                    assert step_np.shape == (1,),  \
                            "global step shape is (1,), the shape is {}".format( step_np.shape )

                    self._learning_rate.step_num = int(step_np[0])
                elif isinstance(global_step, np.ndarray):
                    assert global_step.shape == (1,),  \
                            "global step shape is (1,), the shape is {}".format( global_step.shape )
                    self._learning_rate.step_num = global_step[0]
                else:
                    raise RuntimeError(
                        "Type not supprt, value in state dict must be [VarBase, Variable, numpy], the type is ",
                        type(global_step))

        def _load_state_para(state_dict, param):
            var = param.value()
            tensor = var.get_tensor()
            model_np = np.array(tensor)
            load_para = state_dict[param.name]
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
                                                param.name, model_np.shape, load_para_np.shape)

            assert model_np.dtype == load_para_np.dtype, \
                                        "Parameter dtype not match, Dygraph Parameter [ {} ] need tensor with dtype {}  but load tensor with dtype {}".format(
                                            param.name, model_np.dtype, load_para_np.dtype)

            tensor.set(load_para_np, framework._current_expected_place())

        self._accumulators_holder = state_dict
        for k, v in self._accumulators.items():
            for para_name, var_tmp in v.items():
                assert var_tmp.name in state_dict, \
                        "optimizer variable {} not found".format( var_tmp.name )
                _load_state_para(state_dict, var_tmp)

        for k, v in self._global_accumulators.items():
            assert v.name in state_dict, \
                        "optimizer variable {} not found".format( v.name )
            _load_state_para(state_dict, v)

    # [aliases] Compatible with old method names
    set_dict = set_state_dict

    def get_opti_var_name_list(self):
        return self._opti_name_list

    def _set_auxiliary_var(self, key, val):
        self._auxiliary_vars[key] = val

    def _get_auxiliary_var(self, key):
        if key in self._auxiliary_vars:
            return self._auxiliary_vars[key]
        else:
            return None

    def _create_global_learning_rate(self):
        from paddle.optimizer.lr import LRScheduler
        if isinstance(self._learning_rate, LRScheduler):
            lr_var = self._global_learning_rate()
            # only create global lr_var once
            if not isinstance(lr_var, framework.Variable):
                lr_name = unique_name.generate('learning_rate')
                self._learning_rate._var_name = lr_name
                lr_var = self.helper.create_global_variable(
                    name=lr_name,
                    shape=[1],
                    persistable=True,
                    stop_gradient=True,
                    dtype='float32' if self._dtype is None else self._dtype)
                main_prog = framework.default_main_program()
                main_prog.lr_sheduler = self._learning_rate
                main_prog.lr_var = lr_var
                self._learning_rate_map[framework.default_main_program(
                )] = lr_var

            lr_value = float(self._learning_rate())
            self.helper.set_variable_initializer(
                lr_var, initializer=Constant(value=lr_value))
            return

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
    def set_lr(self, value):
        """
        :api_attr: imperative
        
        Set the value of the learning rate manually in the optimizer. If the optimizer use LearningRateDecay,
        this API cannot be invoked, because it will lead to conflict.

        Args:
            value (float|Variable): the value of learning rate

        Returns:
            None
          
        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                        
                with fluid.dygraph.guard():
                    linear = fluid.dygraph.nn.Linear(10, 10)

                    adam = fluid.optimizer.Adam(0.1, parameter_list=linear.parameters())

                    # set learning rate manually by python float value
                    lr_list = [0.2, 0.3, 0.4, 0.5, 0.6]
                    for i in range(5):
                        adam.set_lr(lr_list[i])
                        lr = adam.current_step_lr()
                        print("current lr is {}".format(lr))
                    # Print:
                    #    current lr is 0.2
                    #    current lr is 0.3
                    #    current lr is 0.4
                    #    current lr is 0.5
                    #    current lr is 0.6


                    # set learning rate manually by framework Variable
                    lr_var = fluid.layers.create_global_var(
                        shape=[1], value=0.7, dtype='float32')
                    adam.set_lr(lr_var)
                    lr = adam.current_step_lr()
                    print("current lr is {}".format(lr))
                    # Print:
                    #    current lr is 0.7



        """
        if not isinstance(value, (framework.Variable, float)):
            raise TypeError(
                "The type of 'value' in optimizer.set_lr must be (float, Variable), but received %s."
                % (type(value)))
        if isinstance(self._learning_rate, LearningRateDecay):
            raise RuntimeError(
                "optimizer's learning rate can't be LearningRateDecay when invoke this API, because this will lead to conflict."
            )
        if isinstance(value, float):
            self._learning_rate = value
            current_lr = self._global_learning_rate()
            if current_lr is not None:
                if framework._non_static_mode():
                    _C_ops.fill_constant(current_lr, 'value',
                                         float(value), 'dtype',
                                         current_lr.dtype, 'shape',
                                         list(current_lr.shape))
                else:
                    global_block = framework.default_main_program(
                    ).global_block()
                    global_block.append_op(
                        type='fill_constant',
                        outputs={'Out': [current_lr]},
                        attrs={
                            'dtype': current_lr.dtype,
                            'shape': list(current_lr.shape),
                            'value': float(value)
                        },
                        stop_gradient=True)
        else:
            assert len(value.shape) == 1 and value.shape[
                0] == 1, "optimizer's learning rate must be 1-D Tensor with shape[1]"
            self._learning_rate_map[framework.default_main_program()] = value

    @framework.dygraph_only
    def current_step_lr(self):
        """
        :api_attr: imperative
        
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
        if isinstance(current_lr, framework.Variable):
            return self._global_learning_rate().numpy()[0]

        if isinstance(self._learning_rate, float):
            return self._learning_rate
        elif isinstance(self._learning_rate, _LearningRateEpochDecay):
            step_lr = self._learning_rate()
            return step_lr.numpy()[0]
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
                         device=None):
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
            if framework._non_static_mode():
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
            type=core.VarDesc.VarType.LOD_TENSOR
            if framework._non_static_mode() else (param.type
                                                  if type is None else type),
            shape=shape,
            belong_to_optimizer=True)
        if device is None:
            device = self._get_device_for_param(param.name)
        with device_guard(device):
            self.helper.set_variable_initializer(
                var, initializer=Constant(value=float(fill_value)))

        if framework._non_static_mode():
            if len(self._accumulators_holder) > 0:
                assert var_name in self._accumulators_holder, \
                        "Optimizer set error, {} should in state dict".format( var_name )
                var.set_value(self._accumulators_holder[var_name])

        self._accumulators[name][param.name] = var
        return var

    def _add_global_accumulator(self,
                                name,
                                dtype=None,
                                fill_value=0.0,
                                shape=None,
                                type=None,
                                device=None):
        """Utility function to add a global accumulator for all parameters in the model

        Args:
            block: the block in which the loss variable is present
            name: name of the accumulator
            dtype: data type of the accumulator variable
            fill_value: value to initialize the accumulator variable
            shape: the shape of the accumulator
            type: the variable type of the accumulator
            device: the target place of the accumulator
        """
        if self._name is not None:
            name = self._name + "_" + name
        if (name in self._global_accumulators):
            if framework._non_static_mode():
                return self._global_accumulators[name]
            raise Exception("Global accumulator {} already exists".format(name))
        if shape == None:
            shape = [1]  # most case, global accumulator is of shape [1]
        assert isinstance(self.helper, LayerHelper)

        var_name = name
        var_name = unique_name.generate(var_name)
        self._opti_name_list.append(var_name)

        var = self.helper.create_global_variable(
            name=var_name,
            persistable=True,
            dtype=dtype if dtype else self._dtype,
            type=type,
            shape=shape,
            belong_to_optimizer=True)
        if device is None:
            device = 'cpu'
        with device_guard(device):
            self.helper.set_variable_initializer(
                var, initializer=Constant(value=float(fill_value)))

        if framework._non_static_mode():
            if len(self._accumulators_holder) > 0:
                assert var_name in self._accumulators_holder, \
                        "Optimizer set error, {} should in state dict".format( var_name )
                var.set_value(self._accumulators_holder[var_name])

        self._global_accumulators[name] = var
        return var

    def _get_accumulator(self, name, param):
        """Utility function to fetch an accumulator for a parameter

        Args:
            name: name of the accumulator
            param: parameter variable for which accumulator is to be fetched

        Returns:
            accumulator variable
        """
        if self._name is not None:
            name = self._name + "_" + name
        if (name not in self._accumulators or
                param.name not in self._accumulators[name]):
            raise Exception("Accumulator {} does not exist for parameter {}".
                            format(name, param.name))
        return self._accumulators[name][param.name]

    def _get_global_accumulator(self, name):
        """Utility function to fetch a global accumulator

        Args:
            name: name of the accumulator

        Returns:
            accumulator variable
        """
        if self._name is not None:
            name = self._name + "_" + name
        if (name not in self._global_accumulators):
            raise Exception("Global accumulator {} does not exist".format(name))
        return self._global_accumulators[name]

    def _update_param_device_map(self, parameters_and_grads, target_block):
        for param_and_grad in parameters_and_grads:
            if param_and_grad[0].trainable is True:
                param_name = param_and_grad[0].name
                ops = target_block.ops
                device_attr_name = core.op_proto_and_checker_maker.kOpDeviceAttrName(
                )
                for op in ops:
                    input_arg_names = op.input_arg_names
                    if param_name in input_arg_names:
                        self._param_device_map[param_name] = op.attr(
                            device_attr_name)
                        break

    def _get_device_for_param(self, param_name):
        device = None
        if param_name in self._param_device_map:
            device = self._param_device_map[param_name]
        return device

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

        self._update_param_device_map(parameters_and_grads, target_block)
        self._create_accumulators(
            target_block,
            [p[0] for p in parameters_and_grads if p[0].trainable])
        self._create_global_learning_rate()

        if framework._non_static_mode():
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
                        device = self._get_device_for_param(param_and_grad[0]
                                                            .name)
                        with device_guard(device):
                            optimize_op = self._append_optimize_op(
                                target_block, param_and_grad)

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
            parameter_list (Iterable, optional): Iterable of ``Variable`` or ``Variable.name`` to update
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
        if framework._non_static_mode():
            pass
        else:
            act_no_grad_set = self._get_no_grad_set(loss, no_grad_set)

        # Infer dtype by loss if None
        if self._dtype is None:
            self._dtype = loss.dtype

        if framework._non_static_mode():
            parameter_list = parameter_list if parameter_list \
                else self._parameter_list

            params_grads = []
            for param in parameter_list:
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
            parameter_list = parameter_list if parameter_list \
                else self._parameter_list
            with program_guard(program, startup_program):
                params_grads = append_backward_new([loss], parameter_list,
                                                   act_no_grad_set, callbacks)
        return params_grads

    def _create_regularization_of_grad(self, param, grad, regularization=None):
        """ Create and add backward regularization Operators
    
        Function helper of append_regularization_ops.
        """
        # If no gradient or no regularization is specified,  then we don't need to do anything
        if grad is None or ((not hasattr(param, 'regularizer') or
                             (hasattr(param, 'regularizer') and
                              param.regularizer is None)) and
                            regularization is None):
            return grad
        regularization_term = None
        if hasattr(param, 'regularizer') and param.regularizer is not None:
            # Add variable for regularization term in grad block
            regularization_term = param.regularizer(param, grad, grad.block)
        elif regularization is not None:
            regularization_term = regularization(param, grad, grad.block)

        assert regularization_term is not None

        if framework._non_static_mode():
            return _C_ops.sum([grad, regularization_term])

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
        grad.block.append_op(type='sum', inputs=inputs, outputs=outputs)

        return new_grad

    def append_regularization_ops(self,
                                  parameters_and_grads,
                                  regularization=None):
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
        if framework._non_static_mode():
            for param, grad in parameters_and_grads:
                new_grad = self._create_regularization_of_grad(param, grad,
                                                               regularization)
                params_and_grads.append((param, new_grad))
        else:
            repeate_regularizer = False
            with framework.name_scope('regularization'):
                for param, grad in parameters_and_grads:
                    if not repeate_regularizer and getattr(
                            param, 'regularizer',
                            None) is not None and regularization is not None:
                        repeate_regularizer = True
                        logging.info(
                            "If regularizer of a Parameter has been set by 'fluid.ParamAttr' or 'fluid.WeightNormParamAttr' already. "
                            "The Regularization[%s] in Optimizer will not take effect, and it will only be applied to other Parameters!"
                            % regularization.__str__())
                    with param.block.program._optimized_guard([param, grad]):
                        new_grad = self._create_regularization_of_grad(
                            param, grad, regularization)
                        params_and_grads.append((param, new_grad))
        return params_and_grads

    def flatten_param_grads(self, params_grads):
        need_flatten_params = []
        need_flatten_grads = []
        for p, g in params_grads:
            if g is None:
                continue
            g.persistable = True
            if getattr(p, 'need_clip', True) is False or getattr(
                    p, 'regularizer', None) is not None:
                warnings.warn(
                    "flatten_param_grads=True will be discarded since paramter '{}''s need_clip is False or "
                    "the regularizer is set".format(p.name))
                self._flatten_param_grads = False
                return params_grads

            need_flatten_params.append(p)
            need_flatten_grads.append(g)

        shape = [np.prod(p.shape) for p in need_flatten_params]
        block = need_flatten_params[0].block

        flatten_param = self.helper.create_global_variable(
            name='flatten_param',
            persistable=True,
            dtype=need_flatten_params[0].dtype,
            shape=[np.sum(shape)],
            belong_to_optimizer=True)

        flatten_param.trainable = True
        flatten_param.optimize_attr = need_flatten_params[0].optimize_attr
        flatten_param.regularizer = need_flatten_params[0].regularizer

        flatten_grad = self.helper.create_global_variable(
            name='flatten_grad',
            persistable=True,
            dtype=need_flatten_grads[0].dtype,
            shape=[np.sum(shape)],
            belong_to_optimizer=True)

        with program_guard(default_main_program()):
            block.append_op(
                type="coalesce_tensor",
                inputs={"Input": need_flatten_params},
                outputs={
                    "Output": need_flatten_params,
                    "FusedOutput": flatten_param
                },
                attrs={
                    "copy_data": True,
                    "use_align": True,
                    "align_size": self._align_size,
                    "dtype": need_flatten_params[0].dtype
                })

            block.append_op(
                type="coalesce_tensor",
                inputs={"Input": need_flatten_grads},
                outputs={
                    "Output": need_flatten_grads,
                    "FusedOutput": flatten_grad
                },
                attrs={
                    "copy_data": True,
                    "use_align": True,
                    "align_size": self._align_size,
                    "dtype": need_flatten_grads[0].dtype
                })

        #NOTE(zhiqiu): the initializer should be set after coalesce_tensor op,
        # so the shape of flatten_param and flatten_grad will be inferred.
        self.helper.set_variable_initializer(
            flatten_param, initializer=Constant(0.0))
        self.helper.set_variable_initializer(
            flatten_grad, initializer=Constant(0.0))

        return [(flatten_param, flatten_grad)]

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

        # NOTE(zhiqiu): currently, only support ClipGradByGlobalNorm and without regularization.
        if self._flatten_param_grads and self.regularization is None:
            if self._grad_clip == None or isinstance(self._grad_clip,
                                                     ClipGradByGlobalNorm):
                params_grads = self.flatten_param_grads(params_grads)

        # 'optimizer(grad_clip)' or 'set_gradient_clip'
        if self._grad_clip is not None:
            params_grads = self._grad_clip(params_grads)
        else:
            params_grads = append_gradient_clip_ops(params_grads)

        # Add regularization if any
        params_grads = self.append_regularization_ops(params_grads,
                                                      self.regularization)

        optimize_ops = self._create_optimization_pass(params_grads)
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
        if framework._non_static_mode():
            with program_guard(framework.default_main_program(),
                               framework.default_startup_program()):
                if self._grad_clip is not None:
                    params_grads = self._grad_clip(params_grads)
                params_grads = self.append_regularization_ops(
                    params_grads, self.regularization)
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

        If not, new gradient will accumulat on previous gradient.
        
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
                 no_grad_set=None):
        """
        Add operations to minimize ``loss`` by updating ``parameter_list``.

        Args:
            loss (Variable): A ``Variable`` containing the value to minimize.
            startup_program (Program, optional): :ref:`api_fluid_Program` for
                initializing parameters in ``parameter_list``. The default value
                is None, at this time :ref:`api_fluid_default_startup_program` will be used.
            parameter_list (Iterable, optional): Iterable of ``Variable`` or ``Variable.name`` to update
                to minimize ``loss``. The default value is None, at this time all parameters
                will be updated.
            no_grad_set (set, optional): Set of ``Variable``  or ``Variable.name`` that don't need
                to be updated. The default value is None.

        Returns:
            tuple: tuple (optimize_ops, params_grads), A list of operators appended
            by minimize and a list of (param, grad) variable pairs, param is
            ``Parameter``, grad is the gradient value corresponding to the parameter.
            The returned tuple can be passed to ``fetch_list`` in ``Executor.run()`` to 
            indicate program pruning. If so, the program will be pruned by ``feed`` and 
            ``fetch_list`` before run, see details in ``Executor``.

        Examples:
            Please refer to the example of current Optimizer.
        """
        assert isinstance(loss, Variable), "The loss should be an Variable."

        parameter_list = parameter_list if parameter_list \
            else self._parameter_list

        params_grads = self.backward(
            loss,
            startup_program=startup_program,
            parameter_list=parameter_list,
            no_grad_set=no_grad_set)

        optimize_ops = self.apply_optimize(
            loss, startup_program=startup_program, params_grads=params_grads)

        return optimize_ops, params_grads


class AdamOptimizer(Optimizer):
    r"""
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
        epsilon (float|Tensor, optional): A small float value for numerical stability.
            It should be a float number or a Variable with shape [1] and data type as float32.
            The default value is 1e-08.
        parameter_list (Iterable, optional):  Iterable of ``Variable`` names to update to minimize ``loss``. \
            This parameter is required in dygraph mode. \
            The default value is None in static mode, at this time all parameters will be updated.
        regularization (WeightDecayRegularizer, optional): The strategy of regularization. There are two method: \
             :ref:`api_fluid_regularizer_L1Decay` , :ref:`api_fluid_regularizer_L2Decay` . If a parameter has set \
            regularizer using :ref:`api_fluid_ParamAttr` already, the regularization setting here in optimizer will be \
            ignored for this parameter. Otherwise, the regularization setting here in optimizer will take effect.  \
            Default None, meaning there is no regularization.
        grad_clip (GradientClipBase, optional): Gradient cliping strategy, it's an instance of 
            some derived class of ``GradientClipBase`` . There are three cliping strategies 
            ( :ref:`api_fluid_clip_GradientClipByGlobalNorm` , :ref:`api_fluid_clip_GradientClipByNorm` , 
            :ref:`api_fluid_clip_GradientClipByValue` ). Default None, meaning there is no gradient clipping.
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
        use_global_beta_pow (bool, optional): Whether to use global beta_pow. If true, Adam will use global beta_pow 
            for whole model instead of creating beta_pow for each parameter. Default is false.
        flatten_param_grads (bool, optional): Whether to flatten all parameters and gradients. Default is false.
        align_size (int, optional): The alignment size when flatten parameters and gradients. Default is -1, which means
            use same align_size as allocator. 

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
                def get_decayed_betas(beta1_init, beta2_init, decay_steps, decay_rate, epsilon_init):
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
                    epsilon = fluid.layers.create_global_var(
                        shape=[1],
                        value=float(epsilon_init),
                        dtype='float32',
                        # set persistable for save checkpoints and resume
                        persistable=True,
                        name="epsilon")

                    div_res = global_step / decay_steps
                    decayed_beta1 = beta1_init * (decay_rate**div_res)
                    decayed_beta2 = beta2_init * (decay_rate**div_res)
                    fluid.layers.assign(decayed_beta1, beta1)
                    fluid.layers.assign(decayed_beta2, beta2)

                    return beta1, beta2, epsilon

                beta1, beta2, epsilon = get_decayed_betas(0.9, 0.99, 1e5, 0.9, 1e-8)
                adam_optimizer = fluid.optimizer.AdamOptimizer(
                                                    learning_rate=0.01,
                                                    beta1=beta1,
                                                    beta2=beta2,
                                                    epsilon=epsilon)
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
                 grad_clip=None,
                 name=None,
                 lazy_mode=False,
                 use_global_beta_pow=False,
                 flatten_param_grads=False,
                 align_size=-1):
        assert learning_rate is not None
        assert beta1 is not None
        assert beta2 is not None
        assert epsilon is not None
        super(AdamOptimizer, self).__init__(
            learning_rate=learning_rate,
            parameter_list=parameter_list,
            regularization=regularization,
            grad_clip=grad_clip,
            flatten_param_grads=flatten_param_grads,
            align_size=align_size,
            name=name)
        self.type = "adam"
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._lazy_mode = lazy_mode
        self._use_global_beta_pow = use_global_beta_pow

    def _create_accumulators(self, block, parameters):
        assert isinstance(block, framework.Block)

        # Create accumulator tensors for first and second moments
        for p in parameters:
            self._add_accumulator(self._moment1_acc_str, p)
            self._add_accumulator(self._moment2_acc_str, p)
            if not self._use_global_beta_pow:
                self._add_accumulator(
                    name=self._beta1_pow_acc_str,
                    param=p,
                    fill_value=0.9 if isinstance(self._beta1, Variable) \
                            else self._beta1,
                    shape=[1],
                    type=core.VarDesc.VarType.LOD_TENSOR, device='cpu')
                self._add_accumulator(
                    name=self._beta2_pow_acc_str,
                    param=p,
                    fill_value=0.999 if isinstance(self._beta2, Variable) \
                            else self._beta2,
                    shape=[1],
                    type=core.VarDesc.VarType.LOD_TENSOR, device='cpu')
        if self._use_global_beta_pow:
            self._add_global_accumulator(
                name=self._beta1_pow_acc_str,
                fill_value=0.9 if isinstance(self._beta1, Variable) \
                        else self._beta1,
                shape=[1],
                type=core.VarDesc.VarType.LOD_TENSOR, device='cpu')
            self._add_global_accumulator(
                name=self._beta2_pow_acc_str,
                fill_value=0.999 if isinstance(self._beta2, Variable) \
                        else self._beta2,
                shape=[1],
                type=core.VarDesc.VarType.LOD_TENSOR, device='cpu')

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, framework.Block)

        moment1 = self._get_accumulator(self._moment1_acc_str,
                                        param_and_grad[0])
        moment2 = self._get_accumulator(self._moment2_acc_str,
                                        param_and_grad[0])
        if self._use_global_beta_pow:
            beta1_pow_acc = self._get_global_accumulator(
                self._beta1_pow_acc_str)
            beta2_pow_acc = self._get_global_accumulator(
                self._beta2_pow_acc_str)
        else:
            beta1_pow_acc = self._get_accumulator(self._beta1_pow_acc_str,
                                                  param_and_grad[0])
            beta2_pow_acc = self._get_accumulator(self._beta2_pow_acc_str,
                                                  param_and_grad[0])
        lr = self._create_param_lr(param_and_grad)
        # create the adam optimize op

        if framework._non_static_mode():
            _beta1 = self._beta1 if not isinstance(
                self._beta1, Variable) else self._beta1.numpy().item(0)
            _beta2 = self._beta2 if not isinstance(
                self._beta2, Variable) else self._beta2.numpy().item(0)
            master_weight = None
            _, _, _, _, _, _ = _C_ops.adam(
                param_and_grad[0], param_and_grad[1], lr, moment1, moment2,
                beta1_pow_acc, beta2_pow_acc, master_weight, param_and_grad[0],
                moment1, moment2, beta1_pow_acc, beta2_pow_acc, master_weight,
                'epsilon', self._epsilon, 'lazy_mode', self._lazy_mode,
                'min_row_size_to_use_multithread', 1000, 'beta1', _beta1,
                'beta2', _beta2, 'use_global_beta_pow',
                self._use_global_beta_pow)

            return None

        inputs = {
            "Param": [param_and_grad[0]],
            "Grad": [param_and_grad[1]],
            "LearningRate": [lr],
            "Moment1": [moment1],
            "Moment2": [moment2],
            "Beta1Pow": [beta1_pow_acc],
            "Beta2Pow": [beta2_pow_acc]
        }

        # Pass found_inf to adam, to skip update for not only param, but also momentum and beta_pow
        found_inf = self._get_auxiliary_var('found_inf')

        if found_inf:
            inputs['SkipUpdate'] = found_inf

        outputs = {
            "ParamOut": [param_and_grad[0]],
            "Moment1Out": [moment1],
            "Moment2Out": [moment2],
            "Beta1PowOut": [beta1_pow_acc],
            "Beta2PowOut": [beta2_pow_acc],
        }
        attrs = {
            "lazy_mode": self._lazy_mode,
            "min_row_size_to_use_multithread": 1000,
            'use_global_beta_pow': self._use_global_beta_pow
        }

        if isinstance(self._beta1, Variable):
            inputs['Beta1Tensor'] = self._beta1
        else:
            attrs['beta1'] = self._beta1
        if isinstance(self._beta2, Variable):
            inputs['Beta2Tensor'] = self._beta2
        else:
            attrs['beta2'] = self._beta2
        if isinstance(self._epsilon, Variable):
            inputs['EpsilonTensor'] = self._epsilon
        else:
            attrs['epsilon'] = self._epsilon

        adam_op = block.append_op(
            type=self.type,
            inputs=inputs,
            outputs=outputs,
            attrs=attrs,
            stop_gradient=True)

        return adam_op

    def _finish_update(self, block, parameters_and_grads):
        r"""Update beta1_pow and beta2_pow accumulator
        """
        assert isinstance(block, framework.Block)
        if self._use_global_beta_pow:
            beta1_pow_acc = self._get_global_accumulator(
                self._beta1_pow_acc_str)
            beta2_pow_acc = self._get_global_accumulator(
                self._beta2_pow_acc_str)

            with block.program._optimized_guard([]):
                inputs = {"X": beta1_pow_acc}
                outputs = {"Out": beta1_pow_acc}
                attrs = {}
                if isinstance(self._beta1, Variable):
                    inputs["Y"] = self._beta1
                    # use elementwise_mul for better performance
                    block.append_op(
                        type="elementwise_mul",
                        inputs=inputs,
                        outputs=outputs,
                        attrs=attrs,
                        stop_gradient=True)
                else:
                    attrs['scale'] = self._beta1
                    block.append_op(
                        type="scale",
                        inputs=inputs,
                        outputs=outputs,
                        attrs=attrs,
                        stop_gradient=True)

                inputs = {"X": beta2_pow_acc}
                outputs = {"Out": beta2_pow_acc}
                attrs = {}
                if isinstance(self._beta2, Variable):
                    inputs["Y"] = self._beta2
                    # use elementwise_mul for better performance
                    block.append_op(
                        type="elementwise_mul",
                        inputs=inputs,
                        outputs=outputs,
                        attrs=attrs,
                        stop_gradient=True)
                else:
                    attrs['scale'] = self._beta2
                    block.append_op(
                        type="scale",
                        inputs=inputs,
                        outputs=outputs,
                        attrs=attrs,
                        stop_gradient=True)
