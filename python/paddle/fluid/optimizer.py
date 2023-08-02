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

import numpy as np
import os
import logging
from collections import defaultdict

import paddle


from paddle.fluid.framework import (
    Program,
    Variable,
    Parameter,
    name_scope,
    default_main_program,
    default_startup_program,
    device_guard,
)

from . import framework
from . import layers
from . import unique_name
from .backward import (
    append_backward,
    _some_in_set_,
    _append_grad_suffix_,
    _get_no_grad_set_name,
)
from .framework import program_guard
from .layer_helper import LayerHelper
from .dygraph import base as imperative_base
from .dygraph import no_grad
from .dygraph.learning_rate_scheduler import (
    LearningRateDecay,
    _LearningRateEpochDecay,
)
from paddle.fluid import core
from functools import reduce
from functools import cmp_to_key
from .wrapped_decorator import signature_safe_contextmanager
import warnings
from paddle import _C_ops, _legacy_C_ops
from ..fluid.framework import (
    in_dygraph_mode,
    _current_expected_place,
)

__all__ = [
    'SGD',
    'Momentum',
    'Adam',
    'SGDOptimizer',
    'MomentumOptimizer',
    'AdamOptimizer',
    'LarsMomentum',
    'LarsMomentumOptimizer',
    'RecomputeOptimizer',
]


class Optimizer:
    """Optimizer Base class.

    Define the common interface of an optimizer.
    User should not use this class directly,
    but need to use one of it's implementation.
    """

    @imperative_base.no_grad
    def __init__(
        self,
        learning_rate,
        parameter_list=None,
        regularization=None,
        grad_clip=None,
        flatten_param_grads=False,
        align_size=-1,
        name=None,
    ):
        """
        Args:
            flatten_param_grads (bool, optional): Whether to flatten all the parameters and grads.
                If true, the parameters and gradients will be coalesce to contiguous mempry,
                and the grad_clip ops / optimizer ops will be fuse to one operator.
        """
        # Because of the loop import, so place it in the function body
        from paddle.optimizer.lr import LRScheduler

        self._parameter_list = (
            list(parameter_list) if parameter_list is not None else None
        )
        self._name = name
        if in_dygraph_mode():
            if not isinstance(
                learning_rate, (float, LearningRateDecay, LRScheduler)
            ):
                raise TypeError(
                    "learning rate should be float or LRScheduler, got %s here"
                    % type(learning_rate)
                )
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
                            % regularization.__str__()
                        )
                        break
        else:
            if not isinstance(
                learning_rate, (float, framework.Variable, LRScheduler)
            ):
                raise TypeError(
                    "learning rate should be float or LRScheduler, got %s here"
                    % type(learning_rate)
                )

        if grad_clip is not None:
            if not isinstance(grad_clip, paddle.nn.clip.GradientClipBase):
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
            self._learning_rate_map[
                framework.default_main_program()
            ] = self._learning_rate
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
                import paddle

                with fluid.dygraph.guard():
                    emb = paddle.nn.Embedding(10, 10)

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
                var_temp = framework._create_tensor(
                    None, name='global_step', dtype='int32'
                )

                paddle.tensor.fill_constant(
                    [1], "int32", self._learning_rate.step_num, out=var_temp
                )

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

                paddle.disable_static()

                emb = paddle.nn.Embedding(10, 10)

                state_dict = emb.state_dict()
                paddle.save(state_dict, "paddle_dy.pdparams")

                scheduler = paddle.optimizer.lr.NoamDecay(
                    d_model=0.01, warmup_steps=100, verbose=True)
                adam = paddle.optimizer.Adam(
                    learning_rate=scheduler,
                    parameters=emb.parameters())
                state_dict = adam.state_dict()
                paddle.save(state_dict, "paddle_dy.pdopt")

                para_state_dict = paddle.load("paddle_dy.pdparams")
                opti_state_dict = paddle.load("paddle_dy.pdopt")
        '''
        from paddle.optimizer.lr import LRScheduler

        if isinstance(self._learning_rate, LRScheduler):
            self._learning_rate.set_dict(state_dict["LR_Scheduler"])

        if isinstance(self._learning_rate, LearningRateDecay):
            self._learning_rate.set_dict(state_dict["LR_Scheduler"])

            if not isinstance(self._learning_rate, _LearningRateEpochDecay):
                assert (
                    'global_step' in state_dict
                ), 'Global step not in state dict, Dygraph use LearningRateDecay, global_step must in state_dict'
                global_step = state_dict['global_step']

                if isinstance(global_step, Variable):
                    step_np = global_step
                    step_np = np.array(step_np.value().get_tensor())
                    assert step_np.shape == (
                        1,
                    ), "global step shape is (1,), the shape is {}".format(
                        step_np.shape
                    )

                    self._learning_rate.step_num = int(step_np[0])
                elif isinstance(global_step, np.ndarray):
                    assert global_step.shape == (
                        1,
                    ), "global step shape is (1,), the shape is {}".format(
                        global_step.shape
                    )
                    self._learning_rate.step_num = global_step[0]
                else:
                    raise RuntimeError(
                        "Type not supprt, value in state dict must be [Tensor, Variable, numpy], the type is ",
                        type(global_step),
                    )

        def _load_state_para(state_dict, param):
            var = param.value()
            tensor = var.get_tensor()
            model_np = np.array(tensor)
            load_para = state_dict[param.name]
            if isinstance(load_para, Variable):
                load_para_np = load_para.numpy()
            elif isinstance(load_para, core.eager.Tensor):
                load_para_np = load_para.numpy()
            elif isinstance(load_para, np.ndarray):
                load_para_np = load_para
            else:
                raise RuntimeError(
                    "State dict type {} not supprt".format(str(type(load_para)))
                )

            assert (
                model_np.shape == load_para_np.shape
            ), "Parameter shape not match, Dygraph Parameter [ {} ] need tensor with shape {} but load tensor with shape {}".format(
                param.name, model_np.shape, load_para_np.shape
            )

            assert (
                model_np.dtype == load_para_np.dtype
            ), "Parameter dtype not match, Dygraph Parameter [ {} ] need tensor with dtype {}  but load tensor with dtype {}".format(
                param.name, model_np.dtype, load_para_np.dtype
            )

            tensor.set(load_para_np, framework._current_expected_place())

        self._accumulators_holder = state_dict
        for k, v in self._accumulators.items():
            for para_name, var_tmp in v.items():
                assert (
                    var_tmp.name in state_dict
                ), "optimizer variable {} not found".format(var_tmp.name)
                _load_state_para(state_dict, var_tmp)

        for k, v in self._global_accumulators.items():
            assert (
                v.name in state_dict
            ), "optimizer variable {} not found".format(v.name)
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
                    dtype='float32' if self._dtype is None else self._dtype,
                )
                main_prog = framework.default_main_program()
                main_prog.lr_scheduler = self._learning_rate
                main_prog.lr_var = lr_var
                self._learning_rate_map[
                    framework.default_main_program()
                ] = lr_var

            lr_value = float(self._learning_rate())
            self.helper.set_variable_initializer(
                lr_var,
                initializer=paddle.nn.initializer.Constant(value=lr_value),
            )
            return

        if imperative_base.enabled():
            # create learning rate Variable
            if isinstance(self._learning_rate, float):
                lr = self._global_learning_rate()

                if isinstance(lr, framework.Variable):
                    return
                else:
                    self._learning_rate_map[
                        framework.default_main_program()
                    ] = paddle.static.create_global_var(
                        name=unique_name.generate("learning_rate"),
                        shape=[1],
                        value=float(self._learning_rate),
                        dtype='float32' if self._dtype is None else self._dtype,
                        persistable=True,
                    )
            # get learning rate Variable from LearningRateDecay
            elif isinstance(self._learning_rate, LearningRateDecay):
                self._learning_rate_map[
                    framework.default_main_program()
                ] = self._learning_rate()
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
            self._learning_rate_map[
                framework.default_main_program()
            ] = paddle.static.create_global_var(
                name=unique_name.generate("learning_rate"),
                shape=[1],
                value=float(self._learning_rate),
                dtype='float32' if self._dtype is None else self._dtype,
                persistable=True,
            )

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

                import paddle
                import paddle.fluid as fluid
                import paddle

                with fluid.dygraph.guard():
                    linear = paddle.nn.Linear(10, 10)

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
                    lr_var = paddle.static.create_global_var(
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
                % (type(value))
            )
        if isinstance(self._learning_rate, LearningRateDecay):
            raise RuntimeError(
                "optimizer's learning rate can't be LearningRateDecay when invoke this API, because this will lead to conflict."
            )
        if isinstance(value, float):
            self._learning_rate = value
            current_lr = self._global_learning_rate()
            if current_lr is not None:
                if in_dygraph_mode():
                    place = _current_expected_place()
                    _C_ops.full_(
                        current_lr,
                        list(current_lr.shape),
                        float(value),
                        current_lr.dtype,
                        place,
                    )
                else:
                    global_block = (
                        framework.default_main_program().global_block()
                    )
                    global_block.append_op(
                        type='fill_constant',
                        outputs={'Out': [current_lr]},
                        attrs={
                            'dtype': current_lr.dtype,
                            'shape': list(current_lr.shape),
                            'value': float(value),
                        },
                        stop_gradient=True,
                    )
        else:
            assert (
                len(value.shape) == 1 and value.shape[0] == 1
            ), "optimizer's learning rate must be 1-D Tensor with shape[1]"
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
                import paddle

                # example1: LearningRateDecay is not used, return value is all the same
                with fluid.dygraph.guard():
                    emb = paddle.nn.Embedding(10, 10)
                    adam = fluid.optimizer.Adam(0.001, parameter_list = emb.parameters())
                    lr = adam.current_step_lr()
                    print(lr) # 0.001

                # example2: PiecewiseDecay is used, return the step learning rate
                with fluid.dygraph.guard():
                    inp = np.random.uniform(-0.1, 0.1, [10, 10]).astype("float32")
                    linear = paddle.nn.Linear(10, 10)
                    inp = fluid.dygraph.to_variable(inp)
                    out = linear(inp)
                    loss = paddle.mean(out)

                    bd = [2, 4, 6, 8]
                    value = [0.2, 0.4, 0.6, 0.8, 1.0]
                    adam = paddle.optimizer.Adam(paddle.optimizer.lr.PiecewiseDecay(bd, value),
                                           parameters=linear.parameters())

                    # first step: learning rate is 0.2
                    np.allclose(adam.get_lr(), 0.2, rtol=1e-06, atol=0.0) # True

                    # learning rate for different steps
                    ret = [0.2, 0.2, 0.4, 0.4, 0.6, 0.6, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0]
                    for i in range(12):
                        adam.minimize(loss)
                        adam.step()
                        lr = adam.get_lr()
                        np.allclose(lr, ret[i], rtol=1e-06, atol=0.0) # True

        """
        current_lr = self._global_learning_rate()
        if isinstance(current_lr, framework.Variable):
            return float(current_lr)

        if isinstance(self._learning_rate, float):
            return self._learning_rate
        elif isinstance(self._learning_rate, _LearningRateEpochDecay):
            step_lr = self._learning_rate()
            return float(step_lr)
        else:
            step_lr = self._learning_rate.step()
            if isinstance(step_lr, (float, int)):
                return step_lr
            else:
                return float(step_lr)

    def _global_learning_rate(self, program=None):
        """
        get global decayed learning rate
        :return:
        """
        if program is None:
            program = framework.default_main_program()
        return self._learning_rate_map.get(program, None)

    def _append_optimize_op(self, block, param_and_grad):
        """append optimize operator to block and return all the added optimize_op"""
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
                    is_with_opt=True
                ), framework.name_scope('scale_with_param_lr'):
                    return self._global_learning_rate() * param_lr

    def _is_dtype_fp16_or_bf16(self, dtype):
        """
        check the dtype is fp16 or the dtype is bf16
        :param dtype: instance of core.VarDesc.VarType
        :return: True if dtype is one of fp16 or bf16, False otherwise
        """
        assert isinstance(
            dtype, core.VarDesc.VarType
        ), "The dtype should be an instance of core.VarDesc.VarType."
        return (
            dtype == core.VarDesc.VarType.FP16
            or dtype == core.VarDesc.VarType.BF16
        )

    def _create_master_weight(self, param):
        if param.name in self._master_weights:
            var = self._master_weights[param.name]
        else:
            assert isinstance(self.helper, LayerHelper)

            var_name = param.name + "_fp32_master"
            var_name = unique_name.generate(var_name)
            var = paddle.static.create_global_var(
                name=var_name,
                shape=param.shape,
                value=0,
                dtype='float32',
                persistable=True,
            )
            block = self.helper.startup_program.global_block()
            block.append_op(
                type="cast",
                inputs={"X": [param]},
                outputs={"Out": [var]},
                attrs={
                    "in_dtype": param.dtype,
                    "out_dtype": core.VarDesc.VarType.FP32,
                },
            )
            self._master_weights[param.name] = var
        return var

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

    def _add_accumulator(
        self,
        name,
        param,
        dtype=None,
        fill_value=0.0,
        shape=None,
        type=None,
        device=None,
    ):
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
        if (
            name in self._accumulators
            and param.name in self._accumulators[name]
        ):
            if in_dygraph_mode():
                return self._accumulators[name][param.name]
            raise Exception(
                "Accumulator {} already exists for parameter {}".format(
                    name, param.name
                )
            )
        if shape is None:
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
            if in_dygraph_mode()
            else (param.type if type is None else type),
            shape=shape,
            belong_to_optimizer=True,
        )
        if device is None:
            device = self._get_device_for_param(param.name)
        with device_guard(device):
            self.helper.set_variable_initializer(
                var,
                initializer=paddle.nn.initializer.Constant(
                    value=float(fill_value)
                ),
            )

        if in_dygraph_mode():
            if len(self._accumulators_holder) > 0:
                assert (
                    var_name in self._accumulators_holder
                ), "Optimizer set error, {} should in state dict".format(
                    var_name
                )
                var.set_value(self._accumulators_holder[var_name])

        self._accumulators[name][param.name] = var
        return var

    def _add_global_accumulator(
        self,
        name,
        dtype=None,
        fill_value=0.0,
        shape=None,
        type=None,
        device=None,
    ):
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
        if name in self._global_accumulators:
            if in_dygraph_mode():
                return self._global_accumulators[name]
            raise Exception("Global accumulator {} already exists".format(name))
        if shape is None:
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
            belong_to_optimizer=True,
        )
        if device is None:
            device = 'cpu'
        with device_guard(device):
            self.helper.set_variable_initializer(
                var,
                initializer=paddle.nn.initializer.Constant(
                    value=float(fill_value)
                ),
            )

        if in_dygraph_mode():
            if len(self._accumulators_holder) > 0:
                assert (
                    var_name in self._accumulators_holder
                ), "Optimizer set error, {} should in state dict".format(
                    var_name
                )
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
        if (
            name not in self._accumulators
            or param.name not in self._accumulators[name]
        ):
            raise Exception(
                "Accumulator {} does not exist for parameter {}".format(
                    name, param.name
                )
            )
        return self._accumulators[name][param.name]

    def _get_accumulator_master(self, name, param):
        """Utility function to fetch an accumulator for a parameter
        Args:
            name: name of the accumulator
            param: parameter variable for which accumulator is to be fetched
        Returns:
            accumulator variable for the parameter
        """
        if self._name is not None:
            name = self._name + "_" + name
        find_master = self._multi_precision and self._is_dtype_fp16_or_bf16(
            param.dtype
        )
        target_param = (
            self._master_weights[param.name] if find_master else param
        )
        target_name = target_param.name
        if (
            name not in self._accumulators
            or target_name not in self._accumulators[name]
        ):
            raise Exception(
                "Accumulator {} does not exist for parameter {}".format(
                    name, target_name
                )
            )
        return self._accumulators[name][target_name]

    def _get_global_accumulator(self, name):
        """Utility function to fetch a global accumulator

        Args:
            name: name of the accumulator

        Returns:
            accumulator variable
        """
        if self._name is not None:
            name = self._name + "_" + name
        if name not in self._global_accumulators:
            raise Exception("Global accumulator {} does not exist".format(name))
        return self._global_accumulators[name]

    def _update_param_device_map(self, parameters_and_grads, target_block):
        for param_and_grad in parameters_and_grads:
            if param_and_grad[0].trainable is True:
                param_name = param_and_grad[0].name
                ops = target_block.ops
                device_attr_name = (
                    core.op_proto_and_checker_maker.kOpDeviceAttrName()
                )
                for op in ops:
                    input_arg_names = op.input_arg_names
                    if param_name in input_arg_names:
                        self._param_device_map[param_name] = op.attr(
                            device_attr_name
                        )
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
            assert (
                current_block.backward_block_idx != -1
            ), "current block is not global_block, but it doesn't have backward block."
            target_block = framework.default_main_program().blocks[
                current_block.backward_block_idx
            ]

        start = len(target_block.ops)

        self._update_param_device_map(parameters_and_grads, target_block)
        self._create_accumulators(
            target_block, [p[0] for p in parameters_and_grads if p[0].trainable]
        )
        self._create_global_learning_rate()

        if in_dygraph_mode():
            found_inf = self._get_auxiliary_var('found_inf')
            if found_inf:
                if isinstance(found_inf, core.eager.Tensor):
                    self._set_auxiliary_var('found_inf', True)
            else:
                if isinstance(found_inf, core.eager.Tensor):
                    self._set_auxiliary_var('found_inf', False)
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
                    param_and_grad
                ), name_scope("optimizer"):
                    if param_and_grad[0].trainable is True:
                        device = self._get_device_for_param(
                            param_and_grad[0].name
                        )
                        with device_guard(device):
                            optimize_op = self._append_optimize_op(
                                target_block, param_and_grad
                            )

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
        from paddle.distributed.distribute_lookup_table import (
            find_distributed_lookup_table,
        )

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
                        "multi dist table var found, only support one now!"
                    )
                table_param = p
                table_grad = g
            else:
                new_param_grads.append((p, g))
        sgd_op = None
        if table_param is not None:
            param_and_grad = [table_param, table_grad]
            with table_param.block.program._optimized_guard(
                param_and_grad
            ), framework.name_scope("optimizer"):
                self._create_global_learning_rate()
                # create the optimize op
                sgd_op = global_block.append_op(
                    type='sgd',
                    inputs={
                        "Param": table_param,
                        "Grad": table_grad,
                        "LearningRate": self._create_param_lr(param_and_grad),
                    },
                    outputs={"ParamOut": param_and_grad[0]},
                )
        return new_param_grads, (table_param, table_grad), sgd_op

    def backward(
        self,
        loss,
        startup_program=None,
        parameter_list=None,
        no_grad_set=None,
        callbacks=None,
    ):
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
        if in_dygraph_mode():
            pass
        else:
            act_no_grad_set = self._get_no_grad_set(loss, no_grad_set)

        # Infer dtype by loss if None
        if self._dtype is None:
            self._dtype = loss.dtype

        if in_dygraph_mode():
            parameter_list = (
                parameter_list if parameter_list else self._parameter_list
            )

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
                callbacks = [paddle.nn.clip.error_clip_callback]
            else:
                assert isinstance(callbacks, list)
            program = loss.block.program
            assert np.prod(loss.shape) == 1, (
                "The number of elements of loss should be 1, but the current loss.shape is {}, whose number of elements is not 1. "
                "Maybe that you should call paddle.mean to process the current loss.".format(
                    loss.shape
                )
            )
            parameter_list = (
                parameter_list if parameter_list else self._parameter_list
            )
            with program_guard(program, startup_program):
                params_grads = append_backward(
                    loss, parameter_list, act_no_grad_set, callbacks
                )
        return params_grads

    def _create_regularization_of_grad(self, param, grad, regularization=None):
        """Create and add backward regularization Operators

        Function helper of append_regularization_ops.
        """
        # If no gradient or no regularization is specified,  then we don't need to do anything
        if grad is None or (
            (
                not hasattr(param, 'regularizer')
                or (hasattr(param, 'regularizer') and param.regularizer is None)
            )
            and regularization is None
        ):
            return grad
        regularization_term = None
        if hasattr(param, 'regularizer') and param.regularizer is not None:
            # Add variable for regularization term in grad block
            regularization_term = param.regularizer(param, grad, grad.block)
        elif regularization is not None:
            regularization_term = regularization(param, grad, grad.block)

        assert regularization_term is not None

        if in_dygraph_mode():
            return _legacy_C_ops.sum([grad, regularization_term])

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
                type=core.VarDesc.VarType.LOD_TENSOR,
            )

        inputs = {"X": [grad, regularization_term]}
        outputs = {"Out": [new_grad]}
        grad.block.append_op(type='sum', inputs=inputs, outputs=outputs)

        return new_grad

    def append_regularization_ops(
        self, parameters_and_grads, regularization=None
    ):
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
        if in_dygraph_mode():
            for param, grad in parameters_and_grads:
                new_grad = self._create_regularization_of_grad(
                    param, grad, regularization
                )
                params_and_grads.append((param, new_grad))
        else:
            repeate_regularizer = False
            with framework.name_scope('regularization'):
                for param, grad in parameters_and_grads:
                    if (
                        not repeate_regularizer
                        and getattr(param, 'regularizer', None) is not None
                        and regularization is not None
                    ):
                        repeate_regularizer = True
                        logging.info(
                            "If regularizer of a Parameter has been set by 'fluid.ParamAttr' or 'fluid.WeightNormParamAttr' already. "
                            "The Regularization[%s] in Optimizer will not take effect, and it will only be applied to other Parameters!"
                            % regularization.__str__()
                        )
                    with param.block.program._optimized_guard([param, grad]):
                        new_grad = self._create_regularization_of_grad(
                            param, grad, regularization
                        )
                        params_and_grads.append((param, new_grad))
        return params_and_grads

    def flatten_param_grads(self, params_grads):
        need_flatten_params = []
        need_flatten_grads = []
        for p, g in params_grads:
            if g is None:
                continue
            g.persistable = True
            if (
                getattr(p, 'need_clip', True) is False
                or getattr(p, 'regularizer', None) is not None
            ):
                warnings.warn(
                    "flatten_param_grads=True will be discarded since paramter '{}''s need_clip is False or "
                    "the regularizer is set".format(p.name)
                )
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
            belong_to_optimizer=True,
        )

        flatten_param.trainable = True
        flatten_param.optimize_attr = need_flatten_params[0].optimize_attr
        flatten_param.regularizer = need_flatten_params[0].regularizer

        flatten_grad = self.helper.create_global_variable(
            name='flatten_grad',
            persistable=True,
            dtype=need_flatten_grads[0].dtype,
            shape=[np.sum(shape)],
            belong_to_optimizer=True,
        )

        with program_guard(default_main_program()):
            block.append_op(
                type="coalesce_tensor",
                inputs={"Input": need_flatten_params},
                outputs={
                    "Output": need_flatten_params,
                    "FusedOutput": flatten_param,
                },
                attrs={
                    "copy_data": True,
                    "use_align": True,
                    "align_size": self._align_size,
                    "dtype": need_flatten_params[0].dtype,
                },
            )

            block.append_op(
                type="coalesce_tensor",
                inputs={"Input": need_flatten_grads},
                outputs={
                    "Output": need_flatten_grads,
                    "FusedOutput": flatten_grad,
                },
                attrs={
                    "copy_data": True,
                    "use_align": True,
                    "align_size": self._align_size,
                    "dtype": need_flatten_grads[0].dtype,
                },
            )

        # NOTE(zhiqiu): the initializer should be set after coalesce_tensor op,
        # so the shape of flatten_param and flatten_grad will be inferred.
        self.helper.set_variable_initializer(
            flatten_param,
            initializer=paddle.nn.initializer.Constant(0.0),
        )
        self.helper.set_variable_initializer(
            flatten_grad,
            initializer=paddle.nn.initializer.Constant(0.0),
        )

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
            if self._grad_clip is None or isinstance(
                self._grad_clip, paddle.nn.ClipGradByGlobalNorm
            ):
                params_grads = self.flatten_param_grads(params_grads)

        # 'optimizer(grad_clip)' or 'set_gradient_clip'
        if self._grad_clip is not None:
            params_grads = self._grad_clip(params_grads)
        else:
            params_grads = paddle.nn.clip.append_gradient_clip_ops(params_grads)

        # Add regularization if any
        params_grads = self.append_regularization_ops(
            params_grads, self.regularization
        )

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
        if in_dygraph_mode():
            with program_guard(
                framework.default_main_program(),
                framework.default_startup_program(),
            ):
                if self._grad_clip is not None:
                    params_grads = self._grad_clip(params_grads)
                params_grads = self.append_regularization_ops(
                    params_grads, self.regularization
                )
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
            [param.name for param in parameters if param.trainable is False]
        )
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
                import paddle
                import numpy as np

                with fluid.dygraph.guard():
                    value = np.arange(26).reshape(2, 13).astype("float32")
                    a = fluid.dygraph.to_variable(value)
                    linear = paddle.nn.Linear(13, 5)
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
    def minimize(
        self, loss, startup_program=None, parameter_list=None, no_grad_set=None
    ):
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

        parameter_list = (
            parameter_list if parameter_list else self._parameter_list
        )

        params_grads = self.backward(
            loss,
            startup_program=startup_program,
            parameter_list=parameter_list,
            no_grad_set=no_grad_set,
        )

        optimize_ops = self.apply_optimize(
            loss, startup_program=startup_program, params_grads=params_grads
        )

        return optimize_ops, params_grads


class SGDOptimizer(Optimizer):
    r"""
    Optimizer of the stochastic gradient descent algorithm.

    .. math::

        param\_out = param - learning\_rate * grad

    Parameters:
        learning_rate (float|Variable): The learning rate used to update parameters. \
            Can be a float value or a Variable with one float value as data element.
        parameter_list (Iterable, optional):  Iterable of ``Variable`` names to update to minimize ``loss``. \
            This parameter is required in dygraph mode. \
            The default value is None in static graph mode, at this time all parameters will be updated.
        regularization (WeightDecayRegularizer, optional): The strategy of regularization. There are two method: \
             :ref:`api_fluid_regularizer_L1Decay` , :ref:`api_fluid_regularizer_L2Decay` . If a parameter has set \
            regularizer using :ref:`api_fluid_ParamAttr` already, the regularization setting here in optimizer will be \
            ignored for this parameter. Otherwise, the regularization setting here in optimizer will take effect.  \
            Default None, meaning there is no regularization.
        grad_clip (GradientClipBase, optional): Gradient cliping strategy, it's an instance of
            some derived class of ``GradientClipBase`` . There are three cliping strategies
            ( :ref:`api_fluid_clip_GradientClipByGlobalNorm` , :ref:`api_fluid_clip_GradientClipByNorm` ,
            :ref:`api_fluid_clip_GradientClipByValue` ). Default None, meaning there is no gradient clipping.
        name (str, optional): This parameter is used by developers to print debugging information. \
            For details, please refer to :ref:`api_guide_Name`. Default is None.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            import numpy as np

            paddle.enable_static()
            place = fluid.CPUPlace()
            main = fluid.Program()
            with fluid.program_guard(main):
                x = paddle.static.data(name='x', shape=[-1, 13], dtype='float32')
                y = paddle.static.data(name='y', shape=[-1, 1], dtype='float32')
                y_predict = paddle.static.nn.fc(x, size=1, activation=None)
                cost = paddle.nn.functional.square_error_cost(input=y_predict, label=y)
                avg_cost = paddle.mean(cost)

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

    def __init__(
        self,
        learning_rate,
        parameter_list=None,
        regularization=None,
        grad_clip=None,
        multi_precision=False,
        name=None,
    ):
        assert learning_rate is not None
        super().__init__(
            learning_rate=learning_rate,
            parameter_list=parameter_list,
            regularization=regularization,
            grad_clip=grad_clip,
            name=name,
        )
        self.type = "sgd"
        self._use_mkldnn = False
        self._multi_precision = multi_precision
        self._master_weights = {}

    def _create_accumulators(self, block, parameters):
        assert isinstance(block, framework.Block)
        if isinstance(parameters, dict):
            parameters = self._update_param_group(parameters)

        # Create accumulator tensors for first and second moments
        for p in parameters:
            if self._multi_precision and self._is_dtype_fp16_or_bf16(p.dtype):
                master_p = self._create_master_weight(p)
                continue
            if (
                self._is_dtype_fp16_or_bf16(p.dtype)
                and not self._multi_precision
            ):
                warnings.warn(
                    "Accumulating with FP16/BF16 in optimizer can lead to poor accuracy or slow convergence."
                    "Consider using multi_precision=True option of the Adam optimizer."
                )

    @no_grad
    def _append_optimize_op(self, block, param_and_grad):
        find_master = self._multi_precision and self._is_dtype_fp16_or_bf16(
            param_and_grad[0].dtype
        )
        master_weight = (
            self._master_weights[param_and_grad[0].name]
            if find_master
            else None
        )

        lr = self._create_param_lr(param_and_grad)
        if in_dygraph_mode():
            _C_ops.sgd_(
                param_and_grad[0],
                lr,
                param_and_grad[1],
                master_weight,
                find_master,
            )
            return None
        else:
            assert isinstance(block, framework.Block)
            # create the optimize op
            inputs = {
                "Param": param_and_grad[0],
                "Grad": param_and_grad[1],
                "LearningRate": lr,
            }

            outputs = {"ParamOut": param_and_grad[0]}

            attrs = {"multi_precision": find_master}

            if find_master:
                inputs["MasterParam"] = master_weight
                outputs["MasterParamOut"] = master_weight

            sgd_op = block.append_op(
                type=self.type,
                inputs=inputs,
                outputs=outputs,
                attrs=attrs,
                stop_gradient=True,
            )

            return sgd_op


class MomentumOptimizer(Optimizer):
    r"""

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
        parameter_list (Iterable, optional):  Iterable of ``Variable`` names to update to minimize ``loss``. \
            This parameter is required in dygraph mode. \
            The default value is None in static graph mode, at this time all parameters will be updated.
        use_nesterov (bool, optional): Enables Nesterov momentum, default is false.
        regularization (WeightDecayRegularizer, optional): The strategy of regularization. There are two method: \
             :ref:`api_fluid_regularizer_L1Decay` , :ref:`api_fluid_regularizer_L2Decay` . If a parameter has set \
            regularizer using :ref:`api_fluid_ParamAttr` already, the regularization setting here in optimizer will be \
            ignored for this parameter. Otherwise, the regularization setting here in optimizer will take effect.  \
            Default None, meaning there is no regularization.
        grad_clip (GradientClipBase, optional): Gradient cliping strategy, it's an instance of
            some derived class of ``GradientClipBase`` . There are three cliping strategies
            ( :ref:`api_fluid_clip_GradientClipByGlobalNorm` , :ref:`api_fluid_clip_GradientClipByNorm` ,
            :ref:`api_fluid_clip_GradientClipByValue` ). Default None, meaning there is no gradient clipping.
        name (str, optional): This parameter is used by developers to print debugging information. \
            For details, please refer to :ref:`api_guide_Name`. Default is None.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            import numpy as np

            paddle.enable_static()
            place = fluid.CPUPlace()
            main = fluid.Program()
            with fluid.program_guard(main):
                x = paddle.static.data(name='x', shape=[-1, 13], dtype='float32')
                y = paddle.static.data(name='y', shape=[-1, 1], dtype='float32')
                y_predict = paddle.static.nn.fc(x, size=1, activation=None)
                cost = paddle.nn.functional.square_error_cost(input=y_predict, label=y)
                avg_cost = paddle.mean(cost)

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

    def __init__(
        self,
        learning_rate,
        momentum,
        parameter_list=None,
        use_nesterov=False,
        regularization=None,
        grad_clip=None,
        name=None,
    ):
        assert learning_rate is not None
        assert momentum is not None
        super().__init__(
            learning_rate=learning_rate,
            parameter_list=parameter_list,
            regularization=regularization,
            grad_clip=grad_clip,
            name=name,
        )
        self.type = "momentum"
        self._momentum = momentum
        self._use_nesterov = bool(use_nesterov)

    def _create_accumulators(self, block, parameters):
        assert isinstance(block, framework.Block)

        for p in parameters:
            self._add_accumulator(self._velocity_acc_str, p)

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, framework.Block)

        velocity_acc = self._get_accumulator(
            self._velocity_acc_str, param_and_grad[0]
        )
        lr = self._create_param_lr(param_and_grad)
        master_weight = None
        if in_dygraph_mode():
            _, _, _ = _legacy_C_ops.momentum(
                param_and_grad[0],
                param_and_grad[1],
                velocity_acc,
                lr,
                master_weight,
                param_and_grad[0],
                velocity_acc,
                master_weight,
                'mu',
                self._momentum,
                'use_nesterov',
                self._use_nesterov,
            )
            return None
        else:
            attrs = {"mu": self._momentum, "use_nesterov": self._use_nesterov}
            inputs = {
                "Param": [param_and_grad[0]],
                "Grad": [param_and_grad[1]],
                "Velocity": [velocity_acc],
                "LearningRate": [lr],
            }

            outputs = {
                "ParamOut": [param_and_grad[0]],
                "VelocityOut": [velocity_acc],
            }
            # create the momentum optimize op
            momentum_op = block.append_op(
                type=self.type,
                inputs=inputs,
                outputs=outputs,
                attrs=attrs,
                stop_gradient=True,
            )

            return momentum_op


class LarsMomentumOptimizer(Optimizer):
    r"""
    Momentum optimizer with LARS support

    The update equations are as follows:

    .. math::

        & local\_learning\_rate = learning\_rate * lars\_coeff * \\
          \\frac{||param||}{||gradient|| + lars\_weight\_decay * ||param||}

        & velocity = mu * velocity + local\_learning\_rate * (gradient + lars\_weight\_decay * param + epsilon)

        & param = param - velocity

    Parameters:
        learning_rate (float|Variable): The learning rate used to update parameters. \
            Can be a float value or a Variable with one float value as data element. \
            momentum (float): momentum factor
        lars_coeff (float): Defines how much we trust the layer to change its weights.
        lars_weight_decay (float): Weight decay coefficient for decaying using LARS.
        parameter_list (Iterable, optional):  Iterable of ``Variable`` names to update to minimize ``loss``. \
            This parameter is required in dygraph mode. \
            The default value is None in static graph mode, at this time all parameters will be updated.
        regularization (WeightDecayRegularizer, optional): The strategy of regularization. There are two method: \
             :ref:`api_fluid_regularizer_L1Decay` , :ref:`api_fluid_regularizer_L2Decay` . If a parameter has set \
            regularizer using :ref:`api_fluid_ParamAttr` already, the regularization setting here in optimizer will be \
            ignored for this parameter. Otherwise, the regularization setting here in optimizer will take effect.  \
            Default None, meaning there is no regularization.
        grad_clip (GradientClipBase, optional): Gradient cliping strategy, it's an instance of
            some derived class of ``GradientClipBase`` . There are three cliping strategies
            ( :ref:`api_fluid_clip_GradientClipByGlobalNorm` , :ref:`api_fluid_clip_GradientClipByNorm` ,
            :ref:`api_fluid_clip_GradientClipByValue` ). Default None, meaning there is no gradient clipping.
        name (str, optional): This parameter is used by developers to print debugging information. \
            For details, please refer to :ref:`api_guide_Name`. Default is None.
        exclude_from_weight_decay (list[str], optional): Name string of layers which will be exclude from lars weight decay. Default is None.
        epsilon (float, optional): Epsilon to avoid Division by Zero when calculate local lr. Default is 0.
        multi_precision (bool, optional): Whether to use multi-precision during weight updating.
        rescale_grad (float, optional): Multiply the gradient with `rescale_grad` \
            before updating. Often choose to be `1.0/batch_size`.

    Examples:
        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            import numpy as np

            paddle.enable_static()
            np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            inp = paddle.static.data(
                name="inp", shape=[2, 2], dtype='float32')
            out = paddle.static.nn.fc(inp, size=3)
            out = paddle.sum(out)
            optimizer = fluid.optimizer.LarsMomentumOptimizer(learning_rate=0.001, momentum=0.9)
            optimizer.minimize(out)

            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(fluid.default_startup_program())
            exe.run(
                feed={"inp": np_inp},
                fetch_list=[out.name])
    """
    _velocity_acc_str = "velocity"

    def __init__(
        self,
        learning_rate,
        momentum,
        lars_coeff=0.001,
        lars_weight_decay=0.0005,
        parameter_list=None,
        regularization=None,
        grad_clip=None,
        name=None,
        exclude_from_weight_decay=None,
        epsilon=0,
        multi_precision=False,
        rescale_grad=1.0,
    ):
        assert learning_rate is not None
        assert momentum is not None
        super().__init__(
            learning_rate=learning_rate,
            parameter_list=parameter_list,
            regularization=regularization,
            grad_clip=grad_clip,
            name=name,
        )
        self.type = "lars_momentum"
        self._momentum = momentum
        self._lars_coeff = float(lars_coeff)
        self._lars_weight_decay = float(lars_weight_decay)
        self._epsilon = float(epsilon)
        if exclude_from_weight_decay is None:
            self._exclude_from_weight_decay = []
        else:
            self._exclude_from_weight_decay = exclude_from_weight_decay
        self._multi_precision = multi_precision
        self._rescale_grad = float(rescale_grad)
        self._master_weights = {}

    def _create_accumulators(self, block, parameters):
        assert isinstance(block, framework.Block)

        for p in parameters:
            if self._multi_precision and self._is_dtype_fp16_or_bf16(p.dtype):
                master_p = self._create_master_weight(p)
                self._add_accumulator(self._velocity_acc_str, master_p)
                continue
            if (
                self._is_dtype_fp16_or_bf16(p.dtype)
                and not self._multi_precision
            ):
                warnings.warn(
                    "Accumulating with FP16/BF16 in optimizer can lead to poor accuracy or slow convergence."
                    "Consider using multi_precision=True option of the Lars optimizer."
                )
            self._add_accumulator(self._velocity_acc_str, p)

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, framework.Block)
        _lars_weight_decay = self._lars_weight_decay
        param_name = param_and_grad[0].name
        if len(self._exclude_from_weight_decay) > 0:
            for name in self._exclude_from_weight_decay:
                if name in param_name:
                    _lars_weight_decay = 0.0
                    break

        velocity_acc = self._get_accumulator_master(
            self._velocity_acc_str, param_and_grad[0]
        )
        lr = self._create_param_lr(param_and_grad)

        find_master = self._multi_precision and self._is_dtype_fp16_or_bf16(
            param_and_grad[0].dtype
        )
        master_weight = (
            self._master_weights[param_and_grad[0].name]
            if find_master
            else None
        )

        attrs = {
            "mu": self._momentum,
            "lars_coeff": self._lars_coeff,
            "lars_weight_decay": [_lars_weight_decay],
            "multi_precision": find_master,
            "epsilon": self._epsilon,
            "rescale_grad": self._rescale_grad,
        }

        inputs = {
            "Param": param_and_grad[0],
            "Grad": param_and_grad[1],
            "Velocity": velocity_acc,
            "LearningRate": lr,
        }

        outputs = {"ParamOut": param_and_grad[0], "VelocityOut": velocity_acc}

        if find_master:
            inputs["MasterParam"] = master_weight
            outputs["MasterParamOut"] = master_weight

        if in_dygraph_mode():
            tmp, tmp2 = _legacy_C_ops.lars_momentum(
                [param_and_grad[0]],
                [param_and_grad[1]],
                [velocity_acc],
                [lr],
                [param_and_grad[0]],
                [velocity_acc],
                "mu",
                self._momentum,
                "lars_coeff",
                self._lars_coeff,
                "lars_weight_decay",
                [_lars_weight_decay],
                "multi_precision",
                find_master,
                "epsilon",
                self._epsilon,
                "rescale_grad",
                self._rescale_grad,
            )
        else:
            # create the momentum optimize op
            momentum_op = block.append_op(
                type=self.type,
                inputs=inputs,
                outputs=outputs,
                attrs=attrs,
                stop_gradient=True,
            )

            return momentum_op


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
            The default value is None in static graph mode, at this time all parameters will be updated.
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

            paddle.enable_static()
            place = fluid.CPUPlace()
            main = fluid.Program()
            with fluid.program_guard(main):
                x = paddle.static.data(name='x', shape=[None, 13], dtype='float32')
                y = paddle.static.data(name='y', shape=[None, 1], dtype='float32')
                y_predict = paddle.static.nn.fc(x, size=1, activation=None)
                cost = paddle.nn.functional.square_error_cost(input=y_predict, label=y)
                avg_cost = paddle.mean(cost)

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

            paddle.enable_static()
            place = fluid.CPUPlace()
            main = fluid.Program()
            with fluid.program_guard(main):
                x = paddle.static.data(name='x', shape=[None, 13], dtype='float32')
                y = paddle.static.data(name='y', shape=[None, 1], dtype='float32')
                y_predict = paddle.static.nn.fc(x, size=1, activation=None)
                cost = paddle.nn.functional.square_error_cost(input=y_predict, label=y)
                avg_cost = paddle.mean(cost)

                # define beta decay variable
                def get_decayed_betas(beta1_init, beta2_init, decay_steps, decay_rate, epsilon_init):
                    global_step = lr_scheduler._decay_step_counter()

                    beta1 = paddle.static.create_global_var(
                        shape=[1],
                        value=float(beta1_init),
                        dtype='float32',
                        # set persistable for save checkpoints and resume
                        persistable=True,
                        name="beta1")
                    beta2 = paddle.static.create_global_var(
                        shape=[1],
                        value=float(beta2_init),
                        dtype='float32',
                        # set persistable for save checkpoints and resume
                        persistable=True,
                        name="beta2")
                    epsilon = paddle.static.create_global_var(
                        shape=[1],
                        value=float(epsilon_init),
                        dtype='float32',
                        # set persistable for save checkpoints and resume
                        persistable=True,
                        name="epsilon")

                    div_res = global_step / decay_steps
                    decayed_beta1 = beta1_init * (decay_rate**div_res)
                    decayed_beta2 = beta2_init * (decay_rate**div_res)
                    paddle.assign(decayed_beta1, beta1)
                    paddle.assign(decayed_beta2, beta2)

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

    def __init__(
        self,
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
        align_size=-1,
    ):
        assert learning_rate is not None
        assert beta1 is not None
        assert beta2 is not None
        assert epsilon is not None
        super().__init__(
            learning_rate=learning_rate,
            parameter_list=parameter_list,
            regularization=regularization,
            grad_clip=grad_clip,
            flatten_param_grads=flatten_param_grads,
            align_size=align_size,
            name=name,
        )
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
                    fill_value=0.9
                    if isinstance(self._beta1, Variable)
                    else self._beta1,
                    shape=[1],
                    type=core.VarDesc.VarType.LOD_TENSOR,
                    device='cpu',
                )
                self._add_accumulator(
                    name=self._beta2_pow_acc_str,
                    param=p,
                    fill_value=0.999
                    if isinstance(self._beta2, Variable)
                    else self._beta2,
                    shape=[1],
                    type=core.VarDesc.VarType.LOD_TENSOR,
                    device='cpu',
                )
        if self._use_global_beta_pow:
            self._add_global_accumulator(
                name=self._beta1_pow_acc_str,
                fill_value=0.9
                if isinstance(self._beta1, Variable)
                else self._beta1,
                shape=[1],
                type=core.VarDesc.VarType.LOD_TENSOR,
                device='cpu',
            )
            self._add_global_accumulator(
                name=self._beta2_pow_acc_str,
                fill_value=0.999
                if isinstance(self._beta2, Variable)
                else self._beta2,
                shape=[1],
                type=core.VarDesc.VarType.LOD_TENSOR,
                device='cpu',
            )

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, framework.Block)

        moment1 = self._get_accumulator(
            self._moment1_acc_str, param_and_grad[0]
        )
        moment2 = self._get_accumulator(
            self._moment2_acc_str, param_and_grad[0]
        )
        if self._use_global_beta_pow:
            beta1_pow_acc = self._get_global_accumulator(
                self._beta1_pow_acc_str
            )
            beta2_pow_acc = self._get_global_accumulator(
                self._beta2_pow_acc_str
            )
        else:
            beta1_pow_acc = self._get_accumulator(
                self._beta1_pow_acc_str, param_and_grad[0]
            )
            beta2_pow_acc = self._get_accumulator(
                self._beta2_pow_acc_str, param_and_grad[0]
            )
        lr = self._create_param_lr(param_and_grad)
        # create the adam optimize op

        if in_dygraph_mode():
            _beta1 = (
                self._beta1
                if not isinstance(self._beta1, Variable)
                else self._beta1.item(0)
            )
            _beta2 = (
                self._beta2
                if not isinstance(self._beta2, Variable)
                else self._beta2.item(0)
            )
            master_weight = None
            _, _, _, _, _, _ = _legacy_C_ops.adam(
                param_and_grad[0],
                param_and_grad[1],
                lr,
                moment1,
                moment2,
                beta1_pow_acc,
                beta2_pow_acc,
                master_weight,
                param_and_grad[0],
                moment1,
                moment2,
                beta1_pow_acc,
                beta2_pow_acc,
                master_weight,
                'epsilon',
                self._epsilon,
                'lazy_mode',
                self._lazy_mode,
                'min_row_size_to_use_multithread',
                1000,
                'beta1',
                _beta1,
                'beta2',
                _beta2,
                'use_global_beta_pow',
                self._use_global_beta_pow,
            )

            return None

        inputs = {
            "Param": [param_and_grad[0]],
            "Grad": [param_and_grad[1]],
            "LearningRate": [lr],
            "Moment1": [moment1],
            "Moment2": [moment2],
            "Beta1Pow": [beta1_pow_acc],
            "Beta2Pow": [beta2_pow_acc],
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
            'use_global_beta_pow': self._use_global_beta_pow,
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
            stop_gradient=True,
        )

        return adam_op

    def _finish_update(self, block, parameters_and_grads):
        r"""Update beta1_pow and beta2_pow accumulator"""
        assert isinstance(block, framework.Block)
        if self._use_global_beta_pow:
            beta1_pow_acc = self._get_global_accumulator(
                self._beta1_pow_acc_str
            )
            beta2_pow_acc = self._get_global_accumulator(
                self._beta2_pow_acc_str
            )

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
                        stop_gradient=True,
                    )
                else:
                    attrs['scale'] = self._beta1
                    block.append_op(
                        type="scale",
                        inputs=inputs,
                        outputs=outputs,
                        attrs=attrs,
                        stop_gradient=True,
                    )

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
                        stop_gradient=True,
                    )
                else:
                    attrs['scale'] = self._beta2
                    block.append_op(
                        type="scale",
                        inputs=inputs,
                        outputs=outputs,
                        attrs=attrs,
                        stop_gradient=True,
                    )


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
Adam = AdamOptimizer
LarsMomentum = LarsMomentumOptimizer


class RecomputeOptimizer(Optimizer):
    """
        :api_attr: Static Graph

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

            import paddle
            import paddle.fluid as fluid
            import numpy as np

            paddle.enable_static()

            def gen_data():
                return {"x": np.random.random(size=(32, 32)).astype('float32'),
                "y": np.random.randint(2, size=(32, 1)).astype('int64')}
            def mlp(input_x, input_y, hid_dim=128, label_dim=2):
                print(input_x)
                fc_1 = paddle.static.nn.fc(x=input_x, size=hid_dim)
                prediction = paddle.static.nn.fc(x=[fc_1], size=label_dim, activation='softmax')
                cost = paddle.nn.functional.cross_entropy(
                    input=prediction, label=input_y,
                    reduction='none', use_softmax=False
                )
                sum_cost = paddle.mean(cost)
                return sum_cost, fc_1, prediction
            input_x = paddle.static.data(name="x", shape=[-1,32], dtype='float32')
            input_y = paddle.static.data(name="y", shape=[-1,1], dtype='int64')
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
        if in_dygraph_mode():
            raise Exception("In dygraph, don't support RecomputeOptimizer.")
        self._optimizer = optimizer
        self._checkpoints = None
        self._learning_rate = self._optimizer._learning_rate
        self._learning_rate_map = self._optimizer._learning_rate_map
        self.enable_offload = False

    def _set_checkpoints(self, checkpoints):
        """
        Args:
            checkpoints (list): List of Variable or string
        """
        assert isinstance(
            checkpoints, list
        ), "_checkpoints should be a list of Variable or a list of String"
        for ckpt in checkpoints:
            assert isinstance(ckpt, str) or isinstance(
                ckpt, Variable
            ), "_checkpoints should be a list of Variable or a list of String"
        self._checkpoints = checkpoints

    # should enable offload before calling backward
    def _enable_offload(self):
        self.enable_offload = True

    @framework.deprecate_stat_dict
    def load(self, state_dict):
        """
            :api_attr: Static Graph

        load function is not supported by Recompute Optimizer for now.
        :return: None

        Args:
            state_dict: the dict load by load_persistable method

        Examples:
            .. code-block:: python

                import paddle
                import paddle.fluid as fluid

                paddle.enable_static()
                def mlp(input_x, input_y, hid_dim=128, label_dim=2):
                    fc_1 = paddle.static.nn.fc(x=input_x, size=hid_dim)
                    prediction = paddle.static.nn.fc(x=[fc_1], size=label_dim, activation='softmax')
                    cost = paddle.nn.functional.cross_entropy(
                        input=prediction, label=input_y,
                        reduction='none', use_softmax=False
                    )
                    sum_cost = paddle.mean(cost)
                    return sum_cost, fc_1, prediction

                input_x = paddle.static.data(name="x", shape=[-1,32], dtype='float32')
                input_y = paddle.static.data(name="y", shape=[-1,1], dtype='int64')
                cost, fc_1, pred = mlp(input_x, input_y)
                print("Finished FF")

                sgd = fluid.optimizer.Adam(learning_rate=0.01)
                sgd = fluid.optimizer.RecomputeOptimizer(sgd)
                sgd._set_checkpoints([fc_1, pred])
                try:
                    state_dict = {}
                    sgd.load(state_dict)
                except NotImplementedError as e:
                    print(e)
        """
        raise NotImplementedError(
            "load function is not supported by Recompute Optimizer for now"
        )

    def apply_gradients(self, params_grads):
        """
        call apply_gradients function of self._optimizer.

        Args:
            params_grads (list): list of (param, grad) pair to do optimization.

        Returns:
            list: A list of operators appended to the current program.

        Examples:
            .. code-block:: python

                import paddle
                import paddle.fluid as fluid
                import paddle.fluid.framework as framework

                paddle.enable_static()

                def mlp(input_x, input_y, hid_dim=128, label_dim=2):
                    fc_1 = paddle.static.nn.fc(x=input_x, size=hid_dim)
                    prediction = paddle.static.nn.fc(x=[fc_1], size=label_dim, activation='softmax')
                    cost = paddle.nn.functional.cross_entropy(
                        input=prediction, label=input_y,
                        reduction='none', use_softmax=False
                    )
                    sum_cost = paddle.mean(cost)
                    return sum_cost, fc_1, prediction


                input_x = paddle.static.data(name="x", shape=[-1,32], dtype='float32')
                input_y = paddle.static.data(name="y", shape=[-1,1], dtype='int64')
                cost, fc_1, pred = mlp(input_x, input_y)
                print("Finished FF")

                sgd = fluid.optimizer.Adam(learning_rate=0.01)
                sgd = fluid.optimizer.RecomputeOptimizer(sgd)
                sgd._set_checkpoints([fc_1, pred])
                params_grads = sgd.backward(
                    cost,
                    startup_program=None,
                    parameter_list=None,
                    no_grad_set=None)

                program = cost.block.program
                with framework.program_guard(program, None):
                    optimize_ops = sgd.apply_gradients(params_grads)

                print("Finished apply gradients")
        """

        return self._optimizer.apply_gradients(params_grads=params_grads)

    def _creat_vars(self, varname):
        pinned_var_name = unique_name.generate(varname + "@Pinned")
        fetched_var_name = unique_name.generate(varname + "@Fetch")

        pinned_var = self._main_program.global_block().create_var(
            name=pinned_var_name,
            shape=self.checkpoint_shape,
            dtype=self._main_program.global_block().var(varname).dtype,
            persistable=False,
            stop_gradient=True,
        )

        fetch_var = self._main_program.global_block().create_var(
            name=fetched_var_name,
            shape=self.checkpoint_shape,
            dtype=self._main_program.global_block().var(varname).dtype,
            persistable=False,
            stop_gradient=False,
        )

        return pinned_var_name, fetched_var_name

    def _append_fill_constant_ops(self, startup_program):
        """
        add fill_constant_ops to the end of the prog

        we should fill the pinned vars before runing the main_prog
        to instantiate their tensor hold_, which could tell us whether
        the host memory could hold all the checkpoints from all the
        GPU devices in this node.
        """
        op_role = 0
        block = startup_program.global_block()
        fill_constant_vars = self.checkpoint_name2pinned_name.values()
        OP_ROLE_KEY = core.op_proto_and_checker_maker.kOpRoleAttrName()
        for varname in fill_constant_vars:
            var = self._main_program.global_block().var(varname)
            # NOTE (JZ-LIANG) to pre-allocate the CUDAPinned MEM
            pinned_var = block.create_var(
                name=varname,
                shape=self.checkpoint_shape,
                dtype=self._main_program.global_block().var(var.name).dtype,
                persistable=False,
                stop_gradient=True,
            )
            block.append_op(
                type='fill_constant',
                outputs={'Out': varname},
                attrs={
                    "shape": var.shape,
                    "dtype": var.dtype,
                    "value": 0.0,
                    "place_type": 2,
                    OP_ROLE_KEY: op_role,
                },
            )

        return

    def _insert_async_memcpy_op(
        self, insert_idx, src_varname, dst_varname, op_role, dst_place_type
    ):
        OP_ROLE_KEY = core.op_proto_and_checker_maker.kOpRoleAttrName()
        self.block._insert_op_without_sync(
            insert_idx,
            type='memcpy',
            inputs={'X': [self._main_program.global_block().var(src_varname)]},
            outputs={
                'Out': [self._main_program.global_block().var(dst_varname)]
            },
            attrs={"dst_place_type": int(dst_place_type), OP_ROLE_KEY: op_role},
        )

    def _insert_fetch_op(self, idx, varname):
        assert (
            varname in self.checkpoint_name2pinned_name
        ), "Try to fetch {} from Pinned Memory, but it is NOT a checkpoint".format(
            varname
        )

        pinned_varname = self.checkpoint_name2pinned_name[varname]
        fetch_varname = self.checkpoint_name2fetch_name[varname]
        self._insert_async_memcpy_op(idx, pinned_varname, fetch_varname, 1, 1)

    def _insert_offload_op(self, idx, varname):
        assert (
            varname in self.checkpoint_name2pinned_name
        ), "Try to offload {} to Pinned Memory, but it is NOT a checkpoint".format(
            varname
        )
        pinned_varname = self.checkpoint_name2pinned_name[varname]
        self._insert_async_memcpy_op(idx, varname, pinned_varname, 0, 2)

    def _insert_sync_op(self, op_idx, checkpoint_name):
        # single stream offload no need sync
        pass

    def _record_fetch_op(self, idx):
        assert (
            len(self.un_fetch_checkpoint_names) > 0
        ), "Could NOT found checkpoint to fetch"
        checkpoint_name = self.un_fetch_checkpoint_names.pop(-1)
        logging.debug("Record fetch [{}]".format(checkpoint_name))
        self.idx2insertions[idx] = ("fetch", checkpoint_name)

        return checkpoint_name

    def _record_offload_op(self, idx, checkpoint_name):
        expected_checkpoint_name = self.un_offload_checkpoint_names.pop(0)
        assert (
            checkpoint_name == expected_checkpoint_name
        ), "expected to offload [{}] but got [{}]".format(
            expected_checkpoint_name, checkpoint_name
        )
        logging.debug("Record offload [{}]".format(checkpoint_name))
        self.idx2insertions[idx] = ("offload", checkpoint_name)

    def _record_sync_op(self, idx, checkpoint_name):
        assert (
            checkpoint_name not in self.synced_checkpoints
        ), "Try to sync the checkpoint [{}] twice".format(checkpoint_name)
        self.synced_checkpoints.add(checkpoint_name)
        logging.debug("Record offload sync [{}]".format(checkpoint_name))
        self.idx2insertions[idx] = ("sync", checkpoint_name)

    def _parse_backward(self):
        self.idx2insertions = {}
        # don't offload the last checkpoints, to favor throughput
        self.un_fetch_checkpoint_names = self.sorted_checkpoint_names[:]
        self.un_fetch_checkpoint_names.pop(-1)
        need_fetch_checkpoint_names = self.un_fetch_checkpoint_names[:]
        self.checkpoint_usage_count = {}
        for checkpoint_name in self.un_fetch_checkpoint_names:
            self.checkpoint_usage_count[checkpoint_name] = 0

        self.bw_strart_op_idx = len(self.block.ops)
        for idx, op in enumerate(self.block.ops):
            if int(op.desc.attr("op_role")) == 1:
                self.bw_strart_op_idx = idx
                break

        assert self.bw_strart_op_idx < len(
            self.block.ops
        ), "Could NOT found backword op in prog"

        # fetch second to last checkpoint at the beginning of BW
        fetched_checkpoint_varname = self._record_fetch_op(
            self.bw_strart_op_idx
        )
        last_last_fetch_checkpoint = None

        for i, op in enumerate(self.block.ops[self.bw_strart_op_idx :]):
            idx = self.bw_strart_op_idx + i
            input_vars = op.desc.input_arg_names()

            for input_var in input_vars:
                if input_var in need_fetch_checkpoint_names:
                    if input_var not in self.un_fetch_checkpoint_names:
                        # fetch the  offloade checkpoint when the first usage of its previous one
                        if self.checkpoint_usage_count[input_var] == 0:
                            # TODO (JZ-LIANG) sync memcpy_stream if extra stream for memcpy
                            second_to_last_fetch_checkpoint = (
                                fetched_checkpoint_varname
                            )
                            # there is NO fetch ahead the first checkpoint
                            if input_var != self.sorted_checkpoint_names[0]:
                                fetched_checkpoint_varname = (
                                    self._record_fetch_op(idx)
                                )

                        # should check the current used checkpoint is ths last fetch one
                        assert (
                            second_to_last_fetch_checkpoint == input_var
                        ), "Current recompute segment should use [{}] BUT got [{}]".format(
                            second_to_last_fetch_checkpoint, input_var
                        )
                        # rename
                        self.block.ops[idx]._rename_input(
                            input_var,
                            self.checkpoint_name2fetch_name[input_var],
                        )
                        self.checkpoint_usage_count[input_var] += 1
                    else:
                        raise ValueError(
                            "use checkpoint [{}] before fetch in BW".format(
                                input_var
                            )
                        )

        assert (
            len(self.un_fetch_checkpoint_names) == 0
        ), "{} checkpoints have NOT been Recorded".format(
            self.un_fetch_checkpoint_names
        )

    def _update_backward(self):
        if len(self.idx2insertions) == 0:
            return
        total_op = len(self.block.ops)
        for op_idx in reversed(range(self.bw_strart_op_idx, total_op)):
            if op_idx in self.idx2insertions:
                operation, checkpoint_name = self.idx2insertions[op_idx]
                if operation == "fetch":
                    self._insert_fetch_op(op_idx, checkpoint_name)
                    logging.debug(
                        "Insert [{}] fetch op.".format(checkpoint_name)
                    )
                    del self.idx2insertions[op_idx]
                elif operation == "sync":
                    self._insert_sync_op(op_idx, checkpoint_name)
                    logging.debug("Sync [{}] fetch op.".format(checkpoint_name))
        self.block._sync_with_cpp()
        assert (
            len(self.idx2insertions) == 0
        ), "{} checkpoints left un-Fecthed".format(
            [ele[1] for ele in self.idx2insertions.values()]
        )

    def _parse_forward(self):
        self.idx2insertions = {}
        # don't offload the last checkpoints, faster, less memory saving
        self.un_offload_checkpoint_names = self.sorted_checkpoint_names[:]
        last_checkpoint = self.un_offload_checkpoint_names.pop(-1)
        need_offload_checkpoint_names = self.un_offload_checkpoint_names[:]
        self.checkpoint_usage_count_and_idx = {}
        for checkpoint_name in self.un_offload_checkpoint_names:
            self.checkpoint_usage_count_and_idx[checkpoint_name] = {
                'count': 0,
                'idx': -1,
            }
        self.synced_checkpoints = set()
        self.fw_strart_op_idx = len(self.block.ops)
        for idx, op in enumerate(self.block.ops):
            if int(op.desc.attr("op_role")) == 0:
                self.fw_strart_op_idx = idx
                break

        assert self.fw_strart_op_idx < len(
            self.block.ops
        ), "Could NOT found Forward op in prog"
        last_offload_checkpoint = None

        for i, op in enumerate(
            self.block.ops[self.fw_strart_op_idx : self.bw_strart_op_idx]
        ):
            idx = self.fw_strart_op_idx + i
            output_vars = op.desc.output_arg_names()
            input_vars = op.desc.input_arg_names()

            for output_var in output_vars:
                if output_var in need_offload_checkpoint_names:
                    assert (
                        len(output_vars) == 1
                    ), "chekpoint should be the only Output of a certain op, but [{}] is from [{}]".format(
                        output_var, op
                    )

                    if output_var in self.un_offload_checkpoint_names:
                        # insert sync op if last checkpoint has not been sync
                        if last_offload_checkpoint is not None:
                            if (
                                self.checkpoint_usage_count_and_idx[
                                    last_offload_checkpoint
                                ]['count']
                                == 0
                            ):
                                self._record_sync_op(
                                    idx, last_offload_checkpoint
                                )
                            else:
                                last_usage_idx = (
                                    self.checkpoint_usage_count_and_idx[
                                        last_offload_checkpoint
                                    ]['idx']
                                )
                                assert (
                                    last_usage_idx > 0
                                ), "last_usage_idx of checkpoint [{}] should large than 0".format(
                                    last_offload_checkpoint
                                )
                                self._record_sync_op(
                                    last_usage_idx + 1, last_offload_checkpoint
                                )
                        # insert offload op after the checkpoint's generation op
                        self._record_offload_op(idx + 1, output_var)
                        last_offload_checkpoint = output_var
                    else:
                        raise ValueError(
                            "There should be just ONE op that output checkpoint [{}]".format(
                                output_var
                            )
                        )
                # need to sync the last need to offload checkpoint before the last checkpoint as output op
                if output_var == last_checkpoint:
                    assert (
                        len(output_vars) == 1
                    ), "chekpoint should be the only Output of a certain op, but [{}] is from [{}]".format(
                        output_var, op
                    )
                    assert (
                        last_offload_checkpoint
                        == self.sorted_checkpoint_names[-2]
                    ), "the last offload chekpoint before [{}] is suppose to be [{}], but got [{}]".format(
                        last_checkpoint,
                        self.sorted_checkpoint_names[-2],
                        last_offload_checkpoint,
                    )
                    # sync if last checkpoint has not been sync
                    if (
                        self.checkpoint_usage_count_and_idx[
                            last_offload_checkpoint
                        ]['idx']
                        == 0
                    ):
                        self._record_sync_op(idx, last_offload_checkpoint)
                    else:
                        last_usage_idx = self.checkpoint_usage_count_and_idx[
                            last_offload_checkpoint
                        ]['idx']
                        assert (
                            last_usage_idx > 0
                        ), "last_usage_idx of checkpoint [{}] should large than 0".format(
                            last_offload_checkpoint
                        )
                        self._record_sync_op(
                            last_usage_idx + 1, last_offload_checkpoint
                        )
            # record checkpoint usage
            for input_var in input_vars:
                if input_var in need_offload_checkpoint_names:
                    assert (
                        input_var not in self.synced_checkpoints
                    ), "checkpoint [{}] used after sync".format(input_var)
                    self.checkpoint_usage_count_and_idx[input_var]['count'] += 1
                    self.checkpoint_usage_count_and_idx[input_var]['idx'] = idx

        assert (
            len(self.un_offload_checkpoint_names) == 0
        ), "{} checkpoints have NOT been Recorded".format(
            self.un_fetch_checkpoint_names
        )
        assert len(self.synced_checkpoints) == len(
            need_offload_checkpoint_names
        ), "{} checkpoints have NOT been Recorded".format(
            set(need_offload_checkpoint_names) - set(self.synced_checkpoints)
        )

    def _update_forward(self):
        if len(self.idx2insertions) == 0:
            return
        for op_idx in reversed(
            range(self.fw_strart_op_idx, self.bw_strart_op_idx)
        ):
            if op_idx in self.idx2insertions:
                operation, checkpoint_name = self.idx2insertions[op_idx]
                if operation == "offload":
                    self._insert_offload_op(op_idx, checkpoint_name)
                    logging.debug(
                        "Insert [{}] offload op.".format(checkpoint_name)
                    )
                    del self.idx2insertions[op_idx]
                elif operation == "sync":
                    self._insert_sync_op(op_idx, checkpoint_name)
                    logging.debug(
                        "Insert [{}] offload_sync op.".format(checkpoint_name)
                    )
                    del self.idx2insertions[op_idx]

        self.block._sync_with_cpp()
        assert (
            len(self.idx2insertions) == 0
        ), "{} checkpoints left un-Offloaded".format(
            [ele[1] for ele in self.idx2insertions.values()]
        )

    def _check_offload_fetch(self):
        # TODO(JZ-LIANG) the single stream offload need no sync
        pass

    def _offload(self, loss, startup_program=None):
        """
        core steps for recompute offload
        1. create pinned vars and temp vars
        2. parse & update Forward pass: offload, sync
        3. parse & update Backward pass: rename, fetch, sync
        4. verify the correctness
        """
        self._main_program = loss.block.program
        self.block = loss.block
        if startup_program is None:
            startup_program = paddle.static.default_startup_program()

        with program_guard(self._main_program, startup_program):
            assert (
                len(self.checkpoint_shape) > 0
            ), "checkpoints shape {} should be an non empty list like: [12, 512, 1024]".format(
                self.checkpoint_shape
            )
            assert all(
                [ele > 0 for ele in self.checkpoint_shape]
            ), "all ele in checkpoints shape {} should be a determined integer larger than 0".format(
                self.checkpoint_shape
            )
            self.checkpoint_name2pinned_name = dict()
            self.checkpoint_name2fetch_name = dict()
            for checkpoint_varname in self.sorted_checkpoint_names:
                pinned_var_name, fetch_var_name = self._creat_vars(
                    checkpoint_varname
                )
                self.checkpoint_name2pinned_name[
                    checkpoint_varname
                ] = pinned_var_name
                self.checkpoint_name2fetch_name[
                    checkpoint_varname
                ] = fetch_var_name
            self._append_fill_constant_ops(startup_program)
            # TODO (JZ-LIANG) to provide two offload stragtegy in future
            # step 2. parse & update FW: rename, offload, sync
            self._parse_backward()
            self._update_backward()
            # step 3. parse & update BW: rename, offload, sync
            self._parse_forward()
            self._update_forward()
            # step 4. verify the correctness
            self._check_offload_fetch()

        return

    def backward(
        self,
        loss,
        startup_program=None,
        parameter_list=None,
        no_grad_set=None,
        callbacks=None,
    ):
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

                import paddle
                import paddle.fluid as fluid

                paddle.enable_static()

                def mlp(input_x, input_y, hid_dim=128, label_dim=2):
                    fc_1 = paddle.static.nn.fc(x=input_x, size=hid_dim)
                    prediction = paddle.static.nn.fc(x=[fc_1], size=label_dim, activation='softmax')
                    cost = paddle.nn.functional.cross_entropy(
                        input=prediction, label=input_y,
                        reduction='none', use_softmax=False
                    )
                    sum_cost = paddle.mean(cost)
                    return sum_cost, fc_1, prediction


                input_x = paddle.static.data(name="x", shape=[-1,32], dtype='float32')
                input_y = paddle.static.data(name="y", shape=[-1,1], dtype='int64')
                cost, fc_1, pred = mlp(input_x, input_y)
                print("Finished FF")

                sgd = fluid.optimizer.Adam(learning_rate=0.01)
                sgd = fluid.optimizer.RecomputeOptimizer(sgd)
                sgd._set_checkpoints([fc_1, pred])
                params_grads = sgd.backward(
                    cost,
                    startup_program=None,
                    parameter_list=None,
                    no_grad_set=None)
                print("Finished backward")
        """
        assert (
            self._checkpoints is not None
        ), "You should call _set_checkpoints first"

        if in_dygraph_mode():
            raise NotImplementedError(
                "DyGraph current does not support recompute"
            )

        self._dtype = loss.dtype
        program = loss.block.program
        with program_guard(program, startup_program):
            checkpoint_vars = []
            for ckpt in self._checkpoints:
                if isinstance(ckpt, Variable):
                    checkpoint_vars.append(ckpt)
                else:
                    checkpoint_vars.append(loss.block.var(ckpt))

            # allow return to non-recompute when checkpoints is empty
            if len(checkpoint_vars) > 0:
                params_grads, sorted_checkpoint_names = append_backward(
                    loss,
                    parameter_list,
                    no_grad_set,
                    checkpoints=checkpoint_vars,
                )
            else:
                params_grads = append_backward(
                    loss,
                    parameter_list,
                    no_grad_set,
                    checkpoints=checkpoint_vars,
                )

        if self.enable_offload:
            self.sorted_checkpoint_names = sorted_checkpoint_names
            self._offload(loss, startup_program=startup_program)

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
                import paddle
                import paddle.fluid as fluid

                paddle.enable_static()

                def mlp(input_x, input_y, hid_dim=128, label_dim=2):
                    fc_1 = paddle.static.nn.fc(x=input_x, size=hid_dim)
                    prediction = paddle.static.nn.fc(x=[fc_1], size=label_dim, activation='softmax')
                    cost = paddle.nn.functional.cross_entropy(
                        input=prediction, label=input_y,
                        reduction='none', use_softmax=False
                    )
                    sum_cost = paddle.mean(cost)
                    return sum_cost, fc_1, prediction

                input_x = paddle.static.data(name="x", shape=[-1,32], dtype='float32')
                input_y = paddle.static.data(name="y", shape=[-1,1], dtype='int64')
                cost, fc_1, pred = mlp(input_x, input_y)
                print("Finished FF")

                sgd = fluid.optimizer.Adam(learning_rate=0.01)
                sgd = fluid.optimizer.RecomputeOptimizer(sgd)
                sgd._set_checkpoints([fc_1, pred])
                params_grads = sgd.backward(
                    cost,
                    startup_program=None,
                    parameter_list=None,
                    no_grad_set=None)

                optimize_ops = sgd.apply_optimize(
                    cost, startup_program=None, params_grads=params_grads)

                print("Finished apply_optimize")
        """

        func = (
            self._optimizer.apply_optimize
            if hasattr(self._optimizer, 'apply_optimize')
            else self._optimizer._apply_optimize
        )
        return func(
            loss, startup_program=startup_program, params_grads=params_grads
        )

    def minimize(
        self, loss, startup_program=None, parameter_list=None, no_grad_set=None
    ):
        assert isinstance(loss, Variable), "The loss should be an Variable."
        assert (
            self._checkpoints is not None
        ), "You should call _set_checkpoints first"
        if in_dygraph_mode():
            raise NotImplementedError(
                "DyGraph current does not support recompute"
            )
        params_grads = self.backward(
            loss,
            startup_program=startup_program,
            parameter_list=parameter_list,
            no_grad_set=no_grad_set,
        )

        optimize_ops = self.apply_optimize(
            loss, startup_program=startup_program, params_grads=params_grads
        )

        return optimize_ops, params_grads
