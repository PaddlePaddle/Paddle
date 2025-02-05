# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from copy import deepcopy

import numpy as np

import paddle
from paddle import _legacy_C_ops
from paddle.amp.auto_cast import _in_amp_guard, _in_pure_fp16_guard
from paddle.base import backward, core, framework, program_guard
from paddle.base.compiler import BuildStrategy
from paddle.base.data_feeder import check_type, convert_dtype
from paddle.base.dygraph.base import switch_to_static_graph
from paddle.base.framework import get_flags
from paddle.optimizer.lr import LRScheduler

from . import logging_utils
from .utils import (
    RETURN_NO_VALUE_MAGIC_NUM,
    backend_guard,
    construct_grad_names,
)

__all__ = []


class NestSequence:
    """
    A wrapper class that easily to flatten and restore the nest structure of
    given sequence.
    """

    def __init__(self, raw_input, need_check=False):
        self.__raw_input = raw_input
        self.__input_list = self.tolist()
        self.__var_ids = self._get_var_ids()
        self._check_non_variable(need_check)

    def tolist(self):
        """
        Flattens the nested sequences into single list.
        """
        return paddle.utils.flatten(self.__raw_input)

    def restore(self, value_list):
        """
        Restores the nested sequence from value list.
        """
        assert len(self.__input_list) == len(value_list)
        return paddle.utils.pack_sequence_as(self.__raw_input, value_list)

    def _get_var_ids(self):
        var_ids = []
        for idx, var in enumerate(self.__input_list):
            if isinstance(var, (framework.Variable, core.eager.Tensor)):
                var_ids.append(idx)

        return var_ids

    def _check_non_variable(self, need_check):
        """
        Raises warning if output of traced function contains non-tensor type values.
        """
        if need_check:
            warning_types = set()
            for var in self.__input_list:
                if not isinstance(var, (framework.Variable, core.eager.Tensor)):
                    warning_types.add(type(var))
            if warning_types:
                logging_utils.warn(
                    f"Output of traced function contains non-tensor type values: {list(warning_types)}. "
                    "Currently, We don't support to update them while training and will return "
                    "what we first saw. Please try to return them as tensor."
                )

    @property
    def var_ids(self):
        return self.__var_ids

    def __getitem__(self, item):
        return self.__input_list[item]


class LazyInitialized:
    """
    Descriptor to implement lazy initialization of property.
    """

    def __init__(self, function):
        self.function = function

    def __get__(self, instance, cls):
        val = self.function(instance)
        setattr(instance, self.function.__name__, val)
        return val


class ProgramInfo:
    """
    A helper class to record Program information
    """

    def __init__(self):
        self.op_size = {
            'fp32': -1,
            'amp': -1,
            'fp16': -1,
        }
        self.programs = {}
        self.mode = "infer"

    def __call__(self, key, prog_creator):
        """
        Record infer program and op size.
        """
        assert key in ['fp32', 'amp', 'fp16']
        if key not in self.programs:
            infer_prog = prog_creator(is_infer_mode=True)
            self.programs[key] = infer_prog
            self.op_size[key] = infer_prog.desc.block(0).op_size()

        return self.programs[key], self.op_size[key]


class PartialProgramLayerHook:
    def before_append_backward(self, forward_program): ...

    def after_append_backward(self, whole_program, backward_start_idx): ...

    def after_infer(self, infer_program): ...


class PartialProgramLayer:
    """
    PartialProgramLayer wraps all the ops from layers decorated by `@to_static`
    and execute them as a static subgraph.

    .. note::
        **1. This is a very low level API. Users should not use this API
             directly. Please use `partial_program_from(concrete_program)`
             to create it.
        **2. DenseTensorArray is not currently supported in the output.

    Args:
        main_program(Program): The main program that contains ops need to be executed.
        inputs(list[Variable]): The input list of the decorated function by `@to_static`.
        outputs(list[Variable]): The output list of the decorated function by `@to_static`.
        parameters(list[Tensor]|None): All trainable parameters included in the program. Default None.

    Returns:
        Layer: A Layer object that run all ops internally in static graph mode.
    """

    def __init__(
        self, main_program, inputs, outputs, parameters=None, **kwargs
    ):
        super().__init__()
        self._inputs = NestSequence(inputs)
        self._outputs = NestSequence(outputs, need_check=True)
        self._params = parameters if parameters is not None else []

        self._build_strategy = kwargs.get('build_strategy', BuildStrategy())
        assert isinstance(self._build_strategy, BuildStrategy)

        self._origin_main_program = self._verify_program(main_program)
        with paddle.base.framework._dygraph_guard(paddle.base.dygraph.Tracer()):
            self._cuda_graph_vec = self._create_cuda_graph_vec()
        self._cuda_graph_capture_mode = ""
        self._cuda_graph_pool_id = 0
        # Set default mode to train
        self.training = True
        self._infer_info = ProgramInfo()
        self._forward_end_index_map = {}

        amp_dtype, custom_white_list, custom_black_list = None, None, None
        tracer = framework._dygraph_tracer()
        if tracer:
            custom_white_list, custom_black_list = tracer._get_amp_op_list()
            amp_dtype = tracer._amp_dtype
        if amp_dtype is not None and amp_dtype in ['float16', 'bfloat16']:
            # For AMP training
            self._amp_list = (
                paddle.static.amp.fp16_lists.AutoMixedPrecisionLists(
                    custom_white_list=custom_white_list,
                    custom_black_list=custom_black_list,
                    dtype=amp_dtype,
                )
            )

        # program_id -> list(scope)
        self._pir_scope_cache = {}
        self._legacy_scope_cache = {}
        self._hooker = None
        self._backend = kwargs.get('backend', None)
        self._grad_var_names = {}

        self._in_var_names = []
        for var in self._inputs:
            if isinstance(var, framework.Variable):
                self._in_var_names.append(var.desc.name())
        self._out_var_descs = [
            self._outputs[var_id].desc for var_id in self._outputs.var_ids
        ]

    def __call__(self, inputs):
        """
        Execute static graph by Interpreter and Return dynamic Tensors.
        """
        in_vars, in_var_names = self._prepare_inputs(inputs)
        out_vars = self._prepare_outputs()
        self._cast_fp16_if_pure_fp16(in_vars)
        attrs = self._prepare_attributes()
        attrs.extend(["x_names", in_var_names])

        self._sync_lr_value_with_scheduler()

        _legacy_C_ops.run_program(
            self._valid_vars(in_vars),
            self._valid_vars(self._params),
            self._valid_vars(out_vars),
            self._create_scope_vec(
                program_id=self.program_id, use_scope_cache=True
            ),
            self._cuda_graph_vec,
            *attrs,
        )

        restored_nest_out = self._restore_out(out_vars)
        restored_nest_out = self._remove_no_value(restored_nest_out)

        return restored_nest_out

    def sot_call(self, inputs):
        """
        In sot, inputs and outputs of partial program only contain tensors, so we can skip some step to speed up
        """
        out_vars = self._prepare_outputs()
        self._cast_fp16_if_pure_fp16(inputs)
        attrs = self._prepare_attributes()
        attrs.extend(["x_names", self._in_var_names])

        self._sync_lr_value_with_scheduler()

        _legacy_C_ops.run_program(
            self._valid_vars(inputs),
            self._valid_vars(self._params),
            self._valid_vars(out_vars),
            self._create_scope_vec(
                program_id=self.program_id, use_scope_cache=True
            ),
            self._cuda_graph_vec,
            *attrs,
        )

        return out_vars

    def _sync_lr_value_with_scheduler(self):
        """Update lr_var value with calculated by lr_scheduler."""
        main_program = self._origin_main_program
        if hasattr(main_program, 'lr_scheduler') and hasattr(
            main_program, 'lr_var'
        ):
            lr_scheduler = main_program.lr_scheduler
            lr_var = main_program.lr_var

            assert isinstance(lr_scheduler, LRScheduler), "must be LRScheduler"
            lr_scheduler = self._origin_main_program.lr_scheduler
            lr_value = lr_scheduler()
            data = np.array(lr_value).astype(convert_dtype(lr_var.dtype))
            lr_var.set_value(data)

    def set_hooker(self, hooker):
        self._hooker = hooker

    def _get_scope(self, program_id=None, use_scope_cache=False):
        if self._in_pir_pt_mode or self._enable_pir_in_executor:
            _scope_cache = self._pir_scope_cache
        else:
            _scope_cache = self._legacy_scope_cache
        if not use_scope_cache:
            return core.Scope()
        if program_id not in _scope_cache:
            _scope_cache[program_id] = []
        cached_scopes = _scope_cache[program_id]
        for scope in cached_scopes:
            if scope._can_reused:
                return scope
        scope = core.Scope()
        cached_scopes.append(scope)
        return scope

    # whole
    @switch_to_static_graph
    def _create_program(self, is_infer_mode=False):
        if is_infer_mode:
            infer_program = self._origin_main_program.clone(
                for_test=is_infer_mode
            )
            if self._hooker:
                infer_program = self._hooker.after_infer(infer_program)
            return infer_program
        else:
            train_program = self._append_backward_desc(
                self._origin_main_program
            )
            # Note: Only set grad type once after initializing train program. So we put it here.
            self._set_grad_type(self._params, train_program)
            return train_program

    @switch_to_static_graph
    def _create_amp_program(self, is_infer_mode=False):
        amp_program = self._origin_main_program.clone(for_test=is_infer_mode)
        with program_guard(amp_program):
            paddle.static.amp.fp16_utils.cast_model_to_fp16(
                amp_program, self._amp_list, use_fp16_guard=False, level='O1'
            )
        if is_infer_mode:
            if self._hooker:
                amp_program = self._hooker.after_infer(amp_program)
            return amp_program
        else:
            train_amp_program = self._append_backward_desc(amp_program)
            self._set_grad_type(self._params, train_amp_program)
            return train_amp_program

    @switch_to_static_graph
    def _create_pure_fp16_program(self, is_infer_mode=False):
        pure_fp16_program = self._origin_main_program.clone(
            for_test=is_infer_mode
        )
        with program_guard(pure_fp16_program):
            paddle.static.amp.fp16_utils.cast_model_to_fp16(
                pure_fp16_program, self._amp_list, use_fp16_guard=False
            )

        if is_infer_mode:
            if self._hooker:
                pure_fp16_program = self._hooker.after_infer(pure_fp16_program)
            return pure_fp16_program
        else:
            train_pure_fp16_program = self._append_backward_desc(
                pure_fp16_program
            )
            self._set_grad_type(self._params, train_pure_fp16_program)
            return train_pure_fp16_program

    @switch_to_static_graph
    def _create_forward_backward_train_program(self):
        whole_program = self._train_program
        forward_end_op_index = self.get_forward_end_op_idx(whole_program)
        assert forward_end_op_index >= 0

        return self._get_forward_backward_program_form(
            whole_program, forward_end_op_index
        )

    @switch_to_static_graph
    def _create_forward_backward_train_amp_program(self):
        whole_program = self._train_amp_program
        forward_end_op_index = self.get_forward_end_op_idx(whole_program)
        assert forward_end_op_index >= 0

        return self._get_forward_backward_program_form(
            whole_program, forward_end_op_index
        )

    @switch_to_static_graph
    def _create_forward_backward_train_pure_fp16_program(self):
        whole_program = self._train_pure_fp16_program
        forward_end_op_index = self.get_forward_end_op_idx(whole_program)
        assert forward_end_op_index >= 0

        return self._get_forward_backward_program_form(
            whole_program, forward_end_op_index
        )

    @LazyInitialized
    def _train_program(self):
        return self._create_program()

    @LazyInitialized
    def _infer_program(self):
        program, op_size = self._infer_info('fp32', self._create_program)
        return self._build_infer_program(program, op_size)

    @LazyInitialized
    def _train_amp_program(self):
        return self._create_amp_program()

    @LazyInitialized
    def _infer_amp_program(self):
        program, op_size = self._infer_info('amp', self._create_amp_program)
        return self._build_infer_program(program, op_size)

    @LazyInitialized
    def _train_pure_fp16_program(self):
        return self._create_pure_fp16_program()

    @LazyInitialized
    def _infer_pure_fp16_program(self):
        program, op_size = self._infer_info(
            'fp16', self._create_pure_fp16_program
        )
        return self._build_infer_program(program, op_size)

    @LazyInitialized
    def _train_forward_backward_program(self):
        program = self._create_forward_backward_train_program()
        return program

    @LazyInitialized
    def _train_amp_forward_backward_program(self):
        program = self._create_forward_backward_train_amp_program()
        return program

    @LazyInitialized
    def _empty_backward_program_for_eval(self):
        return paddle.static.Program()

    @LazyInitialized
    def _train_pure_fp16_forward_backward_program(self):
        program = self._create_forward_backward_train_pure_fp16_program()
        return program

    @LazyInitialized
    def _train_program_id(self):
        program_id = paddle.utils._hash_with_id(self._train_program, self)
        return program_id

    @LazyInitialized
    def _infer_program_id(self):
        return paddle.utils._hash_with_id(self._infer_program, self)

    @LazyInitialized
    def _train_amp_program_id(self):
        program_id = paddle.utils._hash_with_id(self._train_amp_program, self)
        return program_id

    @LazyInitialized
    def _infer_amp_program_id(self):
        return paddle.utils._hash_with_id(self._infer_amp_program, self)

    @LazyInitialized
    def _train_pure_fp16_program_id(self):
        program_id = paddle.utils._hash_with_id(
            self._train_pure_fp16_program, self
        )
        return program_id

    @LazyInitialized
    def _infer_pure_fp16_program_id(self):
        return paddle.utils._hash_with_id(self._infer_pure_fp16_program, self)

    def get_forward_end_op_idx(self, program):
        return self._forward_end_index_map[
            paddle.utils._hash_with_id(program, self)
        ]

    @property
    def program(self):
        """
        Return current train or eval program.
        """
        if self.training:
            return self.train_program
        else:
            return self.infer_program

    @property
    def program_id(self):
        """
        Return current train or eval program hash id.
        """
        if self.training:
            if _in_amp_guard():
                return self._train_amp_program_id
            elif _in_pure_fp16_guard():
                return self._train_pure_fp16_program_id
            else:
                return self._train_program_id
        else:
            if _in_amp_guard():
                return self._infer_amp_program_id
            elif _in_pure_fp16_guard():
                return self._infer_pure_fp16_program_id
            else:
                return self._infer_program_id

    @property
    def train_program(self):
        if _in_amp_guard():
            return self._train_amp_program
        elif _in_pure_fp16_guard():
            return self._train_pure_fp16_program
        else:
            return self._train_program

    @property
    def infer_program(self):
        if _in_amp_guard():
            infer_program = self._infer_amp_program
        elif _in_pure_fp16_guard():
            infer_program = self._infer_pure_fp16_program
        else:
            infer_program = self._infer_program
        return infer_program

    @property
    def forward_program(self):
        forward_program = None
        if self.training:
            if _in_amp_guard():
                progs = self._train_amp_forward_backward_program
            elif _in_pure_fp16_guard():
                progs = self._train_pure_fp16_forward_backward_program
            else:
                progs = self._train_forward_backward_program
            return progs[0]
        else:
            forward_program = self.infer_program
        return forward_program

    @property
    def backward_program(self):
        if self.training:
            if _in_amp_guard():
                progs = self._train_amp_forward_backward_program
            elif _in_pure_fp16_guard():
                progs = self._train_pure_fp16_forward_backward_program
            else:
                progs = self._train_forward_backward_program
            return progs[1]
        else:
            """
            Can't just return paddle.static.Program(), because self.backward_program is a property,
            whenever we call this method, a tmp Program() object is created and is gc immediately
            after executed the following line in PartialProgramLayer.__call__.

            >>> self.backward_program.desc.block(0),

            When we access RunProgramAPI, it's possible to get an invalid backward_program address.
            """
            return self._empty_backward_program_for_eval

    def _verify_program(self, main_program):
        """
        Verify that the program parameter is initialized, prune some unused params,
        and remove redundant op callstack.
        """
        # 1. Check all params from main program can be found in self._params
        self._check_params_all_inited(main_program)
        # 2. Prune the parameters not used anywhere in the program.
        self._prune_unused_params(main_program)

        return main_program

    def prepare_gradient_aggregation(
        self, start_idx, main_program, target_program
    ):
        """
        Why we need add gradient aggregation operation ?
        In some cases, if non leaf nodes are used as output, gradient overwriting will occur, such as
        def forward(self, in):
            x = 2 * in  # <---- x is a non-leaf node in program.
            y = x + 3
            return x, y

        loss = forward(in)[0].sum()
        loss.backward()  # <----- x@grad will be overwritten by elementwise_add_grad Op
        """

        def _need_aggregation(var):
            """
            if exist a op whose inputs is var, then return True
            """
            if not isinstance(var, framework.Variable) or var.type not in [
                core.VarDesc.VarType.DENSE_TENSOR,
                core.VarDesc.VarType.SELECTED_ROWS,
            ]:
                return False
            if var.dtype not in [paddle.float32, paddle.float64]:
                return False
            for op in main_program.block(0).ops:
                for in_arg in op.input_arg_names:
                    if in_arg == var.name:
                        return True
            return False

        def _insert_aggregation_ops_for_var(target_program, var):
            suffix = "@dy2static"
            var_grad_name = var.grad_name
            new_grad_name = var.name + suffix + "@GRAD"
            found_ops = list(
                filter(
                    lambda x: x[0] >= start_idx
                    and any(
                        out_arg == var_grad_name
                        for out_arg in x[1].output_arg_names
                    ),
                    enumerate(target_program.block(0).ops),
                )
            )

            # len(found_ops) may equals zero when stop_gradient works.
            # len(found_ops) may > 1, because we may have fill_constant op.
            if len(found_ops) == 0:
                return None
            # step1: create a new var named var.name@GRAD
            target_program.block(0).create_var(
                name=new_grad_name,
                type=var.type,
                dtype=var.dtype,
                shape=var.shape,
            )
            # step2: rename the var.name@GRAD to var.name@GRAD@dy2static
            for idx, op in found_ops:
                op._rename_input(var_grad_name, new_grad_name)
                op._rename_output(var_grad_name, new_grad_name)
            # step3: insert sum op to aggregate the gradient.
            #        var.name@GRAD = sum(var.name@dy2static@GRAD, var.name@GRAD)
            target_program.block(0)._insert_op(
                found_ops[-1][0] + 1,
                type='sum',
                inputs={'X': [var_grad_name, new_grad_name]},
                outputs={"Out": var_grad_name},
            )
            return None

        to_processed_vars = list(
            filter(_need_aggregation, self._outputs.tolist())
        )
        for _var in to_processed_vars:
            target_program: paddle.static.Program
            target_var = target_program.global_block().var(_var.name)
            _insert_aggregation_ops_for_var(target_program, target_var)

    @switch_to_static_graph
    def _append_backward_desc(self, main_program):
        program = main_program.clone(for_test=False)
        if self._hooker:
            program = self._hooker.before_append_backward(program)
        targets = []
        for out in self._outputs.tolist():
            if isinstance(out, framework.Variable):
                targets.append(program.global_block().var(out.name))

        start_idx = len(program.block(0).ops) + len(self._outputs.tolist())
        if targets:
            start_idx = len(program.block(0).ops) + len(self._outputs.tolist())
            with backend_guard(self._backend):
                check_type(
                    targets,
                    'targets',
                    (framework.Variable, list, tuple),
                    'paddle.static.gradients',
                )
                grad_info_map = backward.calc_gradient_helper(
                    targets=targets, inputs=[]
                )

                x_vars = [
                    program.block(0).var(var.name)
                    for var in self._inputs
                    if isinstance(var, framework.Variable)
                ]
                param_vars = [
                    program.block(0).var(param.name) for param in self._params
                ]
                out_vars = [
                    program.block(0).var(var.name)
                    for var in self._outputs
                    if isinstance(var, framework.Variable)
                ]

                self._grad_var_names = construct_grad_names(
                    grad_info_map, x_vars, param_vars, out_vars
                )

            if self._hooker:
                program, start_idx = self._hooker.after_append_backward(
                    program, start_idx
                )
            self.prepare_gradient_aggregation(
                start_idx + 1, main_program, program
            )

        self._forward_end_index_map[
            paddle.utils._hash_with_id(program, self)
        ] = start_idx - len(self._outputs.tolist())
        return program

    def _prune_unused_params(self, program):
        """
        Prune the parameters not used anywhere in the program.
        The `@to_static` may only decorated a sub function which
        contains some unused parameters created in `__init__`.
        So prune these parameters to avoid unnecessary operations in
        `run_program_op`.
        """
        required_params = []
        for param in self._params:
            found_param = False
            for block in program.blocks:
                for op in block.ops:
                    if (
                        param.name in op.input_arg_names
                        or param.name in op.output_arg_names
                    ):
                        required_params.append(param)
                        found_param = True
                        break
                if found_param:
                    break

        self._params = required_params

    def _cast_fp16_if_pure_fp16(self, in_vars):
        if _in_pure_fp16_guard():
            for i, var in enumerate(in_vars):
                name = var.name
                if (
                    self.program.global_block().has_var(name)
                    and self.program.global_block().var(name).dtype
                    == paddle.float16
                ):
                    in_vars[i] = var.astype('float16')
                    in_vars[i].name = name

    @property
    def _in_pir_pt_mode(self):
        pir_dy2st_flag = 'FLAGS_enable_pir_with_pt_in_dy2st'
        in_pir_pt_mode = get_flags(pir_dy2st_flag)[pir_dy2st_flag]
        is_prim_enabled = (
            core._is_fwd_prim_enabled() or core._is_bwd_prim_enabled()
        )
        in_cinn_backend = self._backend == "CINN"
        is_cinn_enabled = self._build_strategy.build_cinn_pass
        if is_prim_enabled or in_cinn_backend or is_cinn_enabled:
            in_pir_pt_mode = False
        return in_pir_pt_mode

    @property
    def _enable_pir_in_executor(self):
        enable_pir_in_executor_flag = 'FLAGS_enable_pir_in_executor'
        enable_pir_in_executor = get_flags(enable_pir_in_executor_flag)[
            enable_pir_in_executor_flag
        ]
        return enable_pir_in_executor

    def _prepare_attributes(self):
        attrs = [
            'forward_global_block',
            self.forward_program.desc.block(0),
            'backward_global_block',
            self.backward_program.desc.block(0),
            'is_test',
            not self.training,
            'program_id',
            self.program_id,
        ]

        if self.training:
            # NOTE: In the case of higher-order gradient, the names of the parameter grads may be like
            # `grad/grad/grad/linear_0.w_0@GRAD` instead of simply `linear_0.w_0@GRAD`, so we get
            # the correct names of the parameter grads from program. And out grads are similar to above.
            attrs.extend(
                (
                    'param_grad_names',
                    self._grad_var_names.get('param', []),
                    'out_grad_names',
                    self._grad_var_names.get('out', []),
                    'x_grad_names',
                    self._grad_var_names.get('x', []),
                )
            )
        if self._cuda_graph_capture_mode:
            attrs.extend(
                (
                    'cuda_graph_capture_mode',
                    self._cuda_graph_capture_mode,
                    'cuda_graph_pool_id',
                    self._cuda_graph_pool_id,
                )
            )

        in_pir_pt_mode = self._in_pir_pt_mode
        attrs.extend(['in_pir_pt_mode', in_pir_pt_mode])

        return attrs

    @switch_to_static_graph
    def _build_infer_program(self, infer_program, forward_end_op_index):
        forward_skip_vars = self._parse_skip_gc_vars(infer_program)
        built_infer_program = add_build_strategy_for(
            infer_program,
            0,
            forward_end_op_index,
            self._build_strategy,
            forward_skip_vars,
        )
        self._apply_inplace_pass(built_infer_program, None)
        return built_infer_program

    @switch_to_static_graph
    def _get_forward_backward_program_form(
        self, whole_program, forward_end_op_index
    ):
        # NOTE(dev): We apply build_strategy for backward firstly to
        # avoid skipping more gc variables.
        backward_start_op_index = forward_end_op_index + len(
            self._outputs.var_ids
        )
        backward_end_op_index = whole_program.desc.block(0).op_size()
        # For Backward process in CINN, all param@GRAD should be skipped for GC, because
        # they will be shared in scope and used by optimizer.
        backward_skip_vars = self._parse_skip_gc_vars(
            whole_program
        ) + self._grad_var_names.get('param', [])
        backward_built_program = add_build_strategy_for(
            whole_program,
            backward_start_op_index,
            backward_end_op_index,
            self._build_strategy,
            backward_skip_vars,
        )

        forward_skip_vars = self._parse_skip_gc_vars(
            whole_program, backward_built_program
        )
        forward_built_program = add_build_strategy_for(
            whole_program,
            0,
            forward_end_op_index,
            self._build_strategy,
            forward_skip_vars,
        )

        self._apply_inplace_pass(forward_built_program, backward_built_program)

        return [forward_built_program, backward_built_program]

    def _apply_inplace_pass(self, forward_program, backward_program):
        attr_types = {
            "use_cuda": "bool",
            "mem_opt_skip_vars": "list[str]",
            "for_partial_block": "bool",
        }
        empty_startup_program = paddle.static.Program()
        use_cuda = True if core.is_compiled_with_cuda() else False
        # skip data var
        forward_mem_opt_skip_vars = self._parse_skip_gc_vars(
            forward_program, backward_program
        )
        backward_mem_opt_skip_vars = self._parse_skip_gc_vars(forward_program)
        if forward_program:
            attrs = {
                "use_cuda": use_cuda,
                "mem_opt_skip_vars": forward_mem_opt_skip_vars,
                "for_partial_block": True,
            }
            # if not (self._in_pir_pt_mode or self._enable_pir_in_executor):
            #     _apply_pass(
            #         forward_program,
            #         empty_startup_program,
            #         "buffer_shared_inplace_pass",
            #         attrs,
            #         attr_types,
            #     )
        if backward_program:
            attrs = {
                "use_cuda": use_cuda,
                "mem_opt_skip_vars": backward_mem_opt_skip_vars,
                "for_partial_block": True,
            }
            # if not (self._in_pir_pt_mode or self._enable_pir_in_executor):
            #     _apply_pass(
            #         backward_program,
            #         empty_startup_program,
            #         "buffer_shared_inplace_pass",
            #         attrs,
            #         attr_types,
            #     )

    @LazyInitialized
    def _inout_var_names(self):
        """
        Returns Variable Names from self._inputs and self.outputs
        """
        var_names = []
        for var in self._inputs:
            if isinstance(var, paddle.base.framework.Variable):
                var_names.append(var.desc.name())
        for var in self._outputs:
            if isinstance(var, paddle.base.framework.Variable):
                var_names.append(var.desc.name())
        return var_names

    def _parse_skip_gc_vars(self, program, backward_program=None):
        """
        Parse variables that need to skip GC after execute it.
        If specify backward_program, it will keep the variables used in backward.
        """
        # skip data var, DO NOT ignore this deepcopy
        skip_vars = deepcopy(self._inout_var_names)
        for var_name, var in program.global_block().vars.items():
            if var.is_data:
                skip_vars.append(var_name)

        if backward_program:
            for var_name in core.parse_safe_eager_deletion_skip_vars(
                backward_program.desc, True
            ):
                skip_vars.append(var_name)
        return skip_vars

    def _prepare_inputs(self, inputs):
        """
        Prepare inputs, outputs, attrs.
        """
        assert isinstance(inputs, (tuple, list))
        # Flatten inputs with nested structure into single list.
        flatten_inputs = paddle.utils.flatten(inputs)
        # Convert variable into Tensor and feed in training data.
        input_vars = []
        input_var_names = []
        expected_place = framework._current_expected_place()
        for i, value in enumerate(flatten_inputs):
            if isinstance(value, np.ndarray):
                var = None
                var = core.eager.Tensor(
                    value=value,
                    name=self._inputs[i].desc.name(),
                    persistable=False,
                    place=expected_place,
                    zero_copy=True,
                )
            elif isinstance(value, core.eager.Tensor):
                # NOTE(Aurelius84): If var is on CPUPlace, it will be transformed multi times
                # into CUDAPlace when it's as input of multi Ops. so we move it in advance
                # to avoid this problem.
                if value.stop_gradient and not value.place._equals(
                    expected_place
                ):
                    var = value._copy_to(expected_place, False)
                    var.stop_gradient = True
                else:
                    var = value
            else:
                continue
            input_var_names.append(self._inputs[i].desc.name())
            input_vars.append(var)

        return input_vars, input_var_names

    def _prepare_outputs(self):
        return paddle.framework.core.create_empty_tensors_with_var_descs(
            self._out_var_descs
        )

    def _create_scope_vec(self, program_id=None, use_scope_cache=False):
        inner_scope = self._get_scope(
            program_id=program_id, use_scope_cache=use_scope_cache
        )
        return [inner_scope]

    def _create_cuda_graph_vec(self):
        var = core.eager.Tensor(
            core.VarDesc.VarType.FP32,
            [],
            "cuda_graph",
            core.VarDesc.VarType.RAW,
            True,
        )
        var.stop_gradient = True
        return var

    def _update_stop_gradient(self, out_vars):
        # Update stop_gradient for all outputs
        def set_stop_gradient(var_id, eager_tensor):
            var = self._outputs[var_id]
            assert isinstance(var, framework.Variable)
            eager_tensor.stop_gradient = var.stop_gradient

        for idx, var in zip(self._outputs.var_ids, out_vars):
            set_stop_gradient(idx, var)

    def _restore_out(self, out_vars):
        """
        Restores same nested outputs by only replacing the Variable with Tensor.
        """

        flatten_outputs = self._outputs.tolist()
        for i, idx in enumerate(self._outputs.var_ids):
            flatten_outputs[idx] = out_vars[i]
        outs = self._outputs.restore(flatten_outputs)
        if outs is not None and len(outs) == 1:
            outs = outs[0]

        return outs

    @switch_to_static_graph
    def _clone_for_test(self, main_program):
        return main_program.clone(for_test=True)

    def _is_no_value(self, var):
        if isinstance(var, core.eager.Tensor) and var.shape == [1]:
            # NOTE: .numpy() will insert MemcpySync operation, it hits performance.
            if var.numpy()[0] == RETURN_NO_VALUE_MAGIC_NUM:
                return True
        return False

    def _remove_no_value(self, out_vars):
        """
        Removes invalid value for various-length return statement
        """
        if isinstance(out_vars, core.eager.Tensor):
            if self._is_no_value(out_vars):
                return None
            return out_vars
        elif isinstance(out_vars, (tuple, list)):
            if isinstance(out_vars, tuple):
                res = tuple(
                    var for var in out_vars if not self._is_no_value(var)
                )
            else:
                # isinstance(out_vars, list)
                res = [var for var in out_vars if not self._is_no_value(var)]

            has_removed = len(out_vars) > len(res)
            # len(out_vars) > len(res) means we have removed var. This is
            # preventing out_vars is empty or just one element at the beginning
            if len(res) == 0 and has_removed:
                return None
            elif len(res) == 1 and has_removed:
                return res[0]
            return res

        return out_vars

    def _set_grad_type(self, params, train_program):
        # NOTE: if user set sparse gradient mode, the param's gradient
        # will be SelectedRows, not DenseTensor. But tracer will just
        # set param grad Tensor by forward Tensor(DenseTensor)
        # If we don't change grad_var type here, RunProgramOp need
        # transform SelectedRows to DenseTensor forcibly, it may not
        # be user wanted result.
        for param in params:
            grad_name = param.name + core.grad_var_suffix()
            grad_var = train_program.desc.block(0).find_var(grad_name.encode())
            # NOTE: cannot find var desc maybe no problem, such as in batch_norm
            if grad_var is None:
                continue
            param._set_grad_type(grad_var.type())

    def _check_params_all_inited(self, main_program):
        """
        Check all params from main program are already initialized, see details as follows:
            1. all parameters in self._params should be type `framework.EagerParamBase` which are created in dygraph.
            2. all parameters from transformed program can be found in self._params.
               Because they share same data with EagerParamBase of original dygraph.
        """
        if not isinstance(self._params, (list, tuple)):
            raise TypeError(
                f"Type of self._params in PartialProgramLayer should be list or tuple, but received {type(self._params)}."
            )

        param_and_buffer_names_set = set()
        for i, var in enumerate(self._params):
            # self._params contains parameters and buffers with persistable=True.
            if not isinstance(var, core.eager.Tensor):
                raise TypeError(
                    f'Type of self._params[{i}] in PartialProgramLayer should be Parameter or Variable, but received {type(var)}.'
                )
            param_and_buffer_names_set.add(var.name)

        for block in main_program.blocks:
            for name, var in block.vars.items():
                if isinstance(var, framework.Parameter):
                    if name not in param_and_buffer_names_set:
                        raise ValueError(
                            "\n\tWe don't support to define layer with parameters in the function decorated by `@to_static`."
                            f"\n\tBut we found parameter({name}) was created in the decorated function."
                            "\n"
                            "\n\tRevise suggestion: "
                            "\n\t\t1. Please ensure all your sublayers are inherited from nn.Layer."
                            "\n\t\t2. Please use nn.ParameterList and nn.LayerList as container instead of using a native Python container such as List"
                        )

    def _valid_vars(self, vars):
        return vars if vars else None


def partial_program_from(concrete_program, from_method=False):
    inputs = concrete_program.inputs

    # NOTE(SigureMo): Remove the first arg `self` from method args.
    if inputs and from_method:
        inputs = inputs[1:]

    return PartialProgramLayer(
        concrete_program.main_program,
        inputs,
        concrete_program.outputs,
        concrete_program.parameters,
        **concrete_program.kwargs,
    )


@switch_to_static_graph
def add_build_strategy_for(
    program, start_op_index, end_op_index, build_strategy=None, skip_vars=None
):
    if start_op_index < end_op_index:
        compiled_program = paddle.static.CompiledProgram(
            core.Graph(program.desc, start_op_index, end_op_index),
            build_strategy=build_strategy,
        )
        if skip_vars:
            # TODO(Aurelius84): Need to unify name with C++, such as kSkipVarNames.
            compiled_program._graph.set("skip_gc_vars", set(skip_vars))
        compiled_program._compile(
            core.Scope(), framework._current_expected_place()
        )
        ir_graph = framework.IrGraph(compiled_program._graph)
        built_program = ir_graph.to_program()
        if hasattr(compiled_program._program, 'lr_scheduler'):
            built_program.lr_scheduler = compiled_program._program.lr_scheduler
    else:
        # can't just create a new program, we need copy the vardesc.
        built_program = paddle.static.Program()
        for var in program.block(0).vars.values():
            built_program.block(0)._clone_variable(var, False)

    # set back the parent_idx of blocks
    for origin, current in zip(program.blocks, built_program.blocks):
        current.desc.set_parent_idx(origin.desc.parent)

    return built_program
