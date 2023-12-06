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

import itertools

import numpy as np

import paddle
import paddle.pir.core as ir_static
from paddle import _legacy_C_ops
from paddle.amp.auto_cast import _in_amp_guard, _in_pure_fp16_guard
from paddle.autograd.ir_backward import grad
from paddle.base import core, framework
from paddle.base.compiler import BuildStrategy
from paddle.base.data_feeder import check_type, convert_dtype
from paddle.base.dygraph.base import switch_to_static_graph
from paddle.optimizer.lr import LRScheduler
from paddle.pir import OpResult, fake_op_result, is_fake_op_result

from .utils import RETURN_NO_VALUE_MAGIC_NUM, backend_guard

__all__ = []


class cached_property:
    """
    Descriptor to implement lazy initialization of property.
    """

    def __init__(self, function):
        self.function = function

    def __get__(self, instance, cls):
        val = self.function(instance)
        setattr(instance, self.function.__name__, val)
        return val


class NestSequence:
    """
    A wrapper class that easily to flatten and restore the nest structure of
    given sequence. It also remove the duplicate variables in the sequence.
    For example:
    >>> t = [v1, v2, v1]
    >>> m = tolist(t)
    [v1, v2]
    >>> m.restore([t1, t2])
    [t1, t2, t1]
    """

    def __init__(self, raw_input):
        self._raw_input = raw_input
        self._var_map, self._var_list = self._tolist()

    @property
    def var_list(self):
        return self._var_list

    def _tolist(self):
        """
        Flattens the nested sequences into single list and remove duplicate variables + non-variable elements.
        """
        variable_map = {}  # opresult -> list idx
        variable_list = []
        for value in paddle.utils.flatten(self._raw_input):
            if not isinstance(value, OpResult):
                continue
            if value in variable_map:
                # remove duplicate opresults.
                continue
            variable_map[value] = len(variable_list)
            variable_list.append(value)
        return variable_map, variable_list

    def restore(self, value_list):
        """
        Restores the nested sequence from value list.
        """
        assert len(self._var_list) == len(value_list)

        def to_value(x):
            if isinstance(x, OpResult):
                return value_list[self._var_map[x]]
            return x

        return paddle.utils.pack_sequence_as(
            self._raw_input,
            list(map(to_value, paddle.utils.flatten(self._raw_input))),
        )

    def __getitem__(self, item):
        return self._var_list[item]


class RunableProgram:
    """a pir program ready for run_program_op to run. constructed by 3 parts:
    - pir program (pir::Program)
    - in_out_values
        - input_x values ([string | pir::OpResult])
        - input_param values ([string | pir::OpResult])
        - output values ([string | pir::OpResult])
    - forward_backward_ranges
        - forward_range (tuple(Int, Int)) | None
        - backward_range (tuple(Int, Int)) | None
    """

    @cached_property
    def get_value_name_map(self):
        return self._get_value_name_map_from_program(self.program)

    @classmethod
    def _get_value_name_map_from_program(cls, program):
        ret = {}
        ret[fake_op_result()] = "FakeVar"
        for op in program.global_block().ops:
            if op.name() == "pd_op.data":
                ret[op.result(0)] = op.attrs()["name"]
            if op.name() == "builtin.set_parameter":
                ret[op.operand(0).source()] = op.attrs()["parameter_name"]
            if op.name() == "builtin.parameter":
                ret[op.result(0)] = op.attrs()["parameter_name"]
        return ret

    @cached_property
    def get_name_value_map(self):
        return {v: k for k, v in self.get_value_name_map.items()}

    def convert_name(self, values):
        if len(values) == 0:
            return []
        if isinstance(values[0], str):
            return values
        return [self.get_value_name_map[v] for v in values]

    @cached_property
    def x_values(self):
        return [self.get_name_value_map[v] for v in self.x_names]

    @cached_property
    def param_values(self):
        return [self.get_name_value_map[v] for v in self.param_names]

    @cached_property
    def out_values(self):
        return [self.get_name_value_map[v] for v in self.out_names]

    @cached_property
    def x_grad_values(self):
        return [self.get_name_value_map[v] for v in self.x_grad_names]

    @cached_property
    def param_grad_values(self):
        return [self.get_name_value_map[v] for v in self.p_grad_names]

    @cached_property
    def out_grad_values(self):
        return [self.get_name_value_map[v] for v in self.o_grad_names]

    def __init__(
        self,
        program,
        in_out_values,
        grad_in_out_values=None,
        forward_range=None,
        backward_range=None,
    ):
        assert isinstance(
            in_out_values, tuple
        ), "in_out_values must be tuple with len == 3"
        assert (
            len(in_out_values) == 3
        ), "in_out_values must be tuple with len == 3"
        assert isinstance(
            in_out_values[0], list
        ), "in_out_values must be tuple with len == 3"
        self.program = program
        self.x_names = self.convert_name(in_out_values[0])
        self.param_names = self.convert_name(in_out_values[1])
        self.out_names = self.convert_name(in_out_values[2])
        self.forward_range = forward_range
        self.backward_range = backward_range
        self.has_splited = False
        self.finish_pass = False
        if self.forward_range is None:
            self.forward_range = (0, len(self.program.global_block().ops))
        if self.backward_range is None:
            self.backward_range = (
                len(self.program.global_block().ops),
                len(self.program.global_block().ops),
            )
        if grad_in_out_values is None:
            grad_in_out_values = [], [], []
        self.x_grad_names = self.convert_name(grad_in_out_values[0])
        self.p_grad_names = self.convert_name(grad_in_out_values[1])
        self.o_grad_names = self.convert_name(grad_in_out_values[2])

    def clone(self):
        cloned_program, _ = paddle.base.libpaddle.pir.clone_program(
            self.program
        )
        return RunableProgram(
            cloned_program,
            (self.x_names, self.param_names, self.out_names),
            None,
            self.forward_range,
            self.backward_range,
        )

    def split_forward_backward(self):
        assert (
            self.has_splited is False
        ), "Please ensure only split once! don't call split_forward_backward manually."
        self.has_splited = True
        [
            fwd_prog,
            bwd_prog,
        ], prog_attr = paddle.base.libpaddle.pir.split_program(
            self.program,
            self.x_values,
            self.param_values,
            self.out_values,
            self.x_grad_values,
            self.param_grad_values,
            self.out_grad_values,
            list(self.forward_range),
            list(self.backward_range),
        )
        return [fwd_prog, bwd_prog], prog_attr

    def apply_pir_program_pass(self, pass_fn):
        """
        Main entries for pass function, without considering any input/output and forward segmentation.
        pass_fn' signature is:

        1. This function will change forward and backward program.
        2. call self.program_attr means start to run.
        so we can't call this function after program_attr is called.

        def pass_fn(forward_program, backward_program):
            return forward_program, backward_program
        """
        program_name_attr = self.program_name_attr
        origin_fwd = self.forward_program
        origin_bwd = self.backward_program
        self.forward_program, self.backward_program = pass_fn(
            self.forward_program, self.backward_program, program_name_attr
        )

    # cached property can ensure program is splited only once.
    @cached_property
    def _forward_backward_program(self):
        return self.split_forward_backward()

    @cached_property  # shouldn't changed when call this once.
    def program_attr(self):
        assert (
            self.finish_pass is False
        ), "program_attr() is called by PartialProgramLayer, don't call it matually, use program_name_attr instead."
        # can't apply pass after call this function.
        self.finish_pass = True
        fwd_map = {
            v: k
            for k, v in self._get_value_name_map_from_program(
                self.forward_program
            ).items()
        }
        bwd_map = {
            v: k
            for k, v in self._get_value_name_map_from_program(
                self.backward_program
            ).items()
        }
        value_program_attr = {}
        for k, ns in self.program_name_attr.items():
            if k.startswith("f"):
                values = [fwd_map[n] for n in ns]
            elif k.startswith("b"):
                values = [bwd_map[n] for n in ns]
            elif k == "no_need_buffers":
                values = [fwd_map[n] for n in ns]
            else:
                raise ValueError(f"Unknown program attr: {k}")
            value_program_attr[k] = values
        return value_program_attr

    @cached_property
    def program_name_attr(self):
        origin_attr = self._forward_backward_program[1]
        fwd_map = self._get_value_name_map_from_program(self.forward_program)
        bwd_map = self._get_value_name_map_from_program(self.backward_program)
        _program_attr = {}
        for k, vs in origin_attr.items():
            if k.startswith("f"):
                names = [fwd_map[v] for v in vs]
            elif k.startswith("b"):
                names = [bwd_map[v] for v in vs]
            elif k == "no_need_buffers":
                names = [fwd_map[v] for v in vs]
            else:
                raise ValueError(f"Unknown program attr: {k}")
            _program_attr[k] = names
        return _program_attr

    @cached_property
    def forward_program(self):
        return self._forward_backward_program[0][0]

    @cached_property
    def backward_program(self):
        return self._forward_backward_program[0][1]


class PirPassContext:
    """
    PirPassContext is a class that only has staticmethod currently.
    It will create a new RunableProgram after calling apply method.
    """

    INPUT_OP_NAME = "pd_op.data"
    PARM_OP_NAME = "builtin.parameter"
    OUTPUT_OP_NAME = "builtin.set_parameter"

    @classmethod
    def apply(cls, runable_program, build_strategy):
        # TODO(Aurelius84): Currently only support infer mode,
        # and we just use forward_program because backward_program
        # is empty.
        if not build_strategy.build_cinn_pass:
            return runable_program
        elif not paddle.is_compiled_with_cinn():
            raise RuntimeError(
                "Please install PaddlePaddle compiled with CINN while setting build_strategy.build_cinn_pass = True."
            )
        fwd_program, _ = paddle.base.libpaddle.pir.clone_program(
            runable_program.forward_program
        )
        paddle.base.libpaddle.pir.apply_pir_pass(fwd_program)
        in_out_values = cls._prepare_attr(fwd_program)
        return RunableProgram(fwd_program, in_out_values)

    @classmethod
    def _prepare_attr(cls, program):
        """
        After applying Pass, we need to update the Input/Parameter/Output Value
        that refer to the new program.

        NOTE: We assume that Inputs come from INPUT_OP, Params come from
              PARM_OP and Output come from OUTPUT_OP.
        """
        inputs, params, outputs = [], [], []
        for op in program.global_block().ops:
            op_name = op.name()
            if op_name == cls.INPUT_OP_NAME:
                inputs.append(op.result(0))
            elif op_name == cls.PARM_OP_NAME:
                params.append(op.result(0))
            elif op_name == cls.OUTPUT_OP_NAME:
                outputs.append(op.operand(0).source())
        return inputs, params, outputs


class PartialProgramLayerHook:
    def before_append_backward(self, forward_program, src_vars):
        ...

    def after_append_backward(
        self, whole_program, src_vars, backward_start_idx
    ):
        ...

    def after_infer(self, infer_program):
        ...


class PartialProgramLayer:
    """
    PartialProgramLayer wraps all the ops from layers decorated by `@to_static`
    and execute them as a static subgraph.

    .. note::
        **1. This is a very low level API. Users should not use this API
             directly. Please use `partial_program_from(concrete_program)`
             to create it.
        **2. TensorArray is not currently supported in the output.

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
        self._outputs = NestSequence(outputs)
        self._params, self._param_values = (
            parameters if parameters is not None else ([], [])
        )

        self._build_strategy = kwargs.get('build_strategy', BuildStrategy())
        assert isinstance(self._build_strategy, BuildStrategy)

        self._origin_main_program = self._verify_program(main_program)
        self._cuda_graph_vec = self._create_cuda_graph_vec()
        self._cuda_graph_capture_mode = ""
        self._cuda_graph_pool_id = 0
        # Set default mode to train
        self.training = True
        self._program_extra_info = {}

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
        self._scope_cache = {}
        self._hooker = None
        self._backend = kwargs.get('backend', None)
        self._grad_var_names = {}
        self._debug_name = None

    def __call__(self, inputs):
        """
        Execute static graph by Interpreter and Return dynamic Tensors.
        """
        in_vars = self._prepare_inputs(inputs)
        out_vars = self._prepare_outputs()
        attrs = self._prepare_attributes()
        _legacy_C_ops.pir_run_program(
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
        return self._remove_no_value(restored_nest_out)

    def sot_call(self, inputs):
        """
        In sot, inputs and outputs of partial program only contain tensors, so we can skip some step to speed up
        """
        return self.__call__(inputs)
        out_vars = self._prepare_outputs()
        attrs = self._prepare_attributes()
        _legacy_C_ops.pir_run_program(
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

    @cached_property
    def origin_runable_program(self):
        inputs = list(self._inputs.var_list)
        outputs = list(self._outputs.var_list)
        params = self._param_values
        paddle.base.libpaddle.pir.append_set_parameters(
            self._origin_main_program,
            outputs,
            len(self._origin_main_program.global_block().ops),
            "output_",
        )
        return RunableProgram(
            self._origin_main_program, (inputs, params, outputs)
        )

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
        if not use_scope_cache:
            return core.Scope()
        if program_id not in self._scope_cache:
            self._scope_cache[program_id] = []
        cached_scopes = self._scope_cache[program_id]
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
            # TODO(xiongkun) who to transfer the pruning program?
            infer_program = self.origin_runable_program.clone()
            if self._hooker:
                self._hooker.after_infer(infer_program)
            infer_program = PirPassContext.apply(
                infer_program, self._build_strategy
            )
            return infer_program
        else:
            train_program: RunableProgram = self.origin_runable_program.clone()
            train_program = self._append_backward_desc(train_program)
            # Note: Only set grad type once after initializing train program. So we put it here.
            self._set_grad_type(self._params, train_program)

            # (NOTE:@xiongkun) HOW TO APPLY PASS: this is a example for forward/backward clone pass, just replace with your cases.
            def pass_fn(forward_program, backward_program, name_attr):
                fwd, _ = paddle.base.libpaddle.pir.clone_program(
                    forward_program
                )

                if self._build_strategy.build_cinn_pass:
                    paddle.base.libpaddle.pir.apply_pir_pass(fwd)

                bwd, _ = paddle.base.libpaddle.pir.clone_program(
                    backward_program
                )

                if self._build_strategy.build_cinn_pass:
                    paddle.base.libpaddle.pir.apply_pir_pass(bwd)

                return fwd, bwd

            train_program.apply_pir_program_pass(pass_fn)
            return train_program

    @cached_property
    def _train_program_id(self):
        program_id = paddle.utils._hash_with_id(self.train_program, self)
        core._set_cached_executor_build_strategy(
            program_id, self._build_strategy
        )
        return program_id

    @cached_property
    def _infer_program_id(self):
        return paddle.utils._hash_with_id(self.infer_program, self)

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
        if _in_amp_guard() or _in_pure_fp16_guard():
            raise NotImplementedError("not implement error.")
        if self.training:
            return self._train_program_id
        else:
            return self._infer_program_id

    @cached_property
    def train_program(self):
        if _in_amp_guard() or _in_pure_fp16_guard():
            raise NotImplementedError("not implement error.")
        return self._create_program()

    @cached_property
    def infer_program(self):
        if _in_amp_guard() or _in_pure_fp16_guard():
            raise NotImplementedError("not implement error.")
        return self._create_program(is_infer_mode=True)

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
        loss.backward()  # <----- x@grad will be overwrited by elementwise_add_grad Op
        """

        def _need_aggregation(var):
            """
            if exist a op whose inputs is var, then return True
            """
            if var.type not in [
                core.VarDesc.VarType.LOD_TENSOR,
                core.VarDesc.VarType.SELECTED_ROWS,
            ]:
                return False
            if var.dtype not in [paddle.float32, paddle.float64]:
                return False
            for op in main_program.global_block().ops:
                for in_arg in op.input_arg_names:
                    if in_arg == var.name:
                        return True
            return False

        def _insert_aggregation_ops_for_var(target_program, var):
            suffix = "@dy2static"
            var_grad_name = var.grad_name
            new_grad_name = var.name + suffix + "@GRAD"
            finded_ops = list(
                filter(
                    lambda x: x[0] >= start_idx
                    and any(
                        out_arg == var_grad_name
                        for out_arg in x[1].output_arg_names
                    ),
                    enumerate(target_program.global_block().ops),
                )
            )

            # len(finded_ops) may equals zero when stop_gradient works.
            # len(finded_ops) may > 1, because we may have fill_constant op.
            if len(finded_ops) == 0:
                return None
            # step1: create a new var named var.name@GRAD
            target_program.global_block().create_var(
                name=new_grad_name,
                type=var.type,
                dtype=var.dtype,
                shape=var.shape,
            )
            # step2: rename the var.name@GRAD to var.name@GRAD@dy2static
            for idx, op in finded_ops:
                op._rename_input(var_grad_name, new_grad_name)
                op._rename_output(var_grad_name, new_grad_name)
            # step3: insert sum op to aggregate the gradient.
            #        var.name@GRAD = sum(var.name@dy2static@GRAD, var.name@GRAD)
            target_program.global_block()._insert_op(
                finded_ops[-1][0] + 1,
                type='sum',
                inputs={'X': [var_grad_name, new_grad_name]},
                outputs={"Out": var_grad_name},
            )
            return None

        to_processed_vars = list(
            filter(_need_aggregation, self._outputs.var_list)
        )
        for _var in to_processed_vars:
            _insert_aggregation_ops_for_var(target_program, _var)

    @switch_to_static_graph
    def _append_backward_desc(self, train_runnable_program: RunableProgram):
        program = train_runnable_program.program
        targets = train_runnable_program.out_values
        # TODO(@zhuoge): refine the interface, use runable_program to apply passes.
        if self._hooker:
            program, targets = self._hooker.before_append_backward(
                program, targets
            )
        inputs = train_runnable_program.x_values
        params = train_runnable_program.param_values
        combined_inputs = list(itertools.chain(inputs, params))
        forward_end_idx = len(program.global_block().ops)
        grad_info_map = [None] * len(combined_inputs)
        with backend_guard(self._backend):
            check_type(
                targets,
                'targets',
                (OpResult, list, tuple),
                'paddle.static.gradients',
            )
            with ir_static.program_guard(program, None):
                # create outputs_grad for backward to avoid full and full_like op.
                forward_outputs_grads = []
                for out_op_result in targets:
                    if out_op_result.stop_gradient is True:
                        forward_outputs_grads.append(fake_op_result())
                    else:
                        value = paddle.full_like(
                            out_op_result,
                            fill_value=1.0,
                            dtype=out_op_result.dtype,
                        )
                        forward_outputs_grads.append(value)
                paddle.base.libpaddle.pir.append_set_parameters(
                    program,
                    forward_outputs_grads,
                    len(program.global_block().ops),
                    "grad_input_",
                )
                op_between_forward_and_backward = (
                    len(program.global_block().ops) - forward_end_idx
                )

                # call grad to get backward ops.
                if (
                    len(
                        list(
                            filter(lambda x: x.stop_gradient is False, targets)
                        )
                    )
                    > 0
                ):
                    grad_info_map = grad(
                        inputs=combined_inputs,
                        outputs=list(
                            filter(lambda x: x.stop_gradient is False, targets)
                        ),
                        grad_outputs=list(
                            filter(
                                lambda x: not is_fake_op_result(x),
                                forward_outputs_grads,
                            )
                        ),
                    )

            if self._hooker:
                (
                    program,
                    forward_end_idx,
                    targets,
                ) = self._hooker.after_append_backward(
                    program, targets, forward_end_idx
                )
            # TODO: add later
            # self.prepare_gradient_aggregation(
            # start_idx + 1, main_program, program
            # )

        mapping_op_result = (
            lambda x: x if isinstance(x, OpResult) else fake_op_result()
        )
        inputs_size = len(inputs)
        x_grad_value = list(
            map(mapping_op_result, grad_info_map[0:inputs_size])
        )
        p_grad_value = list(map(mapping_op_result, grad_info_map[inputs_size:]))
        o_grad_value = list(map(mapping_op_result, forward_outputs_grads))

        # insert grads name for RunableProgram (we need name for grad_inputs and grad_outputs)
        input_grads_to_append = list(
            filter(lambda x: not is_fake_op_result(x), o_grad_value)
        )
        output_grads_to_append = list(
            filter(
                lambda x: not is_fake_op_result(x), x_grad_value + p_grad_value
            )
        )
        backward_end_op_index = len(program.global_block().ops)
        paddle.base.libpaddle.pir.append_set_parameters(
            program,
            output_grads_to_append,
            backward_end_op_index,
            "grad_output_",
        )

        backward_start_op_index = (
            forward_end_idx + op_between_forward_and_backward
        )
        # construct a runnable program.
        return RunableProgram(
            program,
            (inputs, params, targets),
            (x_grad_value, p_grad_value, o_grad_value),
            (0, forward_end_idx),
            (backward_start_op_index, backward_end_op_index),
        )

    def _prune_unused_params(self, program):
        """
        Prune the parameters not used anywhere in the program.
        The `@to_static` may only decorated a sub function which
        contains some unused parameters created in `__init__`.
        So prune these parameters to avoid unnecessary operations in
        `run_program_op`.
        """
        required_params = []
        required_param_values = []
        block = program.global_block()
        for param, param_value in zip(self._params, self._param_values):
            if not param_value.use_empty():
                required_params.append(param)
                required_param_values.append(param_value)
            else:
                # in pir, we need remove the get_parameter op for unused parameters.
                block.remove_op(param_value.get_defining_op())
        self._params = required_params
        self._param_values = required_param_values

    def _prepare_attributes(self):
        attrs = [
            'forward_global_block',
            self.program.forward_program.global_block(),
            'backward_global_block',
            self.program.backward_program.global_block(),
            'is_test',
            not self.training,
            'program_id',
            self.program_id,
        ]
        for key, val in self.program.program_attr.items():
            attrs.append(key)
            attrs.append(val)

        if self._cuda_graph_capture_mode:
            attrs.extend(
                (
                    'cuda_graph_capture_mode',
                    self._cuda_graph_capture_mode,
                    'cuda_graph_pool_id',
                    self._cuda_graph_pool_id,
                )
            )
        return attrs

    def _prepare_inputs(self, inputs):
        """
        Prepare inputs, outputs, attrs.
        """
        assert isinstance(inputs, (tuple, list))
        # Flatten inputs with nested structure into single list.
        flatten_inputs = paddle.utils.flatten(inputs)
        # Convert variable into Tensor and feed in training data.
        input_vars = []
        expected_place = framework._current_expected_place()
        for i, value in enumerate(flatten_inputs):
            if isinstance(value, np.ndarray):
                var = None
                var = core.eager.Tensor(
                    value=value,
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
            input_vars.append(var)
        return input_vars

    def _prepare_outputs(self):
        return paddle.framework.core.create_empty_tensors_with_op_results(
            self._outputs.var_list
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
        def set_stop_gradient(var, eager_tensor):
            assert isinstance(var, OpResult)
            eager_tensor.stop_gradient = var.stop_gradient

        for idx, var in zip(self._outputs.var_list, out_vars):
            set_stop_gradient(idx, var)

    def _restore_out(self, out_vars):
        """
        Restores same nested outputs by only replacing the Variable with Tensor.
        """
        outs = self._outputs.restore(out_vars)
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

    def _set_grad_type(self, params, train_program: RunableProgram):
        # NOTE: if user set sparse gradient mode, the param's gradient
        # will be SelectedRows, not LoDTensor. But tracer will just
        # set param grad Tensor by forward Tensor(LoDTensor)
        # If we don't change grad_var type here, RunProgramOp need
        # transform SelectedRows to LoDTensor forcibly, it may not
        # be user wanted result.
        forward_params_grads = train_program.param_grad_values
        train_program = train_program.program
        for param, value in zip(params, forward_params_grads):
            if is_fake_op_result(value):
                continue
            if value.is_selected_row_type():
                param._set_grad_type(
                    paddle.base.core.VarDesc.VarType.SELECTED_ROWS
                )
            elif value.is_dense_tensor_type():
                param._set_grad_type(
                    paddle.base.core.VarDesc.VarType.LOD_TENSOR
                )
            else:
                raise NotImplementedError(
                    "only support selected_row and dense_tensor grad type."
                )

    def _check_params_all_inited(self, main_program):
        """
        Check all params from main program are already initialized, see details as follows:
            1. all parameters in self._params should be type `framework.EagerParamBase` which are created in dygraph.
            2. all parameters from transformed program can be found in self._params.
               Because they share same data with EagerParamBase of original dygraph.
        """
        if not isinstance(self._params, (list, tuple)):
            raise TypeError(
                "Type of self._params in PartialProgramLayer should be list or tuple, but received %s."
                % type(self._params)
            )

        param_and_buffer_names_set = set()
        for i, var in enumerate(self._params):
            # self._params constains parameters and buffers with persistable=True.
            if not isinstance(var, core.eager.Tensor):
                raise TypeError(
                    'Type of self._params[{}] in PartialProgramLayer should be Parameter or Variable, but received {}.'.format(
                        i, type(var)
                    )
                )
            param_and_buffer_names_set.add(var.name)

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
