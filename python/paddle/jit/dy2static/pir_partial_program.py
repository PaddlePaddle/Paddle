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
from functools import cached_property

import numpy as np

import paddle
import paddle.pir.core as ir_static
from paddle import _legacy_C_ops
from paddle.autograd.backward_utils import ValueDict
from paddle.autograd.ir_backward import grad
from paddle.base import core, framework
from paddle.base.compiler import BuildStrategy
from paddle.base.data_feeder import check_type, convert_dtype
from paddle.base.dygraph.base import switch_to_static_graph
from paddle.optimizer.lr import LRScheduler
from paddle.pir import Value, fake_value, is_fake_value

from .logging_utils import TranslatorLogger
from .utils import (
    RETURN_NO_VALUE_MAGIC_NUM,
    backend_guard,
    cinn_is_enabled,
    cse_is_enabled,
)

__all__ = []

prog_logger = TranslatorLogger()


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
        variable_map = ValueDict()  # value -> list idx
        variable_list = []
        for value in paddle.utils.flatten(self._raw_input):
            if not isinstance(value, Value):
                continue
            if value in variable_map:
                # remove duplicate values.
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
            if isinstance(x, Value):
                return value_list[self._var_map[x]]
            return x

        return paddle.utils.pack_sequence_as(
            self._raw_input,
            list(map(to_value, paddle.utils.flatten(self._raw_input))),
        )

    def __getitem__(self, item):
        return self._var_list[item]


class UnionFindSet:
    def __init__(self):
        self.father = ValueDict()

    def union(self, x, y):
        # x -> y
        father_x = self.find_root(x)
        father_y = self.find_root(y)
        if not (father_x.is_same(father_y)):
            self.father[father_x] = father_y

    def find_root(self, x):
        if x not in self.father:
            self.father[x] = x
        if self.father[x].is_same(x):
            return x
        self.father[x] = self.find_root(self.father[x])
        return self.father[x]

    def iter_elements(self):
        yield from self.father.keys()


class RunnableProgram:
    """a pir program ready for run_program_op to run. constructed by 3 parts:
    - pir program (pir::Program)
    - in_out_values
        - input_x values ([string | pir::Value])
        - input_param values ([string | pir::Value])
        - output values ([string | pir::Value])
    - forward_backward_ranges
        - forward_range (tuple(Int, Int)) | None
        - backward_range (tuple(Int, Int)) | None
    """

    @cached_property
    def get_value_name_map(self):
        return self._get_value_name_map_from_program(self.program)

    @classmethod
    def _get_value_name_map_from_program(cls, program):
        ret = ValueDict()
        ret[fake_value()] = "FakeVar"
        for keyword, arg in program.global_block().kwargs().items():
            ret[arg] = keyword
        for op in program.global_block().ops:
            if op.name() == "builtin.set_parameter":
                ret[op.operand(0).source()] = op.attrs()["parameter_name"]
            elif op.name() == "builtin.parameter":
                ret[op.result(0)] = op.attrs()["parameter_name"]
            elif op.name() == "builtin.shadow_output":
                ret[op.operand(0).source()] = op.attrs()["output_name"]
            elif op.name() == "pd_op.data":
                ret[op.result(0)] = op.attrs()["name"]
        return ret

    @classmethod
    def _get_name_defining_op(cls, program, value):
        for op in program.global_block().ops:
            if op.name() == "builtin.set_parameter":
                if value.is_same(op.operand(0).source()):
                    return op
            elif op.name() == "builtin.parameter":
                if value.is_same(op.result(0)):
                    return op
            elif op.name() == "builtin.shadow_output":
                if value.is_same(op.operand(0).source()):
                    return op
            elif op.name() == "pd_op.data":
                if value.is_same(op.result(0)):
                    return op
        return None

    @cached_property
    def get_name_value_map(self):
        return {v: k for k, v in self.get_value_name_map.items()}

    def convert_name(self, values):
        if len(values) == 0:
            return []
        if isinstance(values[0], str):
            return values
        return [self.get_value_name_map.get(v, "FakeVar") for v in values]

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
        return RunnableProgram(
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
        origin_fwd = self.forward_program
        origin_bwd = self.backward_program

        prog_logger.log(
            1,
            f"******** [JIT] PIR forward program before PIR PASS ********\n{origin_fwd} ",
        )
        prog_logger.log(
            1,
            f"******** [JIT] PIR backward program before PIR PASS ********\n{origin_bwd} ",
        )
        # NOTE(dev): Add this line to trigger program_name_attr logic
        program_name_attr = self.program_name_attr
        self.forward_program, self.backward_program = pass_fn(
            origin_fwd, origin_bwd
        )
        prog_logger.log(
            1,
            f"******** [JIT] PIR forward program after PIR PASS ********\n{origin_fwd} ",
        )
        prog_logger.log(
            1,
            f"******** [JIT] PIR backward program after PIR PASS ********\n{origin_bwd} ",
        )

    # cached property can ensure program is splited only once.
    @cached_property
    def _forward_backward_program(self):
        return self.split_forward_backward()

    @cached_property  # shouldn't changed when call this once.
    def program_attr(self):
        assert (
            self.finish_pass is False
        ), "program_attr() is called by PartialProgramLayer, don't call it manually, use program_name_attr instead."
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
                values = [fwd_map.get(n, fake_value()) for n in ns]
            elif k.startswith("b"):
                values = [bwd_map.get(n, fake_value()) for n in ns]
            elif k == "no_need_buffers":
                values = [fwd_map.get(n, fake_value()) for n in ns]
            else:
                raise ValueError(f"Unknown program attr: {k}")
            value_program_attr[k] = values
        self.deal_inplace_values(self.forward_program, self.backward_program)
        self.deal_inplace_values(self.backward_program)
        return value_program_attr

    def deal_inplace_values(self, program1, program2=None):
        # deal inplace op and modify program1 inplacely.
        value2name = self._get_value_name_map_from_program(program1)

        def has_name(value):
            if self._get_name_defining_op(program1, value) is not None:
                return True
            return False

        ufset = UnionFindSet()
        for op in program1.global_block().ops:
            for out_idx, in_idx in paddle.core.pir.get_op_inplace_info(
                op
            ).items():
                left = op.result(out_idx)
                right = op.operand(in_idx).source()
                if has_name(left):
                    ufset.union(right, left)
                else:
                    ufset.union(left, right)

        for value in ufset.iter_elements():
            if has_name(ufset.find_root(value)):
                name_defining_op = self._get_name_defining_op(program1, value)
                if (
                    name_defining_op
                    and name_defining_op.name() == 'builtin.shadow_output'
                ):
                    old_name = name_defining_op.attrs()['output_name']
                    new_name = value2name[ufset.find_root(value)]
                    if old_name == new_name:
                        continue
                    paddle.core.pir.reset_shadow_output_name(
                        name_defining_op, new_name
                    )
                    if program2 is None:
                        continue
                    block = program2.global_block()
                    kwargs = block.kwargs()
                    if old_name in kwargs:
                        if new_name not in kwargs:
                            new_value = block.add_kwarg(
                                new_name, kwargs[old_name].type()
                            )
                        else:
                            new_value = kwargs[new_name]
                        kwargs[old_name].replace_all_uses_with(new_value)
                        block.erase_kwarg(old_name)

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


class PartialProgramLayerHook:
    def before_append_backward(self, forward_program, src_vars):
        return forward_program, src_vars

    def after_append_backward(
        self,
        whole_program,
        inputs,
        src_vars,
        grad_outputs,
        forward_end_idx,
        backward_start_idx,
    ):
        return whole_program, forward_end_idx, src_vars

    def after_infer(self, infer_program):
        return infer_program


class OperatorIndexPreservePass:
    OP_NAME_PREFIX = "preserved_index_"
    counter = 0

    def __init__(self, index, pass_fn):
        self.name = f"{OperatorIndexPreservePass.OP_NAME_PREFIX}{OperatorIndexPreservePass.counter}"
        OperatorIndexPreservePass.counter += 1
        self.pass_fn = pass_fn
        self.index = index

    def __call__(self, program):
        if len(program.global_block().ops) == 0:
            assert self.index == 0
            return self.pass_fn(program)
        paddle.base.libpaddle.pir.append_shadow_output(
            program,
            program.global_block().ops[0].result(0),
            self.name,
            self.index,
        )
        program = self.pass_fn(program)
        new_index = 0
        for op in program.global_block().ops:
            if (
                op.name() == "builtin.shadow_output"
                and self.name in op.attrs()["output_name"]
            ):
                break
            new_index += 1
        # remove forward_backward_seperator
        if new_index >= len(program.global_block().ops):
            raise RuntimeError(
                f"Can't find index preserve label {self.name}, don't remove it in pass."
            )
        program.global_block().remove_op(program.global_block().ops[new_index])
        self.index = new_index
        return program


class IndicesPreservePass:
    def __init__(self, indices, pass_fn):
        self.pass_fn = pass_fn
        self.indices = indices
        self.new_indices = None

    def __call__(self, program):
        passes = [self.pass_fn]
        for idx, index in enumerate(self.indices):
            passes.append(OperatorIndexPreservePass(index, passes[idx]))
        new_program = passes[-1](program)

        self.new_indices = [p.index for p in passes[1:]]
        return new_program


class ValuePreservePass:
    OP_NAME_PREFIX = "preserved_value_"

    def __init__(self, values):
        self.values = values

    def apply(self, program):
        raise RuntimeError("Not implemented.")

    def __call__(self, program):
        # create fake values for args
        all_values = list(
            filter(
                lambda x: isinstance(x, Value) and not is_fake_value(x),
                paddle.utils.flatten(self.values),
            )
        )

        value2name = ValueDict()
        for idx, v in enumerate(all_values):
            name = f"{ValuePreservePass.OP_NAME_PREFIX}{idx}"
            if v in value2name:
                continue
            value2name[v] = name
            paddle.base.libpaddle.pir.append_shadow_output(
                program,
                v,
                name,
                len(program.global_block().ops),
            )

        # apply program pass
        program = self.apply(program)

        # collect new value
        name2new_value = {}
        to_remove_op = []
        for op in program.global_block().ops:
            if op.name() == "builtin.shadow_output":
                if op.attrs()["output_name"].startswith(
                    ValuePreservePass.OP_NAME_PREFIX
                ):
                    name2new_value[op.attrs()["output_name"]] = op.operand(
                        0
                    ).source()
                    to_remove_op.append(op)

        # remove old op
        for op in to_remove_op:
            program.global_block().remove_op(op)

        # get new values
        value2new_value = ValueDict(
            {
                v: name2new_value.get(name, fake_value())
                for v, name in value2name.items()
            }
        )

        new_args = paddle.utils.map_structure(
            lambda x: (
                value2new_value[x] if not is_fake_value(x) else fake_value()
            ),
            self.values,
        )
        self.values = new_args
        return program


class FusedBnAddActPass(ValuePreservePass):
    def apply(self, program):
        program = paddle.base.libpaddle.pir.apply_bn_add_act_pass(program)
        return program


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
        if parameters is not None:
            parameters[0][:] = self._params
            parameters[1][:] = self._param_values
        with paddle.base.framework._dygraph_guard(paddle.base.dygraph.Tracer()):
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
        self._hookers = []
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
        restored_nest_out = self._restore_out(out_vars)
        return restored_nest_out

    @cached_property
    def origin_runnable_program(self):
        inputs = list(self._inputs.var_list)
        outputs = list(self._outputs.var_list)
        params = self._param_values
        paddle.base.libpaddle.pir.append_shadow_outputs(
            self._origin_main_program,
            outputs,
            len(self._origin_main_program.global_block().ops),
            "output_",
        )
        return RunnableProgram(
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

    def add_hooker(self, hooker):
        self._hookers.append(hooker)

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

            def pass_fn(forward_program, backward_program):
                # common pass
                pm = paddle.base.libpaddle.pir.PassManager()
                paddle.base.libpaddle.pir.infer_symbolic_shape_pass(
                    pm, forward_program
                )
                pm.run(forward_program)
                if cse_is_enabled():
                    paddle.base.libpaddle.pir.apply_cse_pass(forward_program)

                # if-else pass
                if cinn_is_enabled(self._build_strategy, self._backend):
                    paddle.base.libpaddle.pir.apply_cinn_pass(forward_program)
                else:
                    paddle.base.libpaddle.pir.check_infer_symbolic_if_need(
                        forward_program
                    )

                return forward_program, backward_program

            # TODO(xiongkun) who to transfer the pruning program?
            infer_program = self.origin_runnable_program.clone()
            for hooker in self._hookers:
                hooker.after_infer(infer_program)
            infer_program.apply_pir_program_pass(pass_fn)
            return infer_program
        else:
            train_program: RunnableProgram = (
                self.origin_runnable_program.clone()
            )
            train_program = self._append_backward_desc(train_program)
            # Note: Only set grad type once after initializing train program. So we put it here.
            self._set_grad_type(self._params, train_program)

            def pass_fn(forward_program, backward_program):
                if cse_is_enabled():
                    paddle.base.libpaddle.pir.apply_cse_pass(forward_program)
                    paddle.base.libpaddle.pir.apply_cse_pass(backward_program)
                if cinn_is_enabled(self._build_strategy, self._backend):
                    paddle.base.libpaddle.pir.apply_cinn_pass(forward_program)
                    paddle.base.libpaddle.pir.apply_cinn_pass(backward_program)
                else:
                    paddle.base.libpaddle.pir.check_infer_symbolic_if_need(
                        forward_program
                    )
                return forward_program, backward_program

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
        if self.training:
            return self._train_program_id
        else:
            return self._infer_program_id

    @cached_property
    def train_program(self):
        return self._create_program()

    @cached_property
    def infer_program(self):
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
        loss.backward()  # <----- x@grad will be overwritten by elementwise_add_grad Op
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
            found_ops = list(
                filter(
                    lambda x: x[0] >= start_idx
                    and any(
                        out_arg == var_grad_name
                        for out_arg in x[1].output_arg_names
                    ),
                    enumerate(target_program.global_block().ops),
                )
            )

            # len(found_ops) may equals zero when stop_gradient works.
            # len(found_ops) may > 1, because we may have fill_constant op.
            if len(found_ops) == 0:
                return None
            # step1: create a new var named var.name@GRAD
            target_program.global_block().create_var(
                name=new_grad_name,
                type=var.type,
                dtype=var.dtype,
                shape=var.shape,
            )
            # step2: rename the var.name@GRAD to var.name@GRAD@dy2static
            for _, op in found_ops:
                op._rename_input(var_grad_name, new_grad_name)
                op._rename_output(var_grad_name, new_grad_name)
            # step3: insert sum op to aggregate the gradient.
            #        var.name@GRAD = sum(var.name@dy2static@GRAD, var.name@GRAD)
            target_program.global_block()._insert_op(
                found_ops[-1][0] + 1,
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
    def _append_backward_desc(self, train_runnable_program: RunnableProgram):
        program = train_runnable_program.program
        targets = train_runnable_program.out_values
        # TODO(@zhuoge): refine the interface, use runnable_program to apply passes.
        for hooker in self._hookers:
            program, targets = hooker.before_append_backward(program, targets)
        inputs = train_runnable_program.x_values
        params = train_runnable_program.param_values
        combined_inputs = list(itertools.chain(inputs, params))
        forward_end_idx = len(program.global_block().ops)
        forward_end_op = None
        if forward_end_idx > 0:
            forward_end_op = program.global_block().ops[-1]
        grad_info_map = [None] * len(combined_inputs)
        with backend_guard(self._backend):
            check_type(
                targets,
                'targets',
                (Value, list, tuple),
                'paddle.static.gradients',
            )
            with ir_static.program_guard(program, None):
                # create outputs_grad for backward to avoid full and full_like op.
                forward_outputs_grads = []
                for out_value in targets:
                    if out_value.stop_gradient is True:
                        forward_outputs_grads.append(fake_value())
                    else:
                        value = paddle.full_like(
                            out_value,
                            fill_value=1.0,
                            dtype=out_value.dtype,
                        )
                        forward_outputs_grads.append(value)
                paddle.base.libpaddle.pir.append_shadow_outputs(
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
                                lambda x: not is_fake_value(x),
                                forward_outputs_grads,
                            )
                        ),
                    )
                    if forward_end_op is not None:
                        for idx, op in enumerate(program.global_block().ops):
                            if op == forward_end_op:
                                forward_end_idx = idx + 1
                                break

            for hooker in self._hookers:
                (
                    program,
                    forward_end_idx,
                    targets,
                ) = hooker.after_append_backward(
                    program,
                    combined_inputs,
                    targets,
                    forward_outputs_grads,
                    forward_end_idx,
                    forward_end_idx + op_between_forward_and_backward,
                )
            # TODO: add later
            # self.prepare_gradient_aggregation(
            # start_idx + 1, main_program, program
            # )

        mapping_value = lambda x: x if isinstance(x, Value) else fake_value()
        inputs_size = len(inputs)
        x_grad_value = list(map(mapping_value, grad_info_map[0:inputs_size]))
        p_grad_value = list(map(mapping_value, grad_info_map[inputs_size:]))
        o_grad_value = list(map(mapping_value, forward_outputs_grads))

        # insert grads name for RunnableProgram (we need name for grad_inputs and grad_outputs)
        input_grads_to_append = list(
            filter(lambda x: not is_fake_value(x), o_grad_value)
        )
        output_grads_to_append = list(
            filter(lambda x: not is_fake_value(x), x_grad_value + p_grad_value)
        )
        backward_end_op_index = len(program.global_block().ops)
        paddle.base.libpaddle.pir.append_shadow_outputs(
            program,
            output_grads_to_append,
            backward_end_op_index,
            "grad_output_",
        )

        backward_start_op_index = (
            forward_end_idx + op_between_forward_and_backward
        )

        # construct a runnable program.
        fused_bn_add_act_pass = FusedBnAddActPass(
            [inputs, params, targets, x_grad_value, p_grad_value, o_grad_value]
        )
        forward_index_pass = IndicesPreservePass(
            [forward_end_idx, backward_start_op_index, backward_end_op_index],
            fused_bn_add_act_pass,
        )
        program = forward_index_pass(program)
        (
            inputs,
            params,
            targets,
            x_grad_value,
            p_grad_value,
            o_grad_value,
        ) = fused_bn_add_act_pass.values
        (
            forward_end_idx,
            backward_start_op_index,
            backward_end_op_index,
        ) = forward_index_pass.new_indices

        return RunnableProgram(
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
            'forward_program',
            self.program.forward_program,
            'backward_program',
            self.program.backward_program,
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
        return paddle.framework.core.create_empty_tensors_with_values(
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
            assert isinstance(var, Value)
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

    def _set_grad_type(self, params, train_program: RunnableProgram):
        # NOTE: if user set sparse gradient mode, the param's gradient
        # will be SelectedRows, not LoDTensor. But tracer will just
        # set param grad Tensor by forward Tensor(LoDTensor)
        # If we don't change grad_var type here, RunProgramOp need
        # transform SelectedRows to LoDTensor forcibly, it may not
        # be user wanted result.
        forward_params_grads = train_program.param_grad_values
        train_program = train_program.program
        for param, value in zip(params, forward_params_grads):
            if is_fake_value(value):
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
