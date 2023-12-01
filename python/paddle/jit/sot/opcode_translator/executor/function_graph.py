# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

# This file is specifically used to handle the problem
# of generating a Graph from a linear function call.

from __future__ import annotations

import builtins
import inspect
from collections import namedtuple
from copy import deepcopy
from functools import cached_property
from typing import Any, Callable

from ...infer_meta import InferMetaCache, LayerInferMetaCache, MetaInfo
from ...profiler import EventGuard, event_register
from ...symbolic.statement_ir import Reference, Symbol
from ...symbolic.symbolic_context import SymbolicTraceContext
from ...utils import (
    ENV_SHOW_TRACKERS,
    NameGenerator,
    OrderedSet,
    inner_error_default_handler,
    is_inplace_api,
    is_paddle_api,
    log,
    log_do,
    map_if,
    tmp_name_guard,
)
from ..instruction_utils import get_instructions
from .guard import Guard, StringifyExpression, make_guard
from .mutable_data import MutationDel, MutationNew, MutationSet
from .pycode_generator import PyCodeGen
from .side_effects import (
    DictSideEffectRestorer,
    GlobalDelSideEffectRestorer,
    GlobalSetSideEffectRestorer,
    ListSideEffectRestorer,
    ObjDelSideEffectRestorer,
    ObjSetSideEffectRestorer,
    SideEffectRestorer,
    SideEffects,
)
from .tracker import BuiltinTracker, DummyTracker
from .variables import (
    DictVariable,
    GlobalVariable,
    ListVariable,
    NullVariable,
    PaddleLayerVariable,
    TensorVariable,
    VariableBase,
    VariableFactory,
    find_traceable_vars,
    map_variables,
)


def convert_to_meta(inputs: Any):
    """
    Convert the input variables to meta if it is TensorVariable.
    """

    def func(x):
        if isinstance(x, TensorVariable):
            return x.meta
        if isinstance(x, VariableBase):
            return x.get_py_value()
        return x

    return map_variables(func, inputs)


def convert_to_symbol(inputs: Any):
    """
    Convert the input variables to symbol if it can be symbolic.
    """

    def func(x):
        if isinstance(x, (TensorVariable, PaddleLayerVariable)):
            return x.get_symbol()
        if isinstance(x, VariableBase):
            return x.get_py_value()
        return x

    return map_variables(func, inputs)


class FunctionGraph:
    """
    A Graph representation corresponding to each FunctionFrame
    The input binding diagram containing the current call represents three parts of output settings,
    This Graph can be compiled as a f_locals dependency function which produce the same outputs.
    """

    OUT_VAR_PREFIX = "___SIR_out_"
    Memo = namedtuple(
        "function_graph_memo",
        [
            'inner_out',
            'input_variables',
            "stmt_ir",
            "global_guards",
            "side_effects_state",
            "print_variables",
            "inplace_tensors",
        ],
    )

    def __init__(self, frame, **kwargs):
        self.sir_ctx = SymbolicTraceContext()
        self.inner_out = set()
        self.input_variables = []  # Store variables required within a function
        self.pycode_gen = PyCodeGen(frame, disable_eval_frame=True)
        self.side_effects = SideEffects()
        self._global_guarded_variables: OrderedSet[VariableBase] = OrderedSet()
        self._print_variables = []
        self._inplace_tensors = OrderedSet()
        self.build_strategy = kwargs.get('build_strategy', None)
        self._kwargs = kwargs

    @cached_property
    def _builtins(self):
        builtins_ = {}
        # prepare builtins
        for name, value in builtins.__dict__.items():
            builtins_[name] = VariableFactory.from_value(
                value, self, BuiltinTracker(name), debug_name=name
            )
        return builtins_

    def add_print_variables(self, variable):
        """
        Used to support psdb_print
        """
        self._print_variables.append(variable)

    def add_inplace_tensors(self, variable):
        """
        Used to support psdb_print
        """
        self._inplace_tensors.add(variable)

    def need_add_input(self, var):
        """
        Determine if it is the input of graph.

        Args:
            var: The input variable.

        """
        if var.id in self.inner_out:
            return False
        for v in self.input_variables:
            if v.id == var.id:
                return False
        return True

    def save_memo(self) -> FunctionGraph.Memo:
        """
        Save the state of the current FunctionGraph, for future state recovery, it is used for state recovery during inline call error reporting

        NOTE:
            Why don't use __deepcopy__, because memo is not a deepcopy, i.e inner_out is only a shallow copy, SIR is a deepcopy.
        """
        saved_stmt_ir = deepcopy(self.sir_ctx.TOS)
        return FunctionGraph.Memo(
            inner_out=set(self.inner_out),
            input_variables=list(self.input_variables),
            stmt_ir=saved_stmt_ir,
            global_guards=OrderedSet(self._global_guarded_variables),
            side_effects_state=self.side_effects.get_state(),
            print_variables=list(self._print_variables),
            inplace_tensors=OrderedSet(self._inplace_tensors),
        )

    def restore_memo(self, memo: FunctionGraph.Memo):
        """
        Restore the state of graph to memo.

        Args:
            memo: Previously recorded memo

        """
        self.inner_out = memo.inner_out
        self.input_variables = memo.input_variables
        self.sir_ctx.replace_TOS(memo.stmt_ir)
        self._global_guarded_variables = memo.global_guards
        self.side_effects.restore_state(memo.side_effects_state)
        self._print_variables = memo.print_variables
        self._inplace_tensors = memo.inplace_tensors

    def collect_input_variables(self, inputs: list[VariableBase]):
        """
        Variables required within the method

        Args:
            inputs: Required VariableBase
        """

        def collect(inp):
            if isinstance(inp, VariableBase) and self.need_add_input(inp):
                self.input_variables.append(inp)

        map_variables(
            collect,
            inputs,
        )

    @property
    @event_register("guard_fn")
    def guard_fn(self) -> Guard:
        with tmp_name_guard():
            guards = []
            with EventGuard(
                "guard_fn: find vars and make stringify guard", event_level=1
            ):
                for variable in find_traceable_vars(
                    self.input_variables + list(self._global_guarded_variables)
                ):
                    guards.extend(variable.make_stringify_guard())

            guards = OrderedSet(guards)

            for guard in guards:
                assert isinstance(
                    guard, StringifyExpression
                ), "guard must be StringifyExpression."

            return make_guard(guards)

    def _restore_origin_opcode(self, stack_vars, store_var_info, instr_idx):
        class VariableLoader:
            def __init__(self, store_var_info, pycode_gen):
                self._store_var_info = store_var_info
                self._pycode_gen: PyCodeGen = pycode_gen

            def load(self, var):
                if isinstance(var, NullVariable):
                    var.reconstruct(self._pycode_gen)
                    return
                self._pycode_gen.gen_load(self._store_var_info[var.id])

        origin_instrs = get_instructions(self.pycode_gen._origin_code)

        restore_instrs = origin_instrs[:instr_idx]
        restore_instr_names = [
            instr.opname for instr in restore_instrs[:instr_idx]
        ]
        # NOTE(SigureMo): Trailing KW_NAMES + PRECALL is no need to restore in Python 3.11+
        if restore_instr_names[-2:] == ["KW_NAMES", "PRECALL"]:
            restore_instrs = restore_instrs[:-2]

        self.pycode_gen.extend_instrs(restore_instrs)
        nop = self.pycode_gen._add_instr("NOP")

        for instr in origin_instrs:
            if instr.jump_to == origin_instrs[instr_idx]:
                instr.jump_to = nop

        self.pycode_gen.hooks.append(
            lambda: self.pycode_gen.extend_instrs(
                iter(origin_instrs[instr_idx + 1 :])
            )
        )

        self.pycode_gen.gen_enable_eval_frame()

        name_gen = NameGenerator("__start_compile_saved_orig_")

        for var in stack_vars[::-1]:
            store_var_info[var.id] = name_gen.next()
            self.pycode_gen.gen_store_fast(store_var_info[var.id])

        return VariableLoader(store_var_info, self.pycode_gen)

    def _build_compile_fn_with_name_store(self, ret_vars, to_store_vars):
        class VariableLoader:
            def __init__(self, index_for_load, pycode_gen):
                self._index_for_load = index_for_load
                self._pycode_gen: PyCodeGen = pycode_gen

            def load(self, var, allow_push_null=True):
                if isinstance(var, NullVariable):
                    var.reconstruct(self._pycode_gen)
                    return
                self._pycode_gen.gen_load(self._index_for_load[var.id])

        # var_id -> local_name mapping
        index_for_load = {}
        to_store_vars = list(
            filter(lambda x: not isinstance(x, NullVariable), to_store_vars)
        )
        self.start_compile(*(ret_vars + to_store_vars))
        name_gen = NameGenerator("__start_compile_saved_")

        for var in to_store_vars[::-1]:
            index_for_load[var.id] = name_gen.next()

            def _log_fn():
                print(
                    f"[StartCompile] saved var: {index_for_load[var.id]} = ",
                    var,
                )

            log_do(4, _log_fn)

            self.pycode_gen.gen_store_fast(index_for_load[var.id])

        return VariableLoader(index_for_load, self.pycode_gen)

    @event_register("start_compile", event_level=2)
    def start_compile(self, *ret_vars: VariableBase):
        """
        Generate bytecode based on the information collected by the simulation execution.

        This consists of the following steps:
        - Compile the FunctionGraph into a dy2st StaticFunction and load it in the generated bytecode
        - Load the group network input
        - Calling the generated dy2st StaticFunction
        - Restore the side effects
        - Restore the output
        - Return the top of the stack
        """
        from ..breakpoint import BreakpointManager

        BreakpointManager().on_event("start_compile")

        ret_items = [
            ret_item
            for ret_var in ret_vars
            for ret_item in ret_var.flatten_items()
        ]

        tensor_items = self._find_tensor_outputs(ret_items)
        compiled_fn, statment_ir = self.sir_ctx.compile_fn(
            [Symbol(tensor_var.var_name) for tensor_var in tensor_items],
            **self._kwargs,
        )
        input_names = statment_ir.inputs
        compiled_fn_name = f"__compiled_fn_{statment_ir.name}"
        # prepare function and inputs
        self.pycode_gen.gen_load_object(compiled_fn, compiled_fn_name)
        for name in input_names:
            found = False
            for variable in self.input_variables:
                if (
                    isinstance(variable, TensorVariable)
                    and variable.get_symbol().name == name
                ):
                    variable.tracker.gen_instructions(self.pycode_gen)
                    found = True
                    break
            assert found, f"can't find input {name} in SIR."
        # Pack all args into a tuple, because we don't support *args now.
        self.pycode_gen.gen_build_tuple(count=len(input_names))
        # call the compiled_fn
        self.pycode_gen.gen_call_function(argc=1)

        # Store outputs to f_locals
        self.pycode_gen.gen_unpack_sequence(count=len(tensor_items))
        for tensor_var in tensor_items:
            self.pycode_gen.gen_store_fast(tensor_var.out_var_name)
        # restore the outputs.
        for ret_var in ret_vars:
            ret_var.reconstruct(self.pycode_gen)

        # deal side effect
        self.restore_inplace_tensor(self._inplace_tensors)
        self.restore_print_stmts(self._print_variables)
        self.restore_side_effects(self.side_effects.proxy_variables)
        self.pycode_gen.gen_enable_eval_frame()

        tracker_output_path = ENV_SHOW_TRACKERS.get()
        if tracker_output_path:
            from .tracker_viewer import view_tracker

            view_tracker(list(ret_vars), tracker_output_path, format="png")

    def call_paddle_api(
        self,
        func: Callable[..., Any],
        *args: VariableBase,
        **kwargs: VariableBase,
    ):
        """
        Record Paddle Networking API to SIR

        Args:
            func: paddle api
        """
        assert is_paddle_api(func)
        # not fallback api, start symbolic trace.
        # TODO(xiokgun): may have python buildin object inside metas.
        # TODO(xiokgun): 4 kinds of python arguments. support it !!
        log(3, f"call paddle.api : {func.__name__}", "\n")

        def message_handler(*args, **kwargs):
            return f"Call paddle_api error: {func.__name__}, may be not a operator api ?"

        return inner_error_default_handler(self.symbolic_call, message_handler)(
            InferMetaCache(), self.sir_ctx.call_API, func, *args, **kwargs
        )

    def call_tensor_method(
        self, method_name: str, *args: VariableBase, **kwargs
    ):
        """
        call tensor method, start symbolic trace.

        Args:
            method_name: tensor method name
        """

        def message_handler(*args, **kwargs):
            return f"Call tensor_method error: Tensor.{method_name}, may be not a valid operator api ?"

        return inner_error_default_handler(self.symbolic_call, message_handler)(
            InferMetaCache(),
            self.sir_ctx.call_METHOD,
            method_name,
            *args,
            **kwargs,
        )

    @staticmethod
    def get_opcode_executor_stack():
        # NOTE: only for debug.
        # dependent on OpcodeExecutor.
        from .opcode_executor import OpcodeExecutorBase

        if len(OpcodeExecutorBase.call_stack) == 0:
            # In test case, we can meet this senario.
            return []
        current_executor = OpcodeExecutorBase.call_stack[-1]
        current_line = current_executor._current_line
        filename = current_executor._code.co_filename
        source_lines, start_line = inspect.getsourcelines(
            current_executor._code
        )
        # TODO(SigureMo): In 3.11, lineno maybe changed after multiple breakgraph,
        # We need to find a way to fix this.
        line_idx = min(current_line - start_line, len(source_lines) - 1)
        code_line = source_lines[line_idx]
        stack = []
        stack.append(
            '  File "{}", line {}, in {}'.format(
                filename,
                current_line,
                current_executor._code.co_name,
            )
        )
        stack.append(f'    {code_line}')
        return stack

    def call_layer(
        self,
        layer: PaddleLayerVariable,
        weak_ref: bool,
        *args: VariableBase,
        **kwargs: VariableBase,
    ):
        """
        call paddle layer, start symbolic trace.

        Args:
            layer: paddle layer
        """

        def infer_meta_fn(layer, *metas, **kwmetas):
            metas = LayerInferMetaCache()(layer.value, *metas, **kwmetas)
            return metas

        def compute_fn(layer, inputs, outputs, stacks):
            self.sir_ctx.call_LAYER(
                Reference(layer.value, weak_ref),
                inputs=inputs,
                outputs=outputs,
                stacks=stacks,
            )

        def message_handler(*args, **kwargs):
            return f"Call paddle layer error: {layer}, may be not a valid paddle layer ?"

        return inner_error_default_handler(self.symbolic_call, message_handler)(
            infer_meta_fn, compute_fn, layer, *args, **kwargs
        )

    def symbolic_call(self, infer_meta_fn, compute_fn, func, *args, **kwargs):
        """
        Using infer_meta_fn and compute_fn convert func to symbolic function.

        Args:
            infer_meta_fn: function for infer meta, (func, metas, kwmetas) -> output_metas
            compute_fn   : function for sir compile, (func, input_symbols, outputs_symbols) -> None
            func         : symbolic function
        """
        self.collect_input_variables(list(args))
        self.collect_input_variables(list(kwargs.values()))
        metas = convert_to_meta(args)
        kwmetas = convert_to_meta(kwargs)

        out_metas = infer_meta_fn(func, *metas, **kwmetas)
        inputs_symbols = (
            convert_to_symbol(args),
            convert_to_symbol(kwargs),
        )
        log(3, f"         inputs : {inputs_symbols}", "\n")

        outputs = map_if(
            out_metas,
            pred=lambda x: isinstance(x, MetaInfo),
            true_fn=lambda x: TensorVariable(
                x,
                self,
                tracker=DummyTracker(list(args) + list(kwargs.values())),
            ),
            false_fn=lambda x: x,
        )
        stmt_stacks = []
        log_do(
            3,
            lambda: stmt_stacks.extend(
                FunctionGraph.get_opcode_executor_stack()
            ),
        )
        if outputs is not None:
            if is_inplace_api(func):
                # if we want to use a non-inplace api (static api) to replace an inplace behavior (in simulation)
                # just set it back in SIR, and return outputs to replace tensor meta (it might changes?)
                # in this case, the output will not exactly be used
                compute_fn(
                    func,
                    inputs_symbols,
                    convert_to_symbol(args[0]),
                    stmt_stacks,
                )
            else:
                compute_fn(
                    func,
                    inputs_symbols,
                    convert_to_symbol(outputs),
                    stmt_stacks,
                )  # symbolic only contain symbols.
                self._put_inner(outputs)
            return VariableFactory.from_value(
                outputs, self, DummyTracker(list(args) + list(kwargs.values()))
            )
        else:
            return None

    def _put_inner(self, vars: VariableBase):
        """
        put inner variable to inner_out
        """
        map_if(
            vars,
            pred=lambda x: isinstance(x, VariableBase),
            true_fn=lambda x: self.inner_out.add(x.id),
            false_fn=lambda x: None,
        )

    def add_global_guarded_variable(self, variable: VariableBase):
        """
        Add variable to global guarded variable
        """
        self._global_guarded_variables.add(variable)

    def remove_global_guarded_variable(self, variable: VariableBase):
        """
        Remove variable to global guarded variable
        """
        if variable in self._global_guarded_variables:
            self._global_guarded_variables.remove(variable)

    def _find_tensor_outputs(
        self, outputs: list[VariableBase]
    ) -> OrderedSet[TensorVariable]:
        """
        Return all TensorVariable. find TensorVariables participating in networking from the output Variables

        Args:
            outputs: output variables
        """

        def collect_related_dummy_tensor(var):
            if isinstance(var.tracker, DummyTracker):
                if isinstance(var, TensorVariable):
                    return [var]
                else:
                    retval = []
                    for inp in var.tracker.inputs:
                        retval.extend(collect_related_dummy_tensor(inp))
                    return retval
            return []

        output_tensors: OrderedSet[TensorVariable] = OrderedSet()
        # Find Tensor Variables from outputs.
        for output in outputs:
            if isinstance(output.tracker, DummyTracker):
                if isinstance(output, TensorVariable):
                    output_tensors.add(output)
                else:
                    for inp in output.tracker.inputs:
                        for _var in collect_related_dummy_tensor(inp):
                            output_tensors.add(_var)
                    # Guard output that can not be traced.
                    self.add_global_guarded_variable(output)
        # Find Tensor Variables from side effects Variables.
        for side_effect_var in self.side_effects.proxy_variables:
            if isinstance(side_effect_var, (ListVariable, DictVariable)):
                for var in side_effect_var.flatten_items():
                    if (
                        isinstance(var.tracker, DummyTracker)
                        and isinstance(var, TensorVariable)
                        and side_effect_var.tracker.is_traceable()
                    ):
                        output_tensors.add(var)
            else:
                if isinstance(side_effect_var, GlobalVariable):
                    proxy_records = side_effect_var.proxy.records
                elif side_effect_var.tracker.is_traceable():
                    # for attr side effect
                    proxy_records = side_effect_var.attr_proxy.records
                else:
                    continue
                for record in proxy_records:
                    if isinstance(record, (MutationSet, MutationNew)):
                        for var in record.value.flatten_items():
                            if isinstance(
                                var.tracker, DummyTracker
                            ) and isinstance(var, TensorVariable):
                                output_tensors.add(var)
        # Find Tensor in print_stmts
        for print_stmt in self._print_variables:
            for var in print_stmt.flatten_items():
                if isinstance(var.tracker, DummyTracker) and isinstance(
                    var, TensorVariable
                ):
                    output_tensors.add(var)

        # add inplace tensors into output tensors.
        for inplace_tensor in self._inplace_tensors:
            output_tensors.add(inplace_tensor)

        return output_tensors

    def restore_print_stmts(self, variables: list[VariableBase]):
        for var in variables:
            var.reconstruct(
                self.pycode_gen,
                use_tracker=False,
                add_to_global_guarded_vars=False,
            )

    def restore_inplace_tensor(self, variables: list[VariableBase]):
        for var in variables:
            if not var.tracker.is_traceable():
                continue
            var.reconstruct(
                self.pycode_gen,
                use_tracker=True,
                add_to_global_guarded_vars=False,
            )
            self.pycode_gen.gen_load_method(
                "_inplace_assign"
            )  # NOTE: paddle related logic.
            var.reconstruct(
                self.pycode_gen,
                use_tracker=False,
                add_to_global_guarded_vars=True,
            )
            self.pycode_gen.gen_call_method(1)
            self.pycode_gen.gen_pop_top()

    def restore_side_effects(self, variables: list[VariableBase]):
        """
        Generate side effect recovery code for variables with side effects

        Args:
            variables: Variables that may have side effects.
        """
        restorers: list[SideEffectRestorer] = []

        for var in variables:
            # skip inner variables
            if not var.tracker.is_traceable() and not isinstance(
                var, GlobalVariable
            ):
                continue
            if isinstance(var, DictVariable):
                restorers.append(DictSideEffectRestorer(var))
            elif isinstance(var, ListVariable):
                restorers.append(ListSideEffectRestorer(var))
            else:
                if isinstance(var, GlobalVariable):
                    for record in var.proxy.records[::-1]:
                        if isinstance(record, (MutationSet, MutationNew)):
                            restorers.append(
                                GlobalSetSideEffectRestorer(
                                    record.key,
                                    record.value,
                                )
                            )
                        elif isinstance(record, MutationDel):
                            restorers.append(
                                GlobalDelSideEffectRestorer(record.key)
                            )
                else:
                    for record in var.attr_proxy.records[::-1]:
                        if isinstance(record, (MutationSet, MutationNew)):
                            restorers.append(
                                ObjSetSideEffectRestorer(
                                    var,
                                    record.key,
                                    record.value,
                                )
                            )
                        elif isinstance(record, MutationDel):
                            restorers.append(
                                ObjDelSideEffectRestorer(
                                    var,
                                    record.key,
                                )
                            )

        for restorer in restorers:
            restorer.pre_gen(self.pycode_gen)
        for restorer in restorers[::-1]:
            restorer.post_gen(self.pycode_gen)
