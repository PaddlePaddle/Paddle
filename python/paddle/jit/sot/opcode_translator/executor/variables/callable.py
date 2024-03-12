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

from __future__ import annotations

import inspect
import operator
import types
from functools import reduce
from typing import TYPE_CHECKING, Any, Callable

import paddle

from .... import psdb
from ....profiler import EventGuard
from ....utils import (
    ENV_SOT_EXPORT,
    get_static_function,
    is_break_graph_api,
    is_break_graph_tensor_methods,
    is_builtin_fn,
    is_not_supported_paddle_layer,
    is_paddle_api,
    magic_method_builtin_dispatch,
)
from ....utils.exceptions import BreakGraphError, FallbackError, SotErrorBase
from ..dispatcher import Dispatcher
from ..guard import (
    StringifyExpression,
    check_guard,
    object_equal_stringify_guard,
    union_free_vars,
)
from ..tracker import (
    ConstTracker,
    CreateLayerTracker,
    DanglingTracker,
    DummyTracker,
    GetAttrTracker,
    GetItemTracker,
    GetIterTracker,
    Tracker,
)
from .base import VariableBase, VariableFactory
from .basic import ConstantVariable, PrintStmtVariable, SliceVariable

if TYPE_CHECKING:
    from ..function_graph import FunctionGraph


PD_ALL_CONTAINERS = (paddle.nn.Sequential, paddle.nn.LayerList)
PD_SEQ_CONTAINERS = (paddle.nn.Sequential, paddle.nn.LayerList)


class CallableVariable(VariableBase):
    """
    CallableVariable is a subclass of VariableBase used to wrap a callable variable.

    Args:
        graph(FunctionGraph): The FunctionGraph object that this variable is associated with.
        tracker(Tracker): The Tracker object that tracks the information of this variable.
    """

    def __init__(self, graph: FunctionGraph, tracker: Tracker):
        super().__init__(graph, tracker)

    def __call__(self, /, *args, **kwargs) -> VariableBase:
        """Why we need '/' to make self positional only?

        If kwargs have {'self': xxx}, this function call raise a error.
        See: test_str_format.py for details.
        """
        with EventGuard(f"call_function: {self.__class__.__name__}"):
            return self.call_function(*args, **kwargs)

    def call_function(self, /, *args, **kwargs):
        raise NotImplementedError("call_function is not implemented.")


class FunctionVariable(CallableVariable):
    """
    FunctionVariable is a subclass of CallableVariable used to wrap a function variable.

    Args:
        fn (Callable[..., Any]): The function to be wrapped.
        graph(FunctionGraph): The FunctionGraph object that this variable is associated with.
        tracker(Tracker): The Tracker object that tracks the information of this variable.
    """

    def __init__(
        self, fn: Callable[..., Any], graph: FunctionGraph, tracker: Tracker
    ):
        super().__init__(graph, tracker)
        self.value = fn

    def get_py_value(self, allow_tensor=False):
        return self.value

    def get_code(self) -> types.CodeType:
        return self.value.__code__

    def bind(self, instance: VariableBase, name: str):
        method_var = MethodVariable(
            instance,
            self,
            graph=self.graph,
            tracker=GetAttrTracker(instance, name),
        )
        class_var = VariableFactory.from_value(
            instance.get_py_type(),
            graph=self.graph,
            tracker=GetAttrTracker(instance, "__class__"),
        )
        assert class_var is not None
        self.tracker = GetAttrTracker(class_var, name)
        return method_var

    make_stringify_guard = object_equal_stringify_guard


class UserDefinedFunctionVariable(FunctionVariable):
    """
    UserDefinedFunctionVariable is a subclass of FunctionVariable used to wrap a user-defined function.

    Args:
        fn (Callable[..., Any]): The user-defined function to be wrapped.
        graph(FunctionGraph): The FunctionGraph object that this variable is associated with.
        tracker(Tracker): The Tracker object that tracks the information of this variable.
    """

    def __init__(
        self, fn: Callable[..., Any], graph: FunctionGraph, tracker: Tracker
    ):
        super().__init__(fn, graph, tracker)

    def handle_psdb_function(self, /, *args, **kwargs):
        # special function for inner debug.
        if self.value is psdb.assert_true:
            return ConstantVariable.wrap_literal(
                self.value(args[0].value), self.graph
            )
        elif self.value is psdb.print:
            sot_prefix = ConstantVariable.wrap_literal("[SOT]", self.graph)
            self.graph.add_print_variables(
                PrintStmtVariable(([sot_prefix, *args], kwargs), self.graph)
            )
            return ConstantVariable.wrap_literal(None, self.graph)
        elif self.value is psdb.breakpoint:
            # do nothing. just return None.
            from ...breakpoint import BM

            BM.locate(BM.executors[-1])
            BM.add(BM.cur_exe._code.co_filename, BM.cur_exe._current_line)
            return ConstantVariable.wrap_literal(None, self.graph)
        elif self.value is psdb.breakgraph:
            raise BreakGraphError("breakgraph by psdb.breakgraph")
        elif self.value is psdb.fallback:
            raise FallbackError("fallback by psdb.fallback")
        elif self.value is psdb.in_sot:
            return ConstantVariable.wrap_literal(True, self.graph)
        return None

    def call_function(self, /, *args, **kwargs) -> VariableBase:
        from ..opcode_inline_executor import OpcodeInlineExecutor

        result = self.handle_psdb_function(*args, **kwargs)
        if result is not None:
            return result

        checkpoint = self.graph.save_memo()

        static_function = get_static_function(self.value, "inline_call")
        if static_function is not None:
            output = self.graph.call_ast(static_function, *args, **kwargs)
            if output is not None:
                return output

        try:
            inline_executor = OpcodeInlineExecutor(self, *args, **kwargs)
            with EventGuard(
                f"Inline Call: {inline_executor._code.co_name.replace('<', '(').replace('>', ')')}, file {inline_executor._code.co_filename}, line {int(inline_executor._code.co_firstlineno)}"
            ):
                output = inline_executor.inline_call()
        except SotErrorBase as e:
            self.graph.restore_memo(checkpoint)
            raise BreakGraphError(
                f"({e}) raised while inline call {self.value.__code__}."
            )
        return output

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph, tracker: Tracker):
        if isinstance(value, (types.FunctionType)):
            return UserDefinedFunctionVariable(value, graph, tracker)
        if isinstance(
            value, paddle.jit.dy2static.program_translator.StaticFunction
        ):
            return UserDefinedFunctionVariable(
                value.dygraph_function, graph, tracker
            )
        return None

    @property
    def main_info(self) -> dict[str, Any]:
        return {
            "name": self.value.__name__,
        }


class PaddleApiVariable(FunctionVariable):
    """
    PaddleApiVariable is a subclass of FunctionVariable used to wrap a paddlepaddle API function.

    Args:
        fn (Callable[..., Any]): The paddlepaddle API to be wrapped.
        graph(FunctionGraph): The FunctionGraph object that this variable is associated with.
        tracker(Tracker): The Tracker object that tracks the information of this variable.
    """

    def __init__(
        self, fn: Callable[..., Any], graph: FunctionGraph, tracker: Tracker
    ):
        super().__init__(fn, graph, tracker)

    def call_function(self, /, *args, **kwargs):
        if is_break_graph_api(self.value):
            raise BreakGraphError(
                f"breakgraph by unsupport function: {self.value.__name__}"
            )
        return self.graph.call_paddle_api(self.value, *args, **kwargs)

    @VariableFactory.register_from_value(
        successor="UserDefinedFunctionVariable"
    )
    def from_value(value: Any, graph: FunctionGraph, tracker: Tracker):
        if callable(value) and is_paddle_api(value):
            return PaddleApiVariable(value, graph, tracker)
        return None

    @property
    def main_info(self) -> dict[str, Any]:
        return {
            "name": self.value.__name__,
        }

    make_stringify_guard = object_equal_stringify_guard


class TensorFunctionVariable(FunctionVariable):
    """
    TensorFunctionVariable is a subclass of FunctionVariable used to wrap a method of a tensor.

    Args:
        method_name (str): The name of the tensor method to be wrapped.
        graph(FunctionGraph): The FunctionGraph object that this variable is associated with.
        tracker(Tracker): The Tracker object that tracks the information of this variable.
    """

    def __init__(
        self, method_name: str, graph: FunctionGraph, tracker: Tracker
    ):
        fn = getattr(paddle.static.Variable, method_name)
        super().__init__(fn, graph, tracker)
        self.method_name = method_name

    def call_function(self, /, *args, **kwargs):
        if is_break_graph_tensor_methods(self.method_name):
            raise BreakGraphError("call break_graph_tensor_method.")
        return self.graph.call_tensor_method(self.method_name, *args, **kwargs)

    def bind(self, instance: VariableBase, name: str):
        method_var = MethodVariable(
            instance,
            self,
            graph=self.graph,
            tracker=GetAttrTracker(instance, name),
        )
        class_var = VariableFactory.from_value(
            instance.get_py_type(),
            graph=self.graph,
            tracker=ConstTracker(instance.get_py_type()),
        )
        assert class_var is not None
        self.tracker = GetAttrTracker(class_var, name)
        return method_var

    @property
    def main_info(self) -> dict[str, Any]:
        return {
            "name": self.value.__name__,
        }


class MethodVariable(CallableVariable):
    """
    MethodVariable is a subclass of CallableVariable used to wrap a method variable.

    Args:
        bound_instance (VariableBase): The instance of the method.
        fn (VariableBase): The method to be wrapped.
        graph(FunctionGraph): The FunctionGraph object that this variable is associated with.
        tracker(Tracker): The Tracker object that tracks the information of this variable.
        method_name (str): The name of the method to be wrapped.
    """

    def __init__(
        self,
        bound_instance: VariableBase,
        fn: VariableBase,
        graph: FunctionGraph,
        tracker: Tracker,
        *,
        method_name: str | None = None,
    ):
        super().__init__(graph, tracker)
        self.bound_instance = bound_instance
        self.fn = fn
        self.method_name = method_name

    def get_py_value(self, allow_tensor=False):
        return self.fn.get_py_value().__get__(
            self.bound_instance.get_py_value(allow_tensor),
            self.bound_instance.get_py_value(allow_tensor).__class__,
        )

    def _reconstruct(self, pycode_gen):
        assert self.method_name is not None
        self.tensor.reconstruct(pycode_gen)
        pycode_gen.gen_load_attr(self.method_name)

    def call_function(self, /, *args, **kwargs):
        return self.fn(*(self.bound_instance, *args), **kwargs)

    @staticmethod
    def wrap_method(
        value: types.MethodType,
        *,
        graph: FunctionGraph,
        tracker: Tracker,
        instance: VariableBase | None = None,
        fn: VariableBase | None = None,
        method_name: str | None = None,
    ):
        # NOTE(SigureMo): Since the method_self need method_var as the obj
        # of the tracker, we need to temporarily set the tracker of method_self
        # to DummyTracker, and set it to GetAttrTracker after method_var is created.
        instance_var = (
            VariableFactory.from_value(value.__self__, graph, DanglingTracker())
            if instance is None
            else instance
        )

        fn_var = (
            VariableFactory.from_value(value.__func__, graph, DanglingTracker())
            if fn is None
            else fn
        )

        method_var = MethodVariable(
            instance_var,
            fn_var,
            method_name=method_name,
            graph=graph,
            tracker=tracker,
        )
        if instance is None:
            instance_var.tracker = GetAttrTracker(method_var, "__self__")
        if fn is None:
            fn_var.tracker = GetAttrTracker(method_var, "__func__")
        return method_var

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph, tracker: Tracker):
        if inspect.ismethod(value):
            return MethodVariable.wrap_method(
                value=value, tracker=tracker, graph=graph
            )
        return None

    @property
    def main_info(self) -> dict[str, Any]:
        return {
            "method": self.method_name,
        }


class LayerVariable(CallableVariable):
    """
    LayerVariable is a subclass of CallableVariable used to wrap a layer.

    Args:
        layer (paddle.nn.Layer): The layer to be wrapped.
        graph(FunctionGraph): The FunctionGraph object that this variable is associated with.
        tracker(Tracker): The Tracker object that tracks the information of this variable.
    """

    def __init__(
        self, layer: paddle.nn.Layer, graph: FunctionGraph, tracker: Tracker
    ):
        super().__init__(graph, tracker)
        self.value = layer

    def get_py_value(self, allow_tensor=False):
        return self.value

    def call_function(self, /, *args, **kwargs):
        fn_var = UserDefinedFunctionVariable(
            self.value.__class__.__call__,
            self.graph,
            GetAttrTracker(self, "__call__"),
        )

        return fn_var(*(self, *args), **kwargs)

    @check_guard
    def make_stringify_guard(self) -> list[StringifyExpression]:
        frame_value_tracer = self.tracker.trace_value_from_frame()
        return [
            StringifyExpression(
                f"id({{}}) == {id(self.get_py_value())}",
                [frame_value_tracer],
                union_free_vars(frame_value_tracer.free_vars),
            ),
            StringifyExpression(
                f"{{}}.training == {self.get_py_value().training}",
                [frame_value_tracer],
                union_free_vars(frame_value_tracer.free_vars),
            ),
        ]


class ContainerLayerVariable(LayerVariable):
    def __init__(
        self, layer: paddle.nn.Layer, graph: FunctionGraph, tracker: Tracker
    ):
        super().__init__(layer, graph, tracker)

    def __len__(self):
        return len(self.value)

    def len(self):
        return ConstantVariable(len(self), self.graph, DummyTracker([self]))

    def getitem(self, key):
        if isinstance(self.value, PD_SEQ_CONTAINERS) and isinstance(
            key, SliceVariable
        ):
            try:
                slice_py_value = key.get_py_value()
                new_layer_list = self.value[slice_py_value]
                self.graph.add_global_guarded_variable(key)
                return VariableFactory.from_value(
                    new_layer_list,
                    self.graph,
                    GetItemTracker(self, slice_py_value),
                )
            except Exception as e:
                raise BreakGraphError(
                    f"call {self.value.__class__.__name__}.__getitem__ with slice as key, and slice with py value failed: {e}."
                )

        else:
            return super().getitem(key)

    def get_iter(self):
        if isinstance(self.value, PD_SEQ_CONTAINERS):
            from .iter import SequenceIterVariable

            return SequenceIterVariable(self, self.graph, GetIterTracker(self))
        else:
            return super().get_iter()

    def make_stringify_guard(self) -> list[StringifyExpression]:
        if isinstance(self.value, PD_SEQ_CONTAINERS):
            frame_value_tracer = self.tracker.trace_value_from_frame()

            len_guard = StringifyExpression(
                f"len({{}}) == {len(self.value)}",
                [frame_value_tracer],
                frame_value_tracer.free_vars,
            )

            guards = [len_guard]
            for idx, layer in enumerate(self.value):
                layer_variable = VariableFactory.from_value(
                    layer, self.graph, GetItemTracker(self, idx)
                )
                guards.extend(layer_variable.make_stringify_guard())

            return guards
        else:
            return super().make_stringify_guard()

    @property
    def main_info(self) -> dict[str, Any]:
        return {
            "name": self.value.__class__.__name__,
        }

    @VariableFactory.register_from_value(successor="PaddleLayerVariable")
    def from_value(value: Any, graph: FunctionGraph, tracker: Tracker):
        if isinstance(value, PD_ALL_CONTAINERS):
            return ContainerLayerVariable(value, graph, tracker)
        return None


class PaddleLayerVariable(LayerVariable):
    """
    PaddleLayerVariable is a subclass of LayerVariable used to wrap a paddlepaddle layer.

    Args:
        layer (paddle.nn.Layer): The paddle built-in layer to be wrapped.
        graph(FunctionGraph): The FunctionGraph object that this variable is associated with.
        tracker(Tracker): The Tracker object that tracks the information of this variable.
    """

    def __init__(
        self, layer: paddle.nn.Layer, graph: FunctionGraph, tracker: Tracker
    ):
        super().__init__(layer, graph, tracker)

    def call_function(self, /, *args, **kwargs):
        self.graph.add_global_guarded_variable(self)
        # when layer is created in forward function, we use strong ref because it can't have
        # weights and buffers, see PaddleLayerClassVariable for details.
        weak_ref = not isinstance(self.tracker, CreateLayerTracker)
        return self.graph.call_layer(self, weak_ref, *args, **kwargs)

    def make_stringify_guard(self) -> list[StringifyExpression]:
        if isinstance(self.tracker, CreateLayerTracker):
            return reduce(
                operator.add,
                [var.make_stringify_guard() for var in self.tracker.inputs],
            )
        else:
            return super().make_stringify_guard()

    @property
    def main_info(self) -> dict[str, Any]:
        return {
            "name": self.value.__class__.__name__,
        }

    @VariableFactory.register_from_value(successor="UserDefinedLayerVariable")
    def from_value(value: Any, graph: FunctionGraph, tracker: Tracker):
        # TODO: @wuzhanfei, if we support create sub layer when export, remove this branch
        if ENV_SOT_EXPORT.get() != "":
            return None
        # TODO(SigureMo): Add a more common way to check if a value is a paddle builtin layer.
        if isinstance(value, paddle.nn.Layer):
            # If there is a user-defined behavior, such as a container class layer
            # or a hook on the layer, it needs to be converted to UserDefinedLayerVariable,
            # otherwise converted to PaddleLayerVariable
            if (
                hasattr(value, "_forward_pre_hooks")
                and value._forward_pre_hooks
                or hasattr(value, "_forward_post_hooks")
                and value._forward_post_hooks
                or is_not_supported_paddle_layer(type(value))
            ):
                return None
            if value.__module__.startswith("paddle.nn."):
                return PaddleLayerVariable(value, graph, tracker)
        return None


class UserDefinedLayerVariable(LayerVariable):
    """
    UserDefinedLayerVariable is a subclass of LayerVariable used to wrap a user-defined layer.

    Args:
        layer (paddle.nn.Layer): The user-defined layer to be wrapped.
        graph(FunctionGraph): The FunctionGraph object that this variable is associated with.
        tracker(Tracker): The Tracker object that tracks the information of this variable.
    """

    def __init__(
        self, layer: paddle.nn.Layer, graph: FunctionGraph, tracker: Tracker
    ):
        super().__init__(layer, graph, tracker)

    @property
    def main_info(self) -> dict[str, Any]:
        return {
            "name": self.value.__class__.__name__,
        }

    @VariableFactory.register_from_value(successor="PaddleApiVariable")
    def from_value(value: Any, graph: FunctionGraph, tracker: Tracker):
        if isinstance(value, paddle.nn.Layer):
            return UserDefinedLayerVariable(value, graph, tracker)
        return None


class BuiltinVariable(FunctionVariable):
    """
    BuiltinVariable is a subclass of FunctionVariable used to wrap a built-in function.
    Args:
        fn (Callable[..., Any]): The built-in function to be wrapped.
        graph(FunctionGraph): The FunctionGraph object that this variable is associated with.
        tracker(Tracker): The Tracker object that tracks the information of this variable.
    """

    def __init__(
        self, fn: Callable[..., Any], graph: FunctionGraph, tracker: Tracker
    ):
        super().__init__(fn, graph, tracker)
        self.value = fn

    def call_function(self, /, *args, **kwargs):
        # Lookup the handler from dispatcher
        handler = Dispatcher.dispatch(self.value, *args, **kwargs)
        if handler is not None:
            return handler(*args, **kwargs)

        # Try to inline call the magic function
        magic_methods = magic_method_builtin_dispatch(self.value)
        for magic_method in magic_methods:
            sorted_args = args
            if magic_method.is_reverse:
                sorted_args = sorted_args[::-1]
            arg_type = sorted_args[0].get_py_type()
            if hasattr(arg_type, magic_method.name):
                class_fn = getattr(arg_type, magic_method.name)
                class_var = VariableFactory.from_value(
                    arg_type,
                    self.graph,
                    GetAttrTracker(args[0], "__class__"),
                )
                assert isinstance(class_var, VariableBase)
                fn_var = VariableFactory.from_value(
                    class_fn,
                    self.graph,
                    GetAttrTracker(class_var, class_fn.__name__),
                )
                assert isinstance(fn_var, VariableBase)
                return fn_var(*args)
            # If __bool__ and __len__ method are absent, inline bool calls return True.
            # See https://github.com/python/cpython/blob/3.11/Objects/typeobject.c#L7463
            elif magic_method.name == "__bool__" and not hasattr(
                arg_type, "__len__"
            ):
                return VariableFactory.from_value(
                    True,
                    self.graph,
                    DummyTracker([self] + list(args) + list(kwargs.values())),
                )

        # Break graph if neither of the above conditions is met
        arg_types = ", ".join([type(arg).__name__ for arg in args])
        fn_name = (
            self.value.__name__
            if hasattr(self.value, '__name__')
            else self.value
        )
        raise BreakGraphError(
            f"Not support builtin function: {fn_name} with args: Args({arg_types})"
        )

    @VariableFactory.register_from_value(successor="ClassVariable")
    def from_value(value: Any, graph: FunctionGraph, tracker: Tracker):
        if is_builtin_fn(value):
            return BuiltinVariable(value, graph, tracker)
        return None

    @property
    def main_info(self) -> dict[str, Any]:
        return {
            "name": self.value.__name__,
        }


class UserDefinedGeneratorFunctionVariable(FunctionVariable):
    """
    UserDefinedGeneratorFunctionVariable is a subclass of FunctionVariable used to wrap a user-defined generator.
    Args:
        fn (Callable[..., Any]): The user-defined generator to be wrapped.
        graph(FunctionGraph): The FunctionGraph object that this variable is associated with.
        tracker(Tracker): The Tracker object that tracks the information of this variable.
    """

    def __init__(
        self, fn: Callable[..., Any], graph: FunctionGraph, tracker: Tracker
    ):
        super().__init__(fn, graph, tracker)

    def call_function(self, /, *args, **kwargs):
        iter_ = self.value(*args, **kwargs)
        var = VariableFactory.from_value(
            iter_, self.graph, DummyTracker([self])
        )
        return var

    @property
    def main_info(self) -> dict[str, Any]:
        return {"name": self.value.__name__}

    @VariableFactory.register_from_value(
        successor="UserDefinedFunctionVariable"
    )
    def from_value(value: Any, graph: FunctionGraph, tracker: Tracker):
        if inspect.isgeneratorfunction(value):
            return UserDefinedGeneratorFunctionVariable(value, graph, tracker)
        return None


class ClassVariable(CallableVariable):
    def __init__(self, class_: type, graph: FunctionGraph, tracker: Tracker):
        super().__init__(graph, tracker)
        self.value = class_

    def get_py_value(self, allow_tensor=False):
        return self.value

    def call_function(self, /, *args, **kwargs):
        new_object = self.value.__new__(self.value)

        # do not have init function
        if self.value.__init__ is object.__init__:
            return VariableFactory.from_value(
                new_object, self.graph, DummyTracker([self])
            )

        if not hasattr(self.value.__init__, "__code__"):
            fn_var = BuiltinVariable(
                self.value.__init__,
                self.graph,
                GetAttrTracker(self, "__init__"),
            )
        else:
            fn_var = UserDefinedFunctionVariable(
                self.value.__init__,
                self.graph,
                GetAttrTracker(self, "__init__"),
            )

        # need classify variable type here?
        new_object_variable = VariableFactory.from_value(
            new_object,
            self.graph,
            DummyTracker([self] + list(args) + list(kwargs.values())),
        )
        fn_var(new_object_variable, *args, **kwargs)
        return new_object_variable

    make_stringify_guard = object_equal_stringify_guard

    @VariableFactory.register_from_value()
    def from_value(value: Any, graph: FunctionGraph, tracker: Tracker):
        if inspect.isclass(value):
            return ClassVariable(value, graph, tracker)
        return None


class PaddleLayerClassVariable(ClassVariable):
    def __init__(self, class_: type, graph: FunctionGraph, tracker: Tracker):
        super().__init__(class_, graph, tracker)

    def check_no_weight_and_buffers(self, paddle_layer):
        has_parameters = len(paddle_layer.parameters()) > 0
        has_buffers = len(paddle_layer.buffers()) > 0
        return not has_parameters and not has_buffers

    def call_function(self, /, *args, **kwargs):
        input_py_args = [var.get_py_value() for var in args]
        input_py_kwargs = {k: v.get_py_value() for k, v in kwargs.items()}
        new_layer = self.value(*input_py_args, **input_py_kwargs)
        assert self.check_no_weight_and_buffers(
            new_layer
        ), "You have created a layer in to_static function which may have Potential bugs. please create it in __init__/main function."
        return VariableFactory.from_value(
            new_layer, self.graph, CreateLayerTracker(self, args, kwargs)
        )

    @VariableFactory.register_from_value(successor="ClassVariable")
    def from_value(value: Any, graph: FunctionGraph, tracker: Tracker):
        if (
            inspect.isclass(value)
            and issubclass(value, paddle.nn.Layer)
            and value.__module__.startswith("paddle.nn.")
        ):
            return PaddleLayerClassVariable(value, graph, tracker)
        return None
