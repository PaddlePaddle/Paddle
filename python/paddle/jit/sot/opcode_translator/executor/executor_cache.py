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

import gc
import traceback
from typing import TYPE_CHECKING, List, Tuple

import paddle
from paddle.base.dygraph.base import sot_simulation_mode_guard

from ...profiler import EventGuard, event_register
from ...psdb import NO_FALLBACK_CODES
from ...utils import (
    ENV_SOT_ALLOW_DYNAMIC_SHAPE,
    ENV_SOT_ENABLE_GUARD_TREE,
    ENV_SOT_ENABLE_STRICT_GUARD_CHECK,
    BreakGraphError,
    CompileCountInfo,
    ConditionalFallbackError,
    FallbackError,
    InfoCollector,
    InnerError,
    Singleton,
    is_strict_mode,
    log,
    log_do,
)
from ..custom_code import CustomCode
from .function_graph import FunctionGraph
from .guard import Guard
from .opcode_executor import OpcodeExecutor, OpcodeExecutorBase
from .virtual_frame import VirtualFrame

if TYPE_CHECKING:
    import types

GuardedFunction = Tuple[CustomCode, Guard]
GuardedFunctions = List[GuardedFunction]
GuardChain = List[paddle.framework.core.GuardNode]
GuardChainList = List[GuardChain]

dummy_guard: Guard = lambda frame: True
dummy_guard.expr = "lambda frame: True"
dummy_guard.inlined_expr = "lambda frame: True"
if ENV_SOT_ENABLE_STRICT_GUARD_CHECK.get():
    dummy_guard.mirror_guard = lambda frame: True


class OpcodeExecutorCache(metaclass=Singleton):
    """
    A singleton class that implements a cache for translated instructions.
    This cache is used to store previously translated instructions along with their corresponding guard functions.

    Attributes:
        cache (dict): A dictionary that maps code objects to tuples of a cache getter function and a list of guarded functions.
        translate_count (int): The count of how many instructions have been translated. It is used to test whether the cache hits.
    """

    MAX_CACHE_SIZE = 20
    cache: dict[
        types.CodeType, tuple[GuardedFunctions, paddle.framework.core.GuardTree]
    ]
    translate_count: int
    code_symbolic_inputs: dict[types.CodeType, dict[str, None | dict[int, int]]]

    def __init__(self):
        self.cache = {}
        self.translate_count = 0
        self.code_symbolic_inputs = {}

    def get_symbolic_inputs(
        self, code: types.CodeType
    ) -> dict[str, dict[int, int] | None]:
        self.code_symbolic_inputs.setdefault(code, {})
        return self.code_symbolic_inputs[code]

    def clear(self):
        """
        Clears the cache and resets the translate count.
        """
        self.cache.clear()
        self.translate_count = 0
        self.code_symbolic_inputs.clear()

    def dump_state(self):
        return {
            "cache": self.cache,
            "translate_count": self.translate_count,
            "code_symbolic_inputs": self.code_symbolic_inputs,
        }

    def load_state(self, state):
        self.cache = state["cache"]
        self.translate_count = state["translate_count"]
        self.code_symbolic_inputs = state["code_symbolic_inputs"]

    def __call__(self, frame: types.FrameType, **kwargs) -> CustomCode:
        code: types.CodeType = frame.f_code
        if code not in self.cache:
            log(2, f"[Cache]: Firstly call {code}\n")
            new_custom_code, guard_fn, guard_chain = self.translate(
                frame, **kwargs
            )
            assert guard_fn is not None
            assert guard_chain is not None
            self.cache[code] = [
                (new_custom_code, guard_fn)
            ], paddle.framework.core.GuardTree([guard_chain])
            return new_custom_code
        guarded_fns, guard_tree = self.cache[code]
        return self.lookup(frame, guarded_fns, guard_tree, **kwargs)

    @event_register("lookup")
    def lookup(
        self,
        frame: types.FrameType,
        guarded_fns: GuardedFunctions,
        guard_tree: paddle.framework.core.GuardTree,
        **kwargs,
    ) -> CustomCode:
        """
        Looks up the cache for a matching code object and returns a custom code object if a matching guard function is found, otherwise None.

        Args:
            frame (types.FrameType): The frame whose code object needs to be looked up in the cache.
            guarded_fns (GuardedFunctions): The list of guarded functions associated with the code object.

        Returns:
            CustomCode: The custom code object if a matching guard function is found, otherwise None.
        """

        if len(guarded_fns) >= self.MAX_CACHE_SIZE:
            log(2, "[Cache]: Exceed max cache size, skip it\n")
            return CustomCode(None, False)

        enable_strict_guard = ENV_SOT_ENABLE_STRICT_GUARD_CHECK.get()
        enable_guard_tree = ENV_SOT_ENABLE_GUARD_TREE.get()

        cache_index = None
        if enable_strict_guard or enable_guard_tree:
            log(4, f"[Cache] Guard tree: \n{guard_tree.stringify()}")
            cache_index = guard_tree.lookup(frame)

        if not enable_strict_guard and enable_guard_tree:
            if cache_index is not None:
                # TODO(zrr1999): add a mapping between custom_code and cache_index
                return guarded_fns[cache_index][0]
            else:
                log(2, "[Cache]: all guards missed (guard tree mode)\n")
                new_custom_code, guard_fn, guard_chain = self.translate(
                    frame, **kwargs
                )
                if guard_fn is not None:
                    assert guard_chain is not None
                    guarded_fns.append((new_custom_code, guard_fn))
                    guard_tree.add_guard_chain(guard_chain)
                return new_custom_code

        for index, (custom_code, guard_fn) in enumerate(guarded_fns):
            if enable_strict_guard:
                mirror_guard_error = None
                try:
                    with EventGuard("try mirror guard"):
                        mirror_guard_result = guard_fn.mirror_guard(frame)
                except Exception as e:
                    log(2, f"[Cache] Mirror guard error: {e}\n")
                    mirror_guard_error = e

            try:
                with EventGuard("try guard"):
                    guard_result = guard_fn(frame)
                if enable_strict_guard:
                    assert mirror_guard_result == guard_result, (
                        "faster guard result is not equal to guard result, "
                        f"guard_expr: {getattr(guard_fn, 'expr', 'None')} \n"
                        f"faster_guard_expr: {getattr(guard_fn.mirror_guard, 'expr', 'None')},"
                    )
                if guard_result:
                    log(
                        2,
                        f"[Cache] Cache hit, Guard is \n{getattr(guard_fn, 'expr', 'None')}\n",
                    )
                    # TODO(zrr1999): cache_index should be equal to index when enable_strict_guard.
                    # assert (
                    #     cache_index is None or index == cache_index
                    # ), f"cache_index({cache_index}) is not equal to index({index})"
                    return custom_code
                else:
                    log_do(
                        4,
                        self.analyse_guard_global_object(guard_fn),
                    )
                    log(
                        2,
                        f"[Cache] Cache miss, Guard is \n{getattr(guard_fn, 'expr', 'None')}\n",
                    )
                    log_do(
                        2,
                        self.analyse_guard_error(guard_fn, frame),
                    )
            except Exception as e:
                log(2, f"[Cache] Guard function error: {e}\n")
                log(
                    2,
                    f"[Cache] Guard is \n{getattr(guard_fn, 'expr', 'None')}\n",
                )
                log_do(
                    2,
                    self.analyse_guard_error(guard_fn, frame),
                )
                if enable_strict_guard:
                    assert type(e) == type(mirror_guard_error) and str(
                        e
                    ) == str(mirror_guard_error), (
                        "mirror guard error is not equal to guard error, "
                        f"guard_error: {e} \n"
                        f"mirror_guard_error: {mirror_guard_error},"
                    )

        log(2, "[Cache]: all guards missed\n")
        new_custom_code, guard_fn, guard_chain = self.translate(frame, **kwargs)
        if guard_fn is not None:
            assert guard_chain is not None
            guarded_fns.append((new_custom_code, guard_fn))
            guard_tree.add_guard_chain(guard_chain)
        return new_custom_code

    def before_translate_hook(self, frame: types.FrameType):
        if not ENV_SOT_ALLOW_DYNAMIC_SHAPE.get():
            return

    def translate(
        self, frame: types.FrameType, **kwargs
    ) -> tuple[CustomCode, Guard | None, GuardChain | None]:
        """
        Translates the given frame's code object and returns the cache getter function and a guarded function for the translated code object.

        Args:
            frame (types.FrameType): The frame whose code object needs to be translated.

        Returns:
            tuple[CustomCode, Guard]: The cache getter function and a guarded function for the translated code object.
        """
        self.before_translate_hook(frame)
        self.translate_count += 1
        custom_new_code, guard_fn, guard_chain = start_translate(
            frame, **kwargs
        )
        return custom_new_code, guard_fn, guard_chain

    def analyse_guard_global_object(self, guard_fn):
        def inner():
            for key in guard_fn.__globals__.keys():
                if key.startswith("__object"):
                    print(
                        f"[Cache] meet global object: {key} : {guard_fn.__globals__[key]}",
                    )

        return inner

    def analyse_guard_error(self, guard_fn, frame):
        def inner():
            guard_expr = guard_fn.inlined_expr
            lambda_head = "lambda frame: "
            guard_expr = guard_expr.replace(lambda_head, "")
            guards = guard_expr.split(" and ")
            for guard_str in guards:
                guard = eval(lambda_head + guard_str, guard_fn.__globals__)
                result = False
                try:
                    result = guard(frame)
                except Exception as e:
                    print(
                        f"[Cache] Error occurred when checking guard {guard_str}: {e}"
                    )
                    return
                if result is False:
                    print(f"[Cache]: missed at {guard_str}")
                    return
            print("[Cache]: missed guard not found.")

        return inner


def start_translate(
    frame: types.FrameType,
    **kwargs,
) -> tuple[CustomCode, Guard | None, GuardChain | None]:
    """
    Starts the translation process for the given frame and returns the translated code object, its guard function and its guard tree node, or None if translation fails.

    Args:
        frame: The frame to be translated.

    Returns:
        tuple[CustomCode, Guard | None, GuardChain | None]: The translated code object, its guard function and its guard tree node, or None if translation fails.
    """
    simulator = None
    graph = FunctionGraph(frame.f_code, frame.f_globals, **kwargs)
    try:
        vframe = VirtualFrame.from_real_frame(frame, graph)
        simulator = OpcodeExecutor(vframe, graph)
        simulator.check_code_simulatable()
        InfoCollector().attach(CompileCountInfo, frame.f_code)
        with sot_simulation_mode_guard(True):
            new_custom_code, guard_fn = simulator.transform(frame)
            if ENV_SOT_ENABLE_STRICT_GUARD_CHECK.get():
                assert guard_fn(frame)
                assert guard_fn.mirror_guard(frame)
        if not simulator._graph.need_cache:
            return (
                CustomCode(None, True),
                None,
                None,
            )
        guard_chain = simulator.guard_chain
        return new_custom_code, guard_fn, guard_chain
    # TODO(0x45f): handle BreakGraphError to trigger fallback
    except BreakGraphError as e:
        raise RuntimeError(
            f"Found BreakGraphError raised, it should not be catch at start_translate!\n{e}"
        )
    except FallbackError as e:
        if frame.f_code in NO_FALLBACK_CODES:
            raise InnerError(
                f"{frame.f_code.co_name} should not fallback, but got '{e}'"
            )
        if is_strict_mode():
            raise
        log(
            2,
            f"Unsupported Frame is {frame.f_code}, error message is: \n"
            + "".join(traceback.format_exception(type(e), e, e.__traceback__)),
        )
        dummy_guard_chain = [
            # TODO(zrr1999): GuardNode should support zero-expr constructor
            paddle.framework.core.GuardNode(
                paddle.framework.core.DummyGuard(),
                [paddle.framework.core.ConstantExprNode(True)],
            )
        ]
        guard, guard_chain = dummy_guard, dummy_guard_chain

        if isinstance(e, ConditionalFallbackError):
            # Guard global variables only
            graph.input_variables.clear()
            guard = graph.guard_fn
            guard_chain = graph.guard_chain

        return (
            CustomCode(None, e.disable_eval_frame),
            guard,
            guard_chain,
        )
    except Exception as e:
        raise InnerError(OpcodeExecutorBase.error_message_summary(e)) from e
    finally:
        if simulator is not None:
            simulator.cleanup()
        del simulator
        gc.collect()
