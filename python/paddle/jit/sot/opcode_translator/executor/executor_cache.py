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
import sys
import traceback
import types
from typing import List, Tuple

from ...profiler import EventGuard, event_register
from ...psdb import NO_FALLBACK_CODES
from ...utils import (
    BreakGraphError,
    FallbackError,
    InnerError,
    Singleton,
    is_strict_mode,
    log,
    log_do,
)
from ..custom_code import CustomCode
from .guard import Guard
from .opcode_executor import OpcodeExecutor, OpcodeExecutorBase

GuardedFunction = Tuple[CustomCode, Guard]
GuardedFunctions = List[GuardedFunction]

dummy_guard: Guard = lambda frame: True
dummy_guard.expr = "lambda frame: True"
dummy_guard.lambda_expr = "lambda frame: True"


@Singleton
class OpcodeExecutorCache:
    """
    A singleton class that implements a cache for translated instructions.
    This cache is used to store previously translated instructions along with their corresponding guard functions.

    Attributes:
        cache (dict): A dictionary that maps code objects to tuples of a cache getter function and a list of guarded functions.
        translate_count (int): The count of how many instructions have been translated. It is used to test whether the cache hits.
    """

    MAX_CACHE_SIZE = 20
    cache: dict[types.CodeType, GuardedFunctions]
    translate_count: int

    def __init__(self):
        self.cache = {}
        self.translate_count = 0

    def clear(self):
        """
        Clears the cache and resets the translate count.
        """
        self.cache.clear()
        self.translate_count = 0

    def __call__(self, frame: types.FrameType, **kwargs) -> CustomCode:
        code: types.CodeType = frame.f_code
        if code not in self.cache:
            log(2, f"[Cache]: Firstly call {code}\n")
            new_custom_code, guard_fn = self.translate(frame, **kwargs)
            self.cache[code] = [(new_custom_code, guard_fn)]
            return new_custom_code
        guarded_fns = self.cache[code]
        return self.lookup(frame, guarded_fns, **kwargs)

    @event_register("lookup")
    def lookup(
        self, frame: types.FrameType, guarded_fns: GuardedFunctions, **kwargs
    ) -> CustomCode:
        """
        Looks up the cache for a matching code object and returns a custom code object if a matching guard function is found, otherwise None.

        Args:
            frame (types.FrameType): The frame whose code object needs to be looked up in the cache.
            guarded_fns (GuardedFunctions): The list of guarded functions associated with the code object.

        Returns:
            CustomCode | None: The custom code object if a matching guard function is found, otherwise None.
        """

        if len(guarded_fns) >= self.MAX_CACHE_SIZE:
            log(2, "[Cache]: Exceed max cache size, skip it\n")
            return CustomCode(None, False)

        for custom_code, guard_fn in guarded_fns:
            try:
                with EventGuard("try guard"):
                    guard_result = guard_fn(frame)
                if guard_result:
                    log(
                        2,
                        f"[Cache]: Cache hit, Guard is \n{getattr(guard_fn, 'expr', 'None')}\n",
                    )
                    return custom_code
                else:
                    log_do(
                        4,
                        self.analyse_guard_global_object(guard_fn),
                    )
                    log(
                        2,
                        f"[Cache]: Cache miss, Guard is \n{getattr(guard_fn, 'expr', 'None')}\n",
                    )
                    log_do(
                        2,
                        self.analyse_guard_error(guard_fn, frame),
                    )
            except Exception as e:
                log(2, f"[Cache]: Guard function error: {e}\n")
                continue

        log(2, "[Cache]: all guards missed\n")
        new_custom_code, guard_fn = self.translate(frame, **kwargs)
        guarded_fns.append((new_custom_code, guard_fn))
        return new_custom_code

    def translate(
        self, frame: types.FrameType, **kwargs
    ) -> tuple[CustomCode, Guard]:
        """
        Translates the given frame's code object and returns the cache getter function and a guarded function for the translated code object.

        Args:
            frame (types.FrameType): The frame whose code object needs to be translated.

        Returns:
            tuple[CustomCode, Guard]: The cache getter function and a guarded function for the translated code object.
        """
        code: types.CodeType = frame.f_code
        self.translate_count += 1
        custom_new_code, guard_fn = start_translate(frame, **kwargs)
        return custom_new_code, guard_fn

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
            guard_expr = guard_fn.lambda_expr
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
                        f"[Cache]: skip checking {guard_str}\n         because error occured {e}"
                    )
                if result is False:
                    print(f"[Cache]: missed at {guard_str}")
                    return
            print("[Cache]: missed guard not found.")

        return inner


def start_translate(frame: types.FrameType, **kwargs) -> GuardedFunction:
    """
    Starts the translation process for the given frame and returns the translated code object and its guard function, or None if translation fails.

    Args:
        frame: The frame to be translated.

    Returns:
        GuardedFunction | None: The translated code object and its guard function, or None if translation fails.
    """
    if sys.version_info >= (3, 11):
        for const in frame.f_code.co_consts:
            if isinstance(const, types.CodeType) and const.co_name.startswith(
                "<"
            ):
                log(2, f"Found code object {const.co_name}, skip it\n")
                return CustomCode(None, False), dummy_guard
    simulator = OpcodeExecutor(frame, **kwargs)
    try:
        simulator.check_code_simulatable()
        new_custom_code, guard_fn = simulator.transform()
        return new_custom_code, guard_fn
    # TODO(zrr1999): InnerError maybe place before (FallbackError, BreakGraphError)
    # TODO(0x45f): handle BreakGraphError to trigger fallback
    except BreakGraphError as e:
        raise RuntimeError(
            f"Found BreakGraphError raised, it should not be catch at start_translate!\n{e}"
        )
    except FallbackError as e:
        if simulator._code in NO_FALLBACK_CODES:
            raise InnerError(
                f"{simulator._code.co_name} should not fallback, but got '{e}'"
            )
        # if disable_eval_frame is True, it means we want fallback to speedup rather than error occured
        if is_strict_mode() and e.disable_eval_frame is False:
            raise
        log(
            2,
            f"Unsupport Frame is {frame.f_code}, error message is: \n"
            + "".join(traceback.format_exception(type(e), e, e.__traceback__)),
        )
        # simulation not complete, not sure whether this code has sir, set disable_eval_frame = False
        guard_fn = (
            dummy_guard if e.disable_eval_frame is False else simulator.guard_fn
        )
        return (
            CustomCode(None, e.disable_eval_frame),
            guard_fn,
        )
    except Exception as e:
        raise InnerError(OpcodeExecutorBase.error_message_summary(e)) from e
    finally:
        simulator.cleanup()
        del simulator
        gc.collect()
