# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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


class SotCapturedException(RuntimeError):
    # Represents an exception encountered during bytecode execution simulation.
    # This exception is used by SOT to handle Python exceptions by mapping them to
    # SotCapturedException for consistent exception handling in the simulation process.
    ...


# ------------ LookupError ------------
class SotCapturedLookupError(SotCapturedException): ...


class SotCapturedIndexError(SotCapturedLookupError): ...


class SotCapturedKeyError(SotCapturedLookupError): ...


# ------------ ArithmeticError ------------
class SotCapturedArithmeticError(SotCapturedException): ...


class SotCapturedFloatingPointError(SotCapturedArithmeticError): ...


class SotCapturedOverflowError(SotCapturedArithmeticError): ...


class SotCapturedZeroDivisionError(SotCapturedArithmeticError): ...


# ------------ ImportError ------------
class SotCapturedImportError(SotCapturedException): ...


class SotCapturedModuleNotFoundError(SotCapturedImportError): ...


# ------------ RuntimeError ------------
class SotCapturedRuntimeError(SotCapturedException): ...


class SotCapturedNotImplementedError(SotCapturedRuntimeError): ...


# This dictionary establishes a mapping between built-in Python Exception types
# and their corresponding SotCapturedException, preserving inheritance hierarchy
sot_captured_exception_map = {
    Exception: SotCapturedException,
    LookupError: SotCapturedLookupError,
    IndexError: SotCapturedIndexError,
    KeyError: SotCapturedKeyError,
    ArithmeticError: SotCapturedArithmeticError,
    FloatingPointError: SotCapturedFloatingPointError,
    OverflowError: SotCapturedOverflowError,
    ZeroDivisionError: SotCapturedZeroDivisionError,
    ImportError: SotCapturedImportError,
    ModuleNotFoundError: SotCapturedModuleNotFoundError,
    RuntimeError: SotCapturedRuntimeError,
    NotImplementedError: SotCapturedNotImplementedError,
}


def get_sot_captured_exception(
    exc_type: type[Exception],
) -> type[SotCapturedException]:
    if isinstance(exc_type, Exception):
        # `exc_type` should not be a instance of `Exception`
        exc_type = exc_type.__class__

    if isinstance(exc_type, type) and issubclass(
        exc_type, SotCapturedException
    ):
        return exc_type

    if exc_type not in sot_captured_exception_map:
        name = getattr(exc_type, "__name__", str(exc_type))
        sot_captured_exception_map[exc_type] = type(
            f"SotCaptured{name}", (SotCapturedException,), {}
        )
    return sot_captured_exception_map[exc_type]


def create_sot_captured_exception(
    origin_exc: Exception | None = None,
    exc_type: type[Exception] | None = None,
    args: list | tuple | None = None,
    context: Exception | None = None,
    cause: Exception | None = None,
    suppress_context: bool | None = None,
    traceback: None = None,
) -> SotCapturedException:
    # transform an Exception to SotCapturedException
    args = args or []

    if origin_exc is not None:
        exc_type = origin_exc.__class__
        args = origin_exc.args
        context = origin_exc.__context__
        cause = origin_exc.__cause__
        suppress_context = origin_exc.__suppress_context__
        traceback = origin_exc.__traceback__

    new_exc_type = get_sot_captured_exception(exc_type)
    new_exc = new_exc_type(*args)
    new_exc.__cause__ = cause
    new_exc.__context__ = context
    new_exc.__suppress_context__ = suppress_context
    new_exc.__traceback__ = traceback

    return new_exc
