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

import traceback

from .info_collector import BreakGraphReasonInfo


class BreakGraphReasonBase:
    """Base class for representing reasons why graph execution was interrupted.

    Attributes:
        reason_str (str): Description of the break reason
        file_path (str): Path to the file where break occurred
        line_number (int): Line number where break occurred
    """

    def __init__(
        self,
        reason_str,
        file_path="",
        line_number=-1,
    ):
        self.reason_str = reason_str
        self.file_path = file_path
        self.line_number = line_number

    def __repr__(self) -> str:
        return f"{self.reason_str}"


class DataDependencyBreak(BreakGraphReasonBase):
    pass


class DataDependencyControlFlowBreak(DataDependencyBreak):
    """Break reason for control flow execution."""

    def __init__(self, reason_str=None, file_path="", line_number=-1):
        if reason_str is None:
            reason_str = "OpcodeInlineExecutor want break graph when simulate control flow."

        super().__init__(
            reason_str,
            file_path,
            line_number,
        )


class DataDependencyDynamicShapeBreak(DataDependencyBreak):
    pass


class DataDependencyOperationBreak(DataDependencyBreak):
    pass


class UnsupportedOperationBreak(BreakGraphReasonBase):
    def __init__(
        self,
        *,
        left_type=None,
        right_type=None,
        operator=None,
        reason_str=None,
        file_path="",
        line_number=-1,
    ):
        if reason_str is None:
            reason_str = f"Unsupported operator '{operator}' between {left_type} and {right_type}"
        super().__init__(reason_str, file_path, line_number)


class UnsupportedPaddleAPIBreak(UnsupportedOperationBreak):
    def __init__(
        self,
        *,
        fn_name=None,
        reason_str=None,
        file_path="",
        line_number=-1,
    ):
        if reason_str is None:
            reason_str = f"Not support Paddlepaddle API: {fn_name}"

        super().__init__(
            reason_str=reason_str,
            file_path=file_path,
            line_number=line_number,
        )


class BuiltinFunctionBreak(UnsupportedOperationBreak):
    """Break reason for unsupported built-in function calls.

    Args:
        fn_name (str): Name of the builtin function
        arg_types (list): Types of the arguments passed to the function
        file_path (str): Path to the file where break occurred
        line_number (int): Line number where break occurred
    """

    def __init__(
        self,
        *,
        fn_name=None,
        arg_types=None,
        reason_str=None,
        file_path="",
        line_number=-1,
    ):
        if reason_str is None:
            reason_str = f"Not support builtin function: {fn_name} with args: Args({arg_types})"

        super().__init__(
            reason_str=reason_str,
            file_path=file_path,
            line_number=line_number,
        )


class SideEffectBreak(BreakGraphReasonBase):
    pass


class UnsupportedIteratorBreak(SideEffectBreak):
    pass


class InlineCallBreak(BreakGraphReasonBase):
    pass


class FallbackInlineCallBreak(InlineCallBreak):
    pass


class BreakGraphInlineCallBreak(InlineCallBreak):
    pass


class OtherInlineCallBreak(InlineCallBreak):
    pass


class DygraphInconsistentWithStaticBreak(BreakGraphReasonBase):
    pass


class PsdbBreakReason(BreakGraphReasonBase):
    pass


class InferMetaBreak(BreakGraphReasonBase):
    """Break reason during meta information inference phase."""

    pass


class SotErrorBase(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from ..opcode_translator.breakpoint import BreakpointManager

        BreakpointManager().on_event(f"{self.__class__.__name__}")

    def print(self):
        lines = traceback.format_tb(self.__traceback__)
        print("".join(lines))


class InnerError(SotErrorBase):
    pass


class HasNoAttributeError(InnerError):
    pass


class FallbackError(SotErrorBase):
    def __init__(self, msg, disable_eval_frame=False):
        super().__init__(msg)
        self.disable_eval_frame = disable_eval_frame


# raise in inline function call strategy.
class BreakGraphError(SotErrorBase):
    def __init__(self, reason: BreakGraphReasonBase = None):
        super().__init__(str(reason))

        if not isinstance(reason, BreakGraphReasonBase):
            raise ValueError(
                "reason must be a subclass of BreakGraphReasonBase"
            )

        self.reason = reason
        BreakGraphReasonInfo.collect_break_graph_reason(reason)


def inner_error_default_handler(func, message_fn):
    """Wrap function and an error handling function and throw an InnerError."""

    def impl(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except SotErrorBase as e:
            raise e
        except Exception as e:
            message = message_fn(*args, **kwargs)
            origin_exception_message = "\n".join(
                traceback.format_exception(type(e), e, e.__traceback__)
            )
            raise InnerError(
                f"{message}\nOrigin Exception is: \n {origin_exception_message}"
            ) from e

    return impl


class ExportError(SotErrorBase):
    pass


class SotExtraInfo:
    SOT_EXTRA_INFO_ATTR_NAME = "__SOT_EXTRA_INFO__"

    def __init__(self, *, need_breakgraph: bool = False):
        self.need_breakgraph = need_breakgraph

    def set_need_breakgraph(self, need_breakgraph: bool):
        self.need_breakgraph = need_breakgraph

    def attach(self, err: BaseException):
        setattr(err, SotExtraInfo.SOT_EXTRA_INFO_ATTR_NAME, self)

    @staticmethod
    def default() -> SotExtraInfo:
        return SotExtraInfo()

    @staticmethod
    def from_exception(err: BaseException) -> SotExtraInfo:
        info = getattr(
            err, SotExtraInfo.SOT_EXTRA_INFO_ATTR_NAME, SotExtraInfo.default()
        )
        setattr(err, SotExtraInfo.SOT_EXTRA_INFO_ATTR_NAME, info)
        return info
