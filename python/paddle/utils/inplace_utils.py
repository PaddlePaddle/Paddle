# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os
import warnings
from typing import TYPE_CHECKING, Callable, TypeVar

from typing_extensions import ParamSpec

import paddle  # noqa: F401
from paddle.base.framework import stride_ops
from paddle.base.wrapped_decorator import wrap_decorator
from paddle.framework import in_dynamic_mode, in_pir_mode

_InputT = ParamSpec("_InputT")
_RetT = TypeVar("_RetT")

if TYPE_CHECKING:
    from paddle.pir import Value


def check_view_value(value: Value) -> bool:
    # check if the value is a view tensor
    if value.get_defining_op().name() in stride_ops:
        # TODO(ooooo-create): The `x = stride_op(x)` shouldn't return True.
        return True
    all_used_ops = value.all_used_ops()
    if len(all_used_ops) == 0:
        return False
    for op in all_used_ops:
        if op.name() in stride_ops and op.operand_source(0).is_same(value):
            # TODO(ooooo-create): The `y = stride_op(x).clone()` and `y = stride_op(x) + op` should also return False.
            # Now is True.
            return True
    return False


# NOTE(pangyoki): The Inplace APIs with underline(`_`) is only valid for the method of calling `_C_ops`
# in dygraph mode. If static graph mode is used, the inplace mechanism will not be used, and the static method
# of the original API will be called.
# NOTE(GGBond8488): Simply run the original version of the API under the static graph mode has a low
# probability that the result is inconsistent with the dynamic graph.
def _inplace_apis_in_dygraph_only_(
    func: Callable[_InputT, _RetT]
) -> Callable[_InputT, _RetT]:
    def __impl__(*args: _InputT.args, **kwargs: _InputT.kwargs) -> _RetT:
        if not in_dynamic_mode():
            origin_api_name = func.__name__[:-1]
            warnings.warn(
                f"In static graph mode, {func.__name__}() is the same as {origin_api_name}() and does not perform inplace operation."
            )
            from ..base.dygraph.base import in_to_static_mode

            if in_to_static_mode():
                stride_in_no_check_dy2st_diff = os.environ.get(
                    "stride_in_no_check_dy2st_diff", "0"
                )
                if in_pir_mode():
                    if (
                        stride_in_no_check_dy2st_diff != '1'
                        and check_view_value(args[0])
                    ):
                        raise ValueError(
                            f'Sorry about what\'s happened. In to_static mode, {func.__name__}\'s output variable is a viewed Tensor in dygraph. This will result in inconsistent calculation behavior between dynamic and static graphs. You must find the location of the strided API be called, and call paddle.assign() before inplace input.'
                        )
                else:
                    for arg in args:
                        if hasattr(arg, "is_view_var") and arg.is_view_var:
                            raise ValueError(
                                f'Sorry about what\'s happened. In to_static mode, {func.__name__}\'s output variable {arg.name} is a viewed Tensor in dygraph. This will result in inconsistent calculation behavior between dynamic and static graphs. You must find the location of the strided API be called, and call {arg.name} = paddle.assign({arg.name}).'
                            )

            origin_func = f"{func.__module__}.{origin_api_name}"
            return eval(origin_func)(*args, **kwargs)
        return func(*args, **kwargs)

    return __impl__


inplace_apis_in_dygraph_only = wrap_decorator(_inplace_apis_in_dygraph_only_)
