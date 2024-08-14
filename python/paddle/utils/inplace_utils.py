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

import warnings
from typing import Callable, TypeVar

from typing_extensions import ParamSpec

import paddle
from paddle.base.wrapped_decorator import wrap_decorator
from paddle.framework import check_view_value, in_dynamic_mode, in_pir_mode

_InputT = ParamSpec("_InputT")
_RetT = TypeVar("_RetT")


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
                if in_pir_mode():
                    for arg in args:
                        if isinstance(
                            arg, paddle.pir.Value
                        ) and check_view_value(arg):
                            raise ValueError(
                                f'Sorry about what\'s happened. In to_static mode, {func.__name__}\'s output variable {arg.name} is a viewed Tensor in dygraph. This will result in inconsistent calculation behavior between dynamic and static graphs. You must find the location of the strided API be called, and call {arg.name} = {arg.name}.assign().'
                            )
                else:
                    for arg in args:
                        if hasattr(arg, "is_view_var") and arg.is_view_var:
                            raise ValueError(
                                f'Sorry about what\'s happened. In to_static mode, {func.__name__}\'s output variable {arg.name} is a viewed Tensor in dygraph. This will result in inconsistent calculation behavior between dynamic and static graphs. You must find the location of the strided API be called, and call {arg.name} = {arg.name}.assign().'
                            )
                    # check()
                    # 定义的 op，作为第一个返回值
                    # define_op == "pd_op.transpose" and define_op.result(0) == "x"
                    # 被 op 使用，并作为第一个参数
                    # for all_used_ops, "x" == op.operand_source(0)

            origin_func = f"{func.__module__}.{origin_api_name}"
            return eval(origin_func)(*args, **kwargs)
        return func(*args, **kwargs)

    return __impl__


inplace_apis_in_dygraph_only = wrap_decorator(_inplace_apis_in_dygraph_only_)
