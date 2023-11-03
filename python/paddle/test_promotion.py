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

import inspect
from functools import wraps
from typing import Callable, Optional, Sequence

import numpy as np

import paddle
from paddle.base.framework import Variable
from paddle.common_ops_import import VarDesc

Number = (bool, int, float, complex)


class judge_dtype_for_type_promotion:
    def __init__(
        self,
        *,
        type_promoting_args: Optional[Sequence[str]] = None,
    ):
        self.type_promoting_arg_names = type_promoting_args

    def __call__(self, fn: Callable) -> Callable:
        sig = inspect.signature(fn)

        @wraps(fn)
        def _fn(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            x = bound.arguments['x']
            y = bound.arguments['y']
            got_sclar = False
            got_numpy = False
            if isinstance(x, (Variable, paddle.Tensor)):
                x_dtype = x.dtype
            elif isinstance(x, Number):
                x_dtype = type(x)
                got_sclar = True
            elif isinstance(x, np.ndarray):
                x_dtype = x.dtype
                got_numpy = True
            else:
                print("got unknown type: ", x)
                x_dtype = None

            if isinstance(y, (Variable, paddle.Tensor)):
                y_dtype = y.dtype
            elif isinstance(y, Number):
                y_dtype = type(y)
                got_sclar = True
            elif isinstance(y, np.ndarray):
                y_dtype = y.dtype
                got_numpy = True
            else:
                print("got unknown type: ", y)
                y_dtype = None

            if x_dtype != y_dtype:
                if got_sclar:
                    # sclar + tensor, int=int64, float=float32, bool=bool, complex=complex64
                    if (
                        (
                            isinstance(x, float)
                            and isinstance(y, (paddle.Tensor, Variable))
                            and y.dtype == VarDesc.VarType.FP32
                        )
                        or (
                            isinstance(y, float)
                            and isinstance(x, (paddle.Tensor, Variable))
                            and x.dtype == VarDesc.VarType.FP32
                        )
                        or (
                            isinstance(x, int)
                            and isinstance(y, (paddle.Tensor, Variable))
                            and y.dtype == VarDesc.VarType.INT64
                        )
                        or (
                            isinstance(y, int)
                            and isinstance(x, (paddle.Tensor, Variable))
                            and x.dtype == VarDesc.VarType.INT64
                        )
                        or (
                            isinstance(x, bool)
                            and isinstance(y, (paddle.Tensor, Variable))
                            and y.dtype == VarDesc.VarType.BOOL
                        )
                        or (
                            isinstance(y, bool)
                            and isinstance(x, (paddle.Tensor, Variable))
                            and x.dtype == VarDesc.VarType.BOOL
                        )
                        or (
                            isinstance(x, complex)
                            and isinstance(y, (paddle.Tensor, Variable))
                            and y.dtype == VarDesc.VarType.COMPLEX64
                        )
                        or (
                            isinstance(y, complex)
                            and isinstance(x, (paddle.Tensor, Variable))
                            and x.dtype == VarDesc.VarType.COMPLEX64
                        )
                    ):
                        print(
                            "got sclar compute with tensor, x.dtype: ",
                            x_dtype,
                            " y.dtype: ",
                            y_dtype,
                        )
                        result = fn(**bound.arguments)
                        return result
                if got_numpy:
                    # numpy array + tensor, int=int64, float=float64, bool=bool, complex=complex128
                    if (
                        (
                            isinstance(x, np.ndarray)
                            and x.dtype == np.float64
                            and isinstance(y, (paddle.Tensor, Variable))
                            and y.dtype == VarDesc.VarType.FP64
                        )
                        or (
                            isinstance(y, np.ndarray)
                            and y_dtype.dtype == np.float64
                            and isinstance(x, (paddle.Tensor, Variable))
                            and x.dtype == VarDesc.VarType.FP64
                        )
                        or (
                            isinstance(x, np.ndarray)
                            and x.dtype == np.int64
                            and isinstance(y, (paddle.Tensor, Variable))
                            and y.dtype == VarDesc.VarType.INT64
                        )
                        or (
                            isinstance(y, np.ndarray)
                            and y_dtype.dtype == np.int64
                            and isinstance(x, (paddle.Tensor, Variable))
                            and x.dtype == VarDesc.VarType.INT64
                        )
                        or (
                            isinstance(x, np.ndarray)
                            and x.dtype == np.bool_
                            and isinstance(y, (paddle.Tensor, Variable))
                            and y.dtype == VarDesc.VarType.BOOL
                        )
                        or (
                            isinstance(y, np.ndarray)
                            and y_dtype.dtype == np.bool_
                            and isinstance(x, (paddle.Tensor, Variable))
                            and x.dtype == VarDesc.VarType.BOOL
                        )
                        or (
                            isinstance(x, np.ndarray)
                            and x.dtype == np.complex128
                            and isinstance(y, (paddle.Tensor, Variable))
                            and y.dtype == VarDesc.VarType.COMPLEX128
                        )
                        or (
                            isinstance(y, np.ndarray)
                            and y_dtype.dtype == np.complex128
                            and isinstance(x, (paddle.Tensor, Variable))
                            and x.dtype == VarDesc.VarType.COMPLEX128
                        )
                    ):
                        print(
                            "got numpy array compute with tensor, x.dtype: ",
                            x_dtype,
                            " y.dtype: ",
                            y_dtype,
                        )
                        result = fn(**bound.arguments)
                        return result
                raise ValueError(
                    f"got different dtype for x: ({x_dtype}), y: ({y_dtype})."
                )

            result = fn(**bound.arguments)
            return result

        _fn.__signature__ = sig  # type: ignore[attr-defined]
        return _fn
