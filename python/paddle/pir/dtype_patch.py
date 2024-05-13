# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.base.libpaddle import DataType
from paddle.pir.core import _PADDLE_PIR_DTYPE_2_NUMPY_DTYPE

_already_patch_dtype_repr = False


def monkey_patch_dtype():
    global _already_patch_dtype_repr
    if not _already_patch_dtype_repr:
        # NOTE(zhiqiu): pybind11 will set a default __str__ method of enum class.
        # So, we need to overwrite it to a more readable one.
        # See details in https://github.com/pybind/pybind11/issues/2537.
        origin = DataType.__str__

        def dtype_str(dtype):
            if dtype in _PADDLE_PIR_DTYPE_2_NUMPY_DTYPE:
                numpy_dtype = _PADDLE_PIR_DTYPE_2_NUMPY_DTYPE[dtype]
                if numpy_dtype == 'uint16':
                    numpy_dtype = 'bfloat16'
                prefix = 'paddle.'
                return prefix + numpy_dtype
            else:
                return origin(dtype)

        DataType.__str__ = dtype_str
        _already_patch_dtype_repr = True
