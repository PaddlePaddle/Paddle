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

import typing

from paddle.base import core, libpaddle
from paddle.base.libpaddle import (
    _get_current_raw_stream as _cuda_getCurrentRawStream,  # noqa: F401
)

# Define _GLIBCXX_USE_CXX11_ABI based on compilation flags
_GLIBCXX_USE_CXX11_ABI = getattr(libpaddle, '_GLIBCXX_USE_CXX11_ABI', True)
_PYBIND11_COMPILER_TYPE = getattr(libpaddle, '_PYBIND11_COMPILER_TYPE', "")
_PYBIND11_STDLIB = getattr(libpaddle, '_PYBIND11_STDLIB', "")
_PYBIND11_BUILD_ABI = getattr(libpaddle, '_PYBIND11_BUILD_ABI', "")


def _get_custom_class_python_wrapper(
    namespace_name: str, class_name: str
) -> typing.Any:
    return core.torch_compat._get_custom_class_python_wrapper(
        namespace_name, class_name
    )
