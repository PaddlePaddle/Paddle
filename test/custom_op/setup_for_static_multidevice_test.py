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

from utils import IS_MAC, extra_compile_args, paddle_includes

import paddle
from paddle.utils.cpp_extension import CppExtension, CUDAExtension, setup

if paddle.framework.core.is_compiled_with_xpu():
    source_files = ['custom_relu_op_xpu.cc']
    setup(
        name='custom_setup_op_relu_model_static_multidevices',
        ext_modules=CppExtension(  # XPU don't support GPU
            sources=['custom_relu_op_xpu.cc'],
            include_dirs=paddle_includes,
            extra_compile_args=extra_compile_args,
            verbose=True,
        ),
    )
else:
    source_files = ['custom_relu_op.cc']
    if not IS_MAC:
        source_files.append('custom_relu_op.cu')
    setup(
        name='custom_setup_op_relu_model_static_multidevices',
        ext_modules=CUDAExtension(
            sources=source_files,
            include_dirs=paddle_includes,
            extra_compile_args=extra_compile_args,
            verbose=True,
        ),
    )
