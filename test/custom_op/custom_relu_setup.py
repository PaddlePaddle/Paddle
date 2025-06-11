# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import site
import sys

from utils import IS_MAC, extra_compile_args, paddle_includes

from paddle.utils.cpp_extension import CppExtension, CUDAExtension, setup

# Mac-CI don't support GPU
Extension = CppExtension if IS_MAC else CUDAExtension
sources = ['custom_relu_op.cc', 'custom_relu_op_dup.cc']
if not IS_MAC:
    sources.append('custom_relu_op.cu')

# custom_relu_op_dup.cc is only used for multi ops test,
# not a new op, if you want to test only one op, remove this
# source file
site_dir = site.getsitepackages()[0]
sys.argv.extend(["egg_info", f"--egg-base={site_dir}"])

setup(
    name='custom_relu_module_setup',
    ext_modules=Extension(  # test for not specific name here.
        sources=sources,  # test for multi ops
        include_dirs=paddle_includes,
        extra_compile_args=extra_compile_args,
        verbose=True,
    ),
)
