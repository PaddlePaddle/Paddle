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

from utils import extra_compile_args, paddle_includes

from paddle.utils.cpp_extension import CppExtension, setup

setup(
    name='custom_relu_xpu_module_setup',
    ext_modules=CppExtension(  # XPU don't support GPU
        sources=['custom_relu_op_xpu.cc'],
        include_dirs=paddle_includes,
        extra_compile_args=extra_compile_args,
        verbose=True,
    ),
)
