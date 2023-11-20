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

from paddle.utils.cpp_extension import CppExtension, CUDAExtension, setup

# Mac-CI don't support GPU
Extension = CppExtension if IS_MAC else CUDAExtension
sources = ['inference_gap.cc']
if not IS_MAC:
    sources.append('inference_gap.cu')
    extra_compile_args["cxx"] = ["-DPADDLE_WITH_CUDA", "-DPADDLE_WITH_TENSORRT"]

setup(
    name='gap_op_setup',
    ext_modules=Extension(
        sources=sources,
        include_dirs=paddle_includes,
        extra_compile_args=extra_compile_args,
        verbose=True,
    ),
)
