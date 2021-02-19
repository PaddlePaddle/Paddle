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
import os

from utils import paddle_includes, extra_compile_args
from paddle.utils.cpp_extension import CUDAExtension, setup
from paddle.utils.cpp_extension.extension_utils import use_new_custom_op_load_method

# switch to old custom op method
use_new_custom_op_load_method(False)

setup(
    name='custom_relu2',
    ext_modules=CUDAExtension(  # test for not specific name here.
        sources=['relu_op.cc', 'relu_op.cu', 'relu_op3.cc',
                 'relu_op3.cu'],  # test for multi ops
        include_dirs=paddle_includes,
        extra_compile_args=extra_compile_args))
