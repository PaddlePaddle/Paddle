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

import os
<<<<<<< HEAD

from utils import extra_compile_args, paddle_includes

import paddle
import paddle.fluid.core as core
from paddle.utils.cpp_extension import CppExtension, CUDAExtension, setup
=======
import paddle
import paddle.fluid.core as core
from paddle.utils.cpp_extension import CppExtension, CUDAExtension, setup
from utils import paddle_includes, extra_compile_args
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

if paddle.is_compiled_with_cuda():
    sources = ['custom_raw_op_kernel_op.cc', 'custom_raw_op_kernel_op.cu']
    extension = CUDAExtension
else:
    sources = ['custom_raw_op_kernel_op.cc']
    extension = CppExtension

cwd = os.path.dirname(os.path.abspath(__file__))
os.chdir(cwd)

if os.name == 'nt':
    compile_dir = os.path.join(os.environ['work_dir'], os.environ['BUILD_DIR'])
else:
    compile_dir = os.path.join(os.environ['PADDLE_ROOT'], 'build')

macros = []
if core.is_compiled_with_mkldnn():
    macros.append(("PADDLE_WITH_MKLDNN", None))
if core.is_compiled_with_nccl():
    macros.append(("PADDLE_WITH_NCCL", None))
macros.append(("THRUST_IGNORE_CUB_VERSION_CHECK", None))

include_dirs = list(paddle_includes) + [cwd]
<<<<<<< HEAD
setup(
    name=os.getenv("MODULE_NAME", "custom_raw_op_kernel_op_setup"),
    ext_modules=extension(
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        _compile_dir=compile_dir,
        define_macros=macros,
    ),
)
=======
setup(name=os.getenv("MODULE_NAME", "custom_raw_op_kernel_op_setup"),
      ext_modules=extension(sources=sources,
                            include_dirs=include_dirs,
                            extra_compile_args=extra_compile_args,
                            _compile_dir=compile_dir,
                            define_macros=macros))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
