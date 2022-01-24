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
from paddle.fluid import core
from distutils.sysconfig import get_python_lib
from distutils.core import setup, Extension

# cc flags
paddle_extra_compile_args = ['-std=c++14', '-shared', '-fPIC']
if core.is_compiled_with_npu():
    paddle_extra_compile_args += ['-D_GLIBCXX_USE_CXX11_ABI=0']

# include path
site_packages_path = get_python_lib()
paddle_custom_kernel_include = [
    os.path.join(site_packages_path, 'paddle', 'include'),
]

# libs path
paddle_custom_kernel_library_dir = [
    os.path.join(site_packages_path, 'paddle', 'fluid'),
]

# libs
libs = [':core_avx.so']
if not core.has_avx_core and core.has_noavx_core:
    libs = [':core_noavx.so']

custom_kernel_dot_module = Extension(
    'custom_kernel_dot',
    sources=['custom_kernel_dot.cc'],
    include_dirs=paddle_custom_kernel_include,
    library_dirs=paddle_custom_kernel_library_dir,
    libraries=libs,
    extra_compile_args=paddle_extra_compile_args)

setup(
    name='custom_kernel_dot',
    version='1.0',
    description='custom kernel fot compiling',
    ext_modules=[custom_kernel_dot_module])
