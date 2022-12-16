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
import sys
from distutils.sysconfig import get_python_lib

from paddle.utils.cpp_extension.extension_utils import IS_WINDOWS

IS_MAC = sys.platform.startswith('darwin')

site_packages_path = get_python_lib()
# Note(Aurelius84): We use `add_test` in Cmake to config how to run unittest in CI.
# `PYTHONPATH` will be set as `build/python/paddle` that will make no way to find
# paddle include directory. Because the following path is generated after installing
# PaddlePaddle whl. So here we specific `include_dirs` to avoid errors in CI.
paddle_includes = [
    os.path.join(site_packages_path, 'paddle', 'include'),
    os.path.join(site_packages_path, 'paddle', 'include', 'third_party'),
]

# Test for extra compile args
extra_cc_args = ['-w', '-g'] if not IS_WINDOWS else ['/w']
extra_nvcc_args = ['-O3']
extra_compile_args = {'cc': extra_cc_args, 'nvcc': extra_nvcc_args}
