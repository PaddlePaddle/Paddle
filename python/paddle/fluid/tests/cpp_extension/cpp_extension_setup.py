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

import os
from site import getsitepackages

from paddle.utils.cpp_extension import CppExtension, setup

paddle_includes = []
for site_packages_path in getsitepackages():
    paddle_includes.append(
        os.path.join(site_packages_path, 'paddle', 'include')
    )
    paddle_includes.append(
        os.path.join(site_packages_path, 'paddle', 'include', 'third_party')
    )

setup(
    name='custom_cpp_extension',
    ext_modules=CppExtension(
        sources=["custom_add.cc", "custom_sub.cc"],
        include_dirs=paddle_includes
        + [os.path.dirname(os.path.abspath(__file__))],
        extra_compile_args={'cc': ['-w', '-g']},
        verbose=True,
    ),
)
