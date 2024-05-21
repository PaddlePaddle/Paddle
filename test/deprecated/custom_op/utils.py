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
from site import getsitepackages

import numpy as np

from paddle.utils.cpp_extension.extension_utils import IS_WINDOWS

IS_MAC = sys.platform.startswith('darwin')

# Note(Aurelius84): We use `add_test` in Cmake to config how to run unittest in CI.
# `PYTHONPATH` will be set as `build/python/paddle` that will make no way to find
# paddle include directory. Because the following path is generated after installing
# PaddlePaddle whl. So here we specific `include_dirs` to avoid errors in CI.
paddle_includes = []
paddle_libraries = []
for site_packages_path in getsitepackages():
    paddle_includes.append(
        os.path.join(site_packages_path, 'paddle', 'include')
    )
    paddle_includes.append(
        os.path.join(site_packages_path, 'paddle', 'include', 'third_party')
    )
    paddle_libraries.append(os.path.join(site_packages_path, 'paddle', 'libs'))

# Test for extra compile args
extra_cc_args = ['-w', '-g'] if not IS_WINDOWS else ['/w']
extra_nvcc_args = ['-O3']
extra_compile_args = {'cc': extra_cc_args, 'nvcc': extra_nvcc_args}


def check_output(out, pd_out, name):
    if out is None and pd_out is None:
        return
    assert out is not None, "out value of " + name + " is None"
    assert pd_out is not None, "pd_out value of " + name + " is None"
    if isinstance(out, list) and isinstance(pd_out, list):
        for idx in range(len(out)):
            np.testing.assert_array_equal(
                out[idx],
                pd_out[idx],
                err_msg=f'custom op {name}: {out[idx]},\n paddle api {name}: {pd_out[idx]}',
            )
    else:
        np.testing.assert_array_equal(
            out,
            pd_out,
            err_msg=f'custom op {name}: {out},\n paddle api {name}: {pd_out}',
        )


def check_output_allclose(out, pd_out, name, rtol=5e-5, atol=1e-2):
    if out is None and pd_out is None:
        return
    assert out is not None, "out value of " + name + " is None"
    assert pd_out is not None, "pd_out value of " + name + " is None"
    np.testing.assert_allclose(
        out,
        pd_out,
        rtol,
        atol,
        err_msg=f'custom op {name}: {out},\n paddle api {name}: {pd_out}',
    )
