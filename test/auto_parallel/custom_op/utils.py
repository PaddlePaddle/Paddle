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

import numpy as np
from site import getsitepackages
# Test for extra compile args
extra_cc_args = ['-w', '-g']
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
                err_msg='custom op {}: {},\n paddle api {}: {}'.format(
                    name, out[idx], name, pd_out[idx]
                ),
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


def get_paddle_includes():
    env_dict = os.environ
    paddle_source_dir = env_dict.get("PADDLE_SOURCE_DIR")
    paddle_includes = []
    paddle_includes.append(f"{env_dict.get('CMAKE_BINARY_DIR')}")
    paddle_includes.append(f"{env_dict.get('PROTOBUF_INCLUDE_DIR')}")
    paddle_includes.append(f"{paddle_source_dir}")
    
    # mkldnn
    if env_dict.get("WITH_MKLDNN") == 'ON':
        paddle_includes.append(f"{env_dict.get('MKLDNN_INSTALL_DIR')}/include")
    if env_dict.get("WITH_GPU") == 'ON' or env_dict.get("WITH_ROCM") == 'ON':
        paddle_includes.append(f"{env_dict.get('externalError_INCLUDE_DIR')}")
    paddle_includes.append(f"{env_dict.get('PYBIND_INCLUDE_DIR')}")

    for site_packages_path in getsitepackages():
        paddle_includes.append(
            os.path.join(site_packages_path, 'paddle', 'include')
        )
        paddle_includes.append(
            os.path.join(site_packages_path, 'paddle', 'include', 'third_party')
        )
        
    return paddle_includes


paddle_includes = get_paddle_includes()
