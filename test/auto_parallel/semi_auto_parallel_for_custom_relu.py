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

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.utils.cpp_extension import get_build_directory, load
from paddle.utils.cpp_extension.extension_utils import IS_WINDOWS, run_cmd

# Note(Aurelius84): We use `add_test` in Cmake to config how to run unittest in CI.
# `PYTHONPATH` will be set as `build/python/paddle` that will make no way to find
# paddle include directory. Because the following path is generated after installing
# PaddlePaddle whl. So here we specific `include_dirs` to avoid errors in CI.
paddle_includes = []
for site_packages_path in getsitepackages():
    paddle_includes.append(
        os.path.join(site_packages_path, 'paddle', 'include')
    )
    paddle_includes.append(
        os.path.join(site_packages_path, 'paddle', 'include', 'third_party')
    )

# Test for extra compile args
extra_cc_args = ['-w', '-g'] if not IS_WINDOWS else ['/w']
extra_nvcc_args = ['-O3']

# Because Windows don't use docker, the shared lib already exists in the
# cache dir, it will not be compiled again unless the shared lib is removed.
file = f'{get_build_directory()}\\dist_custom_relu\\dist_custom_relu.pyd'
if os.name == 'nt' and os.path.isfile(file):
    cmd = f'del {file}'
    run_cmd(cmd, True)

if os.name == 'nt':
    test_include = "..\\python\\paddle\\base\\tests\\auto_parallel"
else:
    test_include = "../python/paddle/base/tests/auto_parallel"
paddle_includes.append(test_include)

custom_ops = load(
    name='dist_custom_relu_jit',
    sources=[
        '../custom_op/custom_relu_op.cc',
        '../custom_op/custom_relu_op_dup.cc',
        '../custom_op/custom_relu_op.cu',
    ],
    extra_include_paths=paddle_includes,  # add for Coverage CI
    extra_cxx_cflags=extra_cc_args,  # test for cc flags
    extra_cuda_cflags=extra_nvcc_args,  # test for nvcc flags
    verbose=True,
)


class TestCustomReluForSemiAutoParallel:
    def __init__(self):
        self._dtype = os.getenv("dtype")
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

    def check_tensor_eq(self, a, b):
        np1 = a.numpy()
        np2 = b.numpy()
        np.testing.assert_allclose(np1, np2, rtol=1e-05, verbose=True)

    def test_body(self, x_shape, x_placements):
        paddle.seed(self._seed)
        np.random.seed(self._seed)

        x_np = np.random.random(size=x_shape).astype(self._dtype)
        x = paddle.to_tensor(x_np)
        x.stop_gradient = False

        dist_x = dist.shard_tensor(x_np, self._mesh, x_placements)
        dist_x.stop_gradient = False

        y = paddle.add(x, x)
        dist_y = paddle.add(dist_x, dist_x)
        out = custom_ops.custom_relu(y)
        dist_out = custom_ops.custom_relu(dist_y)
        out.stop_gradient = False
        dist_out.stop_gradient = False

        self.check_tensor_eq(out, dist_out)

        out.backward()
        dist_out.backward()
        self.check_tensor_eq(x.grad, dist_x.grad)

    def test_custom_relu(self):
        self.test_body(
            x_shape=[64, 32],
            x_placements=[dist.Shard(0)],
        )

    def run_test_case(self):
        paddle.set_device("gpu:" + str(dist.get_rank()))
        self.test_custom_relu()


if __name__ == '__main__':
    TestCustomReluForSemiAutoParallel().test_custom_relu()
