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

from semi_auto_parallel_simple_net import TestSimpleNetForSemiAutoParallel

import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle import nn
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

BATCH_SIZE = 16
BATCH_NUM = 4
IMAGE_SIZE = 128
CLASS_NUM = 10


class PPDemoNet(nn.Layer):
    def __init__(self, mesh0, mesh1, param_suffix=""):
        super().__init__()
        self.mesh0 = mesh0
        self.mesh1 = mesh1
        self.w0 = dist.shard_tensor(
            self.create_parameter(
                shape=[IMAGE_SIZE, IMAGE_SIZE],
                attr=paddle.framework.ParamAttr(
                    name="pp_demo_weight_0" + param_suffix,
                    initializer=paddle.nn.initializer.Uniform(0, 1),
                ),
            ),
            mesh0,
            [dist.Replicate(), dist.Replicate()],
        )
        self.w1 = dist.shard_tensor(
            self.create_parameter(
                shape=[IMAGE_SIZE, CLASS_NUM],
                attr=paddle.framework.ParamAttr(
                    name="pp_nemo_weight_1" + param_suffix,
                    initializer=paddle.nn.initializer.Uniform(0, 1),
                ),
            ),
            mesh1,
            [dist.Replicate(), dist.Replicate()],
        )

    def forward(self, x):
        out = F.linear(x, self.w0)
        out = custom_ops.custom_relu(out)
        # out = F.relu(out)
        out = dist.reshard(
            out, self.mesh1, [dist.Replicate(), dist.Replicate()]
        )
        out = F.linear(out, self.w1)
        return out


class TestSimpleNetWithCustomReluForSemiAutoParallel(
    TestSimpleNetForSemiAutoParallel
):
    def __init__(self):
        self._dtype = os.getenv("dtype")
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        self._pp_mesh0 = dist.ProcessMesh([0], dim_names=["x"])
        self._pp_mesh1 = dist.ProcessMesh([1], dim_names=["x"])

        paddle.set_device(self._backend)

    def run_dynamic_custom_relu(self, layer, shard_input=False):
        # create loss
        loss_fn = nn.MSELoss()
        # run forward and backward
        image, label = self.init_input_data()
        if shard_input:
            image = dist.shard_tensor(image, self._mesh, [dist.Shard(0)])
        out = layer(image)
        loss = loss_fn(out, label)

        loss.backward()

    def test_demo_net(self):
        mp_layer = dist.shard_layer(
            PPDemoNet(self._pp_mesh0, self._pp_mesh1),
            self._mesh,
            self.shard_fn,
        )
        self.run_dynamic_custom_relu(mp_layer)

    def run_test_case(self):
        self.test_demo_net()


if __name__ == "__main__":
    TestSimpleNetWithCustomReluForSemiAutoParallel().run_test_case()
