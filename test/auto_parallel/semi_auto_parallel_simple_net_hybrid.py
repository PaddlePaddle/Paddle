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

from semi_auto_parallel_simple_net import (
    CLASS_NUM,
    IMAGE_SIZE,
    TestSimpleNetForSemiAutoParallel,
)

import paddle
import paddle.distributed as dist
from paddle import nn


class DPAndMPDemoNet(nn.Layer):
    def __init__(self, np_w0, np_w1, mesh):
        super().__init__()
        self.mesh = mesh
        self.w0 = dist.shard_tensor(
            self.create_parameter(
                shape=[IMAGE_SIZE, IMAGE_SIZE],
                attr=paddle.framework.ParamAttr(
                    name="dmp_demo_weight_1",
                    initializer=paddle.nn.initializer.Assign(np_w0),
                ),
            ),
            dist_attr=dist.DistAttr(mesh=mesh, sharding_specs=[None, 'y']),
        )
        self.w1 = dist.shard_tensor(
            self.create_parameter(
                shape=[IMAGE_SIZE, CLASS_NUM],
                attr=paddle.framework.ParamAttr(
                    name="dmp_nemo_weight_2",
                    initializer=paddle.nn.initializer.Assign(np_w1),
                ),
            ),
            dist_attr=dist.DistAttr(mesh=mesh, sharding_specs=['y', None]),
        )

    def forward(self, x):
        y = paddle.matmul(
            dist.shard_tensor(
                x,
                dist_attr=dist.DistAttr(
                    mesh=self.mesh, sharding_specs=['x', None]
                ),
            ),
            self.w0,
        )
        z = paddle.matmul(y, self.w1)
        return z


class TestSimpleNetHybridStrategyForSemiAutoParallel(
    TestSimpleNetForSemiAutoParallel
):
    def __init__(self):
        self._dtype = os.getenv("dtype")
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))
        self._mesh = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=["x", "y"])

        paddle.set_device(self._backend)

        self.init_input_data()
        self.init_single_card_net_result()

    def test_dp_mp_demo_net(self):
        (
            self.dp_mp_loss,
            self.dp_mp_w0_grad,
            self.dp_mp_w1_grad,
        ) = self.run_dynamic(DPAndMPDemoNet(self.w0, self.w1, self._mesh))
        self.check_tensor_eq(self.dp_mp_loss, self.base_loss)
        self.check_tensor_eq(self.dp_mp_w0_grad, self.base_w0_grad)
        self.check_tensor_eq(self.dp_mp_w1_grad, self.base_w1_grad)

    def run_test_case(self):
        self.test_dp_mp_demo_net()


if __name__ == '__main__':
    TestSimpleNetHybridStrategyForSemiAutoParallel().run_test_case()
