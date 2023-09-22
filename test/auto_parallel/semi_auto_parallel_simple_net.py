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

import paddle
import paddle.distributed as dist
from paddle import nn

BATCH_SIZE = 16
BATCH_NUM = 4
IMAGE_SIZE = 784
CLASS_NUM = 10


# TODO(chenweihang): update to MLP Layer later
class DemoNet(nn.Layer):
    def __init__(self, np_w0, np_w1):
        super().__init__()
        self.w0 = self.create_parameter(
            shape=[IMAGE_SIZE, IMAGE_SIZE],
            attr=paddle.framework.ParamAttr(
                name="demo_weight_1",
                initializer=paddle.nn.initializer.Assign(np_w0),
            ),
        )
        self.w1 = self.create_parameter(
            shape=[IMAGE_SIZE, CLASS_NUM],
            attr=paddle.framework.ParamAttr(
                name="nemo_weight_2",
                initializer=paddle.nn.initializer.Assign(np_w1),
            ),
        )

    def forward(self, x):
        y = paddle.matmul(x, self.w0)
        z = paddle.matmul(y, self.w1)
        return z


class DPDemoNet(nn.Layer):
    def __init__(self, np_w0, np_w1, mesh):
        super().__init__()
        self.replicate_dist_attr = dist.DistAttr(
            mesh=mesh, sharding_specs=[None, None]
        )
        self.shard_axis0_dist_attr = dist.DistAttr(
            mesh=mesh, sharding_specs=['x', None]
        )
        self.w0 = dist.shard_tensor(
            self.create_parameter(
                shape=[IMAGE_SIZE, IMAGE_SIZE],
                attr=paddle.framework.ParamAttr(
                    name="dp_demo_weight_1",
                    initializer=paddle.nn.initializer.Assign(np_w0),
                ),
            ),
            dist_attr=self.replicate_dist_attr,
        )
        self.w1 = dist.shard_tensor(
            self.create_parameter(
                shape=[IMAGE_SIZE, CLASS_NUM],
                attr=paddle.framework.ParamAttr(
                    name="dp_nemo_weight_2",
                    initializer=paddle.nn.initializer.Assign(np_w1),
                ),
            ),
            dist_attr=self.replicate_dist_attr,
        )

    def forward(self, x):
        y = paddle.matmul(
            dist.shard_tensor(x, dist_attr=self.shard_axis0_dist_attr),
            self.w0,
        )
        z = paddle.matmul(y, self.w1)
        return z


class MPDemoNet(nn.Layer):
    def __init__(self, np_w0, np_w1, mesh):
        super().__init__()
        self.replicate_dist_attr = dist.DistAttr(
            mesh=mesh, sharding_specs=[None, None]
        )
        self.shard_axis0_dist_attr = dist.DistAttr(
            mesh=mesh, sharding_specs=['x', None]
        )
        self.shard_axis1_dist_attr = dist.DistAttr(
            mesh=mesh, sharding_specs=['x', None]
        )
        self.w0 = dist.shard_tensor(
            self.create_parameter(
                shape=[IMAGE_SIZE, IMAGE_SIZE],
                attr=paddle.framework.ParamAttr(
                    name="mp_demo_weight_1",
                    initializer=paddle.nn.initializer.Assign(np_w0),
                ),
            ),
            dist_attr=self.shard_axis1_dist_attr,
        )
        self.w1 = dist.shard_tensor(
            self.create_parameter(
                shape=[IMAGE_SIZE, CLASS_NUM],
                attr=paddle.framework.ParamAttr(
                    name="mp_nemo_weight_2",
                    initializer=paddle.nn.initializer.Assign(np_w1),
                ),
            ),
            dist_attr=self.shard_axis0_dist_attr,
        )

    def forward(self, x):
        y = paddle.matmul(
            dist.shard_tensor(x, dist_attr=self.replicate_dist_attr), self.w0
        )
        z = paddle.matmul(y, self.w1)
        return z


class TestSimpleNetForSemiAutoParallel:
    def __init__(self):
        self._dtype = os.getenv("dtype")
        self._backend = os.getenv("backend")
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

        paddle.set_device(self._backend)

        self.init_input_data()

        self.init_single_card_net_result()

    def init_input_data(self):
        paddle.seed(2023)
        np.random.seed(2023)

        self.image = np.random.random([BATCH_SIZE, IMAGE_SIZE]).astype(
            'float32'
        )
        self.label = np.random.random([BATCH_SIZE, CLASS_NUM]).astype('float32')
        self.w0 = np.random.random([IMAGE_SIZE, IMAGE_SIZE]).astype('float32')
        self.w1 = np.random.random([IMAGE_SIZE, CLASS_NUM]).astype('float32')

    # TODO(chenweihang): optimizer cannot run auto-parallel now
    def run_dynamic(self, layer, parallel=False):
        # create loss
        loss_fn = nn.MSELoss()
        # run forward and backward
        image = paddle.to_tensor(self.image)
        out = layer(image)
        label = (
            dist.shard_tensor(
                self.label,
                dist_attr=dist.DistAttr(
                    mesh=self._mesh, sharding_specs=[None, None]
                ),
            )
            if parallel is True
            else paddle.to_tensor(self.label)
        )
        loss = loss_fn(out, label)
        loss.backward()
        return loss, layer.w0.grad, layer.w1.grad

    def init_single_card_net_result(self):
        self.base_loss, self.base_w0_grad, self.base_w1_grad = self.run_dynamic(
            DemoNet(self.w0, self.w1)
        )

    def check_tensor_eq(self, a, b):
        np1 = a.numpy()
        np2 = b.numpy()
        np.testing.assert_allclose(np1, np2, rtol=1e-05, verbose=True)

    def test_dp_demo_net(self):
        self.dp_loss, self.dp_w0_grad, self.dp_w1_grad = self.run_dynamic(
            DPDemoNet(self.w0, self.w1, self._mesh), parallel=True
        )
        self.check_tensor_eq(self.dp_loss, self.base_loss)
        self.check_tensor_eq(self.dp_w0_grad, self.base_w0_grad)
        self.check_tensor_eq(self.dp_w1_grad, self.base_w1_grad)

    def test_mp_demo_net(self):
        self.mp_loss, self.mp_w0_grad, self.mp_w1_grad = self.run_dynamic(
            MPDemoNet(self.w0, self.w1, self._mesh), parallel=True
        )
        self.check_tensor_eq(self.mp_loss, self.base_loss)
        self.check_tensor_eq(self.mp_w0_grad, self.base_w0_grad)
        self.check_tensor_eq(self.mp_w1_grad, self.base_w1_grad)

    def run_test_case(self):
        self.test_dp_demo_net()
        self.test_mp_demo_net()


if __name__ == '__main__':
    TestSimpleNetForSemiAutoParallel().run_test_case()
