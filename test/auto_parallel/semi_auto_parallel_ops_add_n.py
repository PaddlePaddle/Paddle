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


import numpy as np

import paddle
import paddle.distributed as dist
from paddle import nn

BATCH_SIZE = 16
BATCH_NUM = 4
IMAGE_SIZE = 784
CLASS_NUM = 10


class DemoNet(nn.Layer):
    def __init__(self, np_w0, np_w1, mesh, param_suffix=""):
        super().__init__()
        self.w0 = dist.shard_tensor(
            self.create_parameter(
                shape=[IMAGE_SIZE, IMAGE_SIZE],
                attr=paddle.framework.ParamAttr(
                    name="mp_demo_weight_1" + param_suffix,
                    initializer=paddle.nn.initializer.Assign(np_w0),
                ),
            ),
            dist_attr=dist.DistAttr(mesh=mesh, sharding_specs=[None, 'x']),
        )
        self.w1 = dist.shard_tensor(
            self.create_parameter(
                shape=[IMAGE_SIZE, CLASS_NUM],
                attr=paddle.framework.ParamAttr(
                    name="mp_nemo_weight_2" + param_suffix,
                    initializer=paddle.nn.initializer.Assign(np_w1),
                ),
            ),
            dist_attr=dist.DistAttr(mesh=mesh, sharding_specs=['x', None]),
        )

    def forward(self, x):
        y = paddle.matmul(x, self.w0)
        z = paddle.matmul(y, self.w1)
        a = paddle.assign(z)
        b = paddle.add_n([a, z])
        return b


def run_dynamic(layer, image, label):
    # create loss
    loss_fn = nn.MSELoss()
    # run forward and backward
    image = paddle.to_tensor(image)
    image.stop_gradient = False
    out = layer(image)

    label = paddle.to_tensor(label)
    loss = loss_fn(out, label)

    loss.backward()
    return loss, layer.w0.grad, layer.w1.grad


class TestSemiAutoParallelOpsAddN:
    def test_addn():
        mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        image = np.random.random([BATCH_SIZE, IMAGE_SIZE]).astype('float32')
        label = np.random.random([BATCH_SIZE, CLASS_NUM]).astype('float32')
        w0 = np.random.random([IMAGE_SIZE, IMAGE_SIZE]).astype('float32')
        w1 = np.random.random([IMAGE_SIZE, CLASS_NUM]).astype('float32')
        run_dynamic(layer=DemoNet(w0, w1, mesh), image=image, label=label)


if __name__ == "__main__":
    TestSemiAutoParallelOpsAddN.test_addn()
