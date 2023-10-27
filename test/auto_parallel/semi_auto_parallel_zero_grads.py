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
from semi_auto_parallel_simple_net import MPDemoNet

import paddle
import paddle.distributed as dist
from paddle import nn

BATCH_SIZE = 16
BATCH_NUM = 4
IMAGE_SIZE = 784
CLASS_NUM = 10


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
    layer.w0._zero_grads()
    layer.w1._zero_grads()


class TestSemiAutoParallelZeroGrads:
    def test_zero_grads():
        mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        image = np.random.random([BATCH_SIZE, IMAGE_SIZE]).astype('float32')
        label = np.random.random([BATCH_SIZE, CLASS_NUM]).astype('float32')
        w0 = np.random.random([IMAGE_SIZE, IMAGE_SIZE]).astype('float32')
        w1 = np.random.random([IMAGE_SIZE, CLASS_NUM]).astype('float32')
        run_dynamic(layer=MPDemoNet(w0, w1, mesh), image=image, label=label)


if __name__ == "__main__":
    TestSemiAutoParallelZeroGrads.test_zero_grads()
