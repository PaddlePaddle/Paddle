# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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


import paddle
import paddle.distributed as dist
from paddle import nn


class LinearNet(nn.Layer):
    def __init__(self):
        super().__init__()
        self._linear1 = nn.Linear(10, 10)
        self._linear2 = nn.Linear(10, 1)

    def forward(self, x):
        return self._linear2(self._linear1(x))


def train():
    dist.init_parallel_env()
    layer = paddle.jit.to_static(LinearNet(), full_graph=True, backend='CINN')
    dp_layer = paddle.DataParallel(layer)
    inputs = paddle.randn([10, 10], 'float32')
    # NOTE(dev): Spawn will launch multi-process to run this file in
    # gpu:0 and gpu:1, it's not easy to apply np.testing.allclose
    # between @to_static and dynamic mode.
    dp_layer(inputs)


if __name__ == "__main__":
    dist.spawn(train, nprocs=2)
