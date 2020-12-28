# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import os
import six
import unittest

import paddle
import paddle.nn as nn
import paddle.optimizer as opt
import paddle.distributed as dist

import paddle.fluid.core as core


def multi_gpus_used_check():
    # get all cuda devices status
    res = os.popen("nvidia-smi --query-gpu=memory.used --format=csv").read()
    lines = res.splitlines()
    print("Memory use info: ", lines)
    # get visible cuda devices list
    env_devices = os.getenv("CUDA_VISIBLE_DEVICES", None)
    if env_devices is None or env_devices == "":
        env_devices_list = [
            str(x) for x in six.moves.range(core.get_cuda_device_count())
        ]
    else:
        env_devices_list = env_devices.split(',')
    print("Visible devices list: ", env_devices_list)
    assert len(env_devices_list) <= len(lines) and len(env_devices_list) > 2
    # check memory using
    memory_use = []
    for line in lines[3:]:
        units = line.split(' ')
        memory_use.append(int(units[0]))
    for mem in memory_use:
        if mem <= 10:
            return False
    return True


class LinearNet(nn.Layer):
    def __init__(self):
        super(LinearNet, self).__init__()
        self._linear1 = nn.Linear(10, 10)
        self._linear2 = nn.Linear(10, 1)

    def forward(self, x):
        return self._linear2(self._linear1(x))


def train():
    # initialize parallel environmen
    dist.init_parallel_env()

    # create data parallel layer & optimizer
    layer = LinearNet()
    dp_layer = paddle.DataParallel(layer)

    loss_fn = nn.MSELoss()
    adam = opt.Adam(learning_rate=0.001, parameters=dp_layer.parameters())

    # 4. run layer
    inputs = paddle.randn([10, 10], 'float32')
    outputs = dp_layer(inputs)
    labels = paddle.randn([10, 1], 'float32')
    loss = loss_fn(outputs, labels)

    loss.backward()

    check_result = multi_gpus_used_check()

    adam.step()
    adam.clear_grad()

    return check_result


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestNoSingleProcessUsingMultiGpusInSpawn(unittest.TestCase):
    def test_multi_gpus_used(self):
        context = dist.spawn(train, nprocs=8)
        for res_queue in context.return_queues:
            self.assertFalse(res_queue.get())


if __name__ == '__main__':
    unittest.main()
