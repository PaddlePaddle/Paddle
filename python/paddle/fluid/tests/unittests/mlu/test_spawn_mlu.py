# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import os

import paddle
import paddle.nn as nn
import paddle.optimizer as opt
import paddle.distributed as dist
from paddle.distributed.spawn import _get_subprocess_env_list, _options_valid_check, _get_default_nprocs
from paddle.fluid import core


class LinearNet(nn.Layer):

    def __init__(self):
        super(LinearNet, self).__init__()
        self._linear1 = nn.Linear(10, 10)
        self._linear2 = nn.Linear(10, 1)

    def forward(self, x):
        return self._linear2(self._linear1(x))


def train(print_result=False):
    # 1. initialize parallel environment
    dist.init_parallel_env()

    # 2. create data parallel layer & optimizer
    layer = LinearNet()
    dp_layer = paddle.DataParallel(layer)

    loss_fn = nn.MSELoss()
    adam = opt.Adam(learning_rate=0.001, parameters=dp_layer.parameters())

    # 3. run layer
    inputs = paddle.randn([10, 10], 'float32')
    outputs = dp_layer(inputs)
    labels = paddle.randn([10, 1], 'float32')
    loss = loss_fn(outputs, labels)

    if print_result is True:
        print("Rank:", int(os.getenv("PADDLE_TRAINER_ID")))

    loss.backward()
    adam.step()
    adam.clear_grad()

    return int(os.getenv("PADDLE_TRAINER_ID"))


class TestSpawn(unittest.TestCase):

    def test_nprocs_greater_than_device_num_error(self):
        with self.assertRaises(RuntimeError):
            _get_subprocess_env_list(nprocs=100, options=dict())

    def test_selected_devices_error(self):
        with self.assertRaises(ValueError):
            options = dict()
            options['selected_devices'] = "100,101"
            _get_subprocess_env_list(nprocs=2, options=options)

    def test_get_correct_env(self):
        options = dict()
        options['print_config'] = True
        env_dict = _get_subprocess_env_list(nprocs=1, options=options)[0]
        self.assertEqual(env_dict['PADDLE_TRAINER_ID'], '0')
        self.assertEqual(env_dict['PADDLE_TRAINERS_NUM'], '1')

    def test_nprocs_not_equal_to_selected_devices(self):
        with self.assertRaises(ValueError):
            options = dict()
            options['selected_devices'] = "100,101,102"
            _get_subprocess_env_list(nprocs=2, options=options)

    def test_options_valid_check(self):
        options = dict()
        options['selected_devices'] = "100,101,102"
        _options_valid_check(options)

        with self.assertRaises(ValueError):
            options['error'] = "error"
            _options_valid_check(options)

    def test_get_default_nprocs(self):
        paddle.set_device('mlu')
        nprocs = _get_default_nprocs()
        self.assertEqual(nprocs, core.get_mlu_device_count())

    def test_spawn(self):
        num_devs = core.get_mlu_device_count()
        context = dist.spawn(train, backend='cncl', nprocs=num_devs)
        rank_list = []
        for i in range(num_devs):
            rank_list.append(context.return_queues[i].get())
        rank_list.sort()
        self.assertEqual(rank_list, list(range(num_devs)))


if __name__ == '__main__':
    unittest.main()
