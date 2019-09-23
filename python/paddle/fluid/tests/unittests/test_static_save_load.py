# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np

import numpy
import paddle
import paddle.fluid as fluid
from paddle.fluid.layers.device import get_places
from paddle.fluid.executor import as_numpy


class TestDygraphIO(unittest.TestCase):
    def test_save_load(self):
        img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        hidden = fluid.layers.fc(input=img, size=200, act='tanh')
        hidden = fluid.layers.fc(input=hidden, size=200, act='tanh')
        predict = fluid.layers.fc(input=hidden, size=12, act='softmax')
        cost = fluid.layers.cross_entropy(input=predict, label=label)
        avg_cost = fluid.layers.mean(x=cost)
        optimizer = fluid.optimizer.Adam(learning_rate=1e-3)
        optimizer.minimize(avg_cost)

        if fluid.core.is_compiled_with_cuda():
            place = fluid.core.CUDAPlace(0)
        else:
            place = fluid.core.CPUPlace()

        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        program = fluid.default_main_program()

        before_save_para_dict = fluid.io.get_parameter(program)
        before_save_opt_dict = fluid.io.get_optimizer(program)
        fluid.io.save(program, "save_dir")
        fluid.io.load(program, "save_dir")
        self.assertRaises(IOError, fluid.io.load, program, "not_exist_dir")

        after_load_para_dict = fluid.io.get_parameter(program)
        after_load_opt_dict = fluid.io.get_optimizer(program)

        for k, v in before_save_para_dict.items():
            self.assertTrue(v == after_load_para_dict[k])

        for k, v in before_save_opt_dict.items():
            self.assertTrue(v == after_load_opt_dict[k])


if __name__ == '__main__':
    unittest.main()
