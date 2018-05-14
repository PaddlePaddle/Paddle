#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.executor import Executor

BATCH_SIZE = 20


class TestNetWithDtype(unittest.TestCase):
    def set_network(self):
        self.dtype = "float64"
        self.init_dtype()
        self.x = fluid.layers.data(name='x', shape=[13], dtype=self.dtype)
        self.y = fluid.layers.data(name='y', shape=[1], dtype=self.dtype)
        y_predict = fluid.layers.fc(input=self.x, size=1, act=None)

        cost = fluid.layers.square_error_cost(input=y_predict, label=self.y)
        avg_cost = fluid.layers.mean(cost)
        self.fetch_list = [avg_cost]

        sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
        sgd_optimizer.minimize(avg_cost)

    def run_net_on_place(self, place):
        train_reader = paddle.batch(
            paddle.dataset.uci_housing.train(), batch_size=BATCH_SIZE)
        feeder = fluid.DataFeeder(place=place, feed_list=[self.x, self.y])
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        for data in train_reader():
            exe.run(fluid.default_main_program(),
                    feed=feeder.feed(data),
                    fetch_list=self.fetch_list)
            # the main program is runable, the datatype is fully supported
            break

    def init_dtype(self):
        pass

    def test_cpu(self):
        self.set_network()
        place = fluid.CPUPlace()
        self.run_net_on_place(place)

    def test_gpu(self):
        if not core.is_compiled_with_cuda():
            return
        self.set_network()
        place = fluid.CUDAPlace(0)
        self.run_net_on_place(place)


# TODO(dzhwinter): make sure the fp16 is runable
# class TestFloat16(SimpleNet):
#     def init_dtype(self):
#         self.dtype = "float16"

if __name__ == '__main__':
    unittest.main()
