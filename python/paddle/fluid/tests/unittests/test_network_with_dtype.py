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

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core

BATCH_SIZE = 20


class TestNetWithDtype(unittest.TestCase):
    def setUp(self):
        self.dtype = "float64"
        self.init_dtype()

    def run_net_on_place(self, place):
        main = fluid.Program()
        startup = fluid.Program()
        with fluid.program_guard(main, startup):
            x = fluid.layers.data(name='x', shape=[13], dtype=self.dtype)
            y = fluid.layers.data(name='y', shape=[1], dtype=self.dtype)
            y_predict = fluid.layers.fc(input=x, size=1, act=None)
            cost = paddle.nn.functional.square_error_cost(
                input=y_predict, label=y
            )
            avg_cost = paddle.mean(cost)
            sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
            sgd_optimizer.minimize(avg_cost)

        fetch_list = [avg_cost]
        train_reader = paddle.batch(
            paddle.dataset.uci_housing.train(), batch_size=BATCH_SIZE
        )
        feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
        exe = fluid.Executor(place)
        exe.run(startup)
        for data in train_reader():
            exe.run(main, feed=feeder.feed(data), fetch_list=fetch_list)
            # the main program is runable, the datatype is fully supported
            break

    def init_dtype(self):
        pass

    def test_cpu(self):
        place = fluid.CPUPlace()
        self.run_net_on_place(place)

    def test_gpu(self):
        if not core.is_compiled_with_cuda():
            return
        place = fluid.CUDAPlace(0)
        self.run_net_on_place(place)


# TODO(dzhwinter): make sure the fp16 is runable
# class TestFloat16(TestNetWithDtype):
#     def init_dtype(self):
#         self.dtype = "float16"

if __name__ == '__main__':
    unittest.main()
