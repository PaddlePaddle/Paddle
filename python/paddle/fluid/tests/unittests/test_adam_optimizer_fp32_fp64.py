# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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


def get_places():
    places = [fluid.CPUPlace()]
    if fluid.is_compiled_with_cuda():
        places.append(fluid.CUDAPlace(0))
    return places


def main_test_func(place, dtype):
    main = fluid.Program()
    startup = fluid.Program()
    with fluid.program_guard(main, startup):
        with fluid.scope_guard(fluid.Scope()):
            x = fluid.data(name='x', shape=[None, 13], dtype=dtype)
            y = fluid.data(name='y', shape=[None, 1], dtype=dtype)
            y_predict = fluid.layers.fc(input=x, size=1, act=None)
            cost = paddle.nn.functional.square_error_cost(
                input=y_predict, label=y
            )
            avg_cost = paddle.mean(cost)

            adam_optimizer = fluid.optimizer.AdamOptimizer(0.01)
            adam_optimizer.minimize(avg_cost)

            fetch_list = [avg_cost]
            train_reader = fluid.io.batch(
                paddle.dataset.uci_housing.train(), batch_size=1
            )
            feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for data in train_reader():
                exe.run(main, feed=feeder.feed(data), fetch_list=fetch_list)


class AdamFp32Test(unittest.TestCase):
    def setUp(self):
        self.dtype = 'float32'

    def test_main(self):
        for p in get_places():
            main_test_func(p, self.dtype)


class AdamFp64Test(AdamFp32Test):
    def setUp(self):
        self.dtype = 'float64'


if __name__ == '__main__':
    unittest.main()
