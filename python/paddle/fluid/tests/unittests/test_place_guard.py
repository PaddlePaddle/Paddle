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

from __future__ import print_function

import unittest

import paddle.fluid as fluid
import paddle.fluid.framework as framework


class TestPlaceGuard(unittest.TestCase):
    def test_place(self):
        with framework.place_guard(place="/worker/gpu"):
            x = fluid.layers.data(name='x', shape=[13], dtype='float32')
            y = fluid.layers.data(name='y', shape=[1], dtype='float32')
            y_predict = fluid.layers.fc(input=x, size=128, act=None)
            cost = fluid.layers.square_error_cost(input=y_predict, label=y)
            avg_cost = fluid.layers.mean(cost)

        with framework.place_guard(place="/worker/cpu"):
            sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
            sgd_optimizer.minimize(avg_cost)


if __name__ == '__main__':
    unittest.main()
