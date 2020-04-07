#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np
from collections import Counter

import paddle.fluid as fluid

from paddle.fluid.dygraph.dygraph_to_static import ProgramTranslator
from paddle.fluid.dygraph.dygraph_to_static import convert_function_with_cache

from test_fetch_feed import Pool2D, Linear


class TestCacheProgram(unittest.TestCase):
    def setUp(self):
        self.batch_num = 5
        self.dygraph_class = Pool2D
        self.data = np.random.random((1, 2, 4, 4)).astype('float32')

    def test_cache(self):
        prev_ops, cur_ops = Counter(), Counter()
        prev_out, cur_out = None, None
        main_program = fluid.Program()
        with fluid.program_guard(main_program):
            static_net = self.dygraph_class()
            for batch_id in range(self.batch_num):
                out = static_net(self.data)
                # Check outputs
                prev_out = cur_out
                cur_out = out
                # Check forward ops
                prev_ops = cur_ops
                cur_ops = Counter([
                    op.type for op in fluid.default_main_program().block(0).ops
                ])
                if batch_id > 0:
                    self.assertTrue(
                        np.allclose(prev_out[0], cur_out[0]),
                        msg='Output in previous batch is {}\n Output in current batch is \n{}'
                        .format(prev_out, cur_out))
                    self.assertEqual(prev_ops, cur_ops)


class TestCacheProgram2(TestCacheProgram):
    def setUp(self):
        self.batch_num = 5
        self.dygraph_class = Linear
        self.data = np.random.random((4, 10)).astype('float32')


class TestCacheProgramWithOptimizer(unittest.TestCase):
    def setUp(self):
        self.dygraph_class = Linear
        self.data = np.random.random((4, 10)).astype('float32')
        self.batch_num = 5

    def train_static(self):
        main_program = fluid.Program()
        loss_data = []
        with fluid.program_guard(main_program):
            static_net = self.dygraph_class()
            adam = fluid.optimizer.AdamOptimizer(learning_rate=0.001)
            # set optimizer
            program_translator = ProgramTranslator()
            program_translator.set_optimizer(adam, index_of_loss=1)

            for batch_id in range(self.batch_num):
                # Support to set optimizer in `for` by using cache.
                program_translator.set_optimizer(adam, index_of_loss=1)
                pred, avg_loss = static_net(self.data)
                loss_data.append(np.array(avg_loss))

        return loss_data

    def train_dygraph(self):
        with fluid.dygraph.guard(fluid.CPUPlace()):
            dygraph_net = self.dygraph_class()
            adam = fluid.optimizer.AdamOptimizer(
                learning_rate=0.001, parameter_list=dygraph_net.parameters())
            loss_data = []
            for batch_id in range(self.batch_num):
                pred, avg_loss = dygraph_net(self.data)

                loss_data.append(avg_loss.numpy())
                avg_loss.backward()
                adam.minimize(avg_loss)
                dygraph_net.clear_gradients()

        return loss_data

    def test_with_optimizer(self):
        dygraph_loss = self.train_dygraph()
        static_loss = self.train_static()
        self.assertTrue(
            np.allclose(dygraph_loss, static_loss),
            msg='dygraph is {}\n static_res is \n{}'.format(dygraph_loss,
                                                            static_loss))


def simple_func(x):
    inputs = fluid.dygraph.to_variable(x)
    mean = fluid.layers.mean(inputs)
    return mean


class TestConvertWithCache(unittest.TestCase):
    def test_cache(self):
        static_func = convert_function_with_cache(simple_func)
        # Get transformed function from cache.
        cached_func = convert_function_with_cache(simple_func)
        self.assertTrue(id(static_func), id(cached_func))


if __name__ == '__main__':
    unittest.main()
