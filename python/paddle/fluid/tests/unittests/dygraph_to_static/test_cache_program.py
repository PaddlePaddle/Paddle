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

import unittest
import numpy as np
from collections import Counter
import paddle
import paddle.fluid as fluid

from paddle.fluid.dygraph.jit import declarative
from paddle.fluid.dygraph.dygraph_to_static import ProgramTranslator
from paddle.fluid.dygraph.dygraph_to_static import convert_to_static

from test_fetch_feed import Pool2D, Linear


class TestCacheProgram(unittest.TestCase):

    def setUp(self):
        self.batch_num = 5
        self.dygraph_class = Pool2D
        self.data = np.random.random((1, 2, 4, 4)).astype('float32')

    def test_cache(self):
        prev_ops, cur_ops = Counter(), Counter()
        prev_out, cur_out = None, None
        with fluid.dygraph.guard(fluid.CPUPlace()):
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
                    prev_out_numpy = prev_out[0].numpy() if isinstance(
                        prev_out, (tuple, list)) else prev_out.numpy()
                    cur_out_numpy = cur_out[0].numpy() if isinstance(
                        cur_out, (tuple, list)) else cur_out.numpy()
                    np.testing.assert_allclose(
                        prev_out_numpy,
                        cur_out_numpy,
                        rtol=1e-05,
                        err_msg=
                        'Output in previous batch is {}\n Output in current batch is \n{}'
                        .format(prev_out_numpy, cur_out_numpy))
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
        return self.train(to_static=True)

    def train_dygraph(self):
        return self.train(to_static=False)

    def train(self, to_static=False):
        prog_trans = ProgramTranslator()
        prog_trans.enable(to_static)

        with fluid.dygraph.guard(fluid.CPUPlace()):
            dygraph_net = self.dygraph_class()
            adam = fluid.optimizer.AdamOptimizer(
                learning_rate=0.001, parameter_list=dygraph_net.parameters())
            loss_data = []
            for batch_id in range(self.batch_num):
                input = fluid.dygraph.to_variable(self.data)
                pred, avg_loss = dygraph_net(input)

                loss_data.append(avg_loss.numpy())
                avg_loss.backward()
                adam.minimize(avg_loss)
                dygraph_net.clear_gradients()

        return loss_data

    def test_with_optimizer(self):
        dygraph_loss = self.train_dygraph()
        static_loss = self.train_static()
        np.testing.assert_allclose(
            dygraph_loss,
            static_loss,
            rtol=1e-05,
            err_msg='dygraph is {}\n static_res is \n{}'.format(
                dygraph_loss, static_loss))


def simple_func(x):
    inputs = fluid.dygraph.to_variable(x)
    mean = paddle.mean(inputs)
    return mean


class TestConvertWithCache(unittest.TestCase):

    def test_cache(self):
        static_func = convert_to_static(simple_func)
        # Get transformed function from cache.
        cached_func = convert_to_static(simple_func)
        self.assertTrue(id(static_func), id(cached_func))


@declarative
def sum_even_until_limit(max_len, limit):
    ret_sum = fluid.dygraph.to_variable(np.zeros((1)).astype('int32'))
    for i in range(max_len):
        if i % 2 > 0:
            continue
        elif i > limit:
            break

        ret_sum += i
    return ret_sum


def sum_under_while(limit):
    i = fluid.dygraph.to_variable(np.zeros((1)).astype('int32'))
    ret_sum = fluid.dygraph.to_variable(np.zeros((1)).astype('int32'))
    while i <= limit:
        ret_sum += i
        i += 1
    return ret_sum


class TestToOutputWithCache(unittest.TestCase):

    def test_output(self):
        with fluid.dygraph.guard():
            ret = sum_even_until_limit(80, 10)
            self.assertEqual(ret.numpy(), 30)

            ret = declarative(sum_under_while)(100)
            self.assertEqual(ret.numpy(), 5050)


if __name__ == '__main__':
    unittest.main()
