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

import paddle
import paddle.fluid as fluid
from paddle.fluid import layers
import numpy as np
import unittest


class TestSoftmaxWithXe(unittest.TestCase):
    def setUp(self):
        self.m, self.n = np.random.random_integers(
            low=100, high=2000, size=[2]).astype('int64')

    def softmax_with_xe(self, x, y, place, inplace=True):
        m, n = x.shape
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            with fluid.scope_guard(fluid.Scope()):
                x_d = fluid.layers.data(
                    name='x',
                    shape=[m, n],
                    dtype='float32',
                    append_batch_size=False)
                y_d = fluid.layers.data(
                    name='y',
                    shape=[m, 1],
                    dtype='int64',
                    append_batch_size=False)
                z_d, s_d = fluid.layers.softmax_with_cross_entropy(
                    x_d, y_d, return_softmax=True)

                exe = fluid.Executor(place)

                exe.run(fluid.default_startup_program())

                build_strategy = fluid.BuildStrategy()
                build_strategy.enable_inplace = inplace
                prog = fluid.CompiledProgram(fluid.default_main_program(
                )).with_data_parallel(
                    build_strategy=build_strategy, places=place)

                if inplace and isinstance(place, fluid.CUDAPlace):
                    fetch_list = [z_d.name, x_d.name]
                else:
                    fetch_list = [z_d.name, s_d.name]

                z, s = exe.run(prog,
                               feed={x_d.name: x,
                                     y_d.name: y},
                               fetch_list=fetch_list)
                return z, s

    def main_with_place(self, place):
        x = np.random.random(size=[self.m, self.n]).astype('float32')
        x_range = [(-30, 30), (10, 20), (-1, 1), (2, 3), (0, 0.3), (-200, -100)]

        for a, b in x_range:
            x = ((b - a) * x + a).astype('float32')
            y = np.random.random_integers(
                size=[self.m, 1], low=0, high=self.n - 1).astype('int64')
            z1, s1 = self.softmax_with_xe(x, y, place, False)
            z2, s2 = self.softmax_with_xe(x, y, place, True)

            self.assertTrue((z1 == z2).all())
            self.assertTrue((s1 == s2).all())

    def test_main(self):
        self.main_with_place(fluid.CPUPlace())
        if fluid.core.is_compiled_with_cuda():
            self.main_with_place(fluid.CUDAPlace(0))


if __name__ == '__main__':
    unittest.main()
