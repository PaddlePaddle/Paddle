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

import unittest

import numpy as np

import paddle
import paddle.fluid as fluid


class TestEmbeddingIdStopGradientBase(unittest.TestCase):
    def setUp(self):
        self.reshape_times = 1
        self.iteration = 10

    def get_places(self):
        places = [fluid.CPUPlace()]
        if fluid.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))

        return places

    def test_check_grad(self):
        for p in self.get_places():
            grad_value1 = self.run_program(p, stop_gradient=False)
            grad_value2 = self.run_program(p, stop_gradient=True)
            np.testing.assert_array_equal(grad_value1, grad_value2)

    def run_program(self, place, stop_gradient=False):
        np.random.seed(1)
        paddle.seed(1)
        paddle.framework.random._manual_program_seed(1)

        startup_program = fluid.Program()
        main_program = fluid.Program()

        scope = fluid.Scope()
        with fluid.program_guard(main_program, startup_program):
            with fluid.scope_guard(scope):
                x_1 = fluid.data(name='x1', shape=[4, 1], dtype='int64')
                x_2 = fluid.data(name='x2', shape=[4, 1], dtype='int64')
                x = paddle.concat([x_1, x_2], axis=-1)

                for _ in range(self.reshape_times):
                    x = paddle.reshape(x, [-1, 1])

                x.stop_gradient = stop_gradient

                emb = paddle.static.nn.embedding(
                    x, size=[10, 32], dtype='float32'
                )
                avg_cost = paddle.mean(emb, name='mean_loss')
                optim = fluid.optimizer.SGD(learning_rate=0.001)
                optim.minimize(avg_cost)

                exe = fluid.Executor(place)
                exe.run(startup_program)

                x1_data = np.random.randint(0, 9, x_1.shape).astype('int64')
                x2_data = np.random.randint(0, 9, x_2.shape).astype('int64')

                fetch_val = None
                for _ in range(self.iteration):
                    fetch_val = exe.run(
                        feed={x_1.name: x1_data, x_2.name: x2_data},
                        fetch_list=[emb],
                    )[0]

                return fetch_val


class TestEmbeddingIdStopGradient2(TestEmbeddingIdStopGradientBase):
    def setUp(self):
        self.reshape_times = 100
        self.iteration = 10


if __name__ == '__main__':
    unittest.main()
