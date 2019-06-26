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

import paddle.fluid as fluid
import unittest


def get_default_place():
    return fluid.GPUPlace(0) if fluid.is_compiled_with_cuda(
    ) else fluid.CPUPlace()


class Test(unittest.TestCase):
    def test_executor(self):
        e = fluid.Executor()
        self.assertTrue(e.place._equals(get_default_place()))

        if fluid.is_compiled_with_cuda():
            p = fluid.GPUPlace(0)
            e = fluid.Executor(p)
            self.assertTrue(e.place._equals(p))

        p = fluid.CPUPlace()
        e = fluid.Executor(p)
        self.assertTrue(e.place._equals(p))

    def test_pe(self):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            x = fluid.layers.data(shape=[-1, 32], name='x', dtype='float32')
            y = fluid.layers.fc(x, size=16)
            loss = fluid.layers.reduce_mean(y)
            optimizer = fluid.optimizer.SGD(learning_rate=1e-3)
            optimizer.minimize(loss)

            places = [None, fluid.CPUPlace()]
            if fluid.is_compiled_with_cuda():
                places.append(fluid.GPUPlace(0))

            for p in places:
                if p is None:
                    use_cuda = None
                    use_cuda_actual = isinstance(get_default_place(),
                                                 fluid.GPUPlace)
                else:
                    use_cuda = isinstance(p, fluid.GPUPlace)
                    use_cuda_actual = use_cuda

                pe = fluid.ParallelExecutor(
                    use_cuda=use_cuda, loss_name=loss.name)
                self.assertEquals(pe._compiled_program._exec_strategy.use_cuda,
                                  use_cuda_actual)


if __name__ == '__main__':
    unittest.main()
