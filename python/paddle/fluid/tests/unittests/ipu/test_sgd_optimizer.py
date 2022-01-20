#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import numpy as np
import unittest
import sys
import paddle
import paddle.fluid as fluid
import paddle.fluid.compiler as compiler

paddle.enable_static()
SEED = 2021


@unittest.skipIf(not paddle.is_compiled_with_ipu(),
                 "core is not compiled with IPU")
class TestSGD(unittest.TestCase):
    def _test_sgd(self, run_ipu=True):
        scope = fluid.core.Scope()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED
        np.random.seed(SEED)

        np_image = np.random.rand(1, 3, 10, 10).astype(np.float32)

        with fluid.scope_guard(scope):
            with paddle.static.program_guard(main_prog, startup_prog):
                image = paddle.static.data(
                    name='image', shape=[1, 3, 10, 10], dtype='float32')
                conv1 = paddle.static.nn.conv2d(
                    image, num_filters=3, filter_size=3, bias_attr=False)
                loss = paddle.mean(conv1)

                sgd = paddle.optimizer.SGD(learning_rate=1e-1)
                sgd.minimize(loss)

            if run_ipu:
                place = paddle.IPUPlace()
            else:
                place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            exe.run(startup_prog)

            if run_ipu:
                feed_list = [image.name]
                fetch_list = [loss.name]
                ipu_strategy = paddle.static.IpuStrategy()
                ipu_strategy.SetGraphConfig(is_training=True)
                program = compiler.IPUCompiledProgram(
                    main_prog, ipu_strategy=ipu_strategy).compile(feed_list,
                                                                  fetch_list)
            else:
                program = main_prog

            result = []
            for epoch in range(100):
                loss_res = exe.run(program,
                                   feed={"image": np_image},
                                   fetch_list=[loss])
                result.append(loss_res)

            return np.array(result)

    def test_sgd(self):
        # cpu and ipu dimenstion mismatch, cpu:(100, 1, 1), ipu:(100, 1)
        ipu_loss = self._test_sgd(True).flatten()
        cpu_loss = self._test_sgd(False).flatten()

        self.assertTrue(np.allclose(ipu_loss, cpu_loss, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
