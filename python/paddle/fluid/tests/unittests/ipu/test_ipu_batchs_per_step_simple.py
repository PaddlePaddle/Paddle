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
import paddle
import paddle.fluid as fluid
import paddle.fluid.compiler as compiler

paddle.enable_static()
SEED = 2021


@unittest.skipIf(not paddle.is_compiled_with_ipu(),
                 "core is not compiled with IPU")
class TestFunc(unittest.TestCase):
    def _test_func(self, run_ipu=True):
        scope = fluid.core.Scope()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED
        np.random.seed(SEED)

        bps = 5
        n = 1 if run_ipu else -1
        c, h, w = 3, 10, 10
        np_image = np.random.uniform(size=[1 * bps, c, h, w]).astype(np.float32)

        with fluid.scope_guard(scope):
            with paddle.static.program_guard(main_prog, startup_prog):
                image = paddle.static.data(
                    name='image', shape=[n, c, h, w], dtype='float32')
                conv2d = paddle.static.nn.conv2d(
                    image, num_filters=3, filter_size=3, bias_attr=False)

                # paddle.mean oshape on ipu is [bps], need another mean()
                # paddle.mean oshape on cpu is [1]
                # out = paddle.mean(conv2d)
                out = conv2d

            if run_ipu:
                place = paddle.IPUPlace()
            else:
                place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            exe.run(startup_prog)

            if run_ipu:
                feed_list = [image.name]
                fetch_list = [out.name]
                ipu_strategy = compiler.get_ipu_strategy()
                ipu_strategy.is_training = False
                ipu_strategy.batches_per_step = bps
                program = compiler.IPUCompiledProgram(
                    main_prog, ipu_strategy=ipu_strategy).compile(feed_list,
                                                                  fetch_list)
            else:
                program = main_prog

            result = exe.run(program,
                             feed={image.name: np_image},
                             fetch_list=[out])
            return result[0]

    def test_func(self):
        ipu_res = self._test_func(True)
        cpu_res = self._test_func(False)

        if np.prod(ipu_res.shape) == np.prod(cpu_res.shape):
            ipu_res = ipu_res.reshape(cpu_res.shape)

        self.assertTrue(np.allclose(ipu_res, cpu_res, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
