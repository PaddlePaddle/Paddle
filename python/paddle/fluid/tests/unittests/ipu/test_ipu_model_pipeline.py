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
class TestCastNet(unittest.TestCase):
    def _test(self, run_ipu=True):
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
                with fluid.ipu_shard(ipu_index=0):
                    conv1 = paddle.static.nn.conv2d(
                        image, num_filters=3, filter_size=3, bias_attr=False)
                with fluid.ipu_shard(ipu_index=1):
                    conv2 = paddle.static.nn.conv2d(
                        conv1, num_filters=3, filter_size=3, bias_attr=False)
                    loss = paddle.mean(conv2)

            if run_ipu:
                place = paddle.IPUPlace()
            else:
                place = paddle.CPUPlace()
            executor = paddle.static.Executor(place)
            executor.run(startup_prog)

            if run_ipu:
                feed_list = [image.name]
                fetch_list = [loss.name]
                ipu_strategy = paddle.static.IpuStrategy()
                ipu_strategy.SetGraphConfig(
                    num_ipus=2, is_training=False, enable_manual_shard=True)
                program = compiler.IPUCompiledProgram(
                    main_prog,
                    ipu_strategy=ipu_strategy).compile(feed_list, fetch_list)
            else:
                program = main_prog

            loss_res = executor.run(program,
                                    feed={"image": np_image},
                                    fetch_list=[loss])
            return loss_res

    def test_cast(self):
        cpu_outputs = self._test(False)
        ipu_outputs = self._test(True)

        self.assertTrue(np.allclose(cpu_outputs, ipu_outputs, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
