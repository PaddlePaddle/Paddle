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

import unittest

import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.compiler as compiler
import paddle.optimizer
import paddle.static
from paddle.fluid.tests.unittests.ipu.op_test_ipu import (IPUOpTest,
                                                          np_dtype_to_fluid_str)

paddle.enable_static()


@unittest.skipIf(not paddle.is_compiled_with_ipu(),
                 "core is not compiled with IPU")
class TestBase(IPUOpTest):
    def setUp(self):
        self.set_atol()
        self.set_feed()
        self.set_feed_attr()
        self.set_attrs()

    def set_feed(self):
        np_data = np.random.uniform(low=-1, high=1, size=[1, 3, 100, 100])
        self.feed_ipu = {"x": np_data.astype('float16')}
        self.feed_cpu = {"x": np_data.astype('float32')}

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed_cpu.values()]
        self.feed_list = list(self.feed_cpu.keys())
        self.feed_dtype = [
            np_dtype_to_fluid_str(x.dtype) for x in self.feed_cpu.values()
        ]

    def set_attrs(self):
        self.attrs = {}

    def _test_base(self, run_ipu=True):
        scope = fluid.core.Scope()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        SEED = self.SEED
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED

        with fluid.scope_guard(scope):
            with paddle.static.program_guard(main_prog, startup_prog):
                x = paddle.static.data(
                    name=self.feed_list[0],
                    shape=self.feed_shape[0],
                    dtype=self.feed_dtype[0])
                conv1 = paddle.static.nn.conv2d(
                    x, num_filters=3, filter_size=3, bias_attr=False)
                conv2 = paddle.static.nn.conv2d(
                    x, num_filters=3, filter_size=3, bias_attr=False)
                add1 = conv1 + conv2
                conv3 = paddle.static.nn.conv2d(
                    add1, num_filters=8, filter_size=8, bias_attr=False)
                out = paddle.fluid.layers.relu(conv3, **self.attrs)
                fetch_list = [out.name]
            if run_ipu:
                place = paddle.IPUPlace()
            else:
                place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            exe.run(startup_prog)

            feed = self.feed_ipu if run_ipu else self.feed_cpu
            if run_ipu:
                feed_list = self.feed_list
                ipu_strategy = compiler.get_ipu_strategy()
                ipu_strategy.is_training = False
                ipu_strategy.enable_fp16 = True
                program = compiler.IPUCompiledProgram(
                    main_prog,
                    ipu_strategy=ipu_strategy).compile(feed_list, fetch_list)
            else:
                feed_list = self.feed_list
                program = main_prog
            result = exe.run(program, feed=feed, fetch_list=fetch_list)
            return result[0]

    def test_base(self):
        res0 = self._test_base(False)
        res1 = self._test_base(True)

        self.assertTrue(res0.shape == res1.shape)
        mae = np.mean(np.abs(res0.flatten() - res1.flatten()))
        print("mae is ", mae)
        self.assertTrue(mae < 0.001)


if __name__ == "__main__":
    unittest.main()
