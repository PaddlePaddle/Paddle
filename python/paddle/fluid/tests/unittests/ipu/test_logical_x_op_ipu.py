#  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.static
from paddle.fluid.tests.unittests.ipu.op_test_ipu import IPUOpTest, ExecutionMode


@unittest.skipIf(not paddle.is_compiled_with_ipu(),
                 "core is not compiled with IPU")
class TestLogicalAnd(IPUOpTest):
    def setUp(self):
        self.set_atol()
        self.set_training()
        self.set_test_op()

    @property
    def fp16_enabled(self):
        return False

    def set_test_op(self):
        self.op = paddle.fluid.layers.logical_and

    def set_op_attrs(self):
        self.attrs = {}

    def _test_base(self, exec_mode):
        scope = paddle.static.Scope()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = self.SEED
        startup_prog.random_seed = self.SEED

        with paddle.static.scope_guard(scope):
            with paddle.static.program_guard(main_prog, startup_prog):
                x = paddle.static.data(
                    name=self.feed_list[0],
                    shape=self.feed_shape[0],
                    dtype=self.feed_dtype[0])
                y = paddle.static.data(
                    name=self.feed_list[1],
                    shape=self.feed_shape[1],
                    dtype=self.feed_dtype[1])

                out = self.op(x, y, **self.attrs)

            fetch_list = [out.name]

            if exec_mode == ExecutionMode.CPU_FP32:
                place = paddle.CPUPlace()
            else:
                place = paddle.IPUPlace()

            exe = paddle.static.Executor(place)
            exe.run(startup_prog)

            if exec_mode != ExecutionMode.CPU_FP32:
                feed_list = self.feed_list
                ipu_strategy = paddle.static.IpuStrategy()
                ipu_strategy.set_graph_config(is_training=self.is_training)
                if exec_mode == ExecutionMode.IPU_POPART_FP16:
                    ipu_strategy.set_precision_config(enable_fp16=True)
                program = paddle.static.IpuCompiledProgram(
                    main_prog,
                    ipu_strategy=ipu_strategy).compile(feed_list, fetch_list)
            else:
                program = main_prog

            result = exe.run(program, feed=self.feed, fetch_list=fetch_list)
            return result[0]

    def run_test_base(self):
        output_dict = {}
        for mode in ExecutionMode:
            if mode > ExecutionMode.IPU_FP32 and not self.fp16_enabled:
                break
            output_dict[mode] = self._test_base(mode).astype(np.int32)

        self.check(output_dict, check_shape=True)

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed.values()]
        self.feed_list = list(self.feed.keys())
        self.feed_dtype = ['bool', 'bool']

    def set_data_feed0(self):
        x = np.random.choice([True, False], size=(1, 3, 5, 5))
        y = np.random.choice([True, False], size=(1, 3, 5, 5))
        self.feed = {
            "x": x.astype('bool'),
            "y": y.astype('bool'),
        }
        self.set_feed_attr()

    def test_case0(self):
        self.set_data_feed0()
        self.set_op_attrs()
        self.run_test_base()


class TestLogicalOr(TestLogicalAnd):
    def set_test_op(self):
        self.op = paddle.fluid.layers.logical_or


if __name__ == "__main__":
    unittest.main()
