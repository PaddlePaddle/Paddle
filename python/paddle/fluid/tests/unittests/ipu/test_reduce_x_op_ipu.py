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
import paddle.static
from paddle.fluid.tests.unittests.ipu.op_test_ipu import IPUOpTest, ExecutionMode


@unittest.skipIf(not paddle.is_compiled_with_ipu(),
                 "core is not compiled with IPU")
class TestMean(IPUOpTest):
    def setUp(self):
        self.set_atol()
        self.set_training()
        self.set_test_op()

    @property
    def fp16_enabled(self):
        return True

    def set_test_op(self):
        self.op = paddle.fluid.layers.reduce_mean

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())
        self.feed_dtype = [x.dtype for x in self.feed_fp32.values()]

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
                    dtype='float32')

                out = self.op(x, **self.attrs)

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

            feed = self.feed_fp32
            if exec_mode > ExecutionMode.IPU_FP32:
                feed = self.feed_fp16

            result = exe.run(program, feed=feed, fetch_list=fetch_list)
            return result[0]

    def run_test_base(self):
        output_dict = {}
        for mode in ExecutionMode:
            if mode > ExecutionMode.IPU_FP32 and not self.fp16_enabled:
                break
            output_dict[mode] = self._test_base(mode).flatten()

        self.check(output_dict)

    def set_data_feed0(self):
        data = np.random.uniform(size=[2, 4])
        self.feed_fp32 = {"in_0": data.astype(np.float32)}
        self.feed_fp16 = {"in_0": data.astype(np.float16)}
        self.set_feed_attr()

    def set_data_feed1(self):
        data = np.random.uniform(size=[2, 2, 2])
        self.feed_fp32 = {"in_0": data.astype(np.float32)}
        self.feed_fp16 = {"in_0": data.astype(np.float16)}
        self.set_feed_attr()

    def set_op_attr0(self):
        self.attrs = {}
        self.attrs['dim'] = None
        self.attrs['keep_dim'] = False

    def test_case0(self):
        self.set_data_feed0()
        self.set_op_attr0()
        self.run_test_base()

    def test_case1(self):
        self.set_data_feed0()
        self.set_op_attr0()
        self.attrs['dim'] = 0
        self.run_test_base()

    def test_case2(self):
        self.set_data_feed0()
        self.set_op_attr0()
        self.attrs['dim'] = -1
        self.run_test_base()

    def test_case3(self):
        self.set_data_feed0()
        self.set_op_attr0()
        self.attrs['dim'] = 1
        self.run_test_base()

    def test_case4(self):
        self.set_data_feed0()
        self.attrs = {}
        self.attrs['dim'] = 1
        self.attrs['keep_dim'] = True
        self.run_test_base()

    def test_case5(self):
        self.set_data_feed1()
        self.attrs = {}
        self.attrs['dim'] = [1, 2]
        self.attrs['keep_dim'] = False
        self.run_test_base()

    def test_case6(self):
        self.set_data_feed1()
        self.attrs = {}
        self.attrs['dim'] = [0, 1]
        self.attrs['keep_dim'] = False
        self.run_test_base()

    def test_case7(self):
        self.set_data_feed1()
        self.attrs = {}
        self.attrs['dim'] = [0, 1]
        self.attrs['keep_dim'] = True
        self.run_test_base()


class TestMax(TestMean):
    def set_test_op(self):
        self.op = paddle.fluid.layers.reduce_max


class TestMin(TestMean):
    def set_test_op(self):
        self.op = paddle.fluid.layers.reduce_min


class TestProd(TestMean):
    def set_test_op(self):
        self.op = paddle.fluid.layers.reduce_prod


class TestSum(TestMean):
    def set_test_op(self):
        self.op = paddle.fluid.layers.reduce_sum


if __name__ == "__main__":
    unittest.main()
