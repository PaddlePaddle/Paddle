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
from paddle.fluid.tests.unittests.ipu.op_test_ipu import (ExecutionMode,
                                                          IPUOpTest)


@unittest.skipIf(not paddle.is_compiled_with_ipu(),
                 "core is not compiled with IPU")
class TestMul(IPUOpTest):
    def setUp(self):
        self.set_atol()
        self.set_training()
        self.set_test_op()

    @property
    def fp16_enabled(self):
        if IPUOpTest.use_ipumodel():
            return False
        else:
            return True

    def set_test_op(self):
        self.op = paddle.fluid.layers.elementwise_mul

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())

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
                y = paddle.static.data(
                    name=self.feed_list[1],
                    shape=self.feed_shape[1],
                    dtype='float32')

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

    def test_case0(self):
        data_x = np.random.uniform(size=(2, 3, 4, 5))
        data_y = np.random.uniform(size=(2, 3, 4, 5))

        self.feed_fp32 = {
            "x": data_x.astype('float32'),
            "y": data_y.astype('float32'),
        }
        self.feed_fp16 = {
            "x": data_x.astype('float16'),
            "y": data_y.astype('float16'),
        }
        self.attrs = {}
        self.set_feed_attr()
        self.run_test_base()

    def test_case1(self):
        data_x = np.random.uniform(size=(2, 3, 4, 5))
        data_y = np.random.uniform(size=(3, 4))
        self.feed_fp32 = {
            "x": data_x.astype('float32'),
            "y": data_y.astype('float32'),
        }
        self.feed_fp16 = {
            "x": data_x.astype('float16'),
            "y": data_y.astype('float16'),
        }
        self.set_feed_attr()
        self.attrs = {"axis": 1}
        self.run_test_base()

    def test_case2(self):
        data_x = np.random.uniform(size=(2, 3, 4, 5))
        data_y = np.random.uniform(size=(5))
        self.feed_fp32 = {
            "x": data_x.astype('float32'),
            "y": data_y.astype('float32'),
        }
        self.feed_fp16 = {
            "x": data_x.astype('float16'),
            "y": data_y.astype('float16'),
        }
        self.set_feed_attr()
        self.attrs = {"axis": -1}
        self.run_test_base()

    def test_case3(self):
        data_x = np.random.uniform(size=(2, 3, 4, 5))
        data_y = np.random.uniform(size=(2))
        self.feed_fp32 = {
            "x": data_x.astype('float32'),
            "y": data_y.astype('float32'),
        }
        self.feed_fp16 = {
            "x": data_x.astype('float16'),
            "y": data_y.astype('float16'),
        }
        self.set_feed_attr()
        self.attrs = {"axis": 0}
        self.run_test_base()


class TestAdd(TestMul):
    def set_test_op(self):
        self.op = paddle.fluid.layers.elementwise_add


class TestSub(TestMul):
    def set_test_op(self):
        self.op = paddle.fluid.layers.elementwise_sub


class TestDiv(TestMul):
    def set_test_op(self):
        self.op = paddle.fluid.layers.elementwise_div


class TestMin(TestMul):
    def set_test_op(self):
        self.op = paddle.fluid.layers.elementwise_min


class TestMax(TestMul):
    def set_test_op(self):
        self.op = paddle.fluid.layers.elementwise_max


class TestPow(TestMul):
    def set_test_op(self):
        self.op = paddle.fluid.layers.elementwise_pow


class TestMod(TestMul):
    def set_atol(self):
        self.atol = 1e-7
        self.rtol = 1e-5
        self.atol_fp16 = 1e-2
        self.rtol_fp16 = 1e-3

    def set_test_op(self):
        self.op = paddle.fluid.layers.elementwise_mod


if __name__ == "__main__":
    unittest.main()
