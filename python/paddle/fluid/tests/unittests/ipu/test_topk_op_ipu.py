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
class TestTopKOp(IPUOpTest):
    def setUp(self):
        self.set_atol()
        self.set_training()
        self.set_data_feed()
        self.set_feed_attr()
        self.set_test_op()
        self.set_op_attrs()

    @property
    def fp16_enabled(self):
        return True

    def set_test_op(self):
        self.op = paddle.fluid.layers.topk

    def set_data_feed(self):
        data = np.random.uniform(size=[3, 5])
        self.feed_fp32 = {"in_0": data.astype(np.float32)}
        self.feed_fp16 = {"in_0": data.astype(np.float16)}

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())

    def set_op_attrs(self):
        self.use_k_as_const_variable = False
        self.attrs = {}
        if not self.use_k_as_const_variable:
            self.attrs["k"] = 3

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

                if not self.use_k_as_const_variable:
                    topk_values, topk_indices = self.op(x, **self.attrs)
                else:
                    # !important, popart cannot accept non const tensor
                    K_t = paddle.fluid.layers.fill_constant(
                        shape=[1], dtype='int32', value=self.k, name="in_2")
                    topk_values, topk_indices = self.op(x, K_t, **self.attrs)

                fetch_list = [topk_values.name, topk_indices.name]

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
            return result

    def test_base(self):
        value_dict = {}
        index_dict = {}
        for mode in ExecutionMode:
            if mode > ExecutionMode.IPU_FP32 and not self.fp16_enabled:
                break
            value, index = self._test_base(mode)
            value_dict[mode] = value
            index_dict[mode] = index

        self.check(value_dict)
        self.check(index_dict)


class TestCase2(TestTopKOp):
    def set_test_op(self):
        self.op = paddle.topk


@unittest.skip("Trying to get data as int64 but it is of type int32")
class TestCase3(TestTopKOp):
    def set_op_attrs(self):
        self.use_k_as_const_variable = True
        self.attrs = {}
        self.k = 2


@unittest.skip("Trying to get data as int64 but it is of type int32")
class TestCase4(TestCase3):
    def set_test_op(self):
        self.op = paddle.topk


if __name__ == "__main__":
    unittest.main()
