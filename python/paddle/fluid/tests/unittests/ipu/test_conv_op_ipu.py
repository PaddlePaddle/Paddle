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
class TestBase(IPUOpTest):
    def setUp(self):
        self.set_atol()
        self.set_training()
        self.set_data_feed()
        self.set_feed_attr()
        self.set_op_attrs()

    @property
    def fp16_enabled(self):
        return True

    def set_atol(self):
        self.atol = 1e-6
        self.rtol = 1e-6
        self.atol_fp16 = 1e-3
        self.rtol_fp16 = 1e-3

    def set_data_feed(self):
        data = np.random.uniform(size=[1, 3, 10, 10])
        self.feed_fp32 = {'in_0': data.astype(np.float32)}
        self.feed_fp16 = {'in_0': data.astype(np.float16)}

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())

    def set_op_attrs(self):
        self.attrs = {}
        self.attrs['num_filters'] = 3
        self.attrs['filter_size'] = 3
        self.attrs['stride'] = 1
        self.attrs['padding'] = 0
        self.attrs['dilation'] = 1
        self.attrs['groups'] = 1
        self.attrs['data_format'] = 'NCHW'

    def _test_base(self, exec_mode):
        scope = paddle.static.Scope()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = self.SEED
        startup_prog.random_seed = self.SEED

        with paddle.static.scope_guard(scope):
            with paddle.static.program_guard(main_prog, startup_prog):
                image = paddle.static.data(
                    name=self.feed_list[0],
                    shape=self.feed_shape[0],
                    dtype='float32')

                out = paddle.fluid.layers.conv2d(image, **self.attrs)

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

    def test(self):
        output_dict = {}
        for mode in ExecutionMode:
            if mode > ExecutionMode.IPU_FP32 and not self.fp16_enabled:
                break
            output_dict[mode] = self._test_base(mode).flatten()

        self.check(output_dict)


class TestCase1(TestBase):
    def set_op_attrs(self):
        super().set_op_attrs()
        self.attrs['num_filters'] = 1


class TestCase2(TestBase):
    def set_op_attrs(self):
        super().set_op_attrs()
        self.attrs['filter_size'] = [3, 3]


class TestCase2_1(TestBase):
    def set_op_attrs(self):
        super().set_op_attrs()
        self.attrs['filter_size'] = [3, 2]


class TestCase3(TestBase):
    def set_op_attrs(self):
        super().set_op_attrs()
        self.attrs['stride'] = [2, 3]


class TestCase4(TestBase):
    def set_op_attrs(self):
        super().set_op_attrs()
        self.attrs['dilation'] = [2, 2]


class TestCase5(TestBase):
    def set_op_attrs(self):
        super().set_op_attrs()
        self.attrs['groups'] = 3


class TestCase6(TestBase):
    def set_op_attrs(self):
        super().set_op_attrs()
        self.attrs['padding'] = 2


class TestCase7(TestBase):
    def set_op_attrs(self):
        super().set_op_attrs()
        self.attrs['padding'] = [2, 3]


class TestCase8(TestBase):
    def set_op_attrs(self):
        super().set_op_attrs()
        self.attrs['padding'] = [1, 2, 2, 3]


if __name__ == "__main__":
    unittest.main()
