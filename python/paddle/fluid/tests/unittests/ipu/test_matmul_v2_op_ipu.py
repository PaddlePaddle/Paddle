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

    def set_data_feed(self):
        x = np.random.uniform(size=[2, 3])
        y = np.random.uniform(size=[3, 2])

        self.feed_fp32 = {"x": x.astype(np.float32), "y": y.astype(np.float32)}
        self.feed_fp16 = {"x": x.astype(np.float16), "y": y.astype(np.float16)}

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())

    def set_op_attrs(self):
        self.attrs = {"transpose_x": False, "transpose_y": False}

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

                out = paddle.matmul(x, y, **self.attrs)

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

    def test_base(self):
        output_dict = {}
        for mode in ExecutionMode:
            if mode > ExecutionMode.IPU_FP32 and not self.fp16_enabled:
                break
            output_dict[mode] = self._test_base(mode).flatten()

        self.check(output_dict)


class TestCase1(TestBase):
    def set_op_attrs(self):
        self.attrs = {
            "transpose_x": True,
            "transpose_y": True,
        }


class TestCase3(TestBase):
    def set_data_feed(self):
        x = np.random.uniform(size=[5, 4, 2, 3])
        y = np.random.uniform(size=[5, 4, 3, 2])

        self.feed_fp32 = {"x": x.astype(np.float32), "y": y.astype(np.float32)}
        self.feed_fp16 = {"x": x.astype(np.float16), "y": y.astype(np.float16)}


class TestCase4(TestBase):
    def set_data_feed(self):
        x = np.random.uniform(size=[4, 2, 3])
        y = np.random.uniform(size=[4, 3, 2])

        self.feed_fp32 = {"x": x.astype(np.float32), "y": y.astype(np.float32)}
        self.feed_fp16 = {"x": x.astype(np.float16), "y": y.astype(np.float16)}


class TestCase5(TestBase):
    def set_data_feed(self):
        x = np.random.uniform(size=[4, 2, 3])
        y = np.random.uniform(size=[3, 2])

        self.feed_fp32 = {"x": x.astype(np.float32), "y": y.astype(np.float32)}
        self.feed_fp16 = {"x": x.astype(np.float16), "y": y.astype(np.float16)}


class TestCase6(TestBase):
    def set_data_feed(self):
        x = np.random.uniform(size=[3])
        y = np.random.uniform(size=[3])

        self.feed_fp32 = {"x": x.astype(np.float32), "y": y.astype(np.float32)}
        self.feed_fp16 = {"x": x.astype(np.float16), "y": y.astype(np.float16)}


@unittest.skip("not supported")
class TestCase6_2(TestCase6):
    def set_data_feed(self):
        x = np.random.uniform(size=[3])
        y = np.random.uniform(size=[3])

        self.feed_fp32 = {"x": x.astype(np.float32), "y": y.astype(np.float32)}
        self.feed_fp16 = {"x": x.astype(np.float16), "y": y.astype(np.float16)}

    def set_op_attrs(self):
        self.attrs = {"transpose_x": True, "transpose_y": True}


class TestCase7(TestBase):
    def set_data_feed(self):
        x = np.random.uniform(size=[3, 1])
        y = np.random.uniform(size=[1, 2])

        self.feed_fp32 = {"x": x.astype(np.float32), "y": y.astype(np.float32)}
        self.feed_fp16 = {"x": x.astype(np.float16), "y": y.astype(np.float16)}


@unittest.skip("dim > 4 is not supported")
class TestCase8(TestBase):
    def set_data_feed(self):
        self.feed = {
            "x": np.random.uniform(size=[6, 5, 4, 2, 3]).astype('float32'),
            "y": np.random.uniform(size=[6, 5, 4, 3, 2]).astype('float32'),
        }


if __name__ == "__main__":
    unittest.main()
