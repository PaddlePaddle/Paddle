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

    def set_data_feed(self):
        x = np.random.uniform(size=[3, 7])
        label = np.arange(3).reshape([3, 1])
        self.feed_fp32 = {
            "x": x.astype(np.float32),
            "label": label.astype(np.int64)
        }
        self.feed_fp16 = {
            "x": x.astype(np.float16),
            "label": label.astype(np.int32)
        }

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())

    def set_op_attrs(self):
        self.attrs = {'soft_label': False, }

    def np_nll_loss(self):
        tmp = -np.log(self.feed_fp32['x'])
        label = self.feed_fp32['label']
        indice = [range(label.shape[0]), label.flatten()]
        self.np_ref = tmp[indice]

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
                    dtype="float32")

                if exec_mode != ExecutionMode.CPU_FP32:
                    label = paddle.static.data(
                        name=self.feed_list[1],
                        shape=self.feed_shape[1],
                        dtype='int32')
                else:
                    label = paddle.static.data(
                        name=self.feed_list[1],
                        shape=self.feed_shape[1],
                        dtype='int64')

                out = paddle.fluid.layers.cross_entropy(
                    input=x, label=label, **self.attrs)

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

            if exec_mode != ExecutionMode.CPU_FP32:
                feed['label'] = feed['label'].astype(np.int32)

            result = exe.run(program, feed=feed, fetch_list=fetch_list)
            return result[0]

    def test(self):
        output_dict = {}
        for mode in ExecutionMode:
            if mode > ExecutionMode.IPU_FP32 and not self.fp16_enabled:
                break
            output_dict[mode] = self._test_base(mode).flatten()
        self.np_nll_loss()

        self.check(output_dict)


class TestCase1(TestBase):
    def set_op_attrs(self):
        self.attrs = {
            'soft_label': False,
            'ignore_index': 1,
        }


class TestCase2(TestBase):
    def set_data_feed(self):
        x = np.random.uniform(size=[30, 70])
        label = np.arange(30).reshape([30, 1])

        self.feed_fp32 = {
            "x": x.astype(np.float32),
            "label": label.astype(np.int64)
        }
        self.feed_fp16 = {
            "x": x.astype(np.float16),
            "label": label.astype(np.int32)
        }


@unittest.skip("soft_label=True is not supported")
class TestCase3(TestBase):
    def set_op_attrs(self):
        self.attrs = {'soft_label': True, }


if __name__ == "__main__":
    unittest.main()
