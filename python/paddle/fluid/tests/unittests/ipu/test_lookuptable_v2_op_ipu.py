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
        x = np.array([[[1], [3]], [[2], [4]], [[4], [127]]])
        self.feed_cpu = {"x": x.astype(np.int64)}
        self.feed_ipu = {"x": x.astype(np.int32)}

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed_cpu.values()]
        self.feed_list = list(self.feed_cpu.keys())
        self.feed_dtype = [x.dtype for x in self.feed_cpu.values()]

    def set_op_attrs(self):
        self.attrs = {
            "num_embeddings": 128,
            "embedding_dim": 16,
            "sparse": False,
            "padding_idx": -1,
            "weight_attr": None
        }

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
                    dtype='int64')

                embedding = paddle.nn.Embedding(**self.attrs)
                out = embedding(x)

                if self.is_training:
                    loss = paddle.mean(out)
                    adam = paddle.optimizer.Adam(learning_rate=1e-2)
                    adam.minimize(loss)
                    fetch_list = [loss.name]
                else:
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

            feed = self.feed_cpu
            if exec_mode > ExecutionMode.CPU_FP32:
                feed = self.feed_ipu

            if self.is_training:
                result = []
                for _ in range(self.epoch):
                    loss_res = exe.run(program,
                                       feed=feed,
                                       fetch_list=fetch_list)
                    result.append(loss_res[0])
                return np.array(result)
            else:
                result = exe.run(program, feed=feed, fetch_list=fetch_list)
                return result[0]

    def test(self):
        output_dict = {}
        for mode in ExecutionMode:
            if mode > ExecutionMode.IPU_FP32 and (not self.fp16_enabled or
                                                  self.is_training):
                break
            output_dict[mode] = self._test_base(mode).flatten()

        self.check(output_dict)


class TestTrainCase1(TestBase):
    def set_atol(self):
        self.atol = 1e-7
        self.rtol = 1e-6
        self.atol_fp16 = 1e-3
        self.rtol_fp16 = 1e-3

    def set_training(self):
        self.is_training = True
        self.epoch = 10


if __name__ == "__main__":
    unittest.main()
