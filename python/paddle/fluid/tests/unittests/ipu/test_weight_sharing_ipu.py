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
from paddle.fluid.tests.unittests.ipu.op_test_ipu import IPUOpTest


@unittest.skipIf(not paddle.is_compiled_with_ipu(),
                 "core is not compiled with IPU")
class TestWeightSharing(IPUOpTest):
    def setUp(self):
        self.set_atol()
        self.set_training()
        self.set_data_feed()
        self.set_feed_attr()
        self.set_op_attrs()

    def set_atol(self):
        self.atol = 1e-6
        self.rtol = 1e-5
        self.atol_fp16 = 1e-2
        self.rtol_fp16 = 1e-3

    def set_data_feed(self):
        x = np.random.randint(0, 768, size=(128, 1)).astype(np.int32)
        self.feed_cpu = {"x": x.astype(np.int64)}
        self.feed_ipu = {
            "x": np.tile(x.astype(np.int64)[np.newaxis, :], [3, 1, 1])
        }

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed_cpu.values()]
        self.feed_list = list(self.feed_cpu.keys())

    def set_op_attrs(self):
        self.attrs = {}

    def _test_base(self, run_ipu=True):
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

                with paddle.static.ipu_shard_guard(index=0, stage=0):
                    y = paddle.fluid.layers.embedding(
                        input=x,
                        size=[768, 768],
                        dtype='float32',
                        param_attr=paddle.fluid.ParamAttr(
                            name='word_embedding'),
                        is_sparse=False)

                with paddle.static.ipu_shard_guard(index=1, stage=1):
                    z = paddle.fluid.layers.fc(
                        input=y,
                        size=768,
                        param_attr=paddle.fluid.ParamAttr(name="fc"))

                with paddle.static.ipu_shard_guard(index=0, stage=2):
                    out = paddle.fluid.layers.matmul(
                        x=z,
                        y=main_prog.global_block().var('word_embedding'),
                        transpose_y=True)

            fetch_list = [out.name]

            if run_ipu:
                place = paddle.IPUPlace()
            else:
                place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            exe.run(startup_prog)

            if run_ipu:
                feed_list = self.feed_list
                ipu_strategy = paddle.static.IpuStrategy()
                ipu_strategy.set_graph_config(
                    num_ipus=2,
                    is_training=self.is_training,
                    enable_manual_shard=True)
                ipu_strategy.set_pipelining_config(
                    enable_pipelining=True, batches_per_step=3)
                program = paddle.static.IpuCompiledProgram(
                    main_prog,
                    ipu_strategy=ipu_strategy).compile(feed_list, fetch_list)
            else:
                program = main_prog

            feed = self.feed_ipu if run_ipu else self.feed_cpu
            result = exe.run(program, feed=feed, fetch_list=fetch_list)
            return result[0]

    def test_base(self):
        res0 = self._test_base(False)
        res1 = self._test_base(True)

        self.assertTrue(
            np.allclose(
                res0.flatten(), res1[0].flatten(), atol=self.atol))


if __name__ == "__main__":
    unittest.main()
