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
import paddle.fluid as fluid
import paddle.fluid.compiler as compiler
import paddle.optimizer
import paddle.static
<<<<<<< HEAD
from paddle.fluid.tests.unittests.ipu.op_test_ipu import (
    IPUOpTest,
    np_dtype_to_fluid_str,
)
=======
from paddle.fluid.tests.unittests.ipu.op_test_ipu import (IPUOpTest,
                                                          np_dtype_to_fluid_str)
from paddle.utils.cpp_extension import load
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf

paddle.enable_static()


class TestBase(IPUOpTest):
<<<<<<< HEAD
=======

>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
    def setUp(self):
        self.set_atol()
        self.set_training()
        self.set_feed()
        self.set_feed_attr()
        self.set_op()

    def set_op(self):
        # setup custom op
        self.op = paddle.incubate.identity_loss

    def set_feed(self):
        self.feed = {
<<<<<<< HEAD
            "x": np.random.uniform(low=-2, high=2, size=[3, 5]).astype(
                'float32'
            ),
=======
            "x": np.random.uniform(low=-2, high=2, size=[3,
                                                         5]).astype('float32'),
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
        }

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed.values()]
        self.feed_list = list(self.feed.keys())
        self.feed_dtype = [
            np_dtype_to_fluid_str(x.dtype) for x in self.feed.values()
        ]

    def _test_base(self, reduction):
        scope = fluid.core.Scope()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        SEED = 0
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED

        with fluid.scope_guard(scope):
            with paddle.static.program_guard(main_prog, startup_prog):
<<<<<<< HEAD
                x = paddle.static.data(
                    name=self.feed_list[0],
                    shape=self.feed_shape[0],
                    dtype=self.feed_dtype[0],
                )
=======
                x = paddle.static.data(name=self.feed_list[0],
                                       shape=self.feed_shape[0],
                                       dtype=self.feed_dtype[0])
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf

                out = self.op(x, reduction)
                fetch_list = [out.name]

            place = paddle.IPUPlace()
            exe = paddle.static.Executor(place)
            exe.run(startup_prog)

            feed_list = self.feed_list
            ipu_strategy = paddle.static.IpuStrategy()
            ipu_strategy.set_graph_config(num_ipus=1, is_training=False)
            ipu_compiler = compiler.IpuCompiledProgram(
<<<<<<< HEAD
                main_prog, ipu_strategy=ipu_strategy
            )
=======
                main_prog, ipu_strategy=ipu_strategy)
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
            program = ipu_compiler.compile(feed_list, fetch_list)

            ipu_res = exe.run(program, self.feed, fetch_list)

            if reduction == 0:
                # sum
                cpu_res = self.feed['x'].sum()
            elif reduction == 1:
                # mean
                cpu_res = self.feed['x'].mean()
            else:
                # none
                cpu_res = self.feed['x']

<<<<<<< HEAD
            np.testing.assert_allclose(
                ipu_res[0], cpu_res, rtol=1e-05, atol=self.atol
            )
=======
            self.assertTrue(np.allclose(ipu_res[0], cpu_res, atol=self.atol))
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf

    def test_base(self):
        # TODO: use string instead of int for reduction
        for reduction in [0, 1, 2]:
            self._test_base(reduction)


if __name__ == "__main__":
    unittest.main()
