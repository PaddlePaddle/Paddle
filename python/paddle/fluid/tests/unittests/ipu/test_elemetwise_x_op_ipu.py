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
import paddle.fluid as fluid
import paddle.fluid.compiler as compiler
import paddle.optimizer
import paddle.static
from paddle.fluid.tests.unittests.ipu.op_test_ipu import (IPUOpTest,
                                                          np_dtype_to_fluid_str)

paddle.enable_static()


@unittest.skipIf(not paddle.is_compiled_with_ipu(),
                 "core is not compiled with IPU")
class TestMul(IPUOpTest):
    def setUp(self):
        self.set_atol()
        self.set_training()
        self.init_op()

    def init_op(self):
        self.op = paddle.fluid.layers.elementwise_mul

    def set_feed_attr(self):
        self.feed_shape = [x.shape for x in self.feed.values()]
        self.feed_list = list(self.feed.keys())
        self.feed_dtype = [
            np_dtype_to_fluid_str(x.dtype) for x in self.feed.values()
        ]

    def _test_base(self, run_ipu=True):
        scope = fluid.core.Scope()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        SEED = self.SEED
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED

        with fluid.scope_guard(scope):
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

            if run_ipu:
                place = paddle.IPUPlace()
            else:
                place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            exe.run(startup_prog)

            if run_ipu:
                feed_list = self.feed_list
                ipu_strategy = compiler.get_ipu_strategy()
                ipu_strategy.is_training = self.is_training
                program = compiler.IPUCompiledProgram(
                    main_prog,
                    ipu_strategy=ipu_strategy).compile(feed_list, fetch_list)
            else:
                program = main_prog

            result = exe.run(program, feed=self.feed, fetch_list=fetch_list)
            return result[0]

    def run_test_base(self):
        res0 = self._test_base(True)
        res1 = self._test_base(False)

        self.assertTrue(
            np.allclose(
                res0.flatten(), res1.flatten(), atol=self.atol))

        self.assertTrue(res0.shape == res1.shape)

    def test_case0(self):
        self.feed = {
            "x": np.random.uniform(size=(2, 3, 4, 5)).astype('float32'),
            "y": np.random.uniform(size=(2, 3, 4, 5)).astype('float32'),
        }
        self.attrs = {}
        self.set_feed_attr()
        self.run_test_base()

    def test_case1(self):
        self.feed = {
            "x": np.random.uniform(size=(2, 3, 4, 5)).astype('float32'),
            "y": np.random.uniform(size=(3, 4)).astype('float32'),
        }
        self.set_feed_attr()
        self.attrs = {"axis": 1}
        self.run_test_base()

    def test_case2(self):
        self.feed = {
            "x": np.random.uniform(size=(2, 3, 4, 5)).astype('float32'),
            "y": np.random.uniform(size=(5)).astype('float32'),
        }
        self.set_feed_attr()
        self.attrs = {"axis": -1}
        self.run_test_base()

    def test_case3(self):
        self.feed = {
            "x": np.random.uniform(size=(2, 3, 4, 5)).astype('float32'),
            "y": np.random.uniform(size=(2)).astype('float32'),
        }
        self.set_feed_attr()
        self.attrs = {"axis": 0}
        self.run_test_base()


class TestAdd(TestMul):
    def init_op(self):
        self.op = paddle.fluid.layers.elementwise_add


class TestSub(TestMul):
    def init_op(self):
        self.op = paddle.fluid.layers.elementwise_sub


class TestDiv(TestMul):
    def init_op(self):
        self.op = paddle.fluid.layers.elementwise_div


class TestMin(TestMul):
    def init_op(self):
        self.op = paddle.fluid.layers.elementwise_min


class TestMax(TestMul):
    def init_op(self):
        self.op = paddle.fluid.layers.elementwise_max


class TestPow(TestMul):
    def init_op(self):
        self.op = paddle.fluid.layers.elementwise_pow


class TestMod(TestMul):
    def init_op(self):
        self.op = paddle.fluid.layers.elementwise_mod


if __name__ == "__main__":
    unittest.main()
