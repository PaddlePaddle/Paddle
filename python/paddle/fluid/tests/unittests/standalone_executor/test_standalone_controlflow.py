# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np

import paddle
import paddle.fluid.layers as layers
from paddle.fluid import core, framework
from paddle.fluid.framework import Program, program_guard

paddle.enable_static()


#  test the compatibility of new executor: run old
#  and new executor twice and check the result.
#  please override the _get_feeds() and build_prgram()
class TestCompatibility(unittest.TestCase):
    def setUp(self):
        self.place = (
            paddle.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )
        self.iter_run = 4

    def _get_feed(self):
        """return the feeds"""
        return None

    def build_program(self):
        def true_func():
            return layers.fill_constant(
                shape=[1, 2], dtype='int32', value=1
            ), layers.fill_constant(shape=[2, 3], dtype='bool', value=True)

        def false_func():
            return layers.fill_constant(
                shape=[3, 4], dtype='float32', value=3
            ), layers.fill_constant(shape=[4, 5], dtype='int64', value=2)

        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            x = layers.fill_constant(shape=[1], dtype='float32', value=0.1)
            y = layers.fill_constant(shape=[1], dtype='float32', value=0.23)
            pred = paddle.less_than(x, y)
            out = paddle.static.nn.cond(pred, true_func, false_func)
            # out is a tuple containing 2 tensors
            return main_program, startup_program, out

    def _run(self, feed):
        paddle.seed(2020)

        main_program, startup_program, fetch_vars = self.build_program()

        exe = paddle.static.Executor(self.place)
        exe.run(startup_program)
        ret = []
        for i in range(self.iter_run):
            ret.append(exe.run(main_program, feed=feed, fetch_list=fetch_vars))
        return ret

    def run_raw_executor(self, feed):
        with framework._enable_standalone_executor(False):
            out = self._run(feed)
        return out

    def run_new_executor(self, feed):
        with framework._enable_standalone_executor(True):
            out = self._run(feed)
        return out

    def test_with_feed(self):
        feed = self._get_feed()
        res = self.run_new_executor(feed)
        gt = self.run_raw_executor(feed)
        for x, y in zip(gt, res):
            if isinstance(x, list):
                for tx, ty in zip(x, y):
                    np.testing.assert_array_equal(tx, ty)
            elif isinstance(x, np.ndarray):
                np.testing.assert_array_equal(tx, ty)
            else:
                raise Exception("Not Implement!")


class TestWhile(TestCompatibility):
    def _get_feed(self):
        """return the feeds"""
        return None

    def build_program(self):
        def cond(i, ten):
            return i < ten

        def body(i, ten):
            i = i + 1
            return [i, ten]

        main_program = paddle.static.default_main_program()
        startup_program = paddle.static.default_startup_program()
        with paddle.static.program_guard(main_program, startup_program):
            i = paddle.full(
                shape=[1], fill_value=0, dtype='int64'
            )  # loop counter
            ten = paddle.full(
                shape=[1], fill_value=10, dtype='int64'
            )  # loop length
            i, ten = paddle.static.nn.while_loop(cond, body, [i, ten])

            exe = paddle.static.Executor(paddle.CPUPlace())
        return main_program, startup_program, i


if __name__ == "__main__":
    unittest.main()
