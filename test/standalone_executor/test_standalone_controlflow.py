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
from paddle.base import core

paddle.enable_static()


#  test the compatibility of new executor: run old
#  and new executor twice and check the result.
#  please override the _get_feeds() and build_program(), run_dygraph_once()
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
            return paddle.tensor.fill_constant(
                shape=[1, 2], dtype='float32', value=1
            ), paddle.tensor.fill_constant(shape=[2, 3], dtype='int64', value=1)

        def false_func():
            return paddle.tensor.fill_constant(
                shape=[3, 4], dtype='float32', value=3
            ), paddle.tensor.fill_constant(shape=[4, 5], dtype='int64', value=2)

        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.tensor.fill_constant(
                shape=[1], dtype='float32', value=0.1
            )
            y = paddle.tensor.fill_constant(
                shape=[1], dtype='float32', value=0.23
            )
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

    def run_dygraph_once(self, feed):
        x = paddle.tensor.fill_constant(shape=[1], dtype='float32', value=0.1)
        y = paddle.tensor.fill_constant(shape=[1], dtype='float32', value=0.23)
        if x < y:
            out = [
                paddle.tensor.fill_constant(
                    shape=[1, 2], dtype='float32', value=1
                ).numpy(),
                paddle.tensor.fill_constant(
                    shape=[2, 3], dtype='int64', value=1
                ).numpy(),
            ]
        else:
            out = [
                paddle.tensor.fill_constant(
                    shape=[3, 4], dtype='float32', value=3
                ).numpy(),
                paddle.tensor.fill_constant(
                    shape=[4, 5], dtype='int64', value=2
                ).numpy(),
            ]
        return out

    def run_dygraph(self, feed):
        ret = []
        for _ in range(self.iter_run):
            ret.append(self.run_dygraph_once(feed))
        return ret

    def run_new_executor(self, feed):
        out = self._run(feed)
        return out

    def test_with_feed(self):
        feed = self._get_feed()
        paddle.enable_static()
        res = self.run_new_executor(feed)
        paddle.disable_static()

        gt = self.run_dygraph(feed)

        for x, y in zip(gt, res):
            if isinstance(x, list):
                for tx, ty in zip(x, y):
                    np.testing.assert_array_equal(tx, ty)
            elif isinstance(x, np.ndarray):
                np.testing.assert_array_equal(x, y)
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

    def run_dygraph_once(self, feed):
        i = 1
        while i < 10:
            i = i + 1
        return [i]


if __name__ == "__main__":
    unittest.main()
