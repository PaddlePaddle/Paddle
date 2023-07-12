# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

paddle.enable_static()


class TestNewIr(unittest.TestCase):
    def test_with_new_ir(self):
        place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)

        main_program = paddle.static.Program()
        new_scope = paddle.static.Scope()
        with paddle.static.scope_guard(new_scope):
            with paddle.static.program_guard(main_program):
                x = paddle.ones([2, 2], dtype="float32")
                y = paddle.ones([2, 2], dtype="float32")

                z = x + y
            out = exe.run(main_program, {}, fetch_list=[z.name])

        gold_res = np.ones([2, 2], dtype="float32") * 2

        np.testing.assert_array_equal(out[0], gold_res)


class TestCombineOp(unittest.TestCase):
    def test_with_new_ir(self):
        place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)

        main_program = paddle.static.Program()
        new_scope = paddle.static.Scope()
        with paddle.static.scope_guard(new_scope):
            with paddle.static.program_guard(main_program):
                x = paddle.ones([2, 2], dtype="float32")
                y = paddle.ones([2, 2], dtype="float32")

                z = paddle.linalg.multi_dot([x, y])
            out = exe.run(main_program, {}, fetch_list=[z.name])

        gold_res = np.ones([2, 2], dtype="float32") * 2

        np.testing.assert_array_equal(out[0], gold_res)


class TestFeedOp(unittest.TestCase):
    def test_with_new_ir(self):
        place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)

        main_program = paddle.static.Program()
        new_scope = paddle.static.Scope()
        with paddle.static.scope_guard(new_scope):
            with paddle.static.program_guard(main_program):
                x = paddle.static.data("x", [2, 2], dtype="float32")
                y = paddle.static.data("y", [2, 2], dtype="float32")

                z = x + y

            np_a = np.random.rand(2, 2).astype("float32")
            np_b = np.random.rand(2, 2).astype("float32")
            out = exe.run(
                main_program,
                feed={"x": np_a, "y": np_b},
                fetch_list=[z.name],
            )

        gold_res = np_a + np_b

        np.testing.assert_array_equal(out[0], gold_res)


class TestSelectedRows(unittest.TestCase):
    def test_with_new_ir(self):
        place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)

        main_program = paddle.static.Program()
        new_scope = paddle.static.Scope()
        with paddle.static.scope_guard(new_scope):
            with paddle.static.program_guard(main_program):
                w = paddle.uniform([10, 10], dtype="float32")
                w.stop_gradient = False
                id = paddle.ones([2], dtype="int32")
                t = paddle.nn.functional.embedding(id, w, sparse=True)
                loss = paddle.mean(t)
                paddle.static.gradients(loss, w)

            out = exe.run(
                main_program,
                fetch_list=[loss.name],
            )


class TestAddGradOp(unittest.TestCase):
    def test_with_new_ir(self):
        place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)

        main_program = paddle.static.Program()
        new_scope = paddle.static.Scope()
        with paddle.static.scope_guard(new_scope):
            with paddle.static.program_guard(main_program):
                x = paddle.static.data("x", [2, 2], dtype="float32")
                y = paddle.static.data("y", [2, 2], dtype="float32")
                x.stop_gradient = False

                z = x * y

                paddle.static.gradients(z, x)

            np_a = np.random.rand(2, 2).astype("float32")
            np_b = np.random.rand(2, 2).astype("float32")
            out = exe.run(
                main_program,
                feed={"x": np_a, "y": np_b},
                fetch_list=[z.name],
            )

        gold_res = np_a * np_b

        np.testing.assert_array_equal(out[0], gold_res)


if __name__ == "__main__":
    unittest.main()
