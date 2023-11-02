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

import os
import tempfile
import unittest

import numpy as np

import paddle


class TestNewIr(unittest.TestCase):
    def test_with_new_ir(self):
        paddle.enable_static()
        place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )
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
        paddle.enable_static()
        place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

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
        paddle.enable_static()
        place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )
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
        # TODO(phlrain): support selected rows in GPU
        paddle.enable_static()
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
        paddle.enable_static()
        place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )
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


class TestNewIrDygraph(unittest.TestCase):
    def test_with_new_ir(self):
        paddle.disable_static()

        @paddle.jit.to_static
        def func(x, y):
            return x + y

        x = paddle.ones([2, 2], dtype='float32')
        y = paddle.ones([2, 2], dtype='float32')
        z = func(x, y)

        gold_res = np.ones([2, 2], dtype="float32") * 2
        np.testing.assert_array_equal(z.numpy(), gold_res)


class TestNewIrBackwardDygraph(unittest.TestCase):
    def test_with_new_ir(self):
        paddle.disable_static()
        build_strategy = paddle.static.BuildStrategy()
        build_strategy.enable_inplace = False

        @paddle.jit.to_static(build_strategy=build_strategy)
        def func(x, y):
            return x * y

        x = paddle.ones([2, 2], dtype='float32')
        y = paddle.ones([2, 2], dtype='float32')
        x.stop_gradient = False
        y.stop_gradient = False
        z = func(x, y)
        loss = z.mean()
        loss.backward()
        gold_res = np.ones([2, 2], dtype="float32")
        np.testing.assert_array_equal(z.numpy(), gold_res)

        gold_res = np.ones([2, 2], dtype="float32") * 0.25
        np.testing.assert_array_equal(x.gradient(), gold_res)
        np.testing.assert_array_equal(y.gradient(), gold_res)


class TestNewIrReshapeBackwardDygraph(unittest.TestCase):
    def test_with_new_ir(self):
        paddle.disable_static()
        build_strategy = paddle.static.BuildStrategy()
        build_strategy.enable_inplace = False

        @paddle.jit.to_static(build_strategy=build_strategy)
        def func(x, y):
            x = x.reshape([-1, 2, 2])
            y = y.reshape([-1, 2, 2])
            return x * y

        x = paddle.ones([2, 2], dtype='float32')
        y = paddle.ones([2, 2], dtype='float32')
        x.stop_gradient = False
        y.stop_gradient = False
        z = func(x, y)
        loss = z.mean()
        loss.backward()
        gold_res = np.ones([1, 2, 2], dtype="float32")

        np.testing.assert_array_equal(z.numpy(), gold_res)

        gold_res = np.ones([2, 2], dtype="float32") * 0.25
        np.testing.assert_array_equal(x.gradient(), gold_res)
        np.testing.assert_array_equal(y.gradient(), gold_res)


class TestSplitOp(unittest.TestCase):
    def test_with_new_ir(self):
        paddle.enable_static()
        place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

        exe = paddle.static.Executor(place)

        main_program = paddle.static.Program()
        new_scope = paddle.static.Scope()
        with paddle.static.scope_guard(new_scope):
            with paddle.static.program_guard(main_program):
                x = paddle.static.data("x", [6, 2], dtype="float32")
                out0, out1, out2 = paddle.split(x, num_or_sections=3, axis=0)

            np_a = np.random.rand(6, 2).astype("float32")
            out = exe.run(
                main_program,
                feed={"x": np_a},
                fetch_list=[out0.name],
            )

            np.testing.assert_array_equal(out[0], np_a[0:2])


class TestNewIrPrint(unittest.TestCase):
    def test_with_new_ir(self):
        paddle.enable_static()
        place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )
        exe = paddle.static.Executor(place)

        main_program = paddle.static.Program()
        new_scope = paddle.static.Scope()
        with paddle.static.scope_guard(new_scope):
            with paddle.static.program_guard(main_program):
                x = paddle.ones([2, 2], dtype="float32")
                y = paddle.ones([2, 2], dtype="float32")

                z = x + y
                z = paddle.static.Print(z)

            out = exe.run(main_program, {}, fetch_list=[z.name])

        gold_res = np.ones([2, 2], dtype="float32") * 2

        np.testing.assert_array_equal(out[0], gold_res)


class TestJitSaveOp(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(self.temp_dir.name, "new_ir_save_load")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_with_new_ir(self):
        paddle.disable_static()

        linear = paddle.nn.Linear(10, 10)
        path = os.path.join(self.model_path, "linear")

        paddle.jit.save(
            linear,
            path,
            input_spec=[paddle.static.InputSpec([10, 10], 'float32', 'x')],
        )

        paddle.enable_static()
        place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

        exe = paddle.static.Executor(place)

        [
            inference_program,
            feed_target_names,
            fetch_targets,
        ] = paddle.static.io.load_inference_model(
            self.model_path,
            executor=exe,
            model_filename="linear.pdmodel",
            params_filename="linear.pdiparams",
        )


class TestNewIrConcatDygraph(unittest.TestCase):
    def test_with_new_ir(self):
        paddle.disable_static()

        @paddle.jit.to_static
        def func(x, y):
            return paddle.concat([paddle.shape(x), y], -1)

        x = paddle.ones([2, 2], dtype='float32')
        y = paddle.ones([2], dtype='int32') * 2

        z = func(x, y)

        gold_res = np.ones([4], dtype="float32") * 2
        np.testing.assert_array_equal(z.numpy(), gold_res)


# TODO(phlrain): open this after fix pr(55509) confict
# class TestNewIrLogicalDygraph(unittest.TestCase):
#     def test_with_new_ir(self):
#         paddle.disable_static()

#         @paddle.jit.to_static
#         def func(x, y, z):
#             a = paddle.logical_and(x, y)
#             return z + a.cast("float32")

#         x = paddle.ones([2, 2], dtype='float32')
#         y = paddle.ones([2, 2], dtype='float32')
#         z = paddle.ones([2, 2], dtype='float32')

#         z = func(x, y, z)

#         gold_res = np.ones([2, 2], dtype="float32") * 2
#         np.testing.assert_array_equal(z.numpy(), gold_res)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
