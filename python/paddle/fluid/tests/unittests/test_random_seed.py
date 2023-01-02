#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
"""Test cloud role maker."""

import unittest

import numpy as np

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.generator as generator
from paddle.tensor import random


class TestGeneratorSeed(unittest.TestCase):
    #     """
    #     Test cases for cpu generator seed.
    #     """

    def test_generator_uniform_random_dygraph(self):
        """Test Generator seed."""

        fluid.enable_dygraph()

        gen = paddle.seed(12312321111)
        x = paddle.uniform([10], dtype="float32", min=0.0, max=1.0)

        st1 = gen.get_state()
        x1 = paddle.uniform([10], dtype="float32", min=0.0, max=1.0)

        gen.set_state(st1)
        print(gen.get_state())
        x2 = paddle.uniform([10], dtype="float32", min=0.0, max=1.0)

        paddle.seed(12312321111)
        x3 = paddle.uniform([10], dtype="float32", min=0.0, max=1.0)

        x_np = x.numpy()
        x1_np = x1.numpy()
        x2_np = x2.numpy()
        x3_np = x3.numpy()

        if not core.is_compiled_with_cuda():
            np.testing.assert_allclose(x1_np, x2_np, rtol=1e-05)
            np.testing.assert_allclose(x_np, x3_np, rtol=1e-05)

    def test_generator_uniform_random_static(self):
        fluid.disable_dygraph()

        gen = paddle.seed(123123143)

        startup_program = fluid.Program()
        train_program = fluid.Program()
        with fluid.program_guard(train_program, startup_program):
            # example 1:
            # attr shape is a list which doesn't contain tensor Variable.
            result_1 = paddle.uniform(shape=[3, 4])
            result_2 = paddle.uniform(shape=[3, 4])

            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(startup_program)
            out1 = exe.run(
                train_program, feed={}, fetch_list=[result_1, result_2]
            )
            # gen.set_state(cur_state)
            gen.manual_seed(123123143)
            out2 = exe.run(
                train_program, feed={}, fetch_list=[result_1, result_2]
            )

            out1_res1 = np.array(out1[0])
            out1_res2 = np.array(out1[1])
            out2_res1 = np.array(out2[0])
            out2_res2 = np.array(out2[1])

            if not core.is_compiled_with_cuda():
                np.testing.assert_allclose(out1_res1, out2_res1, rtol=1e-05)
                np.testing.assert_allclose(out1_res2, out2_res2, rtol=1e-05)
                self.assertTrue(not np.allclose(out1_res2, out1_res1))

    def test_gen_dropout_dygraph(self):
        fluid.enable_dygraph()

        gen = paddle.seed(111111111)
        st = gen.get_state()
        # x = np.arange(1,101).reshape(2,50).astype("float32")
        x = paddle.uniform([2, 10], dtype="float32", min=0.0, max=1.0)
        y = paddle.nn.functional.dropout(x, 0.5)
        gen.manual_seed(111111111)
        # gen.set_state(st)
        x1 = paddle.uniform([2, 10], dtype="float32", min=0.0, max=1.0)
        y1 = paddle.nn.functional.dropout(x1, 0.5)
        y_np = y.numpy()
        y1_np = y1.numpy()

        if not core.is_compiled_with_cuda():
            print(">>>>>>> dropout dygraph >>>>>>>")
            np.testing.assert_allclose(y_np, y1_np, rtol=1e-05)

    def test_gen_dropout_static(self):
        fluid.disable_dygraph()

        gen = paddle.seed(123123143)

        startup_program = fluid.Program()
        train_program = fluid.Program()
        with fluid.program_guard(train_program, startup_program):
            # example 1:
            # attr shape is a list which doesn't contain tensor Variable.
            x_1 = paddle.uniform(shape=[2, 10])
            y_1 = paddle.nn.functional.dropout(x_1, 0.5)
            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(startup_program)
            out1 = exe.run(train_program, feed={}, fetch_list=[y_1])
            # gen.set_state(cur_state)
            gen.manual_seed(123123143)
            out2 = exe.run(train_program, feed={}, fetch_list=[y_1])
        out1_np = np.array(out1[0])
        out2_np = np.array(out2[0])

        if not core.is_compiled_with_cuda():
            print(">>>>>>> dropout static >>>>>>>")
            np.testing.assert_allclose(out1_np, out2_np, rtol=1e-05)

    def test_generator_gaussian_random_dygraph(self):
        """Test Generator seed."""
        fluid.enable_dygraph()

        gen = paddle.seed(12312321111)
        x = random.gaussian([10], dtype="float32")
        st1 = gen.get_state()
        x1 = random.gaussian([10], dtype="float32")
        gen.set_state(st1)
        x2 = random.gaussian([10], dtype="float32")
        gen.manual_seed(12312321111)
        x3 = random.gaussian([10], dtype="float32")
        x_np = x.numpy()
        x1_np = x1.numpy()
        x2_np = x2.numpy()
        x3_np = x3.numpy()

        if not core.is_compiled_with_cuda():
            print(">>>>>>> gaussian random dygraph >>>>>>>")
            np.testing.assert_allclose(x1_np, x2_np, rtol=1e-05)
            np.testing.assert_allclose(x_np, x3_np, rtol=1e-05)

    def test_generator_gaussian_random_static(self):
        fluid.disable_dygraph()

        gen = paddle.seed(123123143)

        startup_program = fluid.Program()
        train_program = fluid.Program()
        with fluid.program_guard(train_program, startup_program):
            # example 1:
            # attr shape is a list which doesn't contain tensor Variable.
            result_1 = random.gaussian(shape=[3, 4])
            result_2 = random.gaussian(shape=[3, 4])

            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(startup_program)
            out1 = exe.run(
                train_program, feed={}, fetch_list=[result_1, result_2]
            )
            # gen.set_state(cur_state)
            gen.manual_seed(123123143)
            out2 = exe.run(
                train_program, feed={}, fetch_list=[result_1, result_2]
            )

            out1_res1 = np.array(out1[0])
            out1_res2 = np.array(out1[1])
            out2_res1 = np.array(out2[0])
            out2_res2 = np.array(out2[1])

            if not core.is_compiled_with_cuda():
                print(">>>>>>> gaussian random static >>>>>>>")
                np.testing.assert_allclose(out1_res1, out2_res1, rtol=1e-05)
                np.testing.assert_allclose(out1_res2, out2_res2, rtol=1e-05)
                self.assertTrue(not np.allclose(out1_res2, out1_res1))

    def test_generator_randint_dygraph(self):
        """Test Generator seed."""
        gen = generator.Generator()

        fluid.enable_dygraph()

        gen = paddle.seed(12312321111)
        x = paddle.randint(low=10, shape=[10], dtype="int32")
        st1 = gen.get_state()
        x1 = paddle.randint(low=10, shape=[10], dtype="int32")
        gen.set_state(st1)
        x2 = paddle.randint(low=10, shape=[10], dtype="int32")
        gen.manual_seed(12312321111)
        x3 = paddle.randint(low=10, shape=[10], dtype="int32")
        x_np = x.numpy()
        x1_np = x1.numpy()
        x2_np = x2.numpy()
        x3_np = x3.numpy()

        if not core.is_compiled_with_cuda():
            print(">>>>>>> randint dygraph >>>>>>>")
            np.testing.assert_allclose(x1_np, x2_np, rtol=1e-05)
            np.testing.assert_allclose(x_np, x3_np, rtol=1e-05)

    def test_generator_uniform_random_static_1(self):
        fluid.disable_dygraph()

        gen = paddle.seed(123123143)

        startup_program = fluid.Program()
        train_program = fluid.Program()
        with fluid.program_guard(train_program, startup_program):
            # example 1:
            # attr shape is a list which doesn't contain tensor Variable.
            result_1 = paddle.uniform(shape=[3, 4])
            result_2 = paddle.uniform(shape=[3, 4])

            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(startup_program)
            out1 = exe.run(
                train_program, feed={}, fetch_list=[result_1, result_2]
            )
            # gen.set_state(cur_state)
            gen.manual_seed(123123143)
            out2 = exe.run(
                train_program, feed={}, fetch_list=[result_1, result_2]
            )

            out1_res1 = np.array(out1[0])
            out1_res2 = np.array(out1[1])
            out2_res1 = np.array(out2[0])
            out2_res2 = np.array(out2[1])

            if not core.is_compiled_with_cuda():
                np.testing.assert_allclose(out1_res1, out2_res1, rtol=1e-05)
                np.testing.assert_allclose(out1_res2, out2_res2, rtol=1e-05)
                self.assertTrue(not np.allclose(out1_res2, out1_res1))

    def test_generator_randint_dygraph_1(self):
        """Test Generator seed."""
        fluid.enable_dygraph()

        gen = paddle.seed(12312321111)
        x = paddle.randint(low=1)
        st1 = gen.get_state()
        x1 = paddle.randint(low=1)
        gen.set_state(st1)
        x2 = paddle.randint(low=1)
        gen.manual_seed(12312321111)
        x3 = paddle.randint(low=1)
        x_np = x.numpy()
        x1_np = x1.numpy()
        x2_np = x2.numpy()
        x3_np = x3.numpy()
        if not core.is_compiled_with_cuda():
            np.testing.assert_allclose(x1_np, x2_np, rtol=1e-05)
            np.testing.assert_allclose(x_np, x3_np, rtol=1e-05)

    def test_generator_ranint_static(self):
        fluid.disable_dygraph()

        gen = paddle.seed(123123143)

        startup_program = fluid.Program()
        train_program = fluid.Program()
        with fluid.program_guard(train_program, startup_program):
            # example 1:
            # attr shape is a list which doesn't contain tensor Variable.
            result_1 = paddle.randint(low=10, shape=[3, 4])
            result_2 = paddle.randint(low=10, shape=[3, 4])

            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(startup_program)
            out1 = exe.run(
                train_program, feed={}, fetch_list=[result_1, result_2]
            )
            # gen.set_state(cur_state)
            gen.manual_seed(123123143)
            out2 = exe.run(
                train_program, feed={}, fetch_list=[result_1, result_2]
            )

            out1_res1 = np.array(out1[0])
            out1_res2 = np.array(out1[1])
            out2_res1 = np.array(out2[0])
            out2_res2 = np.array(out2[1])

            if not core.is_compiled_with_cuda():
                print(">>>>>>> randint static >>>>>>>")
                np.testing.assert_allclose(out1_res1, out2_res1, rtol=1e-05)
                np.testing.assert_allclose(out1_res2, out2_res2, rtol=1e-05)
                self.assertTrue(not np.allclose(out1_res2, out1_res1))

    def test_generator_randperm_dygraph(self):
        """Test Generator seed."""

        fluid.enable_dygraph()

        gen = paddle.seed(12312321111)
        x = paddle.randperm(10)
        st1 = gen.get_state()
        x1 = paddle.randperm(10)
        gen.set_state(st1)
        x2 = paddle.randperm(10)
        gen.manual_seed(12312321111)
        x3 = paddle.randperm(10)
        x_np = x.numpy()
        x1_np = x1.numpy()
        x2_np = x2.numpy()
        x3_np = x3.numpy()

        if not core.is_compiled_with_cuda():
            print(">>>>>>> randperm dygraph >>>>>>>")
            np.testing.assert_allclose(x1_np, x2_np, rtol=1e-05)
            np.testing.assert_allclose(x_np, x3_np, rtol=1e-05)

    def test_generator_randperm_static(self):

        fluid.disable_dygraph()

        paddle.seed(123123143)

        startup_program = fluid.Program()
        train_program = fluid.Program()
        with fluid.program_guard(train_program, startup_program):
            # example 1:
            # attr shape is a list which doesn't contain tensor Variable.
            result_1 = paddle.randperm(10)
            result_2 = paddle.randperm(10)

            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(startup_program)
            out1 = exe.run(
                train_program, feed={}, fetch_list=[result_1, result_2]
            )

            paddle.seed(123123143)
            out2 = exe.run(
                train_program, feed={}, fetch_list=[result_1, result_2]
            )

            out1_res1 = np.array(out1[0])
            out1_res2 = np.array(out1[1])
            out2_res1 = np.array(out2[0])
            out2_res2 = np.array(out2[1])

            if not core.is_compiled_with_cuda():
                print(">>>>>>> randperm static >>>>>>>")
                np.testing.assert_allclose(out1_res1, out2_res1, rtol=1e-05)
                np.testing.assert_allclose(out1_res2, out2_res2, rtol=1e-05)
                self.assertTrue(not np.allclose(out1_res2, out1_res1))

    def test_gen_TruncatedNormal_initializer(self):
        fluid.disable_dygraph()

        gen = paddle.seed(123123143)
        cur_state = gen.get_state()

        startup_program = fluid.Program()
        train_program = fluid.Program()
        with fluid.program_guard(train_program, startup_program):
            # example 1:
            # attr shape is a list which doesn't contain tensor Variable.
            x = paddle.uniform(shape=[2, 10])
            result_1 = fluid.layers.fc(
                input=x,
                size=10,
                param_attr=fluid.initializer.TruncatedNormal(
                    loc=0.0, scale=2.0
                ),
            )
            result_2 = fluid.layers.fc(
                input=x,
                size=10,
                param_attr=fluid.initializer.TruncatedNormal(
                    loc=0.0, scale=2.0
                ),
            )

            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(startup_program)
            out1 = exe.run(
                train_program, feed={}, fetch_list=[result_1, result_2]
            )

        gen.manual_seed(123123143)
        with fluid.program_guard(train_program, startup_program):
            exe.run(startup_program)
            out2 = exe.run(
                train_program, feed={}, fetch_list=[result_1, result_2]
            )

        out1_res1 = np.array(out1[0])
        out1_res2 = np.array(out1[1])
        out2_res1 = np.array(out2[0])
        out2_res2 = np.array(out2[1])

        if not core.is_compiled_with_cuda():
            print(">>>>>>> sampling id static >>>>>>>")
            np.testing.assert_allclose(out1_res1, out2_res1, rtol=1e-05)
            np.testing.assert_allclose(out1_res2, out2_res2, rtol=1e-05)
            self.assertTrue(not np.allclose(out1_res2, out1_res1))


if __name__ == "__main__":
    unittest.main()
