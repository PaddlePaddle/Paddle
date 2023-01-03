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

import os
import shutil
import tempfile
import unittest

import numpy as np

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "Only test cuda Random Generator"
)
class TestGeneratorSeed(unittest.TestCase):
    """
    Test cases for cpu generator seed.
    """

    def test_gen_dropout_dygraph(self):
        gen = paddle.seed(12343)

        fluid.enable_dygraph()

        gen.manual_seed(111111111)
        st = paddle.get_cuda_rng_state()

        x = paddle.uniform([2, 10], dtype="float32", min=0.0, max=1.0)
        x_again = paddle.uniform([2, 10], dtype="float32", min=0.0, max=1.0)
        x_third = paddle.uniform([2, 10], dtype="float32", min=0.0, max=1.0)
        print("x: {}".format(x.numpy()))
        print("x_again: {}".format(x_again.numpy()))
        x = x + x_again + x_third
        y = paddle.nn.functional.dropout(x, 0.5)

        paddle.set_cuda_rng_state(st)

        x1 = paddle.uniform([2, 10], dtype="float32", min=0.0, max=1.0)
        x1_again = paddle.uniform([2, 10], dtype="float32", min=0.0, max=1.0)
        x1_third = paddle.uniform([2, 10], dtype="float32", min=0.0, max=1.0)
        x1 = x1 + x1_again + x1_third
        y1 = paddle.nn.functional.dropout(x1, 0.5)
        y_np = y.numpy()
        y1_np = y1.numpy()

        if core.is_compiled_with_cuda():
            print(">>>>>>> dropout dygraph >>>>>>>")
            np.testing.assert_allclose(y_np, y1_np, rtol=1e-05)

    def test_generator_gaussian_random_dygraph(self):
        """Test Generator seed."""
        fluid.enable_dygraph()

        st = paddle.get_cuda_rng_state()
        x1 = paddle.randn([120], dtype="float32")
        paddle.set_cuda_rng_state(st)
        x2 = paddle.randn([120], dtype="float32")
        paddle.set_cuda_rng_state(st)
        x3 = paddle.randn([120], dtype="float32")

        x1_np = x1.numpy()
        x2_np = x2.numpy()
        x3_np = x3.numpy()

        if core.is_compiled_with_cuda():
            print(">>>>>>> gaussian random dygraph >>>>>>>")
            np.testing.assert_allclose(x1_np, x2_np, rtol=1e-05)
            np.testing.assert_allclose(x2_np, x3_np, rtol=1e-05)

    def test_generator_randint_dygraph(self):
        """Test Generator seed."""

        fluid.enable_dygraph()

        paddle.seed(12312321111)
        x = paddle.randint(low=10, shape=[10], dtype="int32")
        st1 = paddle.get_cuda_rng_state()
        x1 = paddle.randint(low=10, shape=[10], dtype="int32")
        paddle.set_cuda_rng_state(st1)
        x2 = paddle.randint(low=10, shape=[10], dtype="int32")
        paddle.seed(12312321111)
        x3 = paddle.randint(low=10, shape=[10], dtype="int32")
        x_np = x.numpy()
        x1_np = x1.numpy()
        x2_np = x2.numpy()
        x3_np = x3.numpy()

        if core.is_compiled_with_cuda():
            print(">>>>>>> randint dygraph >>>>>>>")
            np.testing.assert_allclose(x_np, x3_np, rtol=1e-05)

    def test_gen_TruncatedNormal_initializer(self):
        fluid.disable_dygraph()

        gen = paddle.seed(123123143)
        cur_state = paddle.get_cuda_rng_state()

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

        paddle.seed(123123143)
        with fluid.program_guard(train_program, startup_program):
            exe.run(startup_program)
            out2 = exe.run(
                train_program, feed={}, fetch_list=[result_1, result_2]
            )

        out1_res1 = np.array(out1[0])
        out1_res2 = np.array(out1[1])
        out2_res1 = np.array(out2[0])
        out2_res2 = np.array(out2[1])

        if core.is_compiled_with_cuda():
            print(">>>>>>> truncated normal static >>>>>>>")
            np.testing.assert_allclose(out1_res1, out2_res1, rtol=1e-05)
            np.testing.assert_allclose(out1_res2, out2_res2, rtol=1e-05)
            self.assertTrue(not np.allclose(out1_res2, out1_res1))

    def test_generator_pickle(self):
        output_dir = tempfile.mkdtemp()
        random_file = os.path.join(output_dir, "random.pdmodel")

        fluid.enable_dygraph()
        x0 = paddle.randn([120], dtype="float32")

        st = paddle.get_cuda_rng_state()
        st_dict = {"random_state": st}
        print("state: ", st[0])

        paddle.save(st_dict, random_file)
        x1 = paddle.randn([120], dtype="float32")

        lt_dict = paddle.load(random_file)
        st = lt_dict["random_state"]

        paddle.set_cuda_rng_state(st)
        x2 = paddle.randn([120], dtype="float32")

        lt_dict = paddle.load(random_file)
        st = lt_dict["random_state"]
        paddle.set_cuda_rng_state(st)
        x3 = paddle.randn([120], dtype="float32")

        x1_np = x1.numpy()
        x2_np = x2.numpy()
        x3_np = x3.numpy()

        print(">>>>>>> gaussian random dygraph state load/save >>>>>>>")
        np.testing.assert_equal(x1_np, x2_np)
        np.testing.assert_equal(x1_np, x2_np)

        shutil.rmtree(output_dir)


if __name__ == "__main__":
    unittest.main()
