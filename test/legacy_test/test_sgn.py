#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from utils import static_guard

import paddle


def np_sgn(x: np.ndarray):
    if x.dtype == 'complex128' or x.dtype == 'complex64':
        x_abs = np.abs(x)
        eps = np.finfo(x.dtype).eps
        x_abs = np.maximum(x_abs, eps)
        out = x / x_abs
    else:
        out = np.sign(x)
    return out


class TestSgnError(unittest.TestCase):
    def test_errors_dynamic(self):
        # The input dtype of sgn must be float16, float32, float64,complex64,complex128.
        input2 = paddle.to_tensor(
            np.random.randint(-10, 10, size=[12, 20]).astype('int32')
        )
        input3 = paddle.to_tensor(
            np.random.randint(-10, 10, size=[12, 20]).astype('int64')
        )

        self.assertRaises(TypeError, paddle.sgn, input2)
        self.assertRaises(TypeError, paddle.sgn, input3)

    def test_errors_static_and_pir(self):
        paddle.enable_static()
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()

        with paddle.static.program_guard(main_program, startup_program):
            # The input dtype of sgn must be float16, float32, float64,complex64,complex128.
            input2 = paddle.to_tensor(
                np.random.randint(-10, 10, size=[12, 20]).astype('int32')
            )
            input3 = paddle.to_tensor(
                np.random.randint(-10, 10, size=[12, 20]).astype('int64')
            )

            self.assertRaises(TypeError, paddle.sgn, input2)
            self.assertRaises(TypeError, paddle.sgn, input3)
        paddle.disable_static()


class TestSignAPI(unittest.TestCase):
    def test_complex_dynamic(self):
        for dtype in ['complex64', 'complex128']:
            np_x = np.array(
                [[3 + 4j, 7 - 24j, 0, 1 + 2j], [6 + 8j, 3, 0, -2]], dtype=dtype
            )
            x = paddle.to_tensor(np_x)
            z = paddle.sgn(x)
            np_z = z.numpy()
            z_expected = np_sgn(np_x)
            np.testing.assert_allclose(np_z, z_expected, rtol=1e-05)

    def test_complex_static_and_pir(self):
        with static_guard():
            for dtype in ['complex64', 'complex128']:
                exe = paddle.static.Executor()

                train_program = paddle.static.Program()
                startup_program = paddle.static.Program()
                with paddle.static.program_guard(
                    train_program, startup_program
                ):
                    x = paddle.static.data(name='X', shape=[2, 4], dtype=dtype)
                    z = paddle.sgn(x)

                # Run the startup program once and only once.
                # Not need to optimize/compile the startup program.
                exe.run(startup_program)

                # Run the main program directly without compile.
                x = np.array(
                    [[3 + 4j, 7 - 24j, 0, 1 + 2j], [6 + 8j, 3, 0, -2]],
                    dtype=dtype,
                )
                (z,) = exe.run(train_program, feed={"X": x}, fetch_list=[z])
                z_expected = np_sgn(x)
                np.testing.assert_allclose(z, z_expected, rtol=1e-05)

    def test_float_dynamic(self):
        dtype_list = ['float32', 'float64']
        if paddle.is_compiled_with_cuda():
            dtype_list.append('float16')
        for dtype in dtype_list:
            np_x = np.random.randint(-10, 10, size=[12, 20, 2]).astype(dtype)
            x = paddle.to_tensor(np_x)
            z = paddle.sgn(x)
            np_z = z.numpy()
            z_expected = np_sgn(np_x)
            np.testing.assert_allclose(np_z, z_expected, rtol=1e-05)

    def test_float_static_and_pir(self):
        dtype_list = ['float32', 'float64']
        if paddle.is_compiled_with_cuda():
            dtype_list.append('float16')
        with static_guard():
            for dtype in dtype_list:
                exe = paddle.static.Executor()

                train_program = paddle.static.Program()
                startup_program = paddle.static.Program()
                with paddle.static.program_guard(
                    train_program, startup_program
                ):
                    np_x = np.random.randint(-10, 10, size=[12, 20, 2]).astype(
                        dtype
                    )
                    x = paddle.static.data(
                        name='X', shape=[12, 20, 2], dtype=dtype
                    )
                    z = paddle.sgn(x)

                # Run the startup program once and only once.
                # Not need to optimize/compile the startup program.
                exe.run(startup_program)

                # Run the main program directly without compile.
                (z,) = exe.run(train_program, feed={"X": np_x}, fetch_list=[z])
                z_expected = np_sgn(np_x)
                np.testing.assert_allclose(z, z_expected, rtol=1e-05)


if __name__ == "__main__":
    unittest.main()
