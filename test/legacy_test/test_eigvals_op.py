#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest

import paddle
from paddle.base import core

np.set_printoptions(threshold=np.inf)


def np_eigvals(a):
    res = np.linalg.eigvals(a)
    if a.dtype == np.float32 or a.dtype == np.complex64:
        res = res.astype(np.complex64)
    else:
        res = res.astype(np.complex128)

    return res


class TestEigvalsOp(OpTest):
    def setUp(self):
        np.random.seed(0)
        paddle.enable_static()
        self.python_api = paddle.linalg.eigvals
        self.op_type = "eigvals"
        self.set_dtype()
        self.set_input_dims()
        self.set_input_data()

        np_output = np_eigvals(self.input_data)

        self.inputs = {'X': self.input_data}
        self.outputs = {'Out': np_output}

    def set_dtype(self):
        self.dtype = np.float32

    def set_input_dims(self):
        self.input_dims = (5, 5)

    def set_input_data(self):
        if self.dtype == np.float32 or self.dtype == np.float64:
            self.input_data = np.random.random(self.input_dims).astype(
                self.dtype
            )
        else:
            self.input_data = (
                np.random.random(self.input_dims)
                + np.random.random(self.input_dims) * 1j
            ).astype(self.dtype)

    def test_check_output(self):
        self.__class__.no_need_check_grad = True
        self.check_output_with_place_customized(
            checker=self.verify_output, place=core.CPUPlace(), check_pir=True
        )

    def verify_output(self, outs):
        actual_outs = np.sort(np.array(outs[0]))
        expect_outs = np.sort(np.array(self.outputs['Out']))
        self.assertTrue(
            actual_outs.shape == expect_outs.shape,
            "Output shape has diff.\n"
            "Expect shape "
            + str(expect_outs.shape)
            + "\n"
            + "But Got"
            + str(actual_outs.shape)
            + " in class "
            + self.__class__.__name__,
        )

        n_dim = actual_outs.shape[-1]
        for actual_row, expect_row in zip(
            actual_outs.reshape((-1, n_dim)), expect_outs.reshape((-1, n_dim))
        ):
            is_mapped_index = np.zeros((n_dim,))
            for i in range(n_dim):
                is_mapped = False
                for j in range(n_dim):
                    if is_mapped_index[j] == 0 and np.isclose(
                        np.array(actual_row[i]),
                        np.array(expect_row[j]),
                        atol=1e-5,
                    ):
                        is_mapped_index[j] = True
                        is_mapped = True
                        break
                self.assertTrue(
                    is_mapped,
                    "Output has diff in class "
                    + self.__class__.__name__
                    + "\nExpect "
                    + str(expect_outs)
                    + "\n"
                    + "But Got"
                    + str(actual_outs)
                    + "\nThe data "
                    + str(actual_row[i])
                    + " in "
                    + str(actual_row)
                    + " mismatch.",
                )


class TestEigvalsOpFloat64(TestEigvalsOp):
    def set_dtype(self):
        self.dtype = np.float64


class TestEigvalsOpComplex64(TestEigvalsOp):
    def set_dtype(self):
        self.dtype = np.complex64


class TestEigvalsOpComplex128(TestEigvalsOp):
    def set_dtype(self):
        self.dtype = np.complex128


class TestEigvalsOpLargeScare(TestEigvalsOp):
    def set_input_dims(self):
        self.input_dims = (128, 128)


class TestEigvalsOpLargeScareFloat64(TestEigvalsOpLargeScare):
    def set_dtype(self):
        self.dtype = np.float64


class TestEigvalsOpLargeScareComplex64(TestEigvalsOpLargeScare):
    def set_dtype(self):
        self.dtype = np.complex64


class TestEigvalsOpLargeScareComplex128(TestEigvalsOpLargeScare):
    def set_dtype(self):
        self.dtype = np.complex128


class TestEigvalsOpBatch1(TestEigvalsOp):
    def set_input_dims(self):
        self.input_dims = (1, 2, 3, 4, 4)


class TestEigvalsOpBatch2(TestEigvalsOp):
    def set_input_dims(self):
        self.input_dims = (3, 1, 4, 5, 5)


class TestEigvalsOpBatch3(TestEigvalsOp):
    def set_input_dims(self):
        self.input_dims = (6, 2, 9, 6, 6)


class TestEigvalsAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

        self.small_dims = [6, 6]
        self.large_dims = [128, 128]
        self.batch_dims = [6, 9, 2, 2]

        self.set_dtype()

        self.input_dims = self.small_dims
        self.set_input_data()
        self.small_input = np.copy(self.input_data)

        self.input_dims = self.large_dims
        self.set_input_data()
        self.large_input = np.copy(self.input_data)

        self.input_dims = self.batch_dims
        self.set_input_data()
        self.batch_input = np.copy(self.input_data)

    def set_dtype(self):
        self.dtype = np.float32

    def set_input_data(self):
        if self.dtype == np.float32 or self.dtype == np.float64:
            self.input_data = np.random.random(self.input_dims).astype(
                self.dtype
            )
        else:
            self.input_data = (
                np.random.random(self.input_dims)
                + np.random.random(self.input_dims) * 1j
            ).astype(self.dtype)

    def verify_output(self, actual_outs, expect_outs):
        actual_outs = np.array(actual_outs)
        expect_outs = np.array(expect_outs)
        self.assertTrue(
            actual_outs.shape == expect_outs.shape,
            "Output shape has diff."
            "\nExpect shape "
            + str(expect_outs.shape)
            + "\n"
            + "But Got"
            + str(actual_outs.shape)
            + " in class "
            + self.__class__.__name__,
        )

        n_dim = actual_outs.shape[-1]
        for actual_row, expect_row in zip(
            actual_outs.reshape((-1, n_dim)), expect_outs.reshape((-1, n_dim))
        ):
            is_mapped_index = np.zeros((n_dim,))
            for i in range(n_dim):
                is_mapped = False
                for j in range(n_dim):
                    if is_mapped_index[j] == 0 and np.isclose(
                        np.array(actual_row[i]),
                        np.array(expect_row[j]),
                        atol=1e-5,
                    ):
                        is_mapped_index[j] = True
                        is_mapped = True
                        break
                self.assertTrue(
                    is_mapped,
                    "Output has diff in class "
                    + self.__class__.__name__
                    + "\nExpect "
                    + str(expect_outs)
                    + "\n"
                    + "But Got"
                    + str(actual_outs)
                    + "\nThe data "
                    + str(actual_row[i])
                    + " in "
                    + str(actual_row)
                    + " mismatch.",
                )

    def run_dygraph(self, place):
        paddle.disable_static()
        paddle.set_device("cpu")
        small_input_tensor = paddle.to_tensor(self.small_input, place=place)
        large_input_tensor = paddle.to_tensor(self.large_input, place=place)
        batch_input_tensor = paddle.to_tensor(self.batch_input, place=place)

        paddle_outs = paddle.linalg.eigvals(small_input_tensor, name='small_x')
        np_outs = np_eigvals(self.small_input)
        self.verify_output(paddle_outs, np_outs)

        paddle_outs = paddle.linalg.eigvals(large_input_tensor, name='large_x')
        np_outs = np_eigvals(self.large_input)
        self.verify_output(paddle_outs, np_outs)

        paddle_outs = paddle.linalg.eigvals(batch_input_tensor, name='small_x')
        np_outs = np_eigvals(self.batch_input)
        self.verify_output(paddle_outs, np_outs)

    def run_static(self, place):
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            small_input_tensor = paddle.static.data(
                name='small_x', shape=self.small_dims, dtype=self.dtype
            )
            large_input_tensor = paddle.static.data(
                name='large_x', shape=self.large_dims, dtype=self.dtype
            )
            batch_input_tensor = paddle.static.data(
                name='batch_x', shape=self.batch_dims, dtype=self.dtype
            )

            small_outs = paddle.linalg.eigvals(
                small_input_tensor, name='small_x'
            )
            large_outs = paddle.linalg.eigvals(
                large_input_tensor, name='large_x'
            )
            batch_outs = paddle.linalg.eigvals(
                batch_input_tensor, name='batch_x'
            )

            exe = paddle.static.Executor(place)

            paddle_outs = exe.run(
                feed={
                    "small_x": self.small_input,
                    "large_x": self.large_input,
                    "batch_x": self.batch_input,
                },
                fetch_list=[small_outs, large_outs, batch_outs],
            )

            np_outs = np_eigvals(self.small_input)
            self.verify_output(paddle_outs[0], np_outs)

            np_outs = np_eigvals(self.large_input)
            self.verify_output(paddle_outs[1], np_outs)

            np_outs = np_eigvals(self.batch_input)
            self.verify_output(paddle_outs[2], np_outs)

    def test_cases(self):
        places = [core.CPUPlace()]
        # if core.is_compiled_with_cuda():
        #    places.append(core.CUDAPlace(0))
        for place in places:
            self.run_dygraph(place)
            self.run_static(place)

    def test_error(self):
        paddle.disable_static()
        x = paddle.to_tensor([1])
        with self.assertRaises(ValueError):
            paddle.linalg.eigvals(x)

        self.input_dims = [1, 2, 3, 4]
        self.set_input_data()
        x = paddle.to_tensor(self.input_data)
        with self.assertRaises(ValueError):
            paddle.linalg.eigvals(x)


class TestEigvalsAPIFloat64(TestEigvalsAPI):
    def set_dtype(self):
        self.dtype = np.float64


class TestEigvalsAPIComplex64(TestEigvalsAPI):
    def set_dtype(self):
        self.dtype = np.complex64


class TestEigvalsAPIComplex128(TestEigvalsAPI):
    def set_dtype(self):
        self.dtype = np.complex128


if __name__ == "__main__":
    unittest.main()
