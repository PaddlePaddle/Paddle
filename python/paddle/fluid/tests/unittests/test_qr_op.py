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
import itertools
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.core as core
from op_test import OpTest


class TestQrOp(OpTest):

    def setUp(self):
        paddle.enable_static()
        self.python_api = paddle.linalg.qr
        np.random.seed(7)
        self.op_type = "qr"
        a, q, r = self.get_input_and_output()
        self.inputs = {"X": a}
        self.attrs = {"mode": self.get_mode()}
        self.outputs = {"Q": q, "R": r}

    def get_dtype(self):
        return "float64"

    def get_mode(self):
        return "reduced"

    def get_shape(self):
        return (11, 11)

    def get_input_and_output(self):
        dtype = self.get_dtype()
        shape = self.get_shape()
        mode = self.get_mode()
        assert mode != "r", "Cannot be backward in r mode."
        a = np.random.rand(*shape).astype(dtype)
        m = a.shape[-2]
        n = a.shape[-1]
        min_mn = min(m, n)
        if mode == "reduced":
            k = min_mn
        else:
            k = m
        q_shape = list(a.shape[:-2])
        q_shape.extend([m, k])
        r_shape = list(a.shape[:-2])
        r_shape.extend([k, n])
        q = np.zeros(q_shape).astype(dtype)
        r = np.zeros(r_shape).astype(dtype)
        batch_size = a.size // (a.shape[-1] * a.shape[-2])
        for i in range(batch_size):
            coord = np.unravel_index(i, a.shape[:-2])
            tmp_q, tmp_r = np.linalg.qr(a[coord], mode=mode)
            q[coord] = tmp_q
            r[coord] = tmp_r
        return a, q, r

    def test_check_output(self):
        self.check_output(check_eager=True)

    def test_check_grad_normal(self):
        self.check_grad(['X'], ['Q', 'R'],
                        check_eager=True,
                        numeric_grad_delta=1e-5,
                        max_relative_error=1e-6)


class TestQrOpCase1(TestQrOp):

    def get_shape(self):
        return (10, 12)


class TestQrOpCase2(TestQrOp):

    def get_shape(self):
        return (16, 15)


class TestQrOpCase3(TestQrOp):

    def get_shape(self):
        return (2, 12, 16)


class TestQrOpCase4(TestQrOp):

    def get_shape(self):
        return (3, 16, 15)


class TestQrOpCase5(TestQrOp):

    def get_mode(self):
        return "complete"

    def get_shape(self):
        return (10, 12)


class TestQrOpCase6(TestQrOp):

    def get_mode(self):
        return "complete"

    def get_shape(self):
        return (2, 10, 12)


class TestQrAPI(unittest.TestCase):

    def test_dygraph(self):
        paddle.disable_static()
        np.random.seed(7)

        def run_qr_dygraph(shape, mode, dtype):
            if dtype == "float32":
                np_dtype = np.float32
            elif dtype == "float64":
                np_dtype = np.float64
            a = np.random.rand(*shape).astype(np_dtype)
            m = a.shape[-2]
            n = a.shape[-1]
            min_mn = min(m, n)
            if mode == "reduced" or mode == "r":
                k = min_mn
            else:
                k = m
            np_q_shape = list(a.shape[:-2])
            np_q_shape.extend([m, k])
            np_r_shape = list(a.shape[:-2])
            np_r_shape.extend([k, n])
            np_q = np.zeros(np_q_shape).astype(np_dtype)
            np_r = np.zeros(np_r_shape).astype(np_dtype)
            places = []
            places = [fluid.CPUPlace()]
            if core.is_compiled_with_cuda():
                places.append(fluid.CUDAPlace(0))
            for place in places:
                batch_size = a.size // (a.shape[-1] * a.shape[-2])
                for i in range(batch_size):
                    coord = np.unravel_index(i, a.shape[:-2])
                    if mode == "r":
                        tmp_r = np.linalg.qr(a[coord], mode=mode)
                        np_r[coord] = tmp_r
                    else:
                        tmp_q, tmp_r = np.linalg.qr(a[coord], mode=mode)
                        np_q[coord] = tmp_q
                        np_r[coord] = tmp_r

                x = paddle.to_tensor(a, dtype=dtype)
                if mode == "r":
                    r = paddle.linalg.qr(x, mode=mode)
                    np.testing.assert_allclose(r, np_r, rtol=1e-05, atol=1e-05)
                else:
                    q, r = paddle.linalg.qr(x, mode=mode)
                    np.testing.assert_allclose(q, np_q, rtol=1e-05, atol=1e-05)
                    np.testing.assert_allclose(r, np_r, rtol=1e-05, atol=1e-05)

        tensor_shapes = [
            (3, 5),
            (5, 5),
            (5, 3),  # 2-dim Tensors
            (2, 3, 5),
            (3, 5, 5),
            (4, 5, 3),  # 3-dim Tensors
            (2, 5, 3, 5),
            (3, 5, 5, 5),
            (4, 5, 5, 3)  # 4-dim Tensors
        ]
        modes = ["reduced", "complete", "r"]
        dtypes = ["float32", "float64"]
        for tensor_shape, mode, dtype in itertools.product(
                tensor_shapes, modes, dtypes):
            run_qr_dygraph(tensor_shape, mode, dtype)

    def test_static(self):
        paddle.enable_static()
        np.random.seed(7)

        def run_qr_static(shape, mode, dtype):
            if dtype == "float32":
                np_dtype = np.float32
            elif dtype == "float64":
                np_dtype = np.float64
            a = np.random.rand(*shape).astype(np_dtype)
            m = a.shape[-2]
            n = a.shape[-1]
            min_mn = min(m, n)
            if mode == "reduced" or mode == "r":
                k = min_mn
            else:
                k = m
            np_q_shape = list(a.shape[:-2])
            np_q_shape.extend([m, k])
            np_r_shape = list(a.shape[:-2])
            np_r_shape.extend([k, n])
            np_q = np.zeros(np_q_shape).astype(np_dtype)
            np_r = np.zeros(np_r_shape).astype(np_dtype)
            places = []
            places = [fluid.CPUPlace()]
            if core.is_compiled_with_cuda():
                places.append(fluid.CUDAPlace(0))
            for place in places:
                with fluid.program_guard(fluid.Program(), fluid.Program()):
                    batch_size = a.size // (a.shape[-1] * a.shape[-2])
                    for i in range(batch_size):
                        coord = np.unravel_index(i, a.shape[:-2])
                        if mode == "r":
                            tmp_r = np.linalg.qr(a[coord], mode=mode)
                            np_r[coord] = tmp_r
                        else:
                            tmp_q, tmp_r = np.linalg.qr(a[coord], mode=mode)
                            np_q[coord] = tmp_q
                            np_r[coord] = tmp_r
                    x = paddle.fluid.data(name="input",
                                          shape=shape,
                                          dtype=dtype)
                    if mode == "r":
                        r = paddle.linalg.qr(x, mode=mode)
                        exe = fluid.Executor(place)
                        fetches = exe.run(fluid.default_main_program(),
                                          feed={"input": a},
                                          fetch_list=[r])
                        np.testing.assert_allclose(fetches[0],
                                                   np_r,
                                                   rtol=1e-05,
                                                   atol=1e-05)
                    else:
                        q, r = paddle.linalg.qr(x, mode=mode)
                        exe = fluid.Executor(place)
                        fetches = exe.run(fluid.default_main_program(),
                                          feed={"input": a},
                                          fetch_list=[q, r])
                        np.testing.assert_allclose(fetches[0],
                                                   np_q,
                                                   rtol=1e-05,
                                                   atol=1e-05)
                        np.testing.assert_allclose(fetches[1],
                                                   np_r,
                                                   rtol=1e-05,
                                                   atol=1e-05)

        tensor_shapes = [
            (3, 5),
            (5, 5),
            (5, 3),  # 2-dim Tensors
            (2, 3, 5),
            (3, 5, 5),
            (4, 5, 3),  # 3-dim Tensors
            (2, 5, 3, 5),
            (3, 5, 5, 5),
            (4, 5, 5, 3)  # 4-dim Tensors
        ]
        modes = ["reduced", "complete", "r"]
        dtypes = ["float32", "float64"]
        for tensor_shape, mode, dtype in itertools.product(
                tensor_shapes, modes, dtypes):
            run_qr_static(tensor_shape, mode, dtype)


if __name__ == "__main__":
    unittest.main()
