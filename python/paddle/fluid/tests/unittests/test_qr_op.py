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

from __future__ import print_function

import unittest
import itertools
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.core as core


class TestQrAPI(unittest.TestCase):
    def test_dygraph(self):
        paddle.disable_static()

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
                    self.assertTrue(np.allclose(r, np_r, atol=1e-5))
                else:
                    q, r = paddle.linalg.qr(x, mode=mode)
                    self.assertTrue(np.allclose(q, np_q, atol=1e-5))
                    self.assertTrue(np.allclose(r, np_r, atol=1e-5))

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
        for tensor_shape, mode, dtype in itertools.product(tensor_shapes, modes,
                                                           dtypes):
            run_qr_dygraph(tensor_shape, mode, dtype)

    def test_static(self):
        paddle.enable_static()

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
                    x = paddle.fluid.data(
                        name="input", shape=shape, dtype=dtype)
                    if mode == "r":
                        r = paddle.linalg.qr(x, mode=mode)
                        exe = fluid.Executor(place)
                        fetches = exe.run(fluid.default_main_program(),
                                          feed={"input": a},
                                          fetch_list=[r])
                        self.assertTrue(
                            np.allclose(
                                fetches[0], np_r, atol=1e-5))
                    else:
                        q, r = paddle.linalg.qr(x, mode=mode)
                        exe = fluid.Executor(place)
                        fetches = exe.run(fluid.default_main_program(),
                                          feed={"input": a},
                                          fetch_list=[q, r])
                        self.assertTrue(
                            np.allclose(
                                fetches[0], np_q, atol=1e-5))
                        self.assertTrue(
                            np.allclose(
                                fetches[1], np_r, atol=1e-5))

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
        for tensor_shape, mode, dtype in itertools.product(tensor_shapes, modes,
                                                           dtypes):
            run_qr_static(tensor_shape, mode, dtype)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
