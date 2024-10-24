#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import scipy

import paddle
from paddle import base


class TestBlockDiagError(unittest.TestCase):
    def test_errors(self):
        def test_type_error():
            A = np.array([[1, 2], [3, 4]])
            B = np.array([[5, 6], [7, 8]])
            C = np.array([[9, 10], [11, 12]])
            with paddle.static.program_guard(base.Program()):
                out = paddle.block_diag([A, B, C])

        self.assertRaises(TypeError, test_type_error)

        def test_dime_error():
            A = paddle.to_tensor([[[1, 2], [3, 4]]])
            B = paddle.to_tensor([[[5, 6], [7, 8]]])
            C = paddle.to_tensor([[[9, 10], [11, 12]]])
            with paddle.static.program_guard(base.Program()):
                out = paddle.block_diag([A, B, C])

        self.assertRaises(ValueError, test_dime_error)


class TestBlockDiag(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.type_list = ['int32', 'int64', 'float32', 'float64']
        self.place = [('cpu', paddle.CPUPlace())] + (
            [('gpu', paddle.CUDAPlace(0))]
            if paddle.is_compiled_with_cuda()
            else []
        )

    def test_dygraph(self):
        paddle.disable_static()
        for device, place in self.place:
            paddle.set_device(device)
            for i in self.type_list:
                A = np.random.randn(2, 3).astype(i)
                B = np.random.randn(2).astype(i)
                C = np.random.randn(4, 1).astype(i)
                s_out = scipy.linalg.block_diag(A, B, C)

                A_tensor = paddle.to_tensor(A)
                B_tensor = paddle.to_tensor(B)
                C_tensor = paddle.to_tensor(C)
                out = paddle.block_diag([A_tensor, B_tensor, C_tensor])
                np.testing.assert_allclose(out.numpy(), s_out)

    def test_static(self):
        paddle.enable_static()
        for device, place in self.place:
            paddle.set_device(device)
            for i in self.type_list:
                A = np.random.randn(2, 3).astype(i)
                B = np.random.randn(2).astype(i)
                C = np.random.randn(4, 1).astype(i)
                s_out = scipy.linalg.block_diag(A, B, C)

                with paddle.static.program_guard(paddle.static.Program()):
                    A_tensor = paddle.static.data('A', [2, 3], i)
                    B_tensor = paddle.static.data('B', [2], i)
                    C_tensor = paddle.static.data('C', [4, 1], i)
                    out = paddle.block_diag([A_tensor, B_tensor, C_tensor])
                    exe = paddle.static.Executor(place)
                    res = exe.run(
                        feed={'A': A, 'B': B, 'C': C},
                        fetch_list=[out],
                    )
                    np.testing.assert_allclose(res[0], s_out)


if __name__ == '__main__':
    unittest.main()
