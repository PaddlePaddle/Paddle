# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import numpy as np
import scipy
import scipy.sparse as sp
import unittest
import os
import re

paddle.set_default_dtype('float64')


def get_cuda_version():
    result = os.popen("nvcc --version").read()
    regex = r'release (\S+),'
    match = re.search(regex, result)
    if match:
        num = str(match.group(1))
        integer, decimal = num.split('.')
        return int(integer) * 1000 + int(float(decimal) * 10)
    else:
        return -1


class TestReshape(unittest.TestCase):
    # x: sparse, out: sparse
    def check_result(self, x_shape, out_shape, format):
        if len(x_shape) == 3:
            mask = paddle.randint(0, 2, [x_shape[-2], x_shape[-1]])
        else:
            mask = paddle.randint(0, 2, x_shape)
        origin_x = paddle.rand(x_shape) * mask

        dense_x = origin_x.detach()
        dense_x.stop_gradient = False
        dense_out = paddle.reshape(dense_x, out_shape)

        if format == "coo":
            sp_x = origin_x.detach().to_sparse_coo(len(x_shape))
        else:
            sp_x = origin_x.detach().to_sparse_csr()
        sp_x.stop_gradient = False
        sp_out = paddle.incubate.sparse.reshape(sp_x, out_shape)

        np.testing.assert_allclose(sp_out.numpy(),
                                   dense_out.numpy(),
                                   rtol=1e-05)
        if get_cuda_version() >= 11030:
            dense_out.backward()
            sp_out.backward()
            np.testing.assert_allclose(sp_x.grad.to_dense().numpy(),
                                       (dense_x.grad * mask).numpy(),
                                       rtol=1e-05)

    @unittest.skipIf(not paddle.is_compiled_with_cuda()
                     or get_cuda_version() < 11000, "only support cuda>=11.0")
    def test_reshape_case1(self):
        self.check_result([16, 12], [8, 24], 'coo')
        self.check_result([16, 12], [8, 24], 'csr')

    @unittest.skipIf(not paddle.is_compiled_with_cuda()
                     or get_cuda_version() < 11070, "only support cuda>=11.7")
    def test_reshape_case2(self):
        self.check_result([8, 16, 12], [4, 2, 12, 16], 'coo')
        self.check_result([8, 16, 12], [4, 2, 12, 16], 'csr')

    @unittest.skipIf(not paddle.is_compiled_with_cuda()
                     or get_cuda_version() < 11070, "only support cuda>=11.7")
    def test_reshape_case3(self):
        self.check_result([12], [4, 3], 'coo')
        self.check_result([12], [4, 3], 'csr')

    @unittest.skipIf(not paddle.is_compiled_with_cuda()
                     or get_cuda_version() < 11070, "only support cuda>=11.7")
    def test_reshape_case3(self):
        self.check_result([4, 2, 12, 16], [8, 16, 12], 'coo')
        self.check_result([4, 2, 12, 16], [8, 16, 12], 'csr')


if __name__ == "__main__":
    unittest.main()
