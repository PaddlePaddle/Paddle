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


class TestSparseIsnan(unittest.TestCase):
    """
    Test the API paddle.sparse.isnan on some sparse tensors.
    x: sparse tensor, out: sparse tensor
    """

    def to_sparse(self, x, format):
        if format == 'coo':
            return x.detach().to_sparse_coo(sparse_dim=x.ndim)
        elif format == 'csr':
            return x.detach().to_sparse_csr()

    def check_result(self, x_shape, format, data_type="float32"):
        raw_inp = np.random.randint(-100, 100, x_shape)
        mask = np.random.randint(0, 2, x_shape)
        inp_x = (raw_inp * mask).astype(data_type)
        inp_x[inp_x > 0] = np.nan
        np_out = np.isnan(inp_x[inp_x != 0])

        dense_x = paddle.to_tensor(inp_x)
        sp_x = self.to_sparse(dense_x, format)
        sp_out = paddle.sparse.isnan(sp_x)
        sp_out_values = sp_out.values().numpy()

        np.testing.assert_allclose(np_out, sp_out_values, rtol=1e-05)

    def test_isnan_shape(self):
        self.check_result([20], 'coo')

        self.check_result([4, 5], 'coo')
        self.check_result([4, 5], 'csr')

        self.check_result([8, 16, 32], 'coo')
        self.check_result([8, 16, 32], 'csr')

    def test_isnan_dtype(self):
        self.check_result([4, 5], 'coo', "float32")
        self.check_result([4, 5], 'csr', "float32")

        self.check_result([8, 16, 32], 'coo', "float64")
        self.check_result([8, 16, 32], 'csr', "float64")


if __name__ == "__main__":
    unittest.main()
