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
import paddle
from paddle import sparse

class TestTensorSparseMask(unittest.TestCase):
    """
    Test paddle.Tensor.sparse_mask API (PyTorch migration compatibility)
    """

    def test_csr_sparse_mask(self):
        """Test CSR format sparse mask"""
        # 1. Construct CSR sparse tensor
        crows = [0, 2, 3, 5]
        cols = [1, 3, 2, 0, 1]
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        shape = [3, 4]
        
        mask = sparse.sparse_csr_tensor(crows, cols, values, shape, dtype='float32')
        
        # 2. Construct dense input
        paddle.seed(2024)  # 在测试内设置种子
        x = paddle.rand(shape, dtype='float32')
        x_np = x.numpy()

        # 3. Call new API (instance method style)
        out_method = x.sparse_mask(mask)

        # 4. Verify type
        self.assertIsInstance(out_method, paddle.sparse.SparseCsrTensor)
        
        # 5. Verify numerical logic
        expected_vals = np.array([
            x_np[0, 1], x_np[0, 3], 
            x_np[1, 2], 
            x_np[2, 0], x_np[2, 1]
        ])
        np.testing.assert_allclose(out_method.values.numpy(), expected_vals, rtol=1e-5)

    def test_coo_sparse_mask(self):
        """Test COO format sparse mask"""
        # 1. Construct COO sparse tensor
        indices = [[0, 1, 2], [1, 2, 0]]
        vals = [1.0, 2.0, 3.0]
        shape = [3, 3]
        
        mask = sparse.sparse_coo_tensor(indices, vals, shape, dtype='float32')
        paddle.seed(2024)  # 在测试内设置种子
        x = paddle.rand(shape, dtype='float32')
        x_np = x.numpy()

        # 2. Call new API
        out = x.sparse_mask(mask)

        # 3. Verify type
        self.assertIsInstance(out, paddle.sparse.SparseCooTensor)
        
        # 4. Verify numerical logic
        expected_vals = np.array([x_np[0, 1], x_np[1, 2], x_np[2, 0]])
        np.testing.assert_allclose(out.values.numpy(), expected_vals, rtol=1e-5)

    def test_name_parameter(self):
        """Test if name parameter works (for static graph tracing)"""
        shape = [2, 2]
        mask = sparse.sparse_coo_tensor([[0, 1], [0, 1]], [1.0, 1.0], shape)
        x = paddle.ones(shape)
        
        out = x.sparse_mask(mask, name="my_sparse_op")
        # In dynamic mode, name is mainly for identification, here we simply verify no error
        self.assertIsInstance(out, paddle.Tensor)

    def test_dtype_compatibility(self):
        """Test compatibility of different data types"""
        shape = [2, 2]
        
        # Test float64
        mask64 = sparse.sparse_coo_tensor(
            [[0, 1], [0, 1]], 
            [1.0, 1.0], 
            shape, 
            dtype='float64'
        )
        x64 = paddle.rand(shape, dtype='float64')
        out64 = x64.sparse_mask(mask64)
        self.assertEqual(out64.dtype, paddle.float64)

        # Test int32
        mask32 = sparse.sparse_coo_tensor(
            [[0, 1], [0, 1]], 
            [1.0, 1.0], 
            shape, 
            dtype='int32'
        )
        x32 = paddle.randint(0, 10, shape, dtype='int32')
        out32 = x32.sparse_mask(mask32)
        self.assertEqual(out32.dtype, paddle.int32)

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
