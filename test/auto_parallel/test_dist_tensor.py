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

paddle.disable_signal_handler()


class TestDistTensorv1(unittest.TestCase):
    def check_tensor_eq(self, a, b):
        np1 = a.numpy()
        np2 = b.numpy()
        np.testing.assert_allclose(np1, np2, rtol=1e-05)

    def test_dist_tensor_basic(self):
        shape = [2, 2]

        def init_local_and_dist(value):
            dist_tensor = paddle.Tensor(
                paddle.float32,
                shape,
                "dist_tensor1",
                paddle.framework.dtype.dtype.DIST_TENSOR,
                True,
            )
            self.assertTrue(dist_tensor.local_tensor() is not None)
            local = paddle.full(shape, value, paddle.float32)
            t = dist_tensor.local_tensor()
            paddle.assign(local, t)
            local.stop_gradient = False
            dist_tensor.stop_gradient = False
            return local, dist_tensor

        local1, dist1 = init_local_and_dist(1.0)
        local2, dist2 = init_local_and_dist(2.0)
        local3, dist3 = init_local_and_dist(3.0)

        # test operations
        local_result = (local1 + local2) * local3
        dist_result = (dist1 + dist2) * dist3
        self.check_tensor_eq(local_result, dist_result.local_tensor())

        # test backward
        local_result.backward()
        dist_result.backward()
        self.check_tensor_eq(local1.grad, dist1.grad.local_tensor())
        self.check_tensor_eq(local2.grad, dist2.grad.local_tensor())
        self.check_tensor_eq(local3.grad, dist3.grad.local_tensor())


class TestDistTensor(unittest.TestCase):
    def test_dist_tensor_creation(self):
        shape = [10, 5]
        dist_attr = paddle.fluid.core.TensorDistAttr()

        # create dist tensor using numpy
        dist_tensor_with_numpy = paddle.Tensor(
            np.ones(shape, dtype=np.float32), dist_attr=dist_attr
        )

        # create dist tensor using tensor
        dist_tensor_with_tensor = paddle.Tensor(
            paddle.ones(shape), dist_attr=dist_attr
        )

        # create normal tensor
        tensor = paddle.ones(shape)

        # test dist tensor properties
        self.assertEqual(dist_tensor_with_numpy.shape, shape)
        self.assertEqual(dist_tensor_with_tensor.shape, shape)
        self.assertEqual(dist_tensor_with_numpy.is_dist(), True)
        self.assertEqual(dist_tensor_with_tensor.is_dist(), True)
        self.assertEqual(tensor.is_dist(), False)
        self.assertEqual(
            str(dist_tensor_with_numpy), str(dist_tensor_with_tensor)
        )
        self.assertEqual(dist_tensor_with_numpy.dist_attr, dist_attr)
        self.assertEqual(dist_tensor_with_tensor.dist_attr, dist_attr)


if __name__ == "__main__":
    unittest.main()
