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
import paddle.distributed as dist


class TestDistTensor(unittest.TestCase):
    def test_dist_tensor_creation(self):
        shape = [10, 5]
        mesh = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=["x", "y"])
        dist_attr = dist.DistAttr(mesh=mesh, sharding_specs=[None, None])

        # create dist tensor using numpy
        dist_tensor_with_numpy = dist.shard_tensor(
            np.ones(shape, dtype=np.float32), dist_attr=dist_attr
        )

        # create dist tensor using tensor
        dist_tensor_with_tensor = dist.shard_tensor(
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


class TestDistTensorFromFn(unittest.TestCase):
    def run_dtensor_from_fn(self):
        # Create a dist_attr
        mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        dist_attr = dist.DistAttr(mesh=mesh, sharding_specs=[None])

        # Call the function dtensor_from_fn with dist_attr parameter
        result = dist.dtensor_from_fn(
            paddle.ones, dist_attr=dist_attr, shape=[16]
        )
        # Verify the result
        if paddle.in_dynamic_mode():
            dist_attr.dynamic_dims = []
            self.assertIsInstance(result, paddle.Tensor)
            self.assertEqual(result.shape, [16])
            self.assertEqual(result.dist_attr, dist_attr)
        else:
            dist_attr.dynamic_dims = [0]
            self.assertIsInstance(result, paddle.static.Variable)
            self.assertEqual(result.shape, (16,))
            self.assertEqual(result.dist_attr, dist_attr)

        result_zeros = dist.dtensor_from_fn(
            paddle.zeros, dist_attr=dist_attr, shape=[16]
        )
        if paddle.in_dynamic_mode():
            dist_attr.dynamic_dims = []
            self.assertIsInstance(result, paddle.Tensor)
            self.assertEqual(result.shape, [16])
            self.assertEqual(result.dist_attr, dist_attr)
        else:
            dist_attr.dynamic_dims = [0]
            self.assertIsInstance(result, paddle.static.Variable)
            self.assertEqual(result.shape, (16,))
            self.assertEqual(result.dist_attr, dist_attr)

        result_random = dist.dtensor_from_fn(
            paddle.rand, dist_attr=dist_attr, shape=[16]
        )
        if paddle.in_dynamic_mode():
            dist_attr.dynamic_dims = []
            self.assertIsInstance(result, paddle.Tensor)
            self.assertEqual(result.shape, [16])
            self.assertEqual(result.dist_attr, dist_attr)
        else:
            dist_attr.dynamic_dims = [0]
            self.assertIsInstance(result, paddle.static.Variable)
            self.assertEqual(result.shape, (16,))
            self.assertEqual(result.dist_attr, dist_attr)

        # Test with invalid sharding_specs length
        with self.assertRaises(AssertionError):
            invalid_dist_attr = dist.DistAttr(mesh=mesh, sharding_specs=['x'])
            dist.dtensor_from_fn(
                paddle.ones, dist_attr=invalid_dist_attr, shape=[2, 3]
            )

    def test_dynamic_mode(self):
        self.run_dtensor_from_fn()

    # Test exceptions when running in static mode
    def test_static_mode(self):
        paddle.enable_static()
        self.run_dtensor_from_fn()
        paddle.disable_static()


if __name__ == "__main__":
    unittest.main()
