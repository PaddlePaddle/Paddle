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
import paddle.nn.functional as F


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


class TestDistTensorForDygraphAPI(unittest.TestCase):
    def check_tensor_eq(self, a, b):
        np1 = a.numpy()
        np2 = b.numpy()
        np.testing.assert_allclose(np1, np2, rtol=1e-05)

    def create_local_and_dist_tensor_pair(self, np_array):
        local_t = paddle.to_tensor(np_array, dtype='float32')

        mesh = dist.ProcessMesh([0], dim_names=["x"])
        dist_attr = dist.DistAttr(mesh=mesh, sharding_specs=[None, None])
        dist_t = dist.shard_tensor(np_array, dist_attr=dist_attr)

        local_t.stop_gradient = False
        dist_t.stop_gradient = False

        return local_t, dist_t

    def test_relu_api_for_dist_tensor(self):
        x = np.random.random(size=[4, 4]).astype("float32")
        local_in, dist_in = self.create_local_and_dist_tensor_pair(x)
        local_out = F.relu(local_in)
        dist_out = F.relu(dist_in)
        self.check_tensor_eq(local_out, dist_out)

        # test backward
        local_out.backward()
        dist_out.backward()
        self.check_tensor_eq(local_in.grad, dist_in.grad)

    # input: std::vector<phi::Tensor>, output: phi::Tensor
    def test_concat_for_dist_tensor(self):
        x1 = np.random.random(size=[4, 4]).astype("float32")
        x2 = np.random.random(size=[4, 4]).astype("float32")
        x3 = np.random.random(size=[4, 4]).astype("float32")
        local_in1, dist_in1 = self.create_local_and_dist_tensor_pair(x1)
        local_in2, dist_in2 = self.create_local_and_dist_tensor_pair(x2)
        local_in3, dist_in3 = self.create_local_and_dist_tensor_pair(x3)
        local_out = paddle.concat([local_in1, local_in2, local_in3])
        dist_out = paddle.concat([dist_in1, dist_in2, dist_in3])
        self.check_tensor_eq(local_out, dist_out)
        local_out.backward()
        dist_out.backward()
        self.check_tensor_eq(local_in1.grad, dist_in1.grad)
        self.check_tensor_eq(local_in2.grad, dist_in2.grad)
        self.check_tensor_eq(local_in3.grad, dist_in3.grad)

    # input: std::vector<phi::Tensor>, output: std::vector<phi::Tensor>
    def test_broadcast_tensors_for_dist_tensor(self):
        x1 = np.random.random(size=[4, 4]).astype("float32")
        x2 = np.random.random(size=[4, 4]).astype("float32")
        local_in1, dist_in1 = self.create_local_and_dist_tensor_pair(x1)
        local_in2, dist_in2 = self.create_local_and_dist_tensor_pair(x2)

        local_out1, local_out2 = paddle.broadcast_tensors(
            [local_in1, local_in2]
        )
        dist_out1, dist_out2 = paddle.broadcast_tensors([dist_in1, dist_in2])
        self.check_tensor_eq(local_out1, dist_out1)
        self.check_tensor_eq(local_out2, dist_out2)

        local_out = local_out1 + local_out2
        dist_out = dist_out1 + dist_out2

        local_out.backward()
        dist_out.backward()
        self.check_tensor_eq(local_in1.grad, dist_in1.grad)
        self.check_tensor_eq(local_in2.grad, dist_in2.grad)

    # input: phi::Tensor, output: std::vector<phi::Tensor>
    def test_unbind_api_for_dist_tensor(self):
        x = np.random.random(size=[2, 8]).astype("float32")
        local_in, dist_in = self.create_local_and_dist_tensor_pair(x)
        local_out1, local_out2 = paddle.unbind(local_in, axis=0)
        dist_out1, dist_out2 = paddle.unbind(dist_in, axis=0)
        self.check_tensor_eq(local_out1, dist_out1)
        self.check_tensor_eq(local_out2, dist_out2)

        local_out = local_out1 + local_out2
        dist_out = dist_out1 + dist_out2

        local_out.backward()
        dist_out.backward()
        self.check_tensor_eq(local_in.grad, dist_in.grad)

    def test_matmul_api_for_dist_tensor(self):
        x = np.random.random(size=[4, 4]).astype("float32")
        y = np.random.random(size=[4, 4]).astype("float32")
        local_x, dist_x = self.create_local_and_dist_tensor_pair(x)
        local_y, dist_y = self.create_local_and_dist_tensor_pair(y)
        local_out = paddle.matmul(local_x, local_y)
        dist_out = paddle.matmul(dist_x, dist_y)
        self.check_tensor_eq(local_out, dist_out)

        # test backward
        local_out.backward()
        dist_out.backward()
        self.check_tensor_eq(local_x.grad, dist_x.grad)
        self.check_tensor_eq(local_y.grad, dist_y.grad)


if __name__ == "__main__":
    unittest.main()
