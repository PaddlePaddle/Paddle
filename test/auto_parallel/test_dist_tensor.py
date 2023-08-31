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
from paddle.nn import Linear


class TestDistTensor(unittest.TestCase):
    def test_dist_tensor_creation(self):
        shape = [10, 5]
        mesh = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=["x", "y"])
        dist_attr = dist.DistAttr(mesh=mesh, sharding_specs=['x', 'y'])

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
        # Create a distributed attribute
        mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        dist_attr = dist.DistAttr(mesh=mesh, sharding_specs=[None])

        # Call the function dtensor_from_fn with dist_attr parameter
        result = dist.dtensor_from_fn(
            paddle.ones, dist_attr=dist_attr, shape=[1]
        )

        # Verify the result
        if paddle.in_dynamic_mode():
            self.assertIsInstance(result, paddle.Tensor)
            self.assertEqual(result.shape, [1])
        else:
            self.assertIsInstance(result, paddle.fluid.framework.Variable)
            self.assertEqual(result.shape, (1,))

        # Test with generate_tensor_zeros()
        result_zeros = dist.dtensor_from_fn(
            paddle.zeros, dist_attr=dist_attr, shape=[1]
        )
        if paddle.in_dynamic_mode():
            self.assertIsInstance(result, paddle.Tensor)
            self.assertEqual(result.shape, [1])
        else:
            self.assertIsInstance(result, paddle.fluid.framework.Variable)
            self.assertEqual(result.shape, (1,))

        # Test with generate_tensor_random()
        result_random = dist.dtensor_from_fn(
            paddle.rand, dist_attr=dist_attr, shape=[1]
        )
        if paddle.in_dynamic_mode():
            self.assertIsInstance(result, paddle.Tensor)
            self.assertEqual(result.shape, [1])
        else:
            self.assertIsInstance(result, paddle.fluid.framework.Variable)
            self.assertEqual(result.shape, (1,))

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


class TestShardLayer(unittest.TestCase):
    def test_shard_layer(self):
        # Create a simple linear model
        model = Linear(10, 1)

        # Define shard function
        def shard_fn(name, submod, process_mesh):
            mesh = dist.ProcessMesh(
                [[2, 4, 5], [0, 1, 3]], dim_names=["x", "y"]
            )
            dist_attr = dist.DistAttr(mesh=mesh, sharding_specs=['x', 'y'])
            return dist.shard_tensor(submod.weight, dist_attr=dist_attr)

        # Define input hook
        def input_fn(inputs, process_mesh):
            return inputs[0] * process_mesh.ndim

        # Define output hook
        def output_fn(outputs, process_mesh):
            return outputs * process_mesh.ndim

        # Shard the model
        model = dist.shard_layer(
            model,
            process_mesh=None,
            shard_fn=shard_fn,
            input_fn=input_fn,
            output_fn=output_fn,
        )

        # Verify the parameters
        for name, param in model.named_parameters():
            self.assertIsInstance(param, paddle.Tensor)
            self.assertEqual(
                param.dist_attr.mesh, shard_fn(None, None, None).dist_attr.mesh
            )
            self.assertEqual(
                param.dist_attr.sharding_specs,
                shard_fn(None, None, None).dist_attr.sharding_specs,
            )

        # Verify the hooks
        x = paddle.to_tensor([1.0])
        y = model(x)
        self.assertEqual(
            y.numpy(), [10.0]
        )  # model has 2 dimensions, so input and output are multiplied by 2


if __name__ == '__main__':
    unittest.main()
