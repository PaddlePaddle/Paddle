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
from paddle import nn
from paddle.nn import Linear


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


class TestShardLayer(unittest.TestCase):
    def test_shard_layer(self):
        # Create a simple linear model
        model = Linear(13, 5)
        mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        dist_attr = dist.DistAttr(mesh=mesh, sharding_specs=[None])

        # Define shard function
        def shard_fn(name, module, process_mesh):
            if isinstance(module, nn.Linear):
                for name, param in module.named_parameters():
                    dist_param = paddle.nn.Parameter(
                        dist.shard_tensor(param, process_mesh, shape=[1])
                    )
                    module.register_parameter(name, dist_param)

        # Define input hook
        def input_fn(layer, input):
            input_return = input[0] * 2
            return input_return

        # Define output hook
        def output_fn(layer, input, output):
            return output * 2

        # Verify input_fn
        input_fn_handle = model.register_forward_pre_hook(input_fn)

        value0 = np.arange(26).reshape(2, 13).astype("float32")
        in0 = paddle.to_tensor(value0)
        out0 = model(in0)

        input_fn_handle.remove()

        value1 = value0 * 2
        in1 = paddle.to_tensor(value1)
        out1 = model(in1)

        # hook change the linear's input to input * 2, so out0 is equal to out1.
        assert (out0.numpy() == out1.numpy()).any()

        # Verify output_fn
        output_fn_handle = model.register_forward_post_hook(output_fn)

        value1 = np.arange(26).reshape(2, 13).astype("float32")
        in1 = paddle.to_tensor(value1)

        out0 = model(in1)

        output_fn_handle.remove()

        out1 = model(in1)

        # hook change the linear's output to output * 2, so out0 is equal to out1 * 2.
        assert (out0.numpy() == (out1.numpy()) * 2).any()

        # Shard the model
        model = dist.shard_layer(
            model,
            process_mesh=mesh,
            shard_fn=shard_fn,
            input_fn=input_fn,
            output_fn=output_fn,
        )

        # Verify the parameters
        for name, param in model.named_parameters():
            if param is not None:
                self.assertIsInstance(param, paddle.Tensor)
                self.assertTrue(param.shape == [13, 5] or param.shape == [5])


if __name__ == '__main__':
    unittest.main()
