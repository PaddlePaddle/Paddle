# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import os

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.distributed.auto_parallel.moe_utils import (
    _only_reshard_mesh_shape,
    get_local_slices,
    get_rank2tensor_indices,
    shard_submesh_and_slice,
)


class TestMoEUtils:
    def __init__(self):
        self._dtype = os.getenv("dtype")
        self._seeds = eval(os.getenv("seeds"))
        self._backend = os.getenv("backend")
        self._mesh0 = dist.ProcessMesh([[0], [1]], dim_names=["x", "y"])  # 2x1
        self._mesh1 = dist.ProcessMesh([[0, 1]], dim_names=["x", "y"])  # 1x2
        self._mesh2 = dist.ProcessMesh(
            [0, 1], dim_names=["x"]
        )  # 1D mesh with 2 processes
        paddle.seed(self._seeds)
        # Ensure the environment flag is set for _only_reshard_mesh_shape
        os.environ["FLAGS_enable_moe_utils"] = "true"

    # Existing tests (unchanged)
    def test_local_reshape(self):
        (h, w) = (4, 4)
        src_shape = [h, w]
        tgt_shape = [h // 2, w * 2]
        x = paddle.arange(0, h * w).reshape(src_shape)
        x.stop_gradient = False
        np_x = x.numpy()

        dist_x = dist.shard_tensor(
            x, self._mesh0, [dist.Shard(1), dist.Replicate()]
        )
        dist_y = dist.auto_parallel.moe_utils._dist_reshape(
            dist_x, [-1, w * 2], self._mesh0, [dist.Shard(1), dist.Replicate()]
        )

        splitted_np_x = np.split(np_x, 2, axis=1)
        for i in range(len(splitted_np_x)):
            splitted_np_x[i] = splitted_np_x[i].reshape([h // 2, w])
        np.testing.assert_array_equal(
            splitted_np_x[dist.get_rank()], dist_y._local_value().numpy()
        )

        label = paddle.ones(tgt_shape, dtype=paddle.int64)
        label.stop_gradient = False
        dist_label = dist.shard_tensor(
            label, self._mesh0, [dist.Shard(1), dist.Replicate()]
        )
        loss = dist_y - dist_label
        loss.backward()

        np_grad = np.ones(src_shape, dtype="int64")
        splitted_np_grad = np.split(np_grad, 2, axis=1)
        np.testing.assert_array_equal(
            splitted_np_grad[dist.get_rank()],
            dist_x.grad._local_value().numpy(),
        )

        with np.testing.assert_raises(AssertionError):
            dist_z = dist.auto_parallel.moe_utils._dist_reshape(
                dist_x,
                dist_x.shape,
                self._mesh1,
                [dist.Replicate(), dist.Replicate()],
            )

        dist_z = dist.auto_parallel.moe_utils._dist_reshape(
            dist_x, dist_x.shape, self._mesh0, [dist.Shard(1), dist.Shard(1)]
        )

    # python -m paddle.distributed.launch --devices=0,1 semi_auto_parallel_moe_utils.py
    def test_nd_mesh_alltoall(self):
        if self._backend == "cpu":
            return

        (h, w) = (4, 4)
        src_shape = [h, w]
        x = paddle.arange(0, h * w).reshape(src_shape)
        x.stop_gradient = False

        dist_x = dist.shard_tensor(
            x, self._mesh0, [dist.Shard(1), dist.Replicate()]
        )
        dist_y = dist.reshard(
            dist_x, self._mesh0, [dist.Shard(0), dist.Replicate()]
        )
        dist_y.backward()

        np.testing.assert_equal(
            dist_y.placements, [dist.Shard(0), dist.Replicate()]
        )
        np.testing.assert_equal(
            dist_x.grad.placements, [dist.Shard(1), dist.Replicate()]
        )
        np_grad = np.ones(src_shape, dtype="int64")
        splitted_np_grad = np.split(np_grad, 2, axis=1)
        np.testing.assert_array_equal(
            splitted_np_grad[dist.get_rank()],
            dist_x.grad._local_value().numpy(),
        )

    def test_reshard_mesh_shape(self):
        (h, w) = (4, 4)
        src_shape = [h, w]
        x = paddle.arange(0, h * w).reshape(src_shape)

        dist_x = dist.shard_tensor(
            x, self._mesh0, [dist.Replicate(), dist.Replicate()]
        )
        dist_y = dist.reshard(
            dist_x, self._mesh1, [dist.Replicate(), dist.Replicate()]
        )

        np.testing.assert_equal(dist_y.process_mesh, self._mesh1)
        np.testing.assert_array_equal(
            dist_y._local_value().numpy(), dist_x._local_value().numpy()
        )

    def test_get_local_slices(self):
        (h, w) = (4, 4)
        src_shape = [h, w]
        x = paddle.arange(0, h * w).reshape(src_shape)
        placements = [dist.Shard(0), dist.Partial()]
        dist_x = dist.shard_tensor(x, self._mesh0, placements)
        dist_x_local_slices = get_local_slices(x, self._mesh0, placements)
        np.testing.assert_equal(
            dist_x_local_slices[0]['slice'], [(0, 2), (0, 4)]
        )
        np.testing.assert_equal(
            dist_x_local_slices[0]['partial'][1],
            dist_x.placements[1].reduce_type(),
        )
        np.testing.assert_equal(
            dist_x_local_slices[1]['slice'], [(2, 4), (0, 4)]
        )
        np.testing.assert_equal(
            dist_x_local_slices[1]['partial'][1],
            dist_x.placements[1].reduce_type(),
        )

    # python -m paddle.distributed.launch --devices=0,1 semi_auto_parallel_moe_utils.py
    def test_reshard_general_case(self):
        """Test reshard when _only_reshard_mesh_shape returns False."""
        (h, w) = (4, 4)
        x = paddle.arange(0, h * w, dtype=self._dtype).reshape([h, w])
        dist_x = dist.shard_tensor(x, self._mesh2, [dist.Replicate()])
        dist_y = dist.reshard(dist_x, self._mesh2, [dist.Shard(0)])

        if dist.get_rank() == 0:
            expected_y = x[:2, :]  # Process 0 gets first half of axis 0
            np.testing.assert_array_equal(
                dist_y._local_value().numpy(), expected_y.numpy()
            )
        elif dist.get_rank() == 1:
            expected_y = x[2:, :]  # Process 1 gets second half of axis 0
            np.testing.assert_array_equal(
                dist_y._local_value().numpy(), expected_y.numpy()
            )

    def test_shard_submesh_and_slice(self):
        """Test shard_submesh_and_slice with even and uneven tensor sizes."""
        mesh = dist.ProcessMesh([[0, 1]], dim_names=["x", "y"])  # 1x2 mesh
        tensor_slice = [(0, 4), (0, 4)]
        tensor_dim = 0
        mesh_dim = 1
        new_sub_meshes, new_slices = shard_submesh_and_slice(
            mesh, tensor_slice, tensor_dim, mesh_dim
        )
        np.testing.assert_equal(len(new_sub_meshes), 2)
        np.testing.assert_equal(new_sub_meshes[0].process_ids, [0])
        np.testing.assert_equal(new_sub_meshes[1].process_ids, [1])
        np.testing.assert_equal(new_slices[0], [(0, 2), (0, 4)])
        np.testing.assert_equal(new_slices[1], [(2, 4), (0, 4)])

        # Uneven size
        tensor_slice = [(0, 5), (0, 4)]
        new_sub_meshes, new_slices = shard_submesh_and_slice(
            mesh, tensor_slice, tensor_dim, mesh_dim
        )
        np.testing.assert_equal(
            new_slices[0], [(0, 3), (0, 4)]
        )  # First shard: 3 elements
        np.testing.assert_equal(
            new_slices[1], [(3, 5), (0, 4)]
        )  # Last shard: 2 elements

    def test_get_rank2tensor_indices(self):
        """Test get_rank2tensor_indices mapping."""
        sub_mesh_indices_info = {
            dist.ProcessMesh([0]): [(0, 2), (0, 4)],
            dist.ProcessMesh([1]): [(2, 4), (0, 4)],
        }
        sub_mesh_partial_info = {}
        rank2tensor_indices = get_rank2tensor_indices(
            sub_mesh_indices_info, sub_mesh_partial_info
        )
        np.testing.assert_equal(
            rank2tensor_indices[0], {'slice': [(0, 2), (0, 4)], 'partial': {}}
        )
        np.testing.assert_equal(
            rank2tensor_indices[1], {'slice': [(2, 4), (0, 4)], 'partial': {}}
        )

    def test_get_local_slices_additional(self):
        """Test get_local_slices with different placements."""
        (h, w) = (4, 4)
        x = paddle.arange(0, h * w, dtype=self._dtype).reshape([h, w])

        # Test with [Replicate(), Replicate()]
        placements = [dist.Replicate(), dist.Replicate()]
        slices = get_local_slices(x, self._mesh0, placements)
        for rank in [0, 1]:
            np.testing.assert_equal(slices[rank]['slice'], [(0, 4), (0, 4)])
            np.testing.assert_equal(slices[rank]['partial'], {})

        # Test with [Shard(1), Replicate()] on mesh1
        placements = [dist.Replicate(), dist.Shard(1)]
        slices = get_local_slices(x, self._mesh1, placements)
        np.testing.assert_equal(slices[0]['slice'], [(0, 4), (0, 2)])
        np.testing.assert_equal(slices[1]['slice'], [(0, 4), (2, 4)])

    def test_only_reshard_mesh_shape(self):
        """Test _only_reshard_mesh_shape conditions."""
        (h, w) = (4, 4)
        x = paddle.arange(0, h * w, dtype=self._dtype).reshape([h, w])

        # Case 1: Same mesh, should return False
        dist_x = dist.shard_tensor(
            x, self._mesh0, [dist.Replicate(), dist.Replicate()]
        )
        result = _only_reshard_mesh_shape(
            dist_x, self._mesh0, [dist.Replicate(), dist.Replicate()]
        )
        assert not result

        # Case 2: Different process IDs, should return False
        mesh_diff = dist.ProcessMesh([[2], [3]], dim_names=["x", "y"])
        result = _only_reshard_mesh_shape(
            dist_x, mesh_diff, [dist.Replicate(), dist.Replicate()]
        )
        assert not result

        # Case 3: Same process IDs, different slices
        dist_x = dist.shard_tensor(
            x, self._mesh0, [dist.Shard(0), dist.Replicate()]
        )
        result = _only_reshard_mesh_shape(
            dist_x, self._mesh1, [dist.Replicate(), dist.Shard(1)]
        )
        assert not result

        # Case 4: Same process IDs, same slices
        dist_x = dist.shard_tensor(
            x, self._mesh0, [dist.Replicate(), dist.Replicate()]
        )
        result = _only_reshard_mesh_shape(
            dist_x, self._mesh1, [dist.Replicate(), dist.Replicate()]
        )
        assert result

        # Case 5: Flag disabled
        os.environ["FLAGS_enable_moe_utils"] = "false"
        result = _only_reshard_mesh_shape(
            dist_x, self._mesh1, [dist.Replicate(), dist.Replicate()]
        )
        assert not result
        os.environ["FLAGS_enable_moe_utils"] = "true"  # Reset

    def run_test_case(self):
        if self._backend == "cpu":
            paddle.set_device("cpu")
        self.test_local_reshape()
        self.test_nd_mesh_alltoall()
        self.test_reshard_mesh_shape()
        self.test_get_local_slices()
        self.test_reshard_general_case()
        self.test_shard_submesh_and_slice()
        self.test_get_rank2tensor_indices()
        self.test_get_local_slices_additional()
        self.test_only_reshard_mesh_shape()


if __name__ == '__main__':
    TestMoEUtils().run_test_case()
