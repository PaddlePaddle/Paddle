import os
import unittest
import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed.auto_parallel.moe_utils import (
    get_sub_meshes_for_shard,
    shard_submesh_and_slice,
    get_rank2tensor_indices,
    get_local_slices,
    _only_reshard_mesh_shape,
)

class TestMoEUtils(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self._dtype = os.getenv("dtype", "float32")
        self._seed = eval(os.getenv("seed", "2024"))
        self._backend = os.getenv("backend", "gpu")
        self._mesh0 = dist.ProcessMesh([[0], [1]], dim_names=["x", "y"])  # 2x1
        self._mesh1 = dist.ProcessMesh([[0, 1]], dim_names=["x", "y"])   # 1x2
        self._mesh2 = dist.ProcessMesh([0, 1], dim_names=["x"])          # 1D mesh with 2 processes
        paddle.seed(self._seed)

    def setUp(self):
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

        with self.assertRaises(AssertionError):
            dist_z = dist.auto_parallel.moe_utils._dist_reshape(
                dist_x,
                dist_x.shape,
                self._mesh1,
                [dist.Replicate(), dist.Replicate()],
            )

        dist_z = dist.auto_parallel.moe_utils._dist_reshape(
            dist_x, dist_x.shape, self._mesh0, [dist.Shard(1), dist.Shard(1)]
        )

    def test_nd_mesh_alltoall(self):
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

        self.assertEqual(dist_y.placements, [dist.Shard(0), dist.Replicate()])
        self.assertEqual(dist_x.grad.placements, [dist.Shard(1), dist.Replicate()])
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

        self.assertEqual(dist_y.process_mesh, self._mesh1)
        np.testing.assert_array_equal(
            dist_y._local_value().numpy(), dist_x._local_value().numpy()
        )

    def test_get_local_slices(self):
        (h, w) = (4, 4)
        src_shape = [h, w]
        x = paddle.arange(0, h * w).reshape(src_shape)
        placements = [dist.Shard(0), dist.Partial()]
        dist_x = dist.shard_tensor(
            x, self._mesh0, placements
        )
        dist_x_local_slices = get_local_slices(x, self._mesh0, placements)
        if dist.get_rank() == 0:
            self.assertEqual(dist_x_local_slices[0]['slice'], ((0, 2), (0, 4)))
            self.assertEqual(dist_x_local_slices[0]['partial'][1], dist_x.placements[1].reduce_type())
        if dist.get_rank() == 1:
            self.assertEqual(dist_x_local_slices[1]['slice'], ((2, 4), (0, 4)))
            self.assertEqual(dist_x_local_slices[1]['partial'][1], dist_x.placements[1].reduce_type())

    # New tests to increase coverage
    def test_reshard_general_case(self):
        """Test reshard when _only_reshard_mesh_shape returns False."""
        (h, w) = (4, 4)
        x = paddle.arange(0, h * w, dtype=self._dtype).reshape([h, w])
        dist_x = dist.shard_tensor(x, self._mesh0, [dist.Shard(0), dist.Replicate()])
        dist_y = dist.reshard(dist_x, self._mesh1, [dist.Replicate(), dist.Shard(1)])
        
        if dist.get_rank() == 0:
            expected_y = x[:, :2]  # Process 0 gets first half of axis 1
            np.testing.assert_array_equal(dist_y._local_value().numpy(), expected_y.numpy())
        elif dist.get_rank() == 1:
            expected_y = x[:, 2:]  # Process 1 gets second half of axis 1
            np.testing.assert_array_equal(dist_y._local_value().numpy(), expected_y.numpy())

    def test_get_sub_meshes_for_shard(self):
        """Test get_sub_meshes_for_shard with a 2x2 mesh."""
        mesh = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=["x", "y"])  # 2x2 mesh
        # Shard along dim 0
        sub_meshes = get_sub_meshes_for_shard(mesh, 0)
        self.assertEqual(len(sub_meshes), 2)
        self.assertEqual(sub_meshes[0].process_ids, [0, 1])
        self.assertEqual(sub_meshes[0].shape, [1, 2])
        self.assertEqual(sub_meshes[1].process_ids, [2, 3])
        self.assertEqual(sub_meshes[1].shape, [1, 2])
        # Shard along dim 1
        sub_meshes = get_sub_meshes_for_shard(mesh, 1)
        self.assertEqual(len(sub_meshes), 2)
        self.assertEqual(sub_meshes[0].process_ids, [0, 2])
        self.assertEqual(sub_meshes[0].shape, [2, 1])
        self.assertEqual(sub_meshes[1].process_ids, [1, 3])
        self.assertEqual(sub_meshes[1].shape, [2, 1])

    def test_shard_submesh_and_slice(self):
        """Test shard_submesh_and_slice with even and uneven tensor sizes."""
        mesh = dist.ProcessMesh([[0, 1]], dim_names=["x", "y"])  # 1x2 mesh
        tensor_slice = [(0, 4), (0, 4)]
        tensor_dim = 0
        mesh_dim = 1
        new_sub_meshes, new_slices = shard_submesh_and_slice(mesh, tensor_slice, tensor_dim, mesh_dim)
        self.assertEqual(len(new_sub_meshes), 2)
        self.assertEqual(new_sub_meshes[0].process_ids, [0])
        self.assertEqual(new_sub_meshes[1].process_ids, [1])
        self.assertEqual(new_slices[0], [(0, 2), (0, 4)])
        self.assertEqual(new_slices[1], [(2, 4), (0, 4)])

        # Uneven size
        tensor_slice = [(0, 5), (0, 4)]
        new_sub_meshes, new_slices = shard_submesh_and_slice(mesh, tensor_slice, tensor_dim, mesh_dim)
        self.assertEqual(new_slices[0], [(0, 3), (0, 4)])  # First shard: 3 elements
        self.assertEqual(new_slices[1], [(3, 5), (0, 4)])  # Last shard: 2 elements

    def test_get_rank2tensor_indices(self):
        """Test get_rank2tensor_indices mapping."""
        sub_mesh2tensor_indices = {
            dist.ProcessMesh([0]): {'slice': [(0, 2), (0, 4)], 'partial': {}},
            dist.ProcessMesh([1]): {'slice': [(2, 4), (0, 4)], 'partial': {}},
        }
        rank2tensor_indices = get_rank2tensor_indices(sub_mesh2tensor_indices)
        self.assertEqual(rank2tensor_indices[0], {'slice': [(0, 2), (0, 4)], 'partial': {}})
        self.assertEqual(rank2tensor_indices[1], {'slice': [(2, 4), (0, 4)], 'partial': {}})

    def test_get_local_slices_additional(self):
        """Test get_local_slices with different placements."""
        (h, w) = (4, 4)
        x = paddle.arange(0, h * w, dtype=self._dtype).reshape([h, w])
        
        # Test with [Replicate(), Replicate()]
        placements = [dist.Replicate(), dist.Replicate()]
        slices = get_local_slices(x, self._mesh0, placements)
        for rank in [0, 1]:
            self.assertEqual(slices[rank]['slice'], [(0, 4), (0, 4)])
            self.assertEqual(slices[rank]['partial'], {})

        # Test with [Shard(1), Replicate()] on mesh1
        placements = [dist.Replicate(), dist.Shard(1)]
        slices = get_local_slices(x, self._mesh1, placements)
        self.assertEqual(slices[0]['slice'], [(0, 4), (0, 2)])
        self.assertEqual(slices[1]['slice'], [(0, 4), (2, 4)])

    def test_only_reshard_mesh_shape(self):
        """Test _only_reshard_mesh_shape conditions."""
        (h, w) = (4, 4)
        x = paddle.arange(0, h * w, dtype=self._dtype).reshape([h, w])
        
        # Case 1: Same mesh, should return False
        dist_x = dist.shard_tensor(x, self._mesh0, [dist.Replicate(), dist.Replicate()])
        result = _only_reshard_mesh_shape(dist_x, self._mesh0, [dist.Replicate(), dist.Replicate()])
        self.assertFalse(result)

        # Case 2: Different process IDs, should return False
        mesh_diff = dist.ProcessMesh([[2], [3]], dim_names=["x", "y"])
        result = _only_reshard_mesh_shape(dist_x, mesh_diff, [dist.Replicate(), dist.Replicate()])
        self.assertFalse(result)

        # Case 3: Same process IDs, different slices
        dist_x = dist.shard_tensor(x, self._mesh0, [dist.Shard(0), dist.Replicate()])
        result = _only_reshard_mesh_shape(dist_x, self._mesh1, [dist.Replicate(), dist.Shard(1)])
        self.assertFalse(result)

        # Case 4: Same process IDs, same slices
        dist_x = dist.shard_tensor(x, self._mesh0, [dist.Replicate(), dist.Replicate()])
        result = _only_reshard_mesh_shape(dist_x, self._mesh1, [dist.Replicate(), dist.Replicate()])
        self.assertTrue(result)

        # Case 5: Flag disabled
        os.environ["FLAGS_enable_moe_utils"] = "false"
        result = _only_reshard_mesh_shape(dist_x, self._mesh1, [dist.Replicate(), dist.Replicate()])
        self.assertFalse(result)
        os.environ["FLAGS_enable_moe_utils"] = "true"  # Reset

    def run_test_case(self):
        self.test_local_reshape()
        self.test_nd_mesh_alltoall()
        self.test_reshard_mesh_shape()
        self.test_get_local_slices()
        self.test_reshard_general_case()
        self.test_get_sub_meshes_for_shard()
        self.test_shard_submesh_and_slice()
        self.test_get_rank2tensor_indices()
        self.test_get_local_slices_additional()
        self.test_only_reshard_mesh_shape()

if __name__ == '__main__':
    unittest.main()