# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.distributed as dist
from paddle import Tensor
from paddle.distributed import ProcessMesh


def custom_function(x):
    mask = paddle.zeros_like(x)
    if dist.get_rank() == 0:
        mask[1:2] = 1
    else:
        mask[2:3] = 1

    x = x * mask
    mask_sum = paddle.sum(x)
    mask_sum = mask_sum / mask.sum()
    return mask_sum


class TestLocalMap(unittest.TestCase):
    def test_local_map(self):
        """Test all functionalities of local_map API"""
        dist.init_parallel_env()
        mesh = ProcessMesh([0, 1], dim_names=["x"])

        # Case 1: Basic distributed tensor input/output with custom function
        local_input = paddle.arange(0, 4, dtype="float32")
        local_input = local_input + dist.get_rank()
        input_dist = dist.auto_parallel.api.dtensor_from_local(
            local_input, mesh, [dist.Shard(0)]
        )

        wrapped_func = dist.local_map(
            custom_function,
            out_placements=[[dist.Partial(dist.ReduceType.kRedSum)]],
            in_placements=[[dist.Shard(0)]],
            process_mesh=mesh,
            reshard_inputs=True,
        )
        output_dist = wrapped_func(input_dist)

        # Verify custom function results
        local_value = output_dist._local_value()
        gathered_values: list[Tensor] = []
        dist.all_gather(gathered_values, local_value)

        expected_rank0, expected_rank1 = 1.0, 3.0
        expected_global = 4.0
        self.assertAlmostEqual(
            gathered_values[0].item(),
            expected_rank0,
            delta=1e-6,
            msg=f"Rank 0 value mismatch: got {gathered_values[0].item()}, expected {expected_rank0}",
        )
        self.assertAlmostEqual(
            gathered_values[1].item(),
            expected_rank1,
            delta=1e-6,
            msg=f"Rank 1 value mismatch: got {gathered_values[1].item()}, expected {expected_rank1}",
        )
        self.assertAlmostEqual(
            output_dist.item(),
            expected_global,
            delta=1e-6,
            msg=f"Global value mismatch: got {output_dist.item()}, expected {expected_global}",
        )

        # Case 2: Normal tensor input -> distributed tensor output
        def func2(x):
            return x + 1

        input_normal = paddle.ones([4])
        wrapped_func = dist.local_map(
            func2,
            out_placements=[[dist.Shard(0)]],
            in_placements=None,
            process_mesh=mesh,
        )
        out2 = wrapped_func(input_normal)
        self.assertTrue(dist.auto_parallel.api.is_dist_tensor(out2))

        # Case 3: Mixed tensor and non-tensor outputs
        def func3(x):
            return x.sum(), "hello"

        wrapped_func = dist.local_map(
            func3,
            out_placements=[
                [dist.Replicate()],
                None,
            ],  # None for non-tensor output
            in_placements=[[dist.Shard(0)]],
            process_mesh=mesh,
        )
        out3_tensor, out3_str = wrapped_func(input_dist)
        self.assertTrue(dist.auto_parallel.api.is_dist_tensor(out3_tensor))
        self.assertIsInstance(out3_str, str)

        # Case 4: Mixed distributed and normal tensor inputs
        def func4(x, y):
            return x + y

        wrapped_func = dist.local_map(
            func4,
            out_placements=[[dist.Shard(0)]],
            in_placements=[
                [dist.Shard(0)],
                None,
            ],  # None for normal tensor input
            process_mesh=mesh,
        )
        out4 = wrapped_func(input_dist, input_normal)
        self.assertTrue(dist.auto_parallel.api.is_dist_tensor(out4))

        # Case 5: Test process_mesh inference in both dynamic and static modes
        def func5(x):
            return x * 2

        # Test in dynamic mode
        paddle.disable_static()

        input_dist = dist.auto_parallel.api.dtensor_from_local(
            paddle.ones([4]), mesh, [dist.Shard(0)]
        )
        wrapped_func = dist.local_map(
            func5,
            out_placements=[[dist.Replicate()]],
            in_placements=[[dist.Shard(0)]],
            process_mesh=None,
        )
        out5 = wrapped_func(input_dist)
        self.assertTrue(dist.auto_parallel.api.is_dist_tensor(out5))
        self.assertEqual(out5.process_mesh, input_dist.process_mesh)

        # Test in static mode
        paddle.enable_static()

        input_dist = dist.auto_parallel.api.dtensor_from_local(
            paddle.ones([4]), mesh, [dist.Shard(0)]
        )
        wrapped_func = dist.local_map(
            func5,
            out_placements=[[dist.Replicate()]],
            in_placements=[[dist.Shard(0)]],
            process_mesh=None,
        )
        out5 = wrapped_func(input_dist)
        self.assertTrue(dist.auto_parallel.api.is_dist_tensor(out5))
        self.assertEqual(
            out5.dist_attr().process_mesh, input_dist.dist_attr().process_mesh
        )

        # Restore to dynamic mode
        paddle.disable_static()

        # Case 6: Test reshard_inputs parameter in both dynamic and static modes
        def func6(x):
            return x * 2

        # Test in dynamic mode
        paddle.disable_static()

        input_dist = dist.auto_parallel.api.dtensor_from_local(
            paddle.ones([4]), mesh, [dist.Shard(0)]
        )
        wrapped_func = dist.local_map(
            func6,
            out_placements=[[dist.Replicate()]],
            in_placements=[[dist.Replicate()]],
            process_mesh=mesh,
            reshard_inputs=True,
        )
        out6_resharded = wrapped_func(input_dist)
        self.assertTrue(dist.auto_parallel.api.is_dist_tensor(out6_resharded))
        self.assertEqual(out6_resharded.placements, [dist.Replicate()])

        # Test reshard_inputs=False
        wrapped_func = dist.local_map(
            func6,
            out_placements=[[dist.Replicate()]],
            in_placements=[[dist.Replicate()]],
            process_mesh=mesh,
            reshard_inputs=False,
        )
        with self.assertRaises(ValueError) as ctx:
            _ = wrapped_func(input_dist)
        self.assertIn("in_placement", str(ctx.exception))

        # Test in static mode
        paddle.enable_static()

        input_dist = dist.auto_parallel.api.dtensor_from_local(
            paddle.ones([4]), mesh, [dist.Shard(0)]
        )
        wrapped_func = dist.local_map(
            func6,
            out_placements=[[dist.Replicate()]],
            in_placements=[[dist.Replicate()]],
            process_mesh=mesh,
            reshard_inputs=True,
        )
        out6_resharded = wrapped_func(input_dist)
        self.assertTrue(dist.auto_parallel.api.is_dist_tensor(out6_resharded))
        self.assertTrue(
            isinstance(out6_resharded.dist_attr().placements[0], dist.Replicate)
        )

        # Test reshard_inputs=False in static mode
        wrapped_func = dist.local_map(
            func6,
            out_placements=[[dist.Replicate()]],
            in_placements=[[dist.Replicate()]],
            process_mesh=mesh,
            reshard_inputs=False,
        )
        with self.assertRaises(ValueError) as ctx:
            _ = wrapped_func(input_dist)
        self.assertIn("dist_tensor.dist_attr().placements", str(ctx.exception))

        # Restore to dynamic mode
        paddle.disable_static()

        # Case 7: Test with in_placements=None and distributed tensor input
        def func7(x):
            return x * 2

        # Test in dynamic mode
        paddle.disable_static()

        input_dist = dist.auto_parallel.api.dtensor_from_local(
            paddle.ones([4]), mesh, [dist.Shard(0)]
        )

        wrapped_func = dist.local_map(
            func7,
            out_placements=[[dist.Replicate()]],
            in_placements=[None],
            process_mesh=mesh,
        )
        out7 = wrapped_func(input_dist)
        self.assertTrue(dist.auto_parallel.api.is_dist_tensor(out7))

        # Test in static mode
        paddle.enable_static()

        input_dist = dist.auto_parallel.api.dtensor_from_local(
            paddle.ones([4]), mesh, [dist.Shard(0)]
        )

        wrapped_func = dist.local_map(
            func7,
            out_placements=[[dist.Replicate()]],
            in_placements=[None],
            process_mesh=mesh,
        )
        out7 = wrapped_func(input_dist)
        self.assertTrue(dist.auto_parallel.api.is_dist_tensor(out7))
        # Restore to dynamic mode
        paddle.disable_static()


if __name__ == "__main__":
    unittest.main()
