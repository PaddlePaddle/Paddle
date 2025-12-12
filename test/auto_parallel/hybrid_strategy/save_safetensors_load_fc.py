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

import os

import paddle
import paddle.distributed as dist
from paddle.distributed.flex_checkpoint.dcp.sharded_weight import ShardedWeight


def get_global_tensors():
    """Create fixed test tensors for verification."""
    # tensor1: [[0, 1], [2, 3]]
    tensor1 = paddle.to_tensor([[0, 1], [2, 3]], dtype='float32')
    # tensor2: [[4, 5], [6, 7]]
    tensor2 = paddle.to_tensor([[4, 5], [6, 7]], dtype='float32')
    return {"tensor1": tensor1, "tensor2": tensor2}


def save_safetensors_to_ranks(ckpt_path):
    """Save tensors to different ranks as safetensors files."""
    import safetensors.numpy

    global_tensors = get_global_tensors()

    if dist.get_rank() == 0:
        os.makedirs(ckpt_path, exist_ok=True)
        file_path = os.path.join(ckpt_path, "tensor1.safetensors")

        tensor1_np = global_tensors["tensor1"].numpy()
        safetensors.numpy.save_file({"tensor1": tensor1_np}, file_path)

    elif dist.get_rank() == 1:
        os.makedirs(ckpt_path, exist_ok=True)
        file_path = os.path.join(ckpt_path, "tensor2.safetensors")

        tensor2_np = global_tensors["tensor2"].numpy()
        safetensors.numpy.save_file({"tensor2": tensor2_np}, file_path)

    dist.barrier()


def create_sharded_state_dict_for_loading():
    """Create sharded state dict for tp loading."""
    sharded_state_dict = {}

    if dist.get_rank() == 0:
        local_tensor1 = paddle.zeros([2, 1], dtype='float32')
        sharded_weight1 = ShardedWeight(
            key="tensor1",
            local_tensor=local_tensor1,
            local_shape=(2, 1),
            global_shape=(2, 2),
            global_offset=(0, 0),
            is_flattened=False,
        )
        sharded_state_dict["tensor1"] = sharded_weight1

        local_tensor2 = paddle.zeros([2, 1], dtype='float32')
        sharded_weight2 = ShardedWeight(
            key="tensor2",
            local_tensor=local_tensor2,
            local_shape=(2, 1),
            global_shape=(2, 2),
            global_offset=(0, 0),
            is_flattened=False,
        )
        sharded_state_dict["tensor2"] = sharded_weight2

    elif dist.get_rank() == 1:
        local_tensor1 = paddle.zeros([2, 1], dtype='float32')
        sharded_weight1 = ShardedWeight(
            key="tensor1",
            local_tensor=local_tensor1,
            local_shape=(2, 1),
            global_shape=(2, 2),
            global_offset=(0, 1),
            is_flattened=False,
        )
        sharded_state_dict["tensor1"] = sharded_weight1

        local_tensor2 = paddle.zeros([2, 1], dtype='float32')
        sharded_weight2 = ShardedWeight(
            key="tensor2",
            local_tensor=local_tensor2,
            local_shape=(2, 1),
            global_shape=(2, 2),
            global_offset=(0, 1),
            is_flattened=False,
        )
        sharded_state_dict["tensor2"] = sharded_weight2

    return sharded_state_dict


def test_save_safetensors_load_fc():
    """Test saving safetensors and loading with flex checkpoint."""
    ckpt_path = os.getenv("ckpt_path")
    dist.init_parallel_env()

    save_safetensors_to_ranks(ckpt_path)

    sharded_state_dict = create_sharded_state_dict_for_loading()

    from paddle.distributed.flex_checkpoint.dcp.load_state_dict import (
        load_state_dict,
    )

    load_state_dict(sharded_state_dict, ckpt_path, safetensors=True)

    loaded_tensor1 = sharded_state_dict["tensor1"].local_tensor
    loaded_tensor2 = sharded_state_dict["tensor2"].local_tensor

    if dist.get_rank() == 0:
        # Rank 0 should have first column of both tensors
        # tensor1: [[0], [2]] (first column)
        # tensor2: [[4], [6]] (first column)
        expected_tensor1 = paddle.to_tensor([[0], [2]], dtype='float32')
        expected_tensor2 = paddle.to_tensor([[4], [6]], dtype='float32')

        assert paddle.allclose(loaded_tensor1, expected_tensor1), (
            f"Rank 0 tensor1 mismatch: got {loaded_tensor1}, expected {expected_tensor1}"
        )
        assert paddle.allclose(loaded_tensor2, expected_tensor2), (
            f"Rank 0 tensor2 mismatch: got {loaded_tensor2}, expected {expected_tensor2}"
        )

    elif dist.get_rank() == 1:
        # Rank 1 should have second column of both tensors
        # tensor1: [[1], [3]] (second column)
        # tensor2: [[5], [7]] (second column)
        expected_tensor1 = paddle.to_tensor([[1], [3]], dtype='float32')
        expected_tensor2 = paddle.to_tensor([[5], [7]], dtype='float32')

        assert paddle.allclose(loaded_tensor1, expected_tensor1), (
            f"Rank 1 tensor1 mismatch: got {loaded_tensor1}, expected {expected_tensor1}"
        )
        assert paddle.allclose(loaded_tensor2, expected_tensor2), (
            f"Rank 1 tensor2 mismatch: got {loaded_tensor2}, expected {expected_tensor2}"
        )

    dist.barrier()


if __name__ == "__main__":
    test_save_safetensors_load_fc()
