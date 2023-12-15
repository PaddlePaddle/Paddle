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

import os

import numpy as np

import paddle
import paddle.distributed as dist


class TestSaveStateDict:
    def __init__(self):
        self._ckpt_path = os.getenv("ckpt_path")

    def test_dedup_tesnor(self):
        w1 = paddle.arange(32).reshape([4, 8])
        w2 = paddle.arange(32, 36).reshape([2, 2])
        mesh = dist.ProcessMesh([0, 1])
        dist_w1 = dist.shard_tensor(w1, mesh, [dist.Replicate()])
        dist_w2 = dist.shard_tensor(w2, mesh, [dist.Shard(0)])
        state_dict = {"w1": dist_w1, "w2": dist_w2}
        dist.save_state_dict(state_dict, self._ckpt_path)
        paddle.distributed.barrier()
        # check
        expect_local_state_dict = {}
        for k, v in state_dict.items():
            expect_local_state_dict[k] = v._local_value()
        data_file_path = os.path.join(
            self._ckpt_path, f"{paddle.distributed.get_rank()}_0.distcp"
        )
        metadata_file_path = os.path.join(self._ckpt_path, "0.metadata")
        assert os.path.exists(data_file_path) and os.path.exists(
            metadata_file_path
        )
        local_state_dict = paddle.load(data_file_path)
        metadata = paddle.load(metadata_file_path)
        for tensor_index, file_name in metadata.storage_metadata.items():
            rank = int(file_name.split(".")[0].split("_")[0])
            if rank == paddle.distributed.get_rank():
                assert (
                    tensor_index.tensor_key in state_dict
                    and tensor_index.tensor_key in local_state_dict
                )
                expect_tensor = expect_local_state_dict[tensor_index.tensor_key]
                local_tensor = local_state_dict[tensor_index.tensor_key]
                np.testing.assert_equal(
                    expect_tensor.numpy(), local_tensor.numpy()
                )
            else:
                if tensor_index.tensor_key == "w1":
                    assert (
                        tensor_index.tensor_key not in local_state_dict
                    ), f"tensor_key: {tensor_index.tensor_key} should not in local state dict"

    def run_test_case(self):
        self.test_dedup_tesnor()


if __name__ == '__main__':
    TestSaveStateDict().run_test_case()
