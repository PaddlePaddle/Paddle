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

import paddle
import paddle.distributed as dist
from paddle.distributed import save_state_dict


def get_global_state_dict():
    w1 = paddle.arange(104).reshape([13, 8])
    w2 = paddle.arange(32, 36).reshape([2, 2])
    return {"w1": w1, "w2": w2}


def check_structure_name_mapping(ckpt_path, state_dict):
    metadata_file_path = os.path.join(ckpt_path, "0.metadata")
    data_file_path = os.path.join(
        ckpt_path, f"{paddle.distributed.get_rank()}_0.distcp"
    )
    assert os.path.exists(
        metadata_file_path
    ), f"metadata file {metadata_file_path} is not found"
    assert os.path.exists(
        data_file_path
    ), f"data file {data_file_path} is not found"
    metadata = paddle.load(metadata_file_path)
    cur_rank_state_dict = paddle.load(data_file_path, keep_name_table=True)
    local_structure_name_mapping = cur_rank_state_dict.pop(
        "StructuredToParameterName@@"
    )
    assert isinstance(
        local_structure_name_mapping, dict
    ), f"local_structure_name_mapping:{local_structure_name_mapping} is not dict type"
    for structure_name, param_name in local_structure_name_mapping.items():
        assert (
            structure_name in state_dict
        ), f"tensor key:{structure_name} is not found in state dict:{state_dict}"
        assert (
            param_name == state_dict[structure_name].name
        ), f"param name:{param_name} is not equal to param name in state_dict:{state_dict[structure_name].name}"


class TestSaveStateDict:
    def __init__(self):
        self._ckpt_path = os.getenv("ckpt_path")

    def test_save_state_dict_with_one_device(self):
        global_state_dict = get_global_state_dict()
        keys = list(global_state_dict.keys())
        w1, w2 = list(global_state_dict.values())
        state_dict = dict(zip(keys, [w1, w2]))
        save_state_dict(state_dict, self._ckpt_path)
        check_structure_name_mapping(self._ckpt_path, state_dict)

    def test_save_state_dict_with_four_devices(self):
        global_state_dict = get_global_state_dict()
        keys = list(global_state_dict.keys())
        w1, w2 = list(global_state_dict.values())
        mesh = dist.ProcessMesh([0, 1])
        mesh2 = dist.ProcessMesh([2, 3])
        sharded_w1 = dist.shard_tensor(w1, mesh, [dist.Shard(0)])
        sharded_w2 = dist.shard_tensor(w2, mesh2, [dist.Shard(0)])
        state_dict = dict(zip(keys, [sharded_w1, sharded_w2]))
        save_state_dict(state_dict, self._ckpt_path)
        paddle.distributed.barrier()
        check_structure_name_mapping(self._ckpt_path, state_dict)

    def run_test_case(self):
        device_num = int(os.getenv("device_num"))
        if device_num == 1:
            self.test_save_state_dict_with_one_device()
        elif device_num == 4:
            self.test_save_state_dict_with_four_devices()


if __name__ == "__main__":
    TestSaveStateDict().run_test_case()
