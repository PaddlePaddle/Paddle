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


class TestSemiautoSaveLoad:
    def __init__(self):
        self._ckpt_path = os.getenv("ckpt_path")

    def test_flatten_mapping(self):
        if paddle.distributed.get_rank() == 0:
            state_dict = {
                "model": {
                    "a": paddle.to_tensor([1, 2]),
                    "b": paddle.to_tensor([3, 4]),
                },
                "optimizer": {
                    "c": paddle.to_tensor([5, 6]),
                    "d": paddle.to_tensor([7, 8]),
                },
            }
        else:
            state_dict = {
                "model": {
                    "a": paddle.to_tensor([10, 20]),
                    "b": paddle.to_tensor([30, 40]),
                },
                "optimizer": {
                    "c": paddle.to_tensor([50, 60]),
                    "d": paddle.to_tensor([70, 80]),
                },
            }
        expected_mapping = {
            "model.a": ("model", "a"),
            "model.b": ("model", "b"),
            "optimizer.c": ("optimizer", "c"),
            "optimizer.d": ("optimizer", "d"),
        }
        dist.save_state_dict(state_dict, self._ckpt_path)
        metadata_path = os.path.join(self._ckpt_path, "0.metadata")
        assert os.path.exists(metadata_path)
        metadata = paddle.load(metadata_path)
        assert len(metadata.flat_mapping) == len(
            expected_mapping
        ), f"expect {len(expected_mapping)}, but got {len(metadata.flat_mapping)}"
        for key in metadata.flat_mapping:
            assert (
                key in expected_mapping
            ), f"expect {key} in flatten_mapping, but not found"
            assert (
                metadata.flat_mapping[key] == expected_mapping[key]
            ), f"expect {metadata.flat_mapping[key]} == {expected_mapping[key]}, but not equal"

    def run_test_case(self):
        self.test_flatten_mapping()


if __name__ == '__main__':
    TestSemiautoSaveLoad().run_test_case()
