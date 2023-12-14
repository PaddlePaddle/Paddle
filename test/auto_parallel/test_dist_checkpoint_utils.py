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
from paddle.distributed.checkpoint.utils import (
    flatten_state_dict,
    unflatten_state_dict,
)


class TestDistCheckpointUtils(unittest.TestCase):
    def test_flatten_state_dict(self):
        state_dict = {
            "model": {
                "a.0": paddle.to_tensor([1, 2]),
                "b": paddle.to_tensor([3, 4]),
            },
            "optimizer": {
                "c": paddle.to_tensor([5, 6]),
                "d.2": paddle.to_tensor([7, 8]),
            },
        }
        expected_flat_state_dict = {
            "model.a.0": paddle.to_tensor([1, 2]),
            "model.b": paddle.to_tensor([3, 4]),
            "optimizer.c": paddle.to_tensor([5, 6]),
            "optimizer.d.2": paddle.to_tensor([7, 8]),
        }
        flat_state_dict, mapping = flatten_state_dict(state_dict)
        self.assertTrue(len(expected_flat_state_dict) == len(flat_state_dict))
        for k, v in flat_state_dict.items():
            self.assertTrue(isinstance(v, paddle.Tensor))
            self.assertTrue(k in expected_flat_state_dict)
            np.testing.assert_equal(
                v.numpy(), expected_flat_state_dict[k].numpy()
            )
        recover_state_dict = unflatten_state_dict(flat_state_dict, mapping)

        def check_state_dict(d1, d2):
            self.assertTrue(len(d1) == len(d2))
            self.assertTrue(type(d1) == type(d2))
            if isinstance(d1, dict):
                for k in d1:
                    self.assertTrue(k in d2)
                    check_state_dict(d1[k], d2[k])
            elif isinstance(d1, paddle.Tensor):
                np.testing.assert_equal(d1.numpy(), d2.numpy())
            else:
                raise ValueError(f"Invalid type of state_dict:{d1} != {d2}")

        check_state_dict(recover_state_dict, state_dict)


if __name__ == "__main__":
    unittest.main()
