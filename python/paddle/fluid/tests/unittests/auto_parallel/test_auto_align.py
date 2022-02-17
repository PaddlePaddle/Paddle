# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.distributed.auto_parallel.auto_align import AutoAlign
import paddle
import os

loss_dict = {
    0: {
        'loss': np.array(
            [10], dtype=float)
    },
    1: {
        'loss': np.array(
            [10], dtype=float)
    },
    2: {
        'loss': np.array(
            [10], dtype=float)
    },
    3: {
        'loss': np.array(
            [10], dtype=float)
    },
    4: {
        'loss': np.array(
            [10], dtype=float)
    },
    5: {
        'loss': np.array(
            [108], dtype=float)
    },
    6: {
        'loss': np.array(
            [10], dtype=float)
    },
    7: {
        'loss': np.array(
            [10], dtype=float)
    },
    8: {
        'loss': np.array(
            [10], dtype=float)
    },
    9: {
        'loss': np.array(
            [10], dtype=float)
    },
    10: {
        'loss': np.array(
            [10], dtype=float)
    },
    11: {
        'loss': np.array(
            [10], dtype=float)
    },
    12: {
        'loss': np.array(
            [10], dtype=float)
    },
    13: {
        'loss': np.array(
            [10], dtype=float)
    },
    14: {
        'loss': np.array(
            [10], dtype=float)
    },
    15: {
        'loss': np.array(
            [10], dtype=float)
    },
    16: {
        'loss': np.array(
            [10], dtype=float)
    },
    17: {
        'loss': np.array(
            [10], dtype=float)
    },
    18: {
        'loss': np.array(
            [10], dtype=float)
    },
    19: {
        'loss': np.array(
            [10], dtype=float)
    }
}
auto_loss_dict0 = {
    10: {
        'loss': np.array(
            [10], dtype=float)
    },
    11: {
        'loss': np.array(
            [10], dtype=float)
    },
    12: {
        'loss': np.array(
            [10], dtype=float)
    },
    13: {
        'loss': np.array(
            [10], dtype=float)
    },
    14: {
        'loss': np.array(
            [10], dtype=float)
    },
    15: {
        'loss': np.array(
            [10], dtype=float)
    },
    16: {
        'loss': np.array(
            [10], dtype=float)
    },
    17: {
        'loss': np.array(
            [10], dtype=float)
    },
    18: {
        'loss': np.array(
            [10], dtype=float)
    },
    19: {
        'loss': np.array(
            [10], dtype=float)
    }
}
auto_loss_dict1 = {
    10: {
        'loss': np.array(
            [10], dtype=float)
    },
    11: {
        'loss': np.array(
            [10], dtype=float)
    },
    12: {
        'loss': np.array(
            [10], dtype=float)
    },
    13: {
        'loss': np.array(
            [10], dtype=float)
    },
    14: {
        'loss': np.array(
            [10], dtype=float)
    },
    15: {
        'loss': np.array(
            [10], dtype=float)
    },
    16: {
        'loss': np.array(
            [10], dtype=float)
    },
    17: {
        'loss': np.array(
            [10], dtype=float)
    },
    18: {
        'loss': np.array(
            [10], dtype=float)
    },
    19: {
        'loss': np.array(
            [10], dtype=float)
    }
}


class TestAutoAlign(unittest.TestCase):
    def test_save_serial_tensors(self):
        auto_align = AutoAlign()
        auto_align.save_serial_tensors(loss_dict)
        self.assertIsNotNone(os.path.exists("tensors.pkl"))

    def test_save_dist_tensors(self):
        auto_align = AutoAlign()
        auto_align.save_dist_tensors(auto_loss_dict0, 0)
        auto_align.save_dist_tensors(auto_loss_dict1, 1)

        self.assertIsNotNone(os.path.exists("tensors_0.pkl"))
        self.assertIsNotNone(os.path.exists("tensors_1.pkl"))

    def test_check_loss(self):
        if os.path.exists("tensors_0.pkl") and os.path.exists("tensors.pkl"):
            auto_align = AutoAlign()
            dist_tensor_file_path_list = ["tensors_0.pkl", "tensors_1.pkl"]
            res = auto_align.check_loss(dist_tensor_file_path_list,
                                        "tensors.pkl")[0]
            self.assertTrue(res)


if __name__ == '__main__':
    unittest.main()
