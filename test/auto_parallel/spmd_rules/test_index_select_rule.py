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

# from paddle.distributed.auto_parallel.static.dist_attribute import (
#     DistTensorSpec,
#     TensorDistAttr,
# )
# from paddle.distributed.fleet import auto
# from paddle.framework import core


class TestElementwiseSPMDRule(unittest.TestCase):
    def setUp(self):
        pass

    def test_index_select_forward(self):
        # [-1, -1, -1], [0], axis=1 --> [-1, -1, -1], [0], [-1, 0, -1]
        pass

    def test_index_select_backward(self):
        # [-1, -1, -1], [0], [-1, 0, -1], axis=1 --> [-1, -1, -1], [0], [-1, 0, -1], [-1, -1, -1](partial on axis=1 with 0)
        pass


if __name__ == "__main__":
    unittest.main()
