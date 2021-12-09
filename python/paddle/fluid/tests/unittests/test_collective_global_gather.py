#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function
import unittest
import numpy as np
import paddle

from test_collective_api_base import TestDistBase
import os


class TestCollectiveGlobalGatherAPI(TestDistBase):
    def _setup_config(self):
        pass

    def test_global_gather_nccl(self):
        paddle.enable_static()
        self.check_with_place("collective_global_gather.py", "global_gather",
                              "nccl")

    def test_global_gather_nccl_dygraph(self):
        self.check_with_place(
            "collective_global_gather_dygraph.py",
            "global_gather",
            "nccl",
            static_mode="0")


if __name__ == '__main__':
    unittest.main()
