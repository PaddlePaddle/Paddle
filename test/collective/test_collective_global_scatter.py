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

import unittest

from legacy_test.test_collective_api_base import TestDistBase

import paddle


class TestCollectiveSelectScatterAPI(TestDistBase):
    def _setup_config(self):
        pass

    def test_global_scatter_nccl(self):
        paddle.enable_static()
        self.check_with_place(
            "collective_global_scatter.py", "global_scatter", "nccl"
        )

    def test_global_scatter_nccl_dygraph_eager(self):
        self.check_with_place(
            "collective_global_scatter_dygraph.py",
            "global_scatter",
            "nccl",
            static_mode="0",
            eager_mode=True,
        )

    def test_global_scatter_nccl_new_comm(self):
        self.check_with_place(
            "collective_global_scatter.py",
            "global_scatter",
            "nccl",
            need_envs={"FLAGS_dynamic_static_unified_comm": "1"},
        )


if __name__ == '__main__':
    unittest.main()
