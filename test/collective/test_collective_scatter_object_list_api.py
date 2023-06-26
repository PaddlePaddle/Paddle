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

import legacy_test.test_collective_api_base as test_base


class TestCollectiveScatterObjectListAPI(test_base.TestDistBase):
    def _setup_config(self):
        pass

    def test_scatter_nccl(self):
        self.check_with_place(
            "collective_scatter_object_list_api_dygraph.py",
            "scatter_object_list",
            "nccl",
            static_mode="0",
            dtype="pyobject",
        )

    def test_scatter_gloo_dygraph(self):
        self.check_with_place(
            "collective_scatter_object_list_api_dygraph.py",
            "scatter_object_list",
            "gloo",
            "3",
            static_mode="0",
            dtype="pyobject",
        )


if __name__ == '__main__':
    unittest.main()
