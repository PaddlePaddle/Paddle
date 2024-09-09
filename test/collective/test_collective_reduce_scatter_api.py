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


class TestCollectiveReduceScatterAPI(test_base.TestDistBase):
    def _setup_config(self):
        pass

    def test_reduce_scatter_nccl_with_new_comm(self):
        dtypes_to_test = [
            "float16",
            "float32",
            "float64",
            "int32",
            "int64",
        ]
        if self._nccl_version >= 21000:
            dtypes_to_test.append("bfloat16")
        for dtype in dtypes_to_test:
            self.check_with_place(
                "collective_reduce_scatter_api.py",
                "reduce_scatter",
                "nccl",
                dtype=dtype,
            )

    def test_reduce_scatter_nccl_dygraph(self):
        dtypes_to_test = [
            "float16",
            "float32",
            "float64",
            "int32",
            "int64",
            "int8",
            "uint8",
            "bool",
        ]
        if self._nccl_version >= 21000:
            dtypes_to_test.append("bfloat16")
        for dtype in dtypes_to_test:
            self.check_with_place(
                "collective_reduce_scatter_api_dygraph.py",
                "reduce_scatter",
                "nccl",
                static_mode="0",
                dtype=dtype,
            )


if __name__ == "__main__":
    unittest.main()
