#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

paddle.enable_static()


class TestCollectiveAllgatherAPI(TestDistBase):

    def _setup_config(self):
        pass

    def test_allgather_nccl(self):
        dtypes_to_test = [
            "float16", "float32", "float64", "int32", "int64", "int8", "uint8",
            "bool", "complex64", "complex128"
        ]
        for dtype in dtypes_to_test:
            self.check_with_place("collective_allgather_api.py",
                                  "allgather",
                                  "nccl",
                                  dtype=dtype)

    def test_allgather_gloo(self):
        dtypes_to_test = [
            "float16", "float32", "float64", "int32", "int64", "int8", "uint8",
            "bool", "complex64", "complex128"
        ]
        for dtype in dtypes_to_test:
            self.check_with_place("collective_allgather_api.py",
                                  "allgather",
                                  "gloo",
                                  "3",
                                  dtype=dtype)

    def test_allgatther_nccl_dygraph(self):
        dtypes_to_test = [
            "float16", "float32", "float64", "int32", "int64", "int8", "uint8",
            "bool", "complex64", "complex128"
        ]
        if self._nccl_version >= 2100:
            dtypes_to_test.append("bfloat16")
        for dtype in dtypes_to_test:
            self.check_with_place("collective_allgather_api_dygraph.py",
                                  "allgather",
                                  "nccl",
                                  static_mode="0",
                                  dtype=dtype)

    def test_allgather_gloo_dygraph(self):
        dtypes_to_test = [
            "float16", "float32", "float64", "int32", "int64", "int8", "uint8",
            "bool", "bfloat16", "complex64", "complex128"
        ]
        for dtype in dtypes_to_test:
            self.check_with_place("collective_allgather_api_dygraph.py",
                                  "allgather",
                                  "gloo",
                                  "3",
                                  static_mode="0",
                                  dtype=dtype)


if __name__ == "__main__":
    unittest.main()
