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

<<<<<<< HEAD
import unittest

from test_collective_api_base import TestDistBase

import paddle

=======
from __future__ import print_function
import unittest
import numpy as np
import paddle

from test_collective_api_base import TestDistBase

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
paddle.enable_static()


class TestCollectiveAllgatherAPI(TestDistBase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def _setup_config(self):
        pass

    def test_allgather_nccl(self):
        dtypes_to_test = [
<<<<<<< HEAD
            "float16",
            "float32",
            "float64",
            "int32",
            "int64",
            "int8",
            "uint8",
            "bool",
        ]
        for dtype in dtypes_to_test:
            self.check_with_place(
                "collective_allgather_api.py", "allgather", "nccl", dtype=dtype
            )

    def test_allgather_gloo(self):
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
        for dtype in dtypes_to_test:
            self.check_with_place(
                "collective_allgather_api.py",
                "allgather",
                "gloo",
                "3",
                dtype=dtype,
            )

    def test_allgather_nccl_dygraph(self):
        dtypes_to_test = [
            "float16",
            "float32",
            "float64",
            "int32",
            "int64",
            "int8",
            "uint8",
            "bool",
=======
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
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        ]
        if self._nccl_version >= 2100:
            dtypes_to_test.append("bfloat16")
        for dtype in dtypes_to_test:
<<<<<<< HEAD
            self.check_with_place(
                "collective_allgather_api_dygraph.py",
                "allgather",
                "nccl",
                static_mode="0",
                dtype=dtype,
            )

    def test_allgather_gloo_dygraph(self):
        dtypes_to_test = [
            "float16",
            "float32",
            "float64",
            "int32",
            "int64",
            "int8",
            "uint8",
            "bool",
            "bfloat16",
        ]
        for dtype in dtypes_to_test:
            self.check_with_place(
                "collective_allgather_api_dygraph.py",
                "allgather",
                "gloo",
                "3",
                static_mode="0",
                dtype=dtype,
            )
=======
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
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


if __name__ == "__main__":
    unittest.main()
