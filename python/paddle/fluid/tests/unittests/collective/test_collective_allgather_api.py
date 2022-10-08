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

import unittest
import paddle

from test_collective_api_base import TestDistBase

paddle.enable_static()


class TestCollectiveAllgatherAPI(TestDistBase):

    def _setup_config(self):
        pass

    def test_allgather_nccl(self):
        self.check_with_place("collective_allgather_api.py",
                              "allgather",
                              "nccl",
                              dtype="float16")
        self.check_with_place("collective_allgather_api.py",
                              "allgather",
                              "nccl",
                              dtype="float32")
        self.check_with_place("collective_allgather_api.py",
                              "allgather",
                              "nccl",
                              dtype="float64")
        self.check_with_place("collective_allgather_api.py",
                              "allgather",
                              "nccl",
                              dtype="bool")
        self.check_with_place("collective_allgather_api.py",
                              "allgather",
                              "nccl",
                              dtype="uint8")
        self.check_with_place("collective_allgather_api.py",
                              "allgather",
                              "nccl",
                              dtype="int8")
        self.check_with_place("collective_allgather_api.py",
                              "allgather",
                              "nccl",
                              dtype="int32")
        self.check_with_place("collective_allgather_api.py",
                              "allgather",
                              "nccl",
                              dtype="int64")
        self.check_with_place("collective_allgather_api.py",
                              "allgather",
                              "nccl",
                              dtype="complex64")
        self.check_with_place("collective_allgather_api.py",
                              "allgather",
                              "nccl",
                              dtype="complex128")

    def test_allgather_gloo(self):
        self.check_with_place("collective_allgather_api.py",
                              "allgather",
                              "gloo",
                              "3",
                              dtype="float16")
        self.check_with_place("collective_allgather_api.py",
                              "allgather",
                              "gloo",
                              "3",
                              dtype="float32")
        self.check_with_place("collective_allgather_api.py",
                              "allgather",
                              "gloo",
                              "3",
                              dtype="float64")
        self.check_with_place("collective_allgather_api.py",
                              "allgather",
                              "gloo",
                              "3",
                              dtype="bool")
        self.check_with_place("collective_allgather_api.py",
                              "allgather",
                              "gloo",
                              "3",
                              dtype="uint8")
        self.check_with_place("collective_allgather_api.py",
                              "allgather",
                              "gloo",
                              "3",
                              dtype="int8")
        self.check_with_place("collective_allgather_api.py",
                              "allgather",
                              "gloo",
                              "3",
                              dtype="int32")
        self.check_with_place("collective_allgather_api.py",
                              "allgather",
                              "gloo",
                              "3",
                              dtype="int64")
        self.check_with_place("collective_allgather_api.py",
                              "allgather",
                              "gloo",
                              "3",
                              dtype="complex64")
        self.check_with_place("collective_allgather_api.py",
                              "allgather",
                              "gloo",
                              "3",
                              dtype="complex128")

    def test_allgatther_nccl_dygraph(self):
        self.check_with_place("collective_allgather_api_dygraph.py",
                              "allgather",
                              "nccl",
                              static_mode="0",
                              dtype="float16")
        self.check_with_place("collective_allgather_api_dygraph.py",
                              "allgather",
                              "nccl",
                              static_mode="0",
                              dtype="float32")
        self.check_with_place("collective_allgather_api_dygraph.py",
                              "allgather",
                              "nccl",
                              static_mode="0",
                              dtype="float64")
        self.check_with_place("collective_allgather_api_dygraph.py",
                              "allgather",
                              "nccl",
                              static_mode="0",
                              dtype="bool")
        self.check_with_place("collective_allgather_api_dygraph.py",
                              "allgather",
                              "nccl",
                              static_mode="0",
                              dtype="uint8")
        self.check_with_place("collective_allgather_api_dygraph.py",
                              "allgather",
                              "nccl",
                              static_mode="0",
                              dtype="int8")
        self.check_with_place("collective_allgather_api_dygraph.py",
                              "allgather",
                              "nccl",
                              static_mode="0",
                              dtype="int32")
        self.check_with_place("collective_allgather_api_dygraph.py",
                              "allgather",
                              "nccl",
                              static_mode="0",
                              dtype="int64")
        self.check_with_place("collective_allgather_api_dygraph.py",
                              "allgather",
                              "nccl",
                              static_mode="0",
                              dtype="complex64")
        self.check_with_place("collective_allgather_api_dygraph.py",
                              "allgather",
                              "nccl",
                              static_mode="0",
                              dtype="complex128")

    def test_allgather_gloo_dygraph(self):
        self.check_with_place("collective_allgather_api_dygraph.py",
                              "allgather",
                              "gloo",
                              "3",
                              static_mode="0",
                              dtype="float16")
        self.check_with_place("collective_allgather_api_dygraph.py",
                              "allgather",
                              "gloo",
                              "3",
                              static_mode="0",
                              dtype="float32")
        self.check_with_place("collective_allgather_api_dygraph.py",
                              "allgather",
                              "gloo",
                              "3",
                              static_mode="0",
                              dtype="float64")
        self.check_with_place("collective_allgather_api_dygraph.py",
                              "allgather",
                              "gloo",
                              "3",
                              static_mode="0",
                              dtype="bool")
        self.check_with_place("collective_allgather_api_dygraph.py",
                              "allgather",
                              "gloo",
                              "3",
                              static_mode="0",
                              dtype="uint8")
        self.check_with_place("collective_allgather_api_dygraph.py",
                              "allgather",
                              "gloo",
                              "3",
                              static_mode="0",
                              dtype="int8")
        self.check_with_place("collective_allgather_api_dygraph.py",
                              "allgather",
                              "gloo",
                              "3",
                              static_mode="0",
                              dtype="int32")
        self.check_with_place("collective_allgather_api_dygraph.py",
                              "allgather",
                              "gloo",
                              "3",
                              static_mode="0",
                              dtype="int64")
        self.check_with_place("collective_allgather_api_dygraph.py",
                              "allgather",
                              "gloo",
                              "3",
                              static_mode="0",
                              dtype="complex64")
        self.check_with_place("collective_allgather_api_dygraph.py",
                              "allgather",
                              "gloo",
                              "3",
                              static_mode="0",
                              dtype="complex128")


if __name__ == '__main__':
    unittest.main()
