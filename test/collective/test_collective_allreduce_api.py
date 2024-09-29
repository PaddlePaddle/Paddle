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

from test_collective_api_base import TestDistBase

import paddle

paddle.enable_static()
import paddle.distributed as dist


class TestCollectiveAllreduceAPI(TestDistBase):
    def _setup_config(self):
        pass

    def test_allreduce_nccl(self):
        if paddle.base.core.is_compiled_with_cuda():
            self.check_with_place(
                "collective_allreduce_api.py", "allreduce", "nccl"
            )

    def test_allreduce_nccl_with_comm_context(self):
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
        red_types_to_test = [
            dist.ReduceOp.SUM,
        ]
        if self._nccl_version >= 21000:
            dtypes_to_test.append("bfloat16")
        for dtype in dtypes_to_test:
            for red_type in red_types_to_test:
                self.check_with_place(
                    "collective_allreduce_api.py",
                    "allreduce",
                    "nccl",
                    dtype=dtype,
                    reduce_type=red_type,
                    need_envs={"USE_COMM_CONTEXT": "1"},
                )

    def test_allreduce_nccl_with_new_comm(self):
        dtypes_to_test = [
            "float16",
            "float32",
            "float64",
            "int32",
            "int64",
        ]
        red_types_to_test = [
            dist.ReduceOp.SUM,
        ]
        if self._nccl_version >= 21000:
            dtypes_to_test.append("bfloat16")
        for dtype in dtypes_to_test:
            for red_type in red_types_to_test:
                self.check_with_place(
                    "collective_allreduce_api.py",
                    "allreduce",
                    "nccl",
                    dtype=dtype,
                    reduce_type=red_type,
                )

    def test_allreduce_bkcl(self):
        if paddle.base.core.is_compiled_with_xpu():
            self.check_with_place(
                "collective_allreduce_api.py", "allreduce", "bkcl"
            )

    def test_allreduce_gloo(self):
        self.check_with_place(
            "collective_allreduce_api.py", "allreduce", "gloo", "2"
        )

    def test_allreduce_gloo_with_comm_context(self):
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
        red_types_to_test = [
            dist.ReduceOp.SUM,
        ]
        for dtype in dtypes_to_test:
            for red_type in red_types_to_test:
                self.check_with_place(
                    "collective_allreduce_api.py",
                    "allreduce",
                    "gloo",
                    "2",
                    dtype=dtype,
                    reduce_type=red_type,
                    need_envs={"USE_COMM_CONTEXT": "1"},
                )

    def test_allreduce_nccl_dygraph(self):
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
                "collective_allreduce_api_dygraph.py",
                "allreduce",
                "nccl",
                static_mode="0",
                dtype=dtype,
            )

    def test_allreduce_gloo_dygraph(self):
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
                "collective_allreduce_api_dygraph.py",
                "allreduce",
                "gloo",
                "2",
                static_mode="0",
                dtype=dtype,
            )


if __name__ == "__main__":
    unittest.main()
