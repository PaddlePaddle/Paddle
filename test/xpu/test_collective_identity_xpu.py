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

from test_collective_base_xpu import TestDistBase

import paddle

paddle.enable_static()


class TestCIdentityOp(TestDistBase):
    def _setup_config(self):
        pass

    def test_identity(self):
        dtypes_to_test = [
            "float16",
            "float32",
            "float64",
            "int32",
            "int64",
            "bfloat16",
        ]
        for dtype in dtypes_to_test:
            self.check_with_place(
                "collective_identity_op_xpu.py", "identity", dtype
            )


if __name__ == '__main__':
    unittest.main()
