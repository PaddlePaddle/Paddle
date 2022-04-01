#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import sys
import unittest
import numpy as np
import paddle

from test_collective_base_mlu import TestDistBase

paddle.enable_static()


class TestCBroadcastOp(TestDistBase):
    def _setup_config(self):
        pass

    def test_broadcast_fp32(self):
        self.check_with_place("collective_broadcast_op.py", "broadcast",
                              "float32")

    def test_broadcast_fp16(self):
        self.check_with_place("collective_broadcast_op.py", "broadcast",
                              "float16")

    def test_broadcast_int32(self):
        self.check_with_place("collective_broadcast_op.py", "broadcast",
                              "int32")

    def test_broadcast_int16(self):
        self.check_with_place("collective_broadcast_op.py", "broadcast",
                              "int16")

    def test_broadcast_int8(self):
        self.check_with_place("collective_broadcast_op.py", "broadcast", "int8")

    def test_broadcast_uint8(self):
        self.check_with_place("collective_broadcast_op.py", "broadcast",
                              "uint8")


if __name__ == '__main__':
    unittest.main()
