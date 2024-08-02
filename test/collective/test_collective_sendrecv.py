#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from legacy_test.test_collective_base import TestDistBase

import paddle

paddle.enable_static()


class TestSendRecvOp(TestDistBase):
    def _setup_config(self):
        pass

    def test_sendrecv(self):
        self.check_with_place("collective_sendrecv_op.py", "sendrecv")

    def test_sendrecv_dynamic_shape(self):
        self.check_with_place(
            "collective_sendrecv_op_dynamic_shape.py",
            "sendrecv_dynamic_shape",
            need_envs={"FLAGS_dynamic_static_unified_comm": "0"},
        )

    def test_sendrecv_array(self):
        self.check_with_place(
            "collective_sendrecv_op_array.py",
            "sendrecv_array",
            need_envs={"FLAGS_dynamic_static_unified_comm": "0"},
        )


if __name__ == '__main__':
    unittest.main()
