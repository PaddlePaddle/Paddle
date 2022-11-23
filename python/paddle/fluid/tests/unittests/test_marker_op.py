# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest
from paddle.distributed.fleet.meta_optimizers.common import OpRole


class TestMarkerOp(OpTest):

    def setUp(self):
        self.op_type = "marker"
        self.inputs = {}
        self.attrs = {
            'marker_role': 'forward',
            'marker_pos': 'B',
            'op_role': OpRole.Forward
        }
        self.outputs = {}

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    unittest.main()
