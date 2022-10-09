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
import paddle

from test_collective_base import TestDistBase

paddle.enable_static()


class TestConcatOp(TestDistBase):

    def _setup_config(self):
        pass

    def test_concat(self, col_type="concat"):
        self.check_with_place("collective_concat_op.py", col_type)


if __name__ == '__main__':
    unittest.main()
