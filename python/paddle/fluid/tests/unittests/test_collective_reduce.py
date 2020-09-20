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

from __future__ import print_function
import unittest
import numpy as np

from test_collective_base import TestDistBase


class TestCReduceOp(TestDistBase):
    def _setup_config(self):
        pass

    def test_reduce(self):
        self.check_with_place("collective_reduce_op.py", "reduce")

    def test_reduce_calc_stream(self):
        self.check_with_place("collective_reduce_op_calc_stream.py", "reduce")


if __name__ == '__main__':
    import paddle
    paddle.disable_static()
    unittest.main()
