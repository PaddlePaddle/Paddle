# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import os
import unittest

os.environ["FLAGS_use_cinn"] = "True"

import paddle
import paddle.nn.functional as F


class PrimeNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        res = F.softmax(x)
        return res


class TestCINNFLAG(unittest.TestCase):
    """
    Make sure that FLAGS_use_cinn goes into effect indeed.
    """

    def test_cinn_flag(self):
        """only cinn and test FLAGS_use_cinn"""
        x = paddle.randn([2, 4])
        net = PrimeNet()
        net = paddle.jit.to_static(net)
        _ = net(x)
        self.assertTrue(
            paddle.device.is_run_with_cinn(),
            msg="The test was not running with CINN! Please check.",
        )
        return


if __name__ == '__main__':
    unittest.main()
