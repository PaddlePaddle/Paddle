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
from test_mul_op import TestMulOp, TestMulOp2, TestFP16MulOp1, TestFP16MulOp2


class TestMKLDNNMulOp(TestMulOp):
    def init_op_test(self):
        super(TestMKLDNNMulOp, self).setUp()
        self.attrs = {"use_mkldnn": True}


class TestMKLDNNMulOp2(TestMulOp2):
    def init_op_test(self):
        super(TestMKLDNNMulOp2, self).setUp()
        self.attrs = {"use_mkldnn": True}


class TestMKLDNNFP16MulOp1(TestFP16MulOp1):
    def init_op_test(self):
        super(TestMKLDNNFP16MulOp1, self).setUp()
        self.attrs = {"use_mkldnn": True}


class TestMKLDNNFP16MulOp2(TestFP16MulOp2):
    def init_op_test(self):
        super(TestMKLDNNFP16MulOp2, self).setUp()
        self.attrs = {"use_mkldnn": True}


if __name__ == "__main__":
    unittest.main()
