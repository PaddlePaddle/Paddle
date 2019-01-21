#	Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import unittest
from paddle.fluid.tests.unittests.test_elementwise_add_op import *


class TestNGRAPHElementwiseAddOp(TestElementwiseAddOp):
    def init_input_output(self):
        super(TestNGRAPHElementwiseAddOp, self).init_input_output()


class TestNGRAPHElementwiseAddOp_scalar(TestElementwiseAddOp_scalar):
    def init_input_output(self):
        super(TestNGRAPHElementwiseAddOp_scalar, self).init_input_output()


class TestNGRAPHElementwiseAddOp_scalar2(TestElementwiseAddOp_scalar2):
    def init_input_output(self):
        super(TestNGRAPHElementwiseAddOp_scalar2, self).init_input_output()


class TestNGRAPHElementwiseAddOp_Vector(TestElementwiseAddOp_Vector):
    def init_input_output(self):
        super(TestNGRAPHElementwiseAddOp_Vector, self).init_input_output()


class TesNGRAPHtElementwiseAddOp_broadcast_0(TestElementwiseAddOp_broadcast_0):
    def init_input_output(self):
        super(TesNGRAPHtElementwiseAddOp_broadcast_0, self).init_input_output()


class TestNGRAPHElementwiseAddOp_broadcast_1(TestElementwiseAddOp_broadcast_1):
    def init_input_output(self):
        super(TestNGRAPHElementwiseAddOp_broadcast_1, self).init_input_output()


class TestNGRAPHElementwiseAddOp_broadcast_2(TestElementwiseAddOp_broadcast_2):
    def init_input_output(self):
        super(TestNGRAPHElementwiseAddOp_broadcast_2, self).init_input_output()


class TestNGRAPHElementwiseAddOp_broadcast_3(TestElementwiseAddOp_broadcast_3):
    def init_input_output(self):
        super(TestNGRAPHElementwiseAddOp_broadcast_3, self).init_input_output()


class TestNGRAPHElementwiseAddOp_broadcast_4(TestElementwiseAddOp_broadcast_4):
    def init_input_output(self):
        super(TestNGRAPHElementwiseAddOp_broadcast_4, self).init_input_output()


class TestNGRAPHElementwiseAddOp_rowwise_add_0(
        TestElementwiseAddOp_rowwise_add_0):
    def init_input_output(self):
        super(TestNGRAPHElementwiseAddOp_rowwise_add_0,
              self).init_input_output()


class TestNGRAPHElementwiseAddOp_rowwise_add_1(
        TestElementwiseAddOp_rowwise_add_1):
    def init_input_output(self):
        super(TestNGRAPHElementwiseAddOp_rowwise_add_1,
              self).init_input_output()


class TestNGRAPHElementwiseAddOp_channelwise_add(
        TestElementwiseAddOp_channelwise_add):
    def init_input_output(self):
        super(TestNGRAPHElementwiseAddOp_channelwise_add,
              self).init_input_output()


if __name__ == '__main__':
    unittest.main()
