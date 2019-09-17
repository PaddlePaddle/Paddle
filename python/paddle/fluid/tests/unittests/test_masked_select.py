# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest


class TestMaskedSelectOp(OpTest):
    def setUp(self):
        self.op_type = "masked_select"
        shape = [13, 17]
        data = np.random.random(shape).astype("float32")
        input = np.random.random(shape).astype("float32")

        mask = data > 0.5

        npresult = input[np.where(mask)]

        self.inputs = {'input': input, 'mask': mask}
        self.outputs = {'Out': npresult}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['input', 'mask'], 'Out', max_relative_error=0.005)


class TestMaskedSelectOp_broadcast_3d(TestMaskedSelectOp):
    def setUp(self):
        self.op_type = "masked_select"
        shape = [11, 13, 17]
        data = np.random.random(shape).astype("float32")
        input = np.random.random(shape).astype("float32")

        mask = data > 0.5

        npresult = input[np.where(mask)]

        self.inputs = {'input': input, 'mask': mask}
        self.outputs = {'Out': npresult}


class TestMaskedSelectOp_broadcast_3d_axis_0(TestMaskedSelectOp):
    def setUp(self):
        self.op_type = "masked_select"
        shape = [11, 13, 17]
        broadcast_shape = [1, 13, 17]
        data = np.random.random(broadcast_shape).astype("float32")
        input = np.random.random(shape).astype("float32")

        mask = data > 0.5

        broadcast_mask = np.tile(mask, (11, 1, 1))

        npresult = input[np.where(broadcast_mask)]

        self.inputs = {'input': input, 'mask': mask}
        self.outputs = {'Out': npresult}


class TestMaskedSelectOp_broadcast_3d_axis_1(TestMaskedSelectOp):
    def setUp(self):
        self.op_type = "masked_select"
        shape = [11, 13, 17]
        broadcast_shape = [11, 1, 17]
        data = np.random.random(broadcast_shape).astype("float32")
        input = np.random.random(shape).astype("float32")

        mask = data > 0.5

        broadcast_mask = np.tile(mask, (1, 13, 1))

        npresult = input[np.where(broadcast_mask)]

        self.inputs = {'input': input, 'mask': mask}
        self.outputs = {'Out': npresult}


class TestMaskedSelectOp_broadcast_3d_axis_2(TestMaskedSelectOp):
    def setUp(self):
        self.op_type = "masked_select"
        shape = [11, 13, 17]
        broadcast_shape = [11, 13, 1]
        data = np.random.random(broadcast_shape).astype("float32")
        input = np.random.random(shape).astype("float32")

        mask = data > 0.5

        broadcast_mask = np.tile(mask, (1, 1, 17))

        npresult = input[np.where(broadcast_mask)]

        self.inputs = {'input': input, 'mask': mask}
        self.outputs = {'Out': npresult}


class TestMaskedSelectOp_broadcast_4d_2axis(TestMaskedSelectOp):
    def setUp(self):
        self.op_type = "masked_select"
        shape = [11, 13, 17, 19]
        broadcast_shape = [1, 13, 1, 19]
        input = np.random.random(shape).astype("float32")
        data = np.random.random(broadcast_shape).astype("float32")

        mask = data > 0.5

        broadcast_mask = np.tile(mask, (11, 1, 17, 1))

        npresult = input[np.where(broadcast_mask)]

        self.inputs = {'input': input, 'mask': mask}
        self.outputs = {'Out': npresult}


class TestMaskedSelectOp_broadcast_4d_both(TestMaskedSelectOp):
    def setUp(self):
        self.op_type = "masked_select"
        shape = [11, 1, 17, 1]
        broadcast_shape = [1, 13, 1, 19]
        input = np.random.random(shape).astype("float32")
        data = np.random.random(broadcast_shape).astype("float32")

        mask = data > 0.5

        broadcast_input = np.tile(input, (1, 13, 1, 19))
        broadcast_mask = np.tile(mask, (11, 1, 17, 1))

        npresult = input[np.where(broadcast_mask)]

        self.inputs = {'input': input, 'mask': mask}
        self.outputs = {'Out': npresult}


class TestMaskedSelectOp_broadcast_4d_default(TestMaskedSelectOp):
    def setUp(self):
        self.op_type = "masked_select"
        shape = [11, 1, 17, 1]
        broadcast_shape = [13, 1, 19]
        input = np.random.random(shape).astype("float32")
        data = np.random.random(broadcast_shape).astype("float32")

        mask = data > 0.5

        broadcast_input = np.tile(input, (1, 13, 1, 19))
        broadcast_mask = np.tile(mask, (11, 1, 17, 1))

        npresult = input[np.where(broadcast_mask)]

        self.inputs = {'input': input, 'mask': mask}
        self.outputs = {'Out': npresult}
