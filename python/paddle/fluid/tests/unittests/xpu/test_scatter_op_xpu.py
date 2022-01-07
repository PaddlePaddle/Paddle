#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import unittest
import sys
sys.path.append("..")
from op_test import OpTest
from op_test_xpu import XPUOpTest
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core

paddle.enable_static()


class TestScatterOp(XPUOpTest):
    def setUp(self):
        self.set_xpu()
        self.op_type = "scatter"
        self.place = paddle.XPUPlace(0)

        ref_np = np.ones((3, 50)).astype("float32")
        index_np = np.array([1, 2]).astype("int32")
        updates_np = np.random.random((2, 50)).astype("float32")
        output_np = np.copy(ref_np)
        output_np[index_np] = updates_np
        self.inputs = {'X': ref_np, 'Ids': index_np, 'Updates': updates_np}
        self.outputs = {'Out': output_np}

    def set_xpu(self):
        self.__class__.use_xpu = True

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        pass


class TestScatterOp0(TestScatterOp):
    def setUp(self):
        self.set_xpu()
        self.op_type = "scatter"
        self.place = paddle.XPUPlace(0)

        ref_np = np.ones((3, 3)).astype("float32")
        index_np = np.array([1, 2]).astype("int32")
        updates_np = np.random.random((2, 3)).astype("float32")
        output_np = np.copy(ref_np)
        output_np[index_np] = updates_np
        self.inputs = {'X': ref_np, 'Ids': index_np, 'Updates': updates_np}
        self.attrs = {'overwrite': True}
        self.outputs = {'Out': output_np}


class TestScatterOp1(TestScatterOp):
    def setUp(self):
        self.set_xpu()
        self.op_type = "scatter"
        self.place = paddle.XPUPlace(0)

        ref_np = np.ones((3, 3)).astype("float32")
        zeros_np = np.zeros([2, 3]).astype('float32')
        index_np = np.array([1, 1]).astype("int32")
        updates_np = np.random.random((2, 3)).astype("float32")
        output_np = np.copy(ref_np)
        output_np[index_np] = zeros_np
        for i in range(0, len(index_np)):
            output_np[index_np[i]] += updates_np[i]
        self.attrs = {'overwrite': False}
        self.inputs = {'X': ref_np, 'Ids': index_np, 'Updates': updates_np}
        self.outputs = {'Out': output_np}


class TestScatterOp2(TestScatterOp):
    def setUp(self):
        self.set_xpu()
        self.op_type = "scatter"
        self.place = paddle.XPUPlace(0)

        ref_np = np.ones((3, 3)).astype("float32")
        index_np = np.array([1, 2]).astype("int32")
        updates_np = np.random.random((2, 3)).astype("float32")
        output_np = np.copy(ref_np)
        output_np[index_np] = updates_np
        self.inputs = {'X': ref_np, 'Ids': index_np, 'Updates': updates_np}
        self.outputs = {'Out': output_np}


class TestScatterOp3(TestScatterOp):
    def setUp(self):
        self.set_xpu()
        self.op_type = "scatter"
        self.place = paddle.XPUPlace(0)

        ref_np = np.ones((3, 3)).astype("float32")
        zeros_np = np.zeros([2, 3]).astype('float32')
        index_np = np.array([1, 1]).astype("int32")
        updates_np = np.random.random((2, 3)).astype("float32")
        output_np = np.copy(ref_np)
        output_np[index_np] = zeros_np
        for i in range(0, len(index_np)):
            output_np[index_np[i]] += updates_np[i]
        self.attrs = {'overwrite': False}
        self.inputs = {'X': ref_np, 'Ids': index_np, 'Updates': updates_np}
        self.outputs = {'Out': output_np}


class TestScatterOp4(TestScatterOp):
    def setUp(self):
        self.set_xpu()
        self.op_type = "scatter"
        self.place = paddle.XPUPlace(0)

        ref_np = np.ones((3, 3)).astype("float32")
        index_np = np.array([1, 2]).astype("int64")
        updates_np = np.random.random((2, 3)).astype("float32")
        output_np = np.copy(ref_np)
        output_np[index_np] = updates_np
        self.inputs = {'X': ref_np, 'Ids': index_np, 'Updates': updates_np}
        self.outputs = {'Out': output_np}


class TestScatterOp5(TestScatterOp):
    def setUp(self):
        self.set_xpu()
        self.op_type = "scatter"
        self.place = paddle.XPUPlace(0)

        ref_np = np.ones((3, 3)).astype("float32")
        index_np = np.array([1, 2]).astype("int64")
        updates_np = np.random.random((2, 3)).astype("float32")
        output_np = np.copy(ref_np)
        output_np[index_np] = updates_np
        self.inputs = {'X': ref_np, 'Ids': index_np, 'Updates': updates_np}
        self.outputs = {'Out': output_np}


class TestScatterOp6(TestScatterOp):
    def setUp(self):
        self.set_xpu()
        self.op_type = "scatter"
        self.place = paddle.XPUPlace(0)

        ref_np = np.ones((3, 3)).astype("int64")
        index_np = np.array([1, 2]).astype("int64")
        updates_np = np.random.random((2, 3)).astype("int64")
        output_np = np.copy(ref_np)
        output_np[index_np] = updates_np
        self.inputs = {'X': ref_np, 'Ids': index_np, 'Updates': updates_np}
        self.outputs = {'Out': output_np}


if __name__ == '__main__':
    unittest.main()
