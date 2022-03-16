#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import sys
sys.path.append("..")

import paddle
from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper

paddle.enable_static()


class XPUTestGatherNd(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'gather_nd'

    class XPUTestGatherNdBase(XPUOpTest):
        def setUp(self):
            self.op_type = "gather_nd"
            self.dtype = self.in_type
            self.__class__.no_need_check_grad = True
            self.place = paddle.XPUPlace(0)
            self.init_data()

            self.inputs = {'X': self.xnp, 'Index': self.inp}
            self.outputs = {'Out': self.output, }

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def init_data(self):
            self.xnp = np.random.random((5, 20)).astype(self.in_type)
            self.inp = np.array([[], []]).astype("int32")
            self.output = np.vstack(
                (self.xnp[np.newaxis, :], self.xnp[np.newaxis, :]))

    class XPUTestGatherNdOpWithEmptyIndex1(XPUTestGatherNdBase):
        def init_data(self):
            self.xnp = np.random.random((5, 20)).astype(self.in_type)
            self.inp = np.array([[], []]).astype("int32")
            self.output = np.vstack(
                (self.xnp[np.newaxis, :], self.xnp[np.newaxis, :]))

    class XPUTestGatherNdOpWithEmptyIndex2(XPUTestGatherNdBase):
        def init_data(self):
            self.xnp = np.random.random((5, 20)).astype(self.in_type)
            self.inp = np.array([[], []]).astype("int64")
            self.output = np.vstack(
                (self.xnp[np.newaxis, :], self.xnp[np.newaxis, :]))

    class XPUTestGatherNdOpWithIndex1(XPUTestGatherNdBase):
        def init_data(self):
            self.xnp = np.random.random((5, 20)).astype(self.in_type)
            self.inp = np.array([1]).astype("int32")
            self.output = self.xnp[self.inp]

    class XPUTestGatherNdOpWithIndex2(XPUTestGatherNdBase):
        def init_data(self):
            self.xnp = np.random.random((5, 20)).astype(self.in_type)
            self.inp = np.array([1]).astype("int64")
            self.output = self.xnp[self.inp]

    class XPUTestGatherNdOpWithLowIndex1(XPUTestGatherNdBase):
        def init_data(self):
            self.xnp = np.random.uniform(0, 100, (10, 10)).astype(self.in_type)
            self.inp = np.array([[1], [2]]).astype("int32")
            self.output = self.xnp[tuple(self.inp.T)]

    class XPUTestGatherNdOpWithLowIndex2(XPUTestGatherNdBase):
        def init_data(self):
            self.xnp = np.random.uniform(0, 100, (10, 10)).astype(self.in_type)
            self.inp = np.array([1, 2]).astype("int64")
            self.output = self.xnp[tuple(self.inp.T)]

    class XPUTestGatherNdOpWithHighRankSame1(XPUTestGatherNdBase):
        def init_data(self):
            shape = (5, 2, 3, 1, 10)
            self.xnp = np.random.rand(*shape).astype(self.in_type)
            self.inp = np.vstack(
                [np.random.randint(
                    0, s, size=2) for s in shape]).T.astype("int32")
            self.output = self.xnp[tuple(self.inp.T)]

    class XPUTestGatherNdOpWithHighRankSame2(XPUTestGatherNdBase):
        def init_data(self):
            shape = (5, 2, 3, 1, 10)
            self.xnp = np.random.rand(*shape).astype(self.in_type)
            self.inp = np.vstack(
                [np.random.randint(
                    0, s, size=2) for s in shape]).T.astype("int64")
            self.output = self.xnp[tuple(self.inp.T)]

    class XPUTestGatherNdOpWithHighRankDiff1(XPUTestGatherNdBase):
        def init_data(self):
            shape = (2, 3, 4, 1, 10)
            self.xnp = np.random.rand(*shape).astype(self.in_type)
            self.inp = np.vstack(
                [np.random.randint(
                    0, s, size=200) for s in shape]).T.astype("int32")
            self.output = self.xnp[tuple(self.inp.T)]

    class XPUTestGatherNdOpWithHighRankDiff2(XPUTestGatherNdBase):
        def init_data(self):
            shape = (2, 3, 4, 1, 10)
            self.xnp = np.random.rand(*shape).astype(self.in_type)
            self.inp = np.vstack(
                [np.random.randint(
                    0, s, size=200) for s in shape]).T.astype("int64")
            self.output = self.xnp[tuple(self.inp.T)]

    class XPUTestGatherNdOpWithSameIndexAsX1(XPUTestGatherNdBase):
        def init_data(self):
            self.xnp = np.random.uniform(0, 100, (10, 10)).astype(self.in_type)
            self.inp = np.array([[1, 1], [2, 1]]).astype("int32")
            self.output = self.xnp[tuple(self.inp.T)]

    class XPUTestGatherNdOpWithSameIndexAsX2(XPUTestGatherNdBase):
        def init_data(self):
            self.xnp = np.random.uniform(0, 100, (10, 10)).astype(self.in_type)
            self.inp = np.array([[1, 1], [2, 1]]).astype("int64")
            self.output = self.xnp[tuple(self.inp.T)]

    class XPUTestGatherNdOpIndex1(XPUTestGatherNdBase):
        def init_data(self):
            self.xnp = np.random.uniform(0, 100, (10, 10)).astype(self.in_type)
            self.inp = np.array([1, 2]).astype("int32")
            self.output = self.xnp[tuple(self.inp.T)]

    class XPUTestGatherNdOpIndex2(XPUTestGatherNdBase):
        def init_data(self):
            self.xnp = np.random.uniform(0, 100, (10, 10)).astype(self.in_type)
            self.inp = np.array([1, 2]).astype("int64")
            self.output = self.xnp[tuple(self.inp.T)]


support_types = get_xpu_op_support_types('gather_nd')
for stype in support_types:
    create_test_class(globals(), XPUTestGatherNd, stype)

if __name__ == "__main__":
    unittest.main()
