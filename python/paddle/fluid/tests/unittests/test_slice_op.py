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
import numpy as np
import sys
from op_test import OpTest

def accomplement_axes(dim, axes, starts, ends):
    s = map(int, np.zeros(dim))
    e = map(lambda x:int(x) - 1, np.zeros(dim))

    for a in axes: 
        s[a] = starts[a]
        e[a] = ends[a]

    return (s, e)

def slice_dim_0(x, starts, ends):
    return x[starts[0]:ends[0]]

def slice_dim_1(x, starts, ends):
    return slice_dim_0(x, starts, ends)[starts[-1]:ends[-1]]

def slice_dim_2(x, starts, ends):
    return slice_dim_1(x, starts, ends)[starts[-1]:ends[-1]]

def slice_dim_3(x, starts, ends):
    return slice_dim_2(x, starts, ends)[starts[-1]:ends[-1]]

def slice_dim_4(x, starts, ends):
    return slice_dim_3(x, starts, ends)[starts[-1]:ends[-1]]

def slice_dim_5(x, starts, ends):
    return slice_dim_4(x, starts, ends)[starts[-1]:ends[-1]]

def slice_dim_6(x, starts, ends):
    return slice_dim_5(x, starts, ends)[starts[-1]:ends[-1]]

def slice_dim_7(x, starts, ends):
    return slice_dim_6(x, starts, ends)[starts[-1]:ends[-1]]

def slice_dim_8(x, starts, ends):
    return slice_dim_7(x, starts, ends)[starts[-1]:ends[-1]]

def slice_dim_9(x, starts, ends):
    return slice_dim_8(x, starts, ends)[starts[-1]:ends[-1]]

global slices = [slice_dim_0, slice_dim_1, slice_dim_2, \
                 slice_dim_3, slice_dim_4, slice_dim_5, \
                 slice_dim_6, slice_dim_7, slice_dim_8, \
                 slice_dim_9]

class TestSliceOp(OpTest):
    def set_data(self):
        self.init_test_case()
        x = np.random.random(self.x_dim).astype('float32')
        axes = np.array(self.axes).astype("int64")
        starts = np.array(self.starts).astype("int64")
        ends = np.array(self.ends).astype("int64")

        self.inputs = {'X': x, 'Axes': axes, 'Starts': starts, 'Ends': ends}

        if x_dim > 10:
            raise Exception("x_dim must be less than 10!")

        s, e = accomplement_axes(x_dim, axes, starts, ends)
        outs = slices[x_dim - 1](x, s, e)
        self.outputs = {'Out': outs}

    def init_test_case(self):
        self.x_dim = (100, 4, 5, 6)
        self.axes = [0, 3, 1, 2]
        self.starts = [3, 0, -4, 10000]
        self.ends = [-1, 2, 4, -1]

    def setUp(self):
        self.op_type = "slice"
        self.set_data()

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        #self.check_grad(['X'], 'Out')


if __name__ == '__main__':
    unittest.main()
