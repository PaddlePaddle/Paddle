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

import unittest
import numpy as np
import sys
from op_test import OpTest

def accomplement_axes(dims, axes, starts, ends):
    s = map(int, np.zeros(len(dims)))
    e = map(lambda x:dims[int(x)], range(len(dims)))

    for i, a in enumerate(axes):
        s[a] = starts[i]
        e[a] = ends[i]

    return (s, e)

def slice_dim_0(x, starts, ends):
    return x[starts[0]:ends[0]]

def slice_dim_1(x, starts, ends):
    return slice_dim_0(x, starts, ends)[:,starts[1]:ends[1]]

def slice_dim_2(x, starts, ends):
    return slice_dim_1(x, starts, ends)[:,:,starts[2]:ends[2]]

def slice_dim_3(x, starts, ends):
    return slice_dim_2(x, starts, ends)[:,:,:,starts[3]:ends[3]]

def slice_dim_4(x, starts, ends):
    return slice_dim_3(x, starts, ends)[:,:,:,:,starts[4]:ends[4]]

def slice_dim_5(x, starts, ends):
    return slice_dim_4(x, starts, ends)[:,:,:,:,:,starts[5]:ends[5]]

def slice_dim_6(x, starts, ends):
    return slice_dim_5(x, starts, ends)[:,:,:,:,:,:,starts[6]:ends[6]]

def slice_dim_7(x, starts, ends):
    return slice_dim_6(x, starts, ends)[:,:,:,:,:,:,:,starts[7]:ends[7]]

def slice_dim_8(x, starts, ends):
    return slice_dim_7(x, starts, ends)[:,:,:,:,:,:,:,:,starts[8]:ends[8]]

def slice_dim_9(x, starts, ends):
    return slice_dim_8(x, starts, ends)[:,:,:,:,:,:,:,:,:,starts[9]:ends[9]]

slices = [slice_dim_0, slice_dim_1, slice_dim_2, \
          slice_dim_3, slice_dim_4, slice_dim_5, \
          slice_dim_6, slice_dim_7, slice_dim_8, \
          slice_dim_9]

class TestSliceOp(OpTest):
    def set_data(self):
        self.axes = []
        self.init_test_case()
        x = np.random.random(self.x_dim).astype('float32')
        starts = np.array(self.starts).astype("int64")
        ends = np.array(self.ends).astype("int64")

        if self.axes:
            axes = np.array(self.axes).astype("int64")
            self.inputs = {'X': x, 'Axes': axes, 'Starts': starts, 'Ends': ends}
        else:
            axes = np.array(range(len(self.x_dim))).astype("int64")
            self.inputs = {'X': x, 'Starts': starts, 'Ends': ends}

        if len(self.x_dim) > 10:
            raise Exception("x_dim must be less than 10!")

        s, e = accomplement_axes(self.x_dim, axes, starts, ends)
        outs = slices[len(self.x_dim) - 1](x, s, e)
        #print 'x: %s, s: %s, e: %s, out: %s ' % (x, s, e, outs)
        self.outputs = {'Out': outs}

    def init_test_case(self):
        self.x_dim =  [2, 4]
        self.axes =   [0, 1]
        self.starts = [1, 0]
        self.ends =   [2, 3]

    def setUp(self):
        self.op_type = "slice"
        self.set_data()

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        #self.check_grad(['X'], 'Out')
        pass

class TestSliceOpDefaultOneAxe(TestSliceOp):
    def init_test_case(self):
        self.x_dim =  [ 20, 10, 5]
        self.axes =   [ 0, 1]
        self.starts = [ 0, 0]
        self.ends =   [ 3, 10]


class TestSliceOpDefaultAxes(TestSliceOp):
    def init_test_case(self):
        self.x_dim =  [ 20, 10, 5]
        self.starts = [ 0, 0, 3]
        self.ends =   [ 20, 10, 4]


class TestSliceOpEndOutOfBounds(TestSliceOp):
    def init_test_case(self):
        self.x_dim =  [ 20, 10, 5]
        self.axes = [1]
        self.starts = [1]
        self.ends = [ 1000]


class TestSliceOpNeg(TestSliceOp):
    def init_test_case(self):
        self.x_dim =  [ 20, 10, 5]
        self.axes = [1]
        self.starts = [0]
        self.ends = [-1]


class TestSliceOpStartOutOfBound(TestSliceOp):
    def init_test_case(self):
        self.x_dim =  [ 20, 10, 5]
        self.axes = [1]
        self.starts = [1000]
        self.ends = [1000]


if __name__ == '__main__':
    unittest.main()
