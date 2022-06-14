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
import paddle
import paddle.fluid.core as core
import sys
sys.path.append("..")
from op_test_xpu import XPUOpTest
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard
import time

paddle.enable_static()
"""
@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 'core is not compiled with XPU')
class TestOneHotOp(XPUOpTest):
    def setUp(self):
        self.use_xpu = True
        self.op_type = 'one_hot'
        depth = 10
        depth_np = np.array(10).astype('int32')
        x_lod = [[4, 1, 3, 3]]
        x = [np.random.randint(0, depth - 1) for i in range(sum(x_lod[0]))]
        x = np.array(x).astype('int32').reshape([sum(x_lod[0]), 1])

        out = np.zeros(shape=(np.product(x.shape[:-1]),
                              depth)).astype('float32')

        for i in range(np.product(x.shape)):
            out[i, x[i]] = 1.0

        self.inputs = {'X': (x, x_lod), 'depth_tensor': depth_np}
        self.attrs = {'dtype': int(core.VarDesc.VarType.FP32)}
        self.outputs = {'Out': (out, x_lod)}

    def test_check_output(self):
        place = paddle.XPUPlace(0)
        self.check_output_with_place(place, check_dygraph=False)


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 'core is not compiled with XPU')
class TestOneHotOp_attr(XPUOpTest):
    def setUp(self):
        self.op_type = 'one_hot'
        depth = 10
        x_lod = [[4, 1, 3, 3]]
        x = [np.random.randint(0, depth - 1) for i in range(sum(x_lod[0]))]
        x = np.array(x).astype('int32').reshape([sum(x_lod[0]), 1])

        out = np.zeros(shape=(np.product(x.shape[:-1]),
                              depth)).astype('float32')

        for i in range(np.product(x.shape)):
            out[i, x[i]] = 1.0

        self.inputs = {'X': (x, x_lod)}
        self.attrs = {'dtype': int(core.VarDesc.VarType.FP32), 'depth': depth}
        self.outputs = {'Out': (out, x_lod)}

    def test_check_output(self):
        place = paddle.XPUPlace(0)
        self.check_output_with_place(place, check_dygraph=False)


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 'core is not compiled with XPU')
class TestOneHotOp_default_dtype(XPUOpTest):
    def setUp(self):
        self.op_type = 'one_hot'
        depth = 10
        depth_np = np.array(10).astype('int32')
        x_lod = [[4, 1, 3, 3]]
        x = [np.random.randint(0, depth - 1) for i in range(sum(x_lod[0]))]
        x = np.array(x).astype('int32').reshape([sum(x_lod[0]), 1])

        out = np.zeros(shape=(np.product(x.shape[:-1]),
                              depth)).astype('float32')

        for i in range(np.product(x.shape)):
            out[i, x[i]] = 1.0

        self.inputs = {'X': (x, x_lod), 'depth_tensor': depth_np}
        self.attrs = {}
        self.outputs = {'Out': (out, x_lod)}

    def test_check_output(self):
        place = paddle.XPUPlace(0)
        self.check_output_with_place(place, check_dygraph=False)


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 'core is not compiled with XPU')
class TestOneHotOp_default_dtype_attr(XPUOpTest):
    def setUp(self):
        self.op_type = 'one_hot'
        depth = 10
        x_lod = [[4, 1, 3, 3]]
        x = [np.random.randint(0, depth - 1) for i in range(sum(x_lod[0]))]
        x = np.array(x).astype('int32').reshape([sum(x_lod[0]), 1])

        out = np.zeros(shape=(np.product(x.shape[:-1]),
                              depth)).astype('float32')

        for i in range(np.product(x.shape)):
            out[i, x[i]] = 1.0

        self.inputs = {'X': (x, x_lod)}
        self.attrs = {'depth': depth}
        self.outputs = {'Out': (out, x_lod)}

    def test_check_output(self):
        place = paddle.XPUPlace(0)
        self.check_output_with_place(place, check_dygraph=False)


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 'core is not compiled with XPU')
class TestOneHotOp_out_of_range(XPUOpTest):
    def setUp(self):
        self.op_type = 'one_hot'
        depth = 10
        x_lod = [[4, 1, 3, 3]]
        x = [np.random.choice([-1, depth]) for i in range(sum(x_lod[0]))]
        x = np.array(x).astype('int32').reshape([sum(x_lod[0]), 1])

        out = np.zeros(shape=(np.product(x.shape[:-1]),
                              depth)).astype('float32')

        self.inputs = {'X': (x, x_lod)}
        self.attrs = {'depth': depth, 'allow_out_of_range': True}
        self.outputs = {'Out': (out, x_lod)}

    def test_check_output(self):
        place = paddle.XPUPlace(0)
        self.check_output_with_place(place, check_dygraph=False)


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 'core is not compiled with XPU')
class TestOneHotOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            # the input must be Variable
            in_w = np.random.random((4, 1)).astype('int32')
            self.assertRaises(TypeError, fluid.layers.one_hot, in_w)
            # the input must be int32 or int 64
            in_w2 = fluid.layers.data(
                name='in_w2',
                shape=[4, 1],
                append_batch_size=False,
                dtype='float32')
            self.assertRaises(TypeError, fluid.layers.one_hot, in_w2)
            # the depth must be int, long or Variable
            in_r = fluid.layers.data(
                name='in_r',
                shape=[4, 1],
                append_batch_size=False,
                dtype='int32')
            depth_w = np.array([4])
            self.assertRaises(TypeError, fluid.layers.one_hot, in_r, 4.1)
            self.assertRaises(TypeError, fluid.layers.one_hot, in_r, depth_w)
"""

if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
