#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test_xpu import XPUOpTest

import paddle

paddle.enable_static()


class XPUTestMeanOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'reduce_mean'
        self.use_dynamic_create_class = False

    class TestMeanOp(XPUOpTest):
        def setUp(self):
            self.dtype = self.in_type
            self.place = paddle.XPUPlace(0)
            self.op_type = "reduce_mean"
            self.inputs = {'X': np.random.random((5, 6, 10)).astype(self.dtype)}
            self.attrs = {'use_xpu': True}
            self.outputs = {'Out': self.inputs['X'].mean(axis=0)}

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            self.check_grad_with_place(self.place, ['X'], 'Out')

    class TestMeanOp5D(TestMeanOp):
        def setUp(self):
            super().setUp()
            self.inputs = {
                'X': np.random.random((1, 2, 5, 6, 10)).astype(self.dtype)
            }
            self.attrs = {'use_xpu': True}
            self.outputs = {'Out': self.inputs['X'].mean(axis=0)}

    class TestMeanOp6D(TestMeanOp):
        def setUp(self):
            super().setUp()
            self.inputs = {
                'X': np.random.random((1, 1, 2, 5, 6, 10)).astype(self.dtype)
            }
            self.attrs = {'use_xpu': True}
            self.outputs = {'Out': self.inputs['X'].mean(axis=0)}

    class TestMeanOp8D(TestMeanOp):
        def setUp(self):
            super().setUp()
            self.inputs = {
                'X': np.random.random((1, 3, 1, 2, 1, 4, 3, 10)).astype(
                    self.dtype
                )
            }
            self.attrs = {'dim': (0, 3), 'use_xpu': True}
            self.outputs = {'Out': self.inputs['X'].mean(axis=(0, 3))}


class XPUTestReduce(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'reduce_mean'
        self.use_dynamic_create_class = False

    class Test1DReduce(XPUOpTest):
        def setUp(self):
            self.dtype = self.in_type
            self.place = paddle.XPUPlace(0)
            self.op_type = "reduce_mean"
            self.inputs = {'X': np.random.random(120).astype(self.dtype)}
            self.attrs = {'use_xpu': True}
            self.outputs = {'Out': self.inputs['X'].mean(axis=0)}

        def test_check_output(self):
            self.check_output_with_place(self.place)

        # There is a api bug in checking grad when dim[0] > 0
        # def test_check_grad(self):
        #     self.check_output_with_place(self.place, ['X'], 'Out')

        def test_check_grad(self):
            self.check_grad_with_place(self.place, ['X'], 'Out')

    class Test2DReduce0(Test1DReduce):
        def setUp(self):
            super().setUp()
            self.attrs = {'dim': [0], 'use_xpu': True}
            self.inputs = {'X': np.random.random((20, 10)).astype(self.dtype)}
            self.outputs = {'Out': self.inputs['X'].mean(axis=0)}

    class Test2DReduce1(Test1DReduce):
        def setUp(self):
            super().setUp()
            self.attrs = {'dim': [1], 'use_xpu': True}
            self.inputs = {'X': np.random.random((20, 10)).astype(self.dtype)}
            self.outputs = {
                'Out': self.inputs['X'].mean(axis=tuple(self.attrs['dim']))
            }

    class Test3DReduce0(Test1DReduce):
        def setUp(self):
            super().setUp()
            self.attrs = {'dim': [1], 'use_xpu': True}
            self.inputs = {'X': np.random.random((5, 6, 7)).astype(self.dtype)}
            self.outputs = {
                'Out': self.inputs['X'].mean(axis=tuple(self.attrs['dim']))
            }

    class Test3DReduce1(Test1DReduce):
        def setUp(self):
            super().setUp()
            self.attrs = {'dim': [2], 'use_xpu': True}
            self.inputs = {'X': np.random.random((5, 6, 7)).astype(self.dtype)}
            self.outputs = {
                'Out': self.inputs['X'].mean(axis=tuple(self.attrs['dim']))
            }

    class Test3DReduce2(Test1DReduce):
        def setUp(self):
            super().setUp()
            self.attrs = {'dim': [-2], 'use_xpu': True}
            self.inputs = {'X': np.random.random((5, 6, 7)).astype(self.dtype)}
            self.outputs = {
                'Out': self.inputs['X'].mean(axis=tuple(self.attrs['dim']))
            }

    class Test3DReduce3(Test1DReduce):
        def setUp(self):
            super().setUp()
            self.attrs = {'dim': [1, 2], 'use_xpu': True}
            self.inputs = {'X': np.random.random((5, 6, 7)).astype(self.dtype)}
            self.outputs = {
                'Out': self.inputs['X'].mean(axis=tuple(self.attrs['dim']))
            }

    class Test6DReduce(Test1DReduce):
        def setUp(self):
            super().setUp()
            self.attrs = {'dim': [1, -1], 'use_xpu': True}
            self.inputs = {
                'X': np.random.random((5, 6, 7, 8, 9, 10)).astype(self.dtype)
            }
            self.outputs = {
                'Out': self.inputs['X'].mean(axis=tuple(self.attrs['dim']))
            }

    class TestKeepDimReduce(Test1DReduce):
        def setUp(self):
            super().setUp()
            self.inputs = {'X': np.random.random((5, 6, 10)).astype(self.dtype)}
            self.attrs = {'dim': [1], 'keep_dim': True, 'use_xpu': True}
            self.outputs = {
                'Out': self.inputs['X'].mean(
                    axis=tuple(self.attrs['dim']),
                    keepdims=self.attrs['keep_dim'],
                )
            }

    class TestKeepDim8DReduce(Test1DReduce):
        def setUp(self):
            super().setUp()
            self.inputs = {
                'X': np.random.random((2, 5, 3, 2, 2, 3, 4, 2)).astype(
                    self.dtype
                )
            }
            self.attrs = {'dim': (3, 4, 5), 'keep_dim': True, 'use_xpu': True}
            self.outputs = {
                'Out': self.inputs['X'].mean(
                    axis=tuple(self.attrs['dim']),
                    keepdims=self.attrs['keep_dim'],
                )
            }


support_types = get_xpu_op_support_types('reduce_mean')
for stype in support_types:
    create_test_class(globals(), XPUTestMeanOp, stype)
    create_test_class(globals(), XPUTestReduce, stype)


if __name__ == '__main__':
    unittest.main()
