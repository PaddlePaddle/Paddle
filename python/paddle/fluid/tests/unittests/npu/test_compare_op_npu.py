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
import paddle
import paddle.fluid as fluid

paddle.enable_static()
SEED = 2021

callback_dict = {
    'less_than': lambda _a, _b: _a < _b,
    'less_equal': lambda _a, _b: _a <= _b,
    'greater_than': lambda _a, _b: _a > _b,
    'greater_equal': lambda _a, _b: _a >= _b,
    'equal': lambda _a, _b: _a == _b,
    'not_equal': lambda _a, _b: _a != _b
}

@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestEqual(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "equal"
        self.place = paddle.NPUPlace(0)

        self.init_dtype()
        np.random.seed(SEED)
        x = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        y = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        out = x == y  # all elements are not equal

        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(x),
            'Y': OpTest.np_dtype_to_fluid_dtype(y)
        }
        self.outputs = {'Out': out}

    def set_npu(self):
        self.__class__.use_npu = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestLessthan(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "less_than"
        self.place = paddle.NPUPlace(0)

        self.init_dtype()
        np.random.seed(SEED)
        x = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        y = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        out = x < y

        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(x),
            'Y': OpTest.np_dtype_to_fluid_dtype(y)
        }
        self.outputs = {'Out': out}

    def set_npu(self):
        self.__class__.use_npu = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)


class TestEqual2(TestEqual):
    def setUp(self):
        self.set_npu()
        self.op_type = "equal"
        self.place = paddle.NPUPlace(0)

        self.init_dtype()
        np.random.seed(SEED)
        x = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        y = x.copy()
        y[0][1] = 1
        out = x == y  # all elements are equal, except position [0][1]

        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(x),
            'Y': OpTest.np_dtype_to_fluid_dtype(y)
        }
        self.outputs = {'Out': out}


class TestLessthan2(TestLessthan):
    def setUp(self):
        self.set_npu()
        self.op_type = "less_than"
        self.place = paddle.NPUPlace(0)

        self.init_dtype()
        np.random.seed(SEED)
        x = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        y = x.copy()
        y[0][1] = 1
        out = x < y  # all elements are equal, except position [0][1]

        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(x),
            'Y': OpTest.np_dtype_to_fluid_dtype(y)
        }
        self.outputs = {'Out': out}


class TestEqual2FP16(TestEqual2):
    def init_dtype(self):
        self.dtype = np.float16


class TestEqual2Int(TestEqual2):
    def init_dtype(self):
        self.dtype = np.int32


class TestLessthan2FP16(TestLessthan2):
    def init_dtype(self):
        self.dtype = np.float16


def create_compare_npu_paddleclass(op_type, dtype, inputs, callback, postfix):
    @unittest.skipIf(not paddle.is_compiled_with_npu(),
                     "core is not compiled with NPU")
    class PaddleCls(OpTest):
        def setUp(self):
            self.__class__.use_npu = True
            self.op_type = op_type
            self.dtype = dtype
            self.place = paddle.NPUPlace(0)
            np.random.seed(SEED)
            x = inputs['X'].astype(dtype)
            y = inputs['Y'].astype(dtype)
            self.inputs = {
                'X': OpTest.np_dtype_to_fluid_dtype(x),
                'Y': OpTest.np_dtype_to_fluid_dtype(y)
            }

            out=callback(self.inputs['X'], self.inputs['Y'])
            self.outputs = {'Out': out}

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_dynamic_api(self):
            if self.dtype == 'float16':
                return
            paddle.disable_static()
            x = paddle.to_tensor(self.inputs['X'])
            y = paddle.to_tensor(self.inputs['Y'])
            op = eval("paddle.%s" % (self.op_type))
            out = op(x, y)
            self.assertEqual((out.numpy() == self.outputs['Out']).all(), True)
            paddle.enable_static()

        def test_static_api(self):
            if self.dtype == 'float16':
                return
            paddle.enable_static()
            with fluid.program_guard(fluid.Program(), fluid.Program()):
                x = fluid.data(name='x', shape=self.inputs['X'].shape, dtype=self.dtype)
                y = fluid.data(name='y', shape=self.inputs['Y'].shape, dtype=self.dtype)
                op = eval('paddle.%s' % (self.op_type))
                out = op(x, y)
                exe = fluid.Executor(self.place)
                res, = exe.run(feed={'x': self.inputs['X'],
                                     'y': self.inputs['Y']},
                               fetch_list=[out])
            self.assertEqual((res == self.outputs['Out']).all(), True)

    cls_name = 'TestCase_%s_%s' % (op_type, dtype)
    if postfix is not None:
        cls_name = cls_name + '_' + postfix
    PaddleCls.__name__ = cls_name
    globals()[cls_name] = PaddleCls

def create_compare_npu_case(op_type):
    def gen_inputs(shape_x, shape_y, l=0, h=5):
        return {'X': np.random.uniform(l, h, shape_x), 'Y': np.random.uniform(l, h, shape_y)}

    inputs_lis = []
    # 1 Add same shape inputs
    inputs_lis.append(gen_inputs([11,17],[11,17]))
    # 2-5 Add different shape for broadcast cases
    inputs_lis.append(gen_inputs([1, 2, 1, 3], [1, 2, 3]))
    inputs_lis.append(gen_inputs([1, 2, 3], [1, 2, 1, 3]))
    inputs_lis.append(gen_inputs([5], [3, 1]))
    inputs_lis.append(gen_inputs([3, 1], [1]))
    postfix_lis = [None, 'broadcast1', 'broadcast2', 'broadcast3', 'broadcast4']

    dtype_list = ['float32', 'float16', 'int32']
    callback = callback_dict[op_type]
    for i in range(len(inputs_lis)):
        for dtype in dtype_list:
            inputs = inputs_lis[i]
            postfix = postfix_lis[i]
            create_compare_npu_paddleclass(op_type, dtype, inputs, callback, postfix)

create_compare_npu_case('greater_than')
create_compare_npu_case('not_equal')
create_compare_npu_case('less_equal')

if __name__ == '__main__':
    unittest.main()

