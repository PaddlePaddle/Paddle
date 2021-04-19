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
from __future__ import print_function
import unittest
import numpy as np
import sys
sys.path.append("..")
from paddle.fluid.op import Operator
import paddle.fluid.core as core
import paddle.fluid as fluid
import paddle
from op_test_xpu import XPUOpTest
from paddle.static import Program, program_guard

TEST_META_OP_DATA = [{
    'op_str': 'logical_and',
    'binary_op': True
}, {
    'op_str': 'logical_or',
    'binary_op': True
}, {
    'op_str': 'logical_not',
    'binary_op': False
}]

TEST_META_SHAPE_DATA = {
    'XDimLargerThanYDim1': {
        'x_shape': [2, 3, 4, 5],
        'y_shape': [4, 5]
    },
    'XDimLargerThanYDim2': {
        'x_shape': [2, 3, 4, 5],
        'y_shape': [4, 1]
    },
    'XDimLargerThanYDim3': {
        'x_shape': [2, 3, 4, 5],
        'y_shape': [1, 4, 1]
    },
    'XDimLargerThanYDim4': {
        'x_shape': [2, 3, 4, 5],
        'y_shape': [3, 4, 1]
    },
    'XDimLargerThanYDim5': {
        'x_shape': [2, 3, 1, 5],
        'y_shape': [3, 1, 1]
    },
    'XDimLessThanYDim1': {
        'x_shape': [4, 1],
        'y_shape': [2, 3, 4, 5]
    },
    'XDimLessThanYDim2': {
        'x_shape': [1, 4, 1],
        'y_shape': [2, 3, 4, 5]
    },
    'XDimLessThanYDim3': {
        'x_shape': [3, 4, 1],
        'y_shape': [2, 3, 4, 5]
    },
    'XDimLessThanYDim4': {
        'x_shape': [3, 1, 1],
        'y_shape': [2, 3, 1, 5]
    },
    'XDimLessThanYDim5': {
        'x_shape': [4, 5],
        'y_shape': [2, 3, 4, 5]
    },
    'Axis1InLargerDim': {
        'x_shape': [1, 4, 5],
        'y_shape': [2, 3, 1, 5]
    },
    'EqualDim1': {
        'x_shape': [10, 7],
        'y_shape': [10, 7]
    },
    'EqualDim2': {
        'x_shape': [1, 1, 4, 5],
        'y_shape': [2, 3, 1, 5]
    }
}

TEST_META_WRONG_SHAPE_DATA = {
    'ErrorDim1': {
        'x_shape': [2, 3, 4, 5],
        'y_shape': [3, 4]
    },
    'ErrorDim2': {
        'x_shape': [2, 3, 4, 5],
        'y_shape': [4, 3]
    }
}


def run_static_xpu(x_np, y_np, op_str, binary_op=True):
    paddle.enable_static()
    startup_program = fluid.Program()
    main_program = fluid.Program()
    place = paddle.XPUPlace(0)
    exe = fluid.Executor(place)
    with fluid.program_guard(main_program, startup_program):
        x = paddle.static.data(name='x', shape=x_np.shape, dtype='bool')
        op = getattr(paddle, op_str)
        feed_list = {'x': x_np}
        if not binary_op:
            res = op(x)
        else:
            y = paddle.static.data(name='y', shape=y_np.shape, dtype='bool')
            feed_list['y'] = y_np
            res = op(x, y)
        exe.run(startup_program)
        static_result = exe.run(main_program, feed=feed_list, fetch_list=[res])
    return static_result


def run_dygraph_xpu(x_np, y_np, op_str, binary_op=True):
    place = paddle.XPUPlace(0)
    paddle.disable_static(place)
    op = getattr(paddle, op_str)
    x = paddle.to_tensor(x_np)
    if not binary_op:
        dygraph_result = op(x)
    else:
        y = paddle.to_tensor(y_np)
        dygraph_result = op(x, y)
    return dygraph_result


def np_data_generator(np_shape, *args, **kwargs):
    return np.random.choice(a=[True, False], size=np_shape).astype(bool)


def test_xpu(unit_test, test_error=False):
    for op_data in TEST_META_OP_DATA:
        meta_data = dict(op_data)
        np_op = getattr(np, meta_data['op_str'])
        META_DATA = dict(TEST_META_SHAPE_DATA)
        if test_error:
            META_DATA = dict(TEST_META_WRONG_SHAPE_DATA)
        for shape_data in META_DATA.values():
            meta_data['x_np'] = np_data_generator(shape_data['x_shape'])
            meta_data['y_np'] = np_data_generator(shape_data['y_shape'])
            if meta_data['binary_op'] and test_error:
                # catch C++ Exception
                unit_test.assertRaises(BaseException, run_static_xpu,
                                       **meta_data)
                continue
            static_result = run_static_xpu(**meta_data)
            dygraph_result = run_dygraph_xpu(**meta_data)
            if meta_data['binary_op']:
                np_result = np_op(meta_data['x_np'], meta_data['y_np'])
            else:
                np_result = np_op(meta_data['x_np'])
            unit_test.assertTrue((static_result == np_result).all())
            unit_test.assertTrue((dygraph_result.numpy() == np_result).all())


def test_type_error(unit_test, type_str_map):
    def check_type(op_str, x, y, binary_op):
        op = getattr(paddle, op_str)
        error_type = TypeError
        if isinstance(x, np.ndarray):
            x = paddle.to_tensor(x)
            y = paddle.to_tensor(y)
            error_type = BaseException
        if binary_op:
            if type_str_map['x'] != 'bool' or type_str_map['y'] != 'bool':
                unit_test.assertRaises(error_type, op, x=x, y=y)
            if not fluid.in_dygraph_mode():
                unit_test.assertRaises(error_type, op, x=x, y=y, out=1)
        else:
            if type_str_map['x'] != 'bool':
                unit_test.assertRaises(error_type, op, x=x)
            if not fluid.in_dygraph_mode():
                unit_test.assertRaises(error_type, op, x=x, out=1)

    place = paddle.XPUPlace(0)

    for op_data in TEST_META_OP_DATA:
        meta_data = dict(op_data)
        binary_op = meta_data['binary_op']

        paddle.disable_static(place)
        x = np.random.choice(a=[0, 1], size=[10]).astype(type_str_map['x'])
        y = np.random.choice(a=[0, 1], size=[10]).astype(type_str_map['y'])
        check_type(meta_data['op_str'], x, y, binary_op)

        paddle.enable_static()
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.static.data(
                name='x', shape=[10], dtype=type_str_map['x'])
            y = paddle.static.data(
                name='y', shape=[10], dtype=type_str_map['y'])
            check_type(meta_data['op_str'], x, y, binary_op)


def type_map_factory():
    x_type_list = ['float32', 'float64', 'int32', 'int64', 'bool']
    y_type_list = ['float32', 'float64', 'int32', 'int64', 'bool']
    return [{
        'x': x_type,
        'y': y_type
    } for x_type in x_type_list for y_type in y_type_list]


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestXPU(unittest.TestCase):
    def test(self):
        test_xpu(self, True)

    def test_error(self):
        test_xpu(self, True)

    def test_type_error(self):
        type_map_list = type_map_factory()
        for type_map in type_map_list:
            test_type_error(self, type_map)


if __name__ == '__main__':
    unittest.main()
