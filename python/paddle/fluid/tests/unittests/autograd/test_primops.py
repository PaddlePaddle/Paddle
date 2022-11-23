# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import uuid

import numpy as np
import paddle
from numpy.random import randint, randn
from paddle.incubate.autograd import primops

import config
import utils

paddle.enable_static()


@utils.place(config.DEVICES)
@utils.parameterize(
    (utils.TEST_CASE_NAME, 'op', 'args', 'kwargs', 'expected_shape',
     'expected_dtype'),
    (
        ('add', primops.add, (randn(2, 3), randn(2, 3)), {}, (2, 3), 'float64'),
        ('sub', primops.sub, (randn(2, 3), randn(2, 3)), {}, (2, 3), 'float64'),
        ('mul', primops.mul, (randn(2, 3), randn(2, 3)), {}, (2, 3), 'float64'),
        ('div', primops.div, (randn(2, 3), randn(2, 3)), {}, (2, 3), 'float64'),
        ('sub', primops.sub, (randn(2, 3), randn(2, 3)), {}, (2, 3), 'float64'),
        ('sqrt', primops.sqrt, randn(2, 3), {}, (2, 3), 'float64'),
        ('tanh', primops.tanh, randn(2, 3), {}, (2, 3), 'float64'),
        ('sin', primops.sin, randn(2, 3), {}, (2, 3), 'float64'),
        ('cos', primops.cos, randn(2, 3), {}, (2, 3), 'float64'),
        ('exp', primops.exp, randn(2, 3), {}, (2, 3), 'float64'),
        ('erf', primops.erf, randn(2, 3), {}, (2, 3), 'float64'),
        ('abs', primops.abs, randn(2, 3), {}, (2, 3), 'float64'),
        ('log', primops.log, randn(2, 3), {}, (2, 3), 'float64'),
        ('cast', primops.cast, randn(2, 3), {
            'dtype': paddle.int64
        }, (2, 3), 'int64'),
        ('reshape', primops.reshape, randn(2, 3), {
            'shape': (3, 2)
        }, (3, 2), 'float64'),
        ('broadcast', primops.broadcast, randn(2), {
            'shape': (3, 2)
        }, (3, 2), 'float64'),
        ('transpose', primops.transpose, randn(2, 3), {
            'axis': (1, 0)
        }, (3, 2), 'float64'),
        ('concat_axis0', primops.concat, ((randn(2, 3), randn(2, 3)), ), {
            'axis': 0
        }, (4, 3), 'float64'),
        ('concat_axis1', primops.concat, ((randn(2, 3), randn(2, 3)), ), {
            'axis': 1
        }, (2, 6), 'float64'),
        ('reduce_axis1', primops.reduce_sum, randn(2, 3), {
            'axis': (1, )
        }, (2, ), 'float64'),
        ('reduce_axis01', primops.reduce_sum, randn(2, 3), {
            'axis': (0, 1)
        }, (1, ), 'float64'),
        ('split', primops.split, randn(2, 3), {
            'num_or_sections': [1, 2],
            'axis': 1
        }, ((2, 1), (2, 2)), ('float64', 'float64')),
        ('matmul', primops.matmul, (randn(2, 3), randn(3, 2)), {},
         (2, 2), 'float64'),
        ('slice_select', primops.slice_select, randn(3, 2), {
            'axis': [0],
            'starts': [0],
            'ends': [2],
            'strides': [1]
        }, (2, 2), 'float64'),
        ('slice_assign', primops.slice_assign, (randn(2, 3), randn(2, 2)), {
            'axis': [1],
            'starts': [1],
            'ends': [3],
            'strides': [1]
        }, (2, 3), 'float64'),
        ('gather', primops.gather, (randn(3, 2), randint(0, 2,
                                                         (5, ), np.int32)), {
                                                             'axis': 0
                                                         }, (5, 2), 'float64'),
        ('scatter_add', primops.scatter_add,
         (randn(3, 2), randn(5, 2), randint(0, 2, (5, ), np.int32)), {
             'axis': 0
         }, (3, 2), 'float64'),
        ('fill_const', primops.fill_const, (), {
            'value': 10,
            'shape': (3, 2),
            'dtype': paddle.float32
        }, (3, 2), 'float32'),
        ('neg', primops.neg, randn(2, 3), {}, (2, 3), 'float64'),
        ('select', primops.select,
         (randn(2, 3) > 0, randn(2, 3), randn(2, 3)), {}, (2, 3), 'float64'),
        ('eq', primops.eq, (randn(2, 3), randn(2, 3)), {}, (2, 3), 'bool'),
        ('ne', primops.ne, (randn(2, 3), randn(2, 3)), {}, (2, 3), 'bool'),
        ('gt', primops.gt, (randn(2, 3), randn(2, 3)), {}, (2, 3), 'bool'),
        ('ge', primops.ge, (randn(2, 3), randn(2, 3)), {}, (2, 3), 'bool'),
        ('pow', primops.pow, (randn(2, 3), randn(2, 3)), {}, (2, 3), 'float64'),
        ('max', primops.max, (randn(2, 3), randn(2, 3)), {}, (2, 3), 'float64'),
    ))
class TestPrimops(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        paddle.enable_static()

    @classmethod
    def tearDownClass(cls):
        paddle.disable_static()

    def test_prim_ops(self):
        program = paddle.static.Program()
        with paddle.static.program_guard(program):
            args = self._as_tuple(self.args)
            args = self.arr2var(args)
            results = self.op(*args, **self.kwargs)
            results = self._as_tuple(results)
            expected_shape = self._as_tuple(self.expected_shape)
            expected_dtype = self._as_tuple(self.expected_dtype)

            for r, shape, dtype in zip(results, expected_shape, expected_dtype):
                self.assertEqual(r.shape, shape)
                self.assertEqual(str(r.dtype).split('.')[1], dtype)

    def arr2var(self, arr):
        """convert numpy ndarray to paddle Variable recursively."""
        return [
            paddle.static.data(f'x{uuid.uuid4()}', v.shape, v.dtype)
            if isinstance(v, np.ndarray) else self.arr2var(v) for v in arr
        ]

    def _as_tuple(self, input):
        if isinstance(input, (tuple, list)) and len(input) == 0:
            return input
        if not isinstance(input, (tuple, list)) or all(
                isinstance(i, int) for i in input):
            return (input, )
        return input


if __name__ == '__main__':
    unittest.main()
