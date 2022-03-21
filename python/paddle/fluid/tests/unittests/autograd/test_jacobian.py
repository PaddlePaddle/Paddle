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

import collections
import typing
import unittest
import torch

import numpy as np
import paddle

import config
import funcs
import parameterize as param
import utils


class TestIndex(unittest.TestCase):
    pass


@param.place(config.DEVICES)
@param.parameterize(
    (param.TEST_CASE_NAME, 'func', 'xs'),
    (  # noqa
        ('1d-in-1d-out', funcs.square, np.array([2., 3.])),
        ('3d-in-3d-out', funcs.square, np.random.rand(2, 3, 4)),
        ('multi-input', funcs.square, np.random.rand(10, 20)),
        ('matmul', paddle.matmul,
         (np.random.rand(2, 2), np.random.rand(2, 2))), ))
class TestJacobianNoBatch(unittest.TestCase):
    def setUp(self):
        self._dtype = self.xs[0].dtype if isinstance(
            self.xs, typing.Sequence) else self.xs.dtype
        self._eps = config.EPS.get(str(self._dtype))

        self.xs = [paddle.to_tensor(x) for x in self.xs] if isinstance(
            self.xs, typing.Sequence) else paddle.to_tensor(self.xs)
        self._actual = paddle.autograd.Jacobian(self.func, self.xs, False)
        self._expected = self._expected()

    def test_jacobian(self):
        Index = collections.namedtuple('Index', ('type', 'value'))
        indexes = (Index('all', (slice(0, None, None), slice(0, None, None))),
                   Index('row', (0, slice(0, None, None))),
                   Index('col', (slice(0, None, None), 0)),
                   Index('multi-row', (slice(0, 2, 1), slice(0, None, None))))
        self.assertEqual(self._actual[:].dtype, self._expected.dtype)
        for index in indexes:
            np.testing.assert_allclose(
                self._actual.__getitem__(index.value),
                self._expected.__getitem__(index.value),
                rtol=config.RTOL.get(str(self._dtype)),
                atol=config.ATOL.get(str(self._dtype)),
                err_msg=f'Testcase {index.type} index not passed, value is {index.value}'
            )

    def _expected(self):
        # numerical_jacobian return list of list of tensors, need to concat.
        results = utils._compute_numerical_jacobian(self.func, self.xs,
                                                    self._eps, self._dtype)
        rows = []
        for i in range(len(results)):
            rows.append(
                paddle.concat([paddle.to_tensor(x) for x in results[i]], -1))
        return paddle.concat(rows, 0)


@param.place(config.DEVICES)
@param.parameterize(
    (param.TEST_CASE_NAME, 'func', 'xs'),
    (
        ('1d-in-1d-out', funcs.square, np.array([[1., 2., 3.], [3., 4., 3.]])),
        ('3d-in-3d-out', funcs.square, np.random.rand(2, 3, 4)),
        ('multi-in-single-out', funcs.square, np.random.rand(5, 6)),
        # ('multi-in-multi-out', funcs.o2, (np.random.rand(2,2), np.random.rand(2,2)))
    ))
class TestJacobianBatchFirst(unittest.TestCase):
    def setUp(self):
        self._dtype = self.xs[0].dtype if isinstance(
            self.xs, typing.Sequence) else self.xs.dtype
        self._eps = config.EPS.get(str(self._dtype))

        self.xs = [paddle.to_tensor(x) for x in self.xs] if isinstance(
            self.xs, typing.Sequence) else paddle.to_tensor(self.xs)
        self._actual = paddle.autograd.Jacobian(self.func, self.xs, True, False)
        self._expected = self._expected()

    def test_jacobian(self):
        Index = collections.namedtuple('Index', ('type', 'value'))
        indexes = (
            Index('all', (slice(0, None, None), slice(0, None, None),
                          slice(0, None, None))),
            Index('row', (slice(0, None, None), 0, slice(0, None, None))),
            Index('col',
                  (slice(0, None, None), slice(0, None, None), 0)), Index(
                      'batch', (slice(0, 2, None), slice(0, None, None),
                                slice(0, None, None))),
            Index('multi-row',
                  (slice(0, 1, None), slice(0, 2, 1), slice(0, None, None))))
        self.assertEqual(self._actual[:].dtype, self._expected.dtype)
        for index in indexes:
            np.testing.assert_allclose(
                self._actual.__getitem__(index.value),
                self._expected.__getitem__(index.value),
                rtol=config.RTOL.get(str(self._dtype)),
                atol=config.ATOL.get(str(self._dtype)),
                err_msg=f'Testcase {index.type} index not passed, value is {index.value}'
            )

    def _expected(self):
        results = utils._compute_numerical_batch_jacobian(
            self.func, self.xs, self._eps, self._dtype)
        rows = []
        for i in range(len(results)):
            rows.append(
                paddle.concat([paddle.to_tensor(x) for x in results[i]], -1))
        return paddle.concat(rows, 1)


if __name__ == '__main__':
    unittest.main()
