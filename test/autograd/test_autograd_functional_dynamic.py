# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import sys
import typing
import unittest

sys.path.insert(0, '.')

import config
import numpy as np
import utils

import paddle
from paddle.incubate.autograd.utils import as_tensors


def make_v(f, inputs):
    outputs = as_tensors(f(*inputs))
    return [paddle.ones_like(x) for x in outputs]


@utils.place(config.DEVICES)
@utils.parameterize(
    (utils.TEST_CASE_NAME, 'func', 'xs'),
    (
        (
            'multi_in_single_out',
            paddle.matmul,
            (np.random.rand(2, 2), np.random.rand(2, 2)),
        ),
    ),
)
class TestJacobianNoBatch(unittest.TestCase):
    def setUp(self):
        self._dtype = (
            self.xs[0].dtype
            if isinstance(self.xs, typing.Sequence)
            else self.xs.dtype
        )
        self._eps = (
            config.TOLERANCE.get(str(self._dtype))
            .get("first_order_grad")
            .get("eps")
        )
        self._rtol = (
            config.TOLERANCE.get(str(self._dtype))
            .get("first_order_grad")
            .get("rtol")
        )
        self._atol = (
            config.TOLERANCE.get(str(self._dtype))
            .get("first_order_grad")
            .get("atol")
        )

    def test_jacobian(self):
        xs = (
            [paddle.to_tensor(x) for x in self.xs]
            if isinstance(self.xs, typing.Sequence)
            else paddle.to_tensor(self.xs)
        )
        self._actual = paddle.incubate.autograd.Jacobian(self.func, xs, False)
        self._expected = self._get_expected()

        Index = collections.namedtuple('Index', ('type', 'value'))
        indexes = (
            Index('all', (slice(0, None, None), slice(0, None, None))),
            Index('row', (0, slice(0, None, None))),
            Index('col', (slice(0, None, None), 0)),
            Index('multi-row', (slice(0, 2, 1), slice(0, None, None))),
        )
        self.assertEqual(self._actual[:].numpy().dtype, self._expected.dtype)
        for index in indexes:
            np.testing.assert_allclose(
                self._actual.__getitem__(index.value),
                self._expected.__getitem__(index.value),
                rtol=self._rtol,
                atol=self._atol,
                err_msg=f'Testcase {index.type} index not passed, value is {index.value}',
            )

    def _get_expected(self):
        xs = (
            [paddle.to_tensor(x) for x in self.xs]
            if isinstance(self.xs, typing.Sequence)
            else paddle.to_tensor(self.xs)
        )
        jac = utils._compute_numerical_jacobian(
            self.func, xs, self._eps, self._dtype
        )
        return utils._np_concat_matrix_sequence(jac, utils.MatrixFormat.NM)


if __name__ == "__main__":
    np.random.seed(2022)
    unittest.main()
