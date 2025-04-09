# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import parameterize
from distribution import config

import paddle
from paddle.distribution import lkj_cholesky
from paddle.distribution.lkj_cholesky import (
    tril_matrix_to_vec,
    vec_to_tril_matrix,
)

np.random.seed(2024)
paddle.seed(2024)


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'concentration'),
    [
        (
            'zero-dim',
            parameterize.xrand(
                (1,),
                dtype='float32',
                max=1.0,
                min=0,
            ).reshape([]),
        ),
        (
            'one-dim',
            parameterize.xrand(
                (2,),
                dtype='float32',
                max=1.0,
                min=0,
            ),
        ),
        (
            'one-dim2',
            parameterize.xrand(
                (1,),
                dtype='float32',
                max=1.0,
                min=0,
            ),
        ),
    ],
)
class TestLKJCholeskyShape(unittest.TestCase):
    def gen_cases(self):
        extra_shape = []
        extra_shape.extend(self.concentration.shape)
        extra_shape.extend(
            [self._paddle_lkj_cholesky.dim, self._paddle_lkj_cholesky.dim]
        )
        cases = [
            {
                'input': (),
                'expect': tuple(extra_shape),
            },
        ]
        return cases

    def test_onion_sample_shape(self):
        sample_method = 'onion'
        self._test_sample_shape_dim(sample_method)

    def test_cvine_sample_shape(self):
        sample_method = 'cvine'
        self._test_sample_shape_dim(sample_method)

    def _test_sample_shape_dim(self, sample_method):
        self._test_sample_shape(2, sample_method)

    def _test_sample_shape(self, dim, sample_method):
        self._paddle_lkj_cholesky = lkj_cholesky.LKJCholesky(
            dim, self.concentration, sample_method
        )
        cases = self.gen_cases()
        for case in cases:
            data = self._paddle_lkj_cholesky.sample(case.get('input'))
            self.assertTrue(tuple(data.shape) == case.get('expect'))


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME),
    [
        ('test_log_prob'),
    ],
)
class TestLKJCholeskyLogProb(unittest.TestCase):
    def test_log_prob(self):
        self.dim = 2
        self._paddle_lkj_cholesky = lkj_cholesky.LKJCholesky(
            self.dim, [1.0], 'onion'
        )
        self._test_log_prob()

    def _test_log_prob(self):
        log_probs = []
        for i in range(2):
            sample = self._paddle_lkj_cholesky.sample()
            log_prob = self._paddle_lkj_cholesky.log_prob(sample)
            sample_tril = tril_matrix_to_vec(sample, diag=-1)
            # log_abs_det_jacobian
            logabsdet = []
            logabsdet.append(self._compute_jacobian(sample_tril)[1])
            logabsdet = paddle.to_tensor(logabsdet)

            log_probs.append((log_prob - logabsdet).numpy())
        np.testing.assert_allclose(
            log_probs[0],
            log_probs[1],
            rtol=0.1,
            atol=config.ATOL.get('float32'),
        )

    def _tril_cholesky_to_tril_corr(self, x):
        last_dim = self.dim * (self.dim - 1) // 2
        x = x.reshape((last_dim,))
        x = vec_to_tril_matrix(x, self.dim, last_dim, last_dim, (1,), -1)
        diag = (1 - (x * x).sum(-1)).sqrt().diag_embed()
        x = x + diag
        x = x.reshape((self.dim, self.dim))
        return tril_matrix_to_vec(x @ x.T, -1)

    def _compute_jacobian(self, x):
        if x.stop_gradient is not False:
            x.stop_gradient = False
        jacobian_matrix = []
        outputs = self._tril_cholesky_to_tril_corr(x)
        for i in range(outputs.shape[0]):
            grad = paddle.grad(
                outputs=outputs[i], inputs=x, create_graph=False
            )[0]
            jacobian_matrix.append(grad)
        J = paddle.stack(jacobian_matrix, axis=0)
        logabsdet = paddle.linalg.slogdet(J)
        return logabsdet


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME),
    [
        ('lkj_cholesky_test_err'),
    ],
)
class LKJCholeskyTestError(unittest.TestCase):
    @parameterize.parameterize_func(
        [
            (1, 1.0, ValueError),  # dim < 2
            (3.0, 1.0, TypeError),  # dim is float
            (3, -1.0, ValueError),  # concentration < 0
        ]
    )
    def test_bad_parameter(self, dim, concentration, error):
        with paddle.base.dygraph.guard(self.place):
            self.assertRaises(
                error, lkj_cholesky.LKJCholesky, dim, concentration
            )

    @parameterize.parameterize_func([(10,)])  # not sequence object sample shape
    def test_bad_sample_shape(self, shape):
        with paddle.base.dygraph.guard(self.place):
            lkj = lkj_cholesky.LKJCholesky(3)
            self.assertRaises(TypeError, lkj.sample, shape)


if __name__ == '__main__':
    unittest.main()
