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

paddle.enable_static()

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
                dtype='float64',
                max=1.0,
                min=0,
            ),
        ),
    ],
)
class TestLKJCholeskyShape(unittest.TestCase):
    def setUp(self):
        self.program = paddle.static.Program()
        self.executor = paddle.static.Executor(self.place)
        with paddle.static.program_guard(self.program):
            conc = paddle.static.data(
                'concentration',
                self.concentration.shape,
                self.concentration.dtype,
            )

            self._paddle_lkj_cholesky_onion = lkj_cholesky.LKJCholesky(
                2, conc, 'onion'
            )

            self._paddle_lkj_cholesky_cvine = lkj_cholesky.LKJCholesky(
                2, conc, 'cvine'
            )

            self.feeds = {
                'concentration': self.concentration,
            }

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
            {
                'input': (2, 2),
                'expect': (2, 2, *extra_shape),
            },
        ]
        return cases

    def test_onion_sample_shape(self):
        sample_method = 'onion'
        self._test_sample_shape(sample_method)

    def test_cvine_sample_shape(self):
        sample_method = 'cvine'
        self._test_sample_shape(sample_method)

    def _test_sample_shape(self, sample_method):
        with paddle.static.program_guard(self.program):
            if sample_method == 'cvine':
                self._paddle_lkj_cholesky = self._paddle_lkj_cholesky_cvine
            else:
                self._paddle_lkj_cholesky = self._paddle_lkj_cholesky_onion

            cases = self.gen_cases()
            for case in cases:
                [data] = self.executor.run(
                    self.program,
                    feed=self.feeds,
                    fetch_list=self._paddle_lkj_cholesky.sample(
                        case.get('input')
                    ),
                )
            self.assertTrue(tuple(data.shape) == case.get('expect'))


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME),
    [
        ('test_log_prob'),
    ],
)
class TestLKJCholeskyLogProb(unittest.TestCase):
    def setUp(self):
        self.program = paddle.static.Program()
        self.executor = paddle.static.Executor(self.place)
        with paddle.static.program_guard(self.program):
            self.concentration = [1.0]
            self.feeds = {
                'concentration': self.concentration,
            }

    def test_log_prob(self):
        self.dim = 2
        self._test_log_prob('onion')

    def _test_log_prob(self, sample_method):
        with paddle.static.program_guard(self.program):
            self._paddle_lkj_cholesky = lkj_cholesky.LKJCholesky(
                self.dim, self.concentration, sample_method
            )
            log_probs = []
            for i in range(2):
                [sample] = self.executor.run(
                    self.program,
                    feed=self.feeds,
                    fetch_list=self._paddle_lkj_cholesky.sample(),
                )
                if paddle.framework.use_pir_api():
                    block = self.program.global_block()
                    for op in block.ops:
                        if op.name() == 'pd_op.fetch':
                            block.remove_op(op)
                sample = paddle.to_tensor(sample)
                [log_prob] = self.executor.run(
                    self.program,
                    feed=self.feeds,
                    fetch_list=self._paddle_lkj_cholesky.log_prob(sample),
                )
                if paddle.framework.use_pir_api():
                    block = self.program.global_block()
                    for op in block.ops:
                        if op.name() == 'pd_op.fetch':
                            block.remove_op(op)

                sample_tril = tril_matrix_to_vec(sample, diag=-1)

                # log_abs_det_jacobian
                logabsdet = []

                [logabsdet_value, J] = self.executor.run(
                    self.program,
                    feed={'sample_tril': sample_tril},
                    fetch_list=self._compute_jacobian(sample_tril),
                )
                if paddle.framework.use_pir_api():
                    block = self.program.global_block()
                    for op in block.ops:
                        if op.name() == 'pd_op.fetch':
                            block.remove_op(op)

                logabsdet.append(logabsdet_value[1])

                log_probs.append(log_prob - logabsdet)
            max_abs_error = np.max(np.abs(log_probs[0] - log_probs[1]))
            self.assertAlmostEqual(max_abs_error, 0, places=3)

    def _tril_cholesky_to_tril_corr(self, x, last_dim):
        x = x.reshape((last_dim,))
        x = vec_to_tril_matrix(x, self.dim, last_dim, last_dim, (1,), -1)
        diag = (1 - (x * x).sum(-1)).sqrt().diag_embed()
        x = x + diag
        x = x.reshape((self.dim, self.dim))
        return tril_matrix_to_vec(x @ x.T, -1)

    def _compute_jacobian(self, x):
        last_dim = self.dim * (self.dim - 1) // 2
        if x.stop_gradient is not False:
            x.stop_gradient = False
        jacobian_matrix = []
        outputs = self._tril_cholesky_to_tril_corr(x, last_dim)
        for i in range(outputs.shape[0]):
            grad = paddle.static.gradients(outputs[i], x)[0]
            grad = grad.reshape((last_dim,))
            jacobian_matrix.append(grad)
        J = paddle.stack(jacobian_matrix, axis=0)
        logabsdet = paddle.linalg.slogdet(J)
        return logabsdet, J


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME),
    [
        ('lkj_cholesky_test_err'),
    ],
)
class LKJCholeskyTestError(unittest.TestCase):
    def setUp(self):
        self.program = paddle.static.Program()

    @parameterize.parameterize_func(
        [
            (1, ValueError),  # dim < 2
            (3.0, TypeError),  # dim is float
        ]
    )
    def test_bad_parameter(self, dim, error):
        with paddle.static.program_guard(self.program):
            self.assertRaises(error, lkj_cholesky.LKJCholesky, dim)

    @parameterize.parameterize_func([(10,)])  # not sequence object sample shape
    def test_bad_sample_shape(self, shape):
        with paddle.static.program_guard(self.program):
            lkj = lkj_cholesky.LKJCholesky(3)
            self.assertRaises(TypeError, lkj.sample, shape)


if __name__ == '__main__':
    unittest.main()
