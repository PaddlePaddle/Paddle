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


import unittest

import numpy as np
import scipy.stats
from op_test import OpTest, convert_float_to_uint16, convert_uint16_to_float

import paddle
from paddle.base import core

paddle.enable_static()


class TestDirichletOp(OpTest):
    # Because dirichlet random sample have not gradient, we skip gradient check.
    no_need_check_grad = True

    def setUp(self):
        self.op_type = "dirichlet"
        self.alpha = np.array((1.0, 2.0))
        self.sample_shape = (100000, 2)

        self.inputs = {'Alpha': np.broadcast_to(self.alpha, self.sample_shape)}
        self.attrs = {}
        self.outputs = {'Out': np.zeros(self.sample_shape)}

    def test_check_output(self):
        self.check_output_customized(self._hypothesis_testing)

    def _hypothesis_testing(self, outs):
        self.assertEqual(outs[0].shape, self.sample_shape)
        self.assertTrue(np.all(outs[0] > 0.0))
        self.assertLess(
            scipy.stats.kstest(
                outs[0][:, 0],
                # scipy dirichlet have not cdf, use beta to replace it.
                scipy.stats.beta(a=self.alpha[0], b=self.alpha[1]).cdf,
            )[0],
            0.01,
        )


class TestDirichletFP16Op(OpTest):
    # Because dirichlet random sample have not gradient, we skip gradient check.
    no_need_check_grad = True

    def setUp(self):
        self.op_type = "dirichlet"
        self.alpha = np.array((1.0, 2.0))
        self.sample_shape = (100000, 2)
        self.dtype = np.float16

        self.inputs = {
            'Alpha': np.broadcast_to(self.alpha, self.sample_shape).astype(
                self.dtype
            )
        }
        self.attrs = {}
        self.outputs = {'Out': np.zeros(self.sample_shape).astype(self.dtype)}

    def test_check_output(self):
        self.check_output_customized(self._hypothesis_testing)

    def _hypothesis_testing(self, outs):
        self.assertEqual(outs[0].shape, self.sample_shape)
        self.assertTrue(np.all(outs[0] > 0.0))
        self.assertLess(
            scipy.stats.kstest(
                outs[0][:, 0],
                # scipy dirichlet have not cdf, use beta to replace it.
                scipy.stats.beta(a=self.alpha[0], b=self.alpha[1]).cdf,
            )[0],
            0.01,
        )


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestDirichletBF16Op(OpTest):
    # Because dirichlet random sample have not gradient, we skip gradient check.
    no_need_check_grad = True

    def setUp(self):
        self.op_type = "dirichlet"
        self.alpha = np.array((1.0, 2.0))
        self.sample_shape = (10000, 2)
        self.dtype = np.uint16
        self.np_dtype = np.float32

        self.inputs = {
            'Alpha': np.broadcast_to(self.alpha, self.sample_shape).astype(
                self.np_dtype
            )
        }
        self.attrs = {}
        self.outputs = {
            'Out': np.zeros(self.sample_shape).astype(self.np_dtype)
        }
        self.inputs['Alpha'] = convert_float_to_uint16(self.inputs['Alpha'])
        self.outputs['Out'] = convert_float_to_uint16(self.outputs['Out'])
        self.place = core.CUDAPlace(0)

    def test_check_output(self):
        self.check_output_with_place_customized(
            self._hypothesis_testing, place=core.CUDAPlace(0)
        )

    def _hypothesis_testing(self, outs):
        outs = convert_uint16_to_float(outs)
        self.assertEqual(outs[0].shape, self.sample_shape)
        self.assertTrue(np.all(outs[0] > 0.0))
        self.assertLess(
            scipy.stats.kstest(
                outs[0][:, 0],
                # scipy dirichlet have not cdf, use beta to replace it.
                scipy.stats.beta(a=self.alpha[0], b=self.alpha[1]).cdf,
            )[0],
            0.3,  # The bfloat16 test difference is below 0.3
        )


if __name__ == '__main__':
    unittest.main()
