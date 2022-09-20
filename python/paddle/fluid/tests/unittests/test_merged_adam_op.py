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
import paddle
import numpy as np
from paddle import _C_ops, _legacy_C_ops
from paddle.fluid.framework import in_dygraph_mode


def run_adam_op(params,
                grads,
                lrs,
                moment1s,
                moment2s,
                beta1_pows,
                beta2_pows,
                master_params,
                epsilon,
                beta1,
                beta2,
                place,
                multi_precision=False,
                use_merged=False):
    assert len(params) == len(grads)
    assert len(params) == len(lrs)
    assert len(params) == len(moment1s)
    assert len(params) == len(moment2s)
    assert len(params) == len(beta1_pows)
    assert len(params) == len(beta1_pows)
    assert len(params) == len(master_params)
    paddle.disable_static()
    paddle.set_device(place)

    param_vars = [paddle.fluid.dygraph.to_variable(p) for p in params]
    grad_vars = [paddle.fluid.dygraph.to_variable(g) for g in grads]
    lr_vars = [paddle.fluid.dygraph.to_variable(l) for l in lrs]
    moment1_vars = [paddle.fluid.dygraph.to_variable(m) for m in moment1s]
    moment2_vars = [paddle.fluid.dygraph.to_variable(m) for m in moment2s]
    beta1_pow_vars = [paddle.fluid.dygraph.to_variable(b) for b in beta1_pows]
    beta2_pow_vars = [paddle.fluid.dygraph.to_variable(b) for b in beta2_pows]
    master_param_vars = [
        paddle.fluid.dygraph.to_variable(m_p) for m_p in master_params
    ]

    if not use_merged:
        for i in range(len(param_vars)):
            _, _, _, _, _, _ = _legacy_C_ops.adam(
                param_vars[i], grad_vars[i], lr_vars[i], moment1_vars[i],
                moment2_vars[i], beta1_pow_vars[i], beta2_pow_vars[i],
                master_param_vars[i], param_vars[i], moment1_vars[i],
                moment2_vars[i], beta1_pow_vars[i], beta2_pow_vars[i],
                master_param_vars[i], 'epsilon', epsilon, 'beta1', beta1,
                'beta2', beta2, 'multi_precision', multi_precision)
    else:
        if in_dygraph_mode():
            _, _, _, _, _, _ = _C_ops.merged_adam_(
                param_vars, grad_vars, lr_vars, moment1_vars, moment2_vars,
                beta1_pow_vars, beta2_pow_vars, master_param_vars, beta1, beta2,
                epsilon, multi_precision, False)
        else:
            _, _, _, _, _, _ = _legacy_C_ops.merged_adam(
                param_vars, grad_vars, lr_vars, moment1_vars, moment2_vars,
                beta1_pow_vars, beta2_pow_vars, master_param_vars, param_vars,
                moment1_vars, moment2_vars, beta1_pow_vars, beta2_pow_vars,
                master_param_vars, 'epsilon', epsilon, 'beta1', beta1, 'beta2',
                beta2, 'multi_precision', multi_precision)

    outputs = {
        'ParamOut': param_vars,
        'Moment1Out': moment1_vars,
        'Moment2Out': moment2_vars,
        'Beta1PowOut': beta1_pow_vars,
        'Beta2PowOut': beta2_pow_vars,
        'MasterParamOut': master_param_vars
    }

    return outputs


class TestMergedAdam(unittest.TestCase):

    def setUp(self):
        paddle.disable_static()
        self.shapes = [[3, 4], [2, 7], [5, 6], [7, 8]]
        self.seed = 10

    def gen_rand_data(self, shapes, dtype):
        return [np.random.random(s).astype(dtype) for s in shapes]

    def prepare_data(self, shapes, multi_precision, seed, place):
        np.random.seed(seed)
        mp_dtype = np.float32
        dtype = np.float16 if multi_precision and place == 'gpu' else np.float32
        params = self.gen_rand_data(shapes, dtype)
        grads = self.gen_rand_data(shapes, dtype)
        lrs = self.gen_rand_data([[1], [1], [1], [1]], mp_dtype)
        moment1s = self.gen_rand_data(shapes, mp_dtype)
        moment2s = self.gen_rand_data(shapes, mp_dtype)
        beta1_pows = self.gen_rand_data([[1], [1], [1], [1]], mp_dtype)
        beta2_pows = self.gen_rand_data([[1], [1], [1], [1]], mp_dtype)
        master_params = [p.astype(mp_dtype) for p in params]
        return params, grads, lrs, moment1s, moment2s, beta1_pows, beta2_pows, master_params

    def check_with_place(self, place, multi_precision):
        params, grads, lrs, moment1s, moment2s, beta1_pows, beta2_pows, master_params = self.prepare_data(
            self.shapes, multi_precision, self.seed, place)

        def run_op(use_merged):
            return run_adam_op(params=params,
                               grads=grads,
                               lrs=lrs,
                               moment1s=moment1s,
                               moment2s=moment2s,
                               beta1_pows=beta1_pows,
                               beta2_pows=beta2_pows,
                               master_params=master_params,
                               epsilon=0.9,
                               beta1=0.9,
                               beta2=0.99,
                               place=place,
                               multi_precision=multi_precision,
                               use_merged=use_merged)

        outs1 = run_op(True)
        outs2 = run_op(False)
        self.assertEqual(len(outs1), len(outs2))

        for key in outs1.keys():
            value1 = outs1[key]
            value2 = outs2[key]
            for i in range(len(value1)):
                if place == 'gpu':
                    np.testing.assert_array_equal(value1[i], value2[i])
                else:
                    np.testing.assert_allclose(value1[i],
                                               value2[i],
                                               rtol=1e-05,
                                               atol=1e-07)

    def get_places(self):
        places = ['cpu']
        if paddle.is_compiled_with_cuda():
            places.append('gpu')
        return places

    def test_main(self):
        for multi_precision in [False, True]:
            for place in self.get_places():
                self.check_with_place(place, multi_precision)


if __name__ == "__main__":
    unittest.main()
