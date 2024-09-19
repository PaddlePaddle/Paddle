# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import paddle

# from op import Operator
# from op_test_xpu import XPUOpTest
from paddle import _C_ops, _legacy_C_ops


def run_adam_op(
    params,
    grads,
    lrs,
    moment1s,
    moment2s,
    moment2s_max,
    beta1_pows,
    beta2_pows,
    master_params,
    epsilon,
    beta1,
    beta2,
    place,
    multi_precision=False,
    use_merged=False,
    amsgrad=False,
):
    assert len(params) == len(grads)
    assert len(params) == len(lrs)
    assert len(params) == len(moment1s)
    assert len(params) == len(moment2s)
    assert len(params) == len(beta1_pows)
    assert len(params) == len(beta1_pows)
    assert len(params) == len(master_params)
    paddle.disable_static()
    paddle.set_device(place)

    param_vars = [paddle.to_tensor(p) for p in params]
    grad_vars = [paddle.to_tensor(g) for g in grads]
    lr_vars = [paddle.to_tensor(l) for l in lrs]
    moment1_vars = [paddle.to_tensor(m) for m in moment1s]
    moment2_vars = [paddle.to_tensor(m) for m in moment2s]
    moment2_max_vars = [paddle.to_tensor(m) for m in moment2s_max]
    beta1_pow_vars = [paddle.to_tensor(b) for b in beta1_pows]
    beta2_pow_vars = [paddle.to_tensor(b) for b in beta2_pows]
    master_param_vars = [paddle.to_tensor(m_p) for m_p in master_params]

    if not use_merged:
        for i in range(len(param_vars)):
            _, _, _, _, _, _, _ = _legacy_C_ops.adam(
                param_vars[i],
                grad_vars[i],
                lr_vars[i],
                moment1_vars[i],
                moment2_vars[i],
                moment2_max_vars[i],
                beta1_pow_vars[i],
                beta2_pow_vars[i],
                master_param_vars[i],
                param_vars[i],
                moment1_vars[i],
                moment2_vars[i],
                moment2_max_vars[i],
                beta1_pow_vars[i],
                beta2_pow_vars[i],
                master_param_vars[i],
                'epsilon',
                epsilon,
                'beta1',
                beta1,
                'beta2',
                beta2,
                'multi_precision',
                False,
                'amsgrad',
                amsgrad,
            )
    else:
        _, _, _, _, _, _, _ = _C_ops.merged_adam_(
            param_vars,
            grad_vars,
            lr_vars,
            moment1_vars,
            moment2_vars,
            moment2_max_vars,
            beta1_pow_vars,
            beta2_pow_vars,
            master_param_vars,
            beta1,
            beta2,
            epsilon,
            False,
            False,
            amsgrad,
        )

    outputs = {
        'ParamOut': param_vars,
        'Moment1Out': moment1_vars,
        'Moment2Out': moment2_vars,
        'Moment2MaxOut': moment2_max_vars,
        'Beta1PowOut': beta1_pow_vars,
        'Beta2PowOut': beta2_pow_vars,
        'MasterParamOut': master_param_vars,
    }

    return outputs


class XPUTestMergedAdamWrapper(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'merged_adam'
        self.use_dynamic_create_class = False

    class XPUTestMergedAdamBase(unittest.TestCase):
        def setUp(self):
            self.shapes = [[3, 4], [2, 7], [5, 6], [7, 8]]
            self.seed = 10

        def gen_rand_data(self, shapes, dtype):
            return [np.random.random(s).astype(dtype) for s in shapes]

        def gen_zero_data(self, shapes, dtype):
            return [np.zeros(s).astype(dtype) for s in shapes]

        def prepare_data(self, shapes, seed):
            np.random.seed(seed)
            mp_dtype = np.float32
            dtype = np.float32
            params = self.gen_rand_data(shapes, dtype)
            grads = self.gen_rand_data(shapes, dtype)
            learning_rate = self.gen_rand_data([[1]], mp_dtype)
            lrs = [learning_rate.copy() for _ in shapes]
            moment1s = self.gen_rand_data(shapes, mp_dtype)
            moment2s = self.gen_rand_data(shapes, mp_dtype)
            moment2s_max = self.gen_zero_data(shapes, mp_dtype)
            beta1_pow = self.gen_rand_data([[1]], mp_dtype)
            beta2_pow = self.gen_rand_data([[1]], mp_dtype)
            beta1_pows = [beta1_pow.copy() for _ in shapes]
            beta2_pows = [beta2_pow.copy() for _ in shapes]
            master_params = [p.astype(mp_dtype) for p in params]
            return (
                params,
                grads,
                lrs,
                moment1s,
                moment2s,
                moment2s_max,
                beta1_pows,
                beta2_pows,
                master_params,
            )

        def check_with_place(self):
            (
                params,
                grads,
                lrs,
                moment1s,
                moment2s,
                moment2s_max,
                beta1_pows,
                beta2_pows,
                master_params,
            ) = self.prepare_data(self.shapes, self.seed)

            def run_op(use_merged, place):
                return run_adam_op(
                    params=params,
                    grads=grads,
                    lrs=lrs,
                    moment1s=moment1s,
                    moment2s=moment2s,
                    moment2s_max=moment2s_max,
                    beta1_pows=beta1_pows,
                    beta2_pows=beta2_pows,
                    master_params=master_params,
                    epsilon=0.9,
                    beta1=0.9,
                    beta2=0.99,
                    place=place,
                    multi_precision=False,
                    use_merged=use_merged,
                    amsgrad=False,  # Currently, xpu NOT support amsgrad.
                )

            outs1 = run_op(True, "xpu")
            outs2 = run_op(True, "cpu")
            outs3 = run_op(False, "xpu")
            outs4 = run_op(False, "cpu")

            self.assertEqual(len(outs1), len(outs2))
            self.assertEqual(len(outs1), len(outs3))
            self.assertEqual(len(outs1), len(outs4))

            for key in outs1.keys():
                if key in ['Moment2MaxOut']:
                    continue

                value1 = outs1[key]
                value2 = outs2[key]
                value3 = outs3[key]
                value4 = outs4[key]
                for i in range(len(value1)):
                    np.testing.assert_allclose(
                        value1[i], value2[i], rtol=1e-05, atol=1e-07
                    )
                    np.testing.assert_allclose(
                        value1[i], value3[i], rtol=1e-05, atol=1e-07
                    )
                    np.testing.assert_allclose(
                        value1[i], value4[i], rtol=1e-05, atol=1e-07
                    )

    class TestMergedAdamOp(XPUTestMergedAdamBase):
        def setUp(self):
            super().setUp()
            self.set_case()

        def set_case(self):
            self.shapes = [[3, 4], [2, 7], [5, 6], [7, 8]]
            self.seed = 10

        def testalltype(self):
            self.check_with_place()

    class TestMergedAdam1(TestMergedAdamOp):
        def set_case(self):
            self.shapes = [[3, 4]]

    class TestMergedAdam2(TestMergedAdamOp):
        def set_case(self):
            self.shapes = [[3, 4], [2, 7]]

    class TestMergedAdam3(TestMergedAdamOp):
        def set_case(self):
            self.shapes = [[3, 4], [2, 4], [3, 4]]

    class TestMergedAdam4(TestMergedAdamOp):
        def set_case(self):
            self.shapes = [[3, 4], [2, 7], [5, 6], [9, 9]]


support_types = get_xpu_op_support_types('merged_adam')
for stype in support_types:
    create_test_class(globals(), XPUTestMergedAdamWrapper, stype)

if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
