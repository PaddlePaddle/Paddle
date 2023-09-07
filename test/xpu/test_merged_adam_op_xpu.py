import unittest

import numpy as np
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
# from op import Operator
# from op_test_xpu import XPUOpTest
from paddle import _C_ops, _legacy_C_ops
from paddle.fluid.framework import in_dygraph_mode
import paddle
from paddle.fluid import core

def run_adam_op(
    params,
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
    use_merged=False,
):
    assert len(params) == len(grads)
    assert len(params) == len(lrs)
    assert len(params) == len(moment1s)
    assert len(params) == len(moment2s)
    assert len(params) == len(beta1_pows)
    assert len(params) == len(beta1_pows)
    assert len(params) == len(master_params)
    paddle.disable_static()
    paddle.set_device("xpu")

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
                param_vars[i],
                grad_vars[i],
                lr_vars[i],
                moment1_vars[i],
                moment2_vars[i],
                beta1_pow_vars[i],
                beta2_pow_vars[i],
                master_param_vars[i],
                param_vars[i],
                moment1_vars[i],
                moment2_vars[i],
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
            )
    else:
        if in_dygraph_mode():
            _, _, _, _, _, _ = _C_ops.merged_adam_(
                param_vars,
                grad_vars,
                lr_vars,
                moment1_vars,
                moment2_vars,
                beta1_pow_vars,
                beta2_pow_vars,
                master_param_vars,
                beta1,
                beta2,
                epsilon,
                False,
                False,
            )
        else:
            _, _, _, _, _, _ = _legacy_C_ops.merged_adam(
                param_vars,
                grad_vars,
                lr_vars,
                moment1_vars,
                moment2_vars,
                beta1_pow_vars,
                beta2_pow_vars,
                master_param_vars,
                param_vars,
                moment1_vars,
                moment2_vars,
                beta1_pow_vars,
                beta2_pow_vars,
                master_param_vars,
                'epsilon',
                epsilon,
                'beta1',
                beta1,
                'beta2',
                beta2,
                'multi_precision',
                multi_precision,
            )

    outputs = {
        'ParamOut': param_vars,
        'Moment1Out': moment1_vars,
        'Moment2Out': moment2_vars,
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
            paddle.disable_static()
            self.shapes = [[3, 4], [2, 7], [5, 6], [7, 8]]
            self.seed = 10
            self.place = paddle.fluid.XPUPlace(0)
            self.__class__.use_xpu = True

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
            return (
                params,
                grads,
                lrs,
                moment1s,
                moment2s,
                beta1_pows,
                beta2_pows,
                master_params,
            )

        def check_with_place(self, place, multi_precision):
            (
                params,
                grads,
                lrs,
                moment1s,
                moment2s,
                beta1_pows,
                beta2_pows,
                master_params,
            ) = self.prepare_data(self.shapes, multi_precision, self.seed, place)

            def run_op(use_merged):
                return run_adam_op(
                    params=params,
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
                    use_merged=use_merged,
                )

            outs1 = run_op(True)
            outs2 = run_op(False)
            self.assertEqual(len(outs1), len(outs2))
    
    class TestMergedAdamOp(XPUTestMergedAdamBase):
        def setUp(self):
            super().setUp()
            self.set_case()

        def set_case(self):
            self.shapes = [[3, 4], [2, 7], [5, 6], [7, 8]]
            self.place = paddle.fluid.XPUPlace(0)
            self.seed = 1

        def testalltype(self):
            self.check_with_place(self.place, self.in_type)

    class TestMergedAdam1(TestMergedAdamOp):
        def set_case(self):
            self.shapes = [[3, 4], [2, 7], [5, 6], [7, 8]]

    class TestMergedAdam2(TestMergedAdamOp):
        def set_case(self):
            self.shapes = [[3, 4], [2, 7], [5, 6], [3, 4]]

    class TestMergedAdam3(TestMergedAdamOp):
        def set_case(self):
            self.shapes = [[3, 4], [2, 4], [3, 4], [3, 4]]

    class TestMergedAdam4(TestMergedAdamOp):
        def set_case(self):
            self.shapes = [[3, 4], [2, 7], [5, 6,], [9, 9]]


support_types = get_xpu_op_support_types('merged_adam')
for stype in support_types:
    create_test_class(globals(), XPUTestMergedAdamWrapper, stype)

if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()