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

import os
import sys
import unittest

import numpy as np
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test_xpu import XPUOpTest

sys.path.append("../deprecated/legacy_test")
from test_attribute_var import UnittestBase
from utils import static_guard

import paddle
from paddle.base import Program, program_guard


def pad_wrapper(x, paddings, pad_value):
    return paddle.nn.functional.pad(
        x, pad=list(paddings), mode='constant', value=pad_value
    )


paddle.enable_static()


class XPUTestPadOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = "pad"
        self.use_dynamic_create_class = False

    class TestPadOp(XPUOpTest):
        def setUp(self):
            self.op_type = "pad"
            self.place = paddle.XPUPlace(0)
            self.python_api = pad_wrapper
            self.public_python_api = pad_wrapper
            self.init_dtype()
            self.init_test_case()
            self.init_data()

        def init_dtype(self):
            self.dtype = self.in_type

        def init_test_case(self):
            self.shape = (16, 16)
            self.paddings = [(0, 1), (2, 3)]
            self.pad_value = 0.0

        def init_data(self):
            self.inputs = {'X': np.random.random(self.shape).astype(self.dtype)}
            self.outputs = {
                'Out': np.pad(
                    self.inputs['X'],
                    self.paddings,
                    mode='constant',
                    constant_values=self.pad_value,
                )
            }
            self.attrs = {
                'paddings': list(np.array(self.paddings).flatten()),
                'pad_value': self.pad_value,
            }

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad_normal(self):
            self.check_grad_with_place(self.place, ['X'], 'Out')

    class TestCase1(TestPadOp):
        def init_test_case(self):
            self.shape = (2, 3, 4, 5)
            self.paddings = [(0, 1), (2, 3), (2, 1), (1, 1)]
            self.pad_value = 0.5

    class TestCase2(TestPadOp):
        def init_test_case(self):
            self.shape = (5, 5, 5)
            self.paddings = [(0, 0), (0, 0), (1, 2)]
            self.pad_value = 1.0

    class TestCase3(TestPadOp):
        def init_test_case(self):
            self.shape = 100
            self.paddings = [(0, 1)]
            self.pad_value = 0.9

    class TestPadOpError(unittest.TestCase):
        def test_errors(self):
            with static_guard():
                with program_guard(Program(), Program()):
                    input_data = np.random.random((2, 2)).astype("float32")

                def test_Variable():
                    paddle.nn.functional.pad(x=input_data, pad=[1, 1, 1, 1])

                self.assertRaises(TypeError, test_Variable)

                data = paddle.static.data(
                    name='data', shape=[4], dtype='float16'
                )
                paddle.nn.functional.pad(x=data, pad=[0, 1])

    class TestPaddingValueTensor(UnittestBase):
        def init_info(self):
            self.shapes = [[2, 4]]
            self.save_path = os.path.join(
                self.temp_dir.name, self.path_prefix()
            )

        def test_static(self):
            with static_guard():
                main_prog = Program()
                startup_prog = Program()
                with program_guard(main_prog, startup_prog):
                    fc = paddle.nn.Linear(4, 10)
                    x = paddle.randn([2, 4])
                    x.stop_gradient = False
                    feat = fc(x)  # [2,3,10]

                    out = self.call_func(feat)

                    sgd = paddle.optimizer.SGD()
                    sgd.minimize(paddle.mean(out))
                    if not paddle.framework.use_pir_api():
                        self.assertTrue(self.var_prefix() in str(main_prog))
                    exe = paddle.static.Executor(paddle.XPUPlace(0))
                    exe.run(startup_prog)
                    res = exe.run(fetch_list=[feat, out])
                    gt = np.pad(
                        res[0], [1, 1], 'constant', constant_values=[1.0, 1.0]
                    )
                    np.testing.assert_allclose(res[1], gt)
                    paddle.static.save_inference_model(
                        self.save_path, [x], [feat, out], exe
                    )
                    # Test for Inference Predictor
                    infer_outs = self.infer_prog()
                    gt = np.pad(
                        infer_outs[0],
                        [1, 1],
                        'constant',
                        constant_values=[1.0, 1.0],
                    )
                    np.testing.assert_allclose(infer_outs[1], gt)

        def path_prefix(self):
            return 'padding_value'

        def var_prefix(self):
            return "Var["

        def call_func(self, x):
            padding_value = paddle.assign([1.0])
            out = paddle.nn.functional.pad(
                x, pad=[1, 1, 1, 1], value=padding_value, mode='constant'
            )
            return out

    class TestPaddingValueTensor2(TestPaddingValueTensor):
        def call_func(self, x):
            padding_value = paddle.assign([1.0])
            # test for int value
            tmp = paddle.nn.functional.pad(x, pad=[1, 1, 1, 1], value=1)
            out = paddle.nn.functional.pad(
                x, pad=[1, 1, 1, 1], value=padding_value
            )
            return out

    class TestPaddingValueTensor3(unittest.TestCase):
        def test_static(self):
            with static_guard():
                np_x = np.random.random((16, 16)).astype('float32')
                main_prog = Program()
                startup_prog = Program()
                with program_guard(main_prog, startup_prog):
                    x = paddle.assign(np_x).astype('float32')
                    pad_value = paddle.assign([0.0]).astype('float64')
                    y = paddle.nn.functional.pad(
                        x, [0, 1, 2, 3], value=pad_value
                    )
                    loss = y.sum()
                    optimize_ops, params_grads = paddle.optimizer.SGD(
                        0.01
                    ).minimize(loss)

                exe = paddle.static.Executor(paddle.XPUPlace(0))
                res = exe.run(
                    main_prog, fetch_list=[y] + [g for p, g in params_grads]
                )
                pd_out = res[0]
                np_out = np.pad(np_x, [(0, 1), (2, 3)], constant_values=0.0)
                np.testing.assert_allclose(pd_out, np_out)


support_types = get_xpu_op_support_types("pad")
for stype in support_types:
    create_test_class(globals(), XPUTestPadOp, stype)

if __name__ == "__main__":
    unittest.main()
