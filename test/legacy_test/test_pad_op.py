#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import unittest

import numpy as np
from op_test import OpTest, convert_float_to_uint16
from test_attribute_var import UnittestBase
from utils import static_guard

import paddle
from paddle.base import Program, core, program_guard


def pad_wrapper(x, paddings, pad_value):
    return paddle.nn.functional.pad(
        x, pad=list(paddings), mode='constant', value=pad_value
    )


class TestPadOp(OpTest):
    def setUp(self):
        self.initTestCase()
        self.dtype = self.get_dtype()
        self.op_type = "pad"
        self.python_api = pad_wrapper
        self.inputs = {
            'X': np.random.random(self.shape).astype(self.dtype),
        }
        self.attrs = {}
        self.attrs['paddings'] = list(np.array(self.paddings).flatten())
        self.attrs['pad_value'] = self.pad_value
        self.outputs = {
            'Out': np.pad(
                self.inputs['X'],
                self.paddings,
                mode='constant',
                constant_values=self.pad_value,
            )
        }
        self.prim_op_type = "prim"
        self.public_python_api = pad_wrapper

    def get_dtype(self):
        return np.float64

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X'], 'Out', check_prim=True)

    def initTestCase(self):
        self.shape = (16, 16)
        self.paddings = [(0, 1), (2, 3)]
        self.pad_value = 0.0


class TestCase1(TestPadOp):
    def initTestCase(self):
        self.shape = (2, 3, 4, 5)
        self.paddings = [(0, 1), (2, 3), (2, 1), (1, 1)]
        self.pad_value = 0.5


class TestCase2(TestPadOp):
    def initTestCase(self):
        self.shape = (5, 5, 5)
        self.paddings = [(0, 0), (0, 0), (1, 2)]
        self.pad_value = 1.0


class TestCase3(TestPadOp):
    def initTestCase(self):
        self.shape = 100
        self.paddings = [(0, 1)]
        self.pad_value = 0.9


# ----------------Pad Fp16----------------


def create_test_fp16(parent):
    @unittest.skipIf(
        not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
    )
    class TestPadFp16(parent):
        def get_dtype(self):
            return np.float16

        def test_check_grad_normal(self):
            self.check_grad(['X'], 'Out', check_prim=True)

    cls_name = "{}_{}".format(parent.__name__, "Fp16")
    TestPadFp16.__name__ = cls_name
    globals()[cls_name] = TestPadFp16


create_test_fp16(TestPadOp)
create_test_fp16(TestCase1)
create_test_fp16(TestCase2)
create_test_fp16(TestCase3)


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
        self.save_path = os.path.join(self.temp_dir.name, self.path_prefix())

    def test_static(self):
        with static_guard():
            main_prog = Program()
            starup_prog = Program()
            with program_guard(main_prog, starup_prog):
                fc = paddle.nn.Linear(4, 10)
                x = paddle.randn([2, 4])
                x.stop_gradient = False
                feat = fc(x)  # [2,3,10]

                out = self.call_func(feat)

                sgd = paddle.optimizer.SGD()
                sgd.minimize(paddle.mean(out))
                self.assertTrue(self.var_prefix() in str(main_prog))

                exe = paddle.static.Executor()
                exe.run(starup_prog)
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
        out = paddle.nn.functional.pad(x, pad=[1, 1, 1, 1], value=padding_value)
        return out


class TestPaddingValueTensor3(unittest.TestCase):
    def test_static(self):
        with static_guard():
            np_x = np.random.random((16, 16)).astype('float32')
            main_prog = Program()
            starup_prog = Program()
            with program_guard(main_prog, starup_prog):
                x = paddle.assign(np_x).astype('float32')
                pad_value = paddle.assign([0.0]).astype('float64')
                y = paddle.nn.functional.pad(x, [0, 1, 2, 3], value=pad_value)
                loss = y.sum()
                optimize_ops, params_grads = paddle.optimizer.SGD(
                    0.01
                ).minimize(loss)

            exe = paddle.static.Executor(paddle.CPUPlace())
            res = exe.run(
                main_prog, fetch_list=[y] + [g for p, g in params_grads]
            )
            pd_out = res[0]
            np_out = np.pad(np_x, [(0, 1), (2, 3)], constant_values=0.0)
            np.testing.assert_allclose(pd_out, np_out)


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestPadBP16Op(OpTest):
    def setUp(self):
        self.initTestCase()
        self.dtype = np.uint16
        self.op_type = "pad"
        self.python_api = pad_wrapper
        x = np.random.random(self.shape).astype(np.float32)
        self.attrs = {}
        self.attrs['paddings'] = list(np.array(self.paddings).flatten())
        self.attrs['pad_value'] = self.pad_value
        out = np.pad(
            x, self.paddings, mode='constant', constant_values=self.pad_value
        )
        self.inputs = {'X': convert_float_to_uint16(x)}
        self.outputs = {'Out': convert_float_to_uint16(out)}
        self.prim_op_type = "prim"
        self.public_python_api = pad_wrapper
        self.if_enable_cinn()

    def if_enable_cinn(self):
        pass

    def initTestCase(self):
        self.shape = (16, 16)
        self.paddings = [(0, 1), (2, 3)]
        self.pad_value = 0.0

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(place)

    def test_check_grad(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(place, ['X'], 'Out', check_prim=True)


if __name__ == '__main__':
    # paddle.enable_static()
    unittest.main()
