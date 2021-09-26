#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from op_test import OpTest, skip_check_grad_ci
import unittest
from paddle.fluid.op import Operator
from paddle.fluid import compiler, Program, program_guard


# cast output to complex for numpy.linalg.eig
def cast_to_complex(input, output):
    if (input.dtype == np.float32):
        output = output.astype(np.complex64)
    elif (input.dtype == np.float64):
        output = output.astype(np.complex128)
    return output


@skip_check_grad_ci(
    reason="Run backward without check grad for new as backward logic is not implemented yet"
)
class TestEigOp(OpTest):
    def setUp(self):
        paddle.enable_static()
        paddle.device.set_device("cpu")
        self.op_type = "eig"
        self.__class__.op_type = self.op_type
        self.init_input()
        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(self.x)}
        self.outputs = {'Eigenvalues': self.out[0], 'Eigenvectors': self.out[1]}

    def init_input(self):
        self.set_dtype()
        self.set_dims()
        self.x = np.random.random(self.shape).astype(self.dtype)
        self.out = np.linalg.eig(self.x)
        self.out = (cast_to_complex(self.x, self.out[0]),
                    cast_to_complex(self.x, self.out[1]))

    # for the real input, a customized checker is needed
    def checker(self, outs):
        actual_out_w = np.sort(outs[0].flatten())
        expect_out_w = np.sort(self.out[0].flatten())
        actual_out_v = outs[1].flatten()
        expect_out_v = self.out[1].flatten()

        length_w = len(expect_out_w)

        for i in range(length_w):
            # print(actual_out_w[i].real, "  ", expect_out_w[i].real)
            self.assertTrue(
                np.allclose(actual_out_w[i].real, expect_out_w[i].real, 1e-6,
                            1e-5),
                "The eigenvalues real part have diff: \nExpected " +
                str(actual_out_w[i].real) + "\n" + "But got: " +
                str(expect_out_w[i].real))
            self.assertTrue(
                np.allclose(actual_out_w[i].imag, expect_out_w[i].imag, 1e-6,
                            1e-5),
                "The eigenvalues image part have diff: \nExpected " +
                str(actual_out_w[i].imag) + "\n" + "But got: " +
                str(expect_out_w[i].imag))

        length_v = len(expect_out_v)
        act_v_real = np.sort(
            np.array([np.abs(actual_out_v[i].real) for i in range(length_v)]))
        act_v_imag = np.sort(
            np.array([np.abs(actual_out_v[i].imag) for i in range(length_v)]))
        exp_v_real = np.sort(
            np.array([np.abs(expect_out_v[i].real) for i in range(length_v)]))
        exp_v_imag = np.sort(
            np.array([np.abs(expect_out_v[i].imag) for i in range(length_v)]))

        for i in range(length_v):
            # print(act_v_real[i], "  ",  exp_v_real[i])
            self.assertTrue(
                np.allclose(act_v_real[i], exp_v_real[i], 1e-6, 1e-5),
                "The eigenvectors real part have diff: \nExpected " +
                str(act_v_real[i]) + "\n" + "But got: " + str(exp_v_real[i]))
            self.assertTrue(
                np.allclose(act_v_imag[i], exp_v_imag[i], 1e-6, 1e-5),
                "The eigenvectors image part have diff: \nExpected " +
                str(act_v_imag[i]) + "\n" + "But got: " + str(exp_v_imag[i]))

    def set_dtype(self):
        self.dtype = np.complex64

    def set_dims(self):
        self.shape = (3, 3)

    def test_check_output(self):
        self.check_output()

    def test_grad(self):
        pass


class TestEigBatchMarices(TestEigOp):
    def set_dims(self):
        self.shape = (3, 3, 3)


class TestComplex128(TestEigOp):
    def set_dtype(self):
        self.dtype = np.complex128


class TestFloat(TestEigOp):
    def set_dtype(self):
        self.dtype = np.float32

    def set_dims(self):
        self.shape = (32, 32)

    def test_check_output(self):
        self.check_output_with_place_customized(
            checker=self.checker, place=core.CPUPlace())


class TestDouble(TestEigOp):
    def set_dtype(self):
        self.dtype = np.float64

    def test_check_output(self):
        self.check_output_with_place_customized(
            checker=self.checker, place=core.CPUPlace())


class TestEigStatic(TestEigOp):
    def test_check_output_with_place(self):
        paddle.enable_static()
        place = core.CPUPlace()
        input_np = np.random.random([3, 3]).astype('float32')
        expect_val, expect_vec = np.linalg.eig(input_np)
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            input = fluid.data(name="input", shape=[3, 3], dtype='float32')
            act_val, act_vec = paddle.linalg.eig(input)

            exe = fluid.Executor(place)
            fetch_val, fetch_vec = exe.run(fluid.default_main_program(),
                                           feed={"input": input_np},
                                           fetch_list=[act_val, act_vec])
        self.assertTrue(
            np.allclose(expect_val, fetch_val, 1e-6, 1e-6),
            "The eigen values have diff: \nExpected " + str(expect_val) + "\n" +
            "But got: " + str(fetch_val))
        self.assertTrue(
            np.allclose(np.abs(expect_vec), np.abs(fetch_vec), 1e-6, 1e-6),
            "The eigen vectors have diff: \nExpected " +
            str(np.abs(expect_vec)) + "\n" + "But got: " +
            str(np.abs(fetch_vec)))


class TestEigWrongDimsError(unittest.TestCase):
    def test_error(self):
        paddle.device.set_device("cpu")
        paddle.disable_static()
        a = np.random.random((3)).astype('float32')
        x = paddle.to_tensor(a)
        self.assertRaises(ValueError, paddle.linalg.eig, x)


class TestEigNotSquareError(unittest.TestCase):
    def test_error(self):
        paddle.device.set_device("cpu")
        paddle.disable_static()
        a = np.random.random((1, 2, 3)).astype('float32')
        x = paddle.to_tensor(a)
        self.assertRaises(ValueError, paddle.linalg.eig, x)


class TestEigUnsupportedDtypeError(unittest.TestCase):
    def test_error(self):
        paddle.device.set_device("cpu")
        paddle.disable_static()
        a = (np.random.random((3, 3)) * 10).astype('int64')
        x = paddle.to_tensor(a)
        self.assertRaises(ValueError, paddle.linalg.eig, x)


class TestGrad(unittest.TestCase):
    def test_run(self):
        paddle.device.set_device("cpu")
        paddle.disable_static()
        dx_expectd = np.array([[0.9482, 0.0767, -0.0162],
                               [-0.0459, 1.0393, -0.0342],
                               [-0.0456, 0.0863, 1.0125]]).astype("float32")
        a = np.array([[1.6707249, 7.2249975, 6.5045543],
                      [9.956216, 8.749598, 6.066444],
                      [4.4251957, 1.7983172, 0.370647]]).astype("float32")
        a_pd = paddle.to_tensor(a)
        a_pd.stop_gradient = False
        w2, v2 = paddle.linalg.eig(a_pd)
        dx_actual = paddle.grad([w2, v2], a_pd)
        self.assertTrue(
            np.allclose(
                dx_actual, dx_expectd, rtol=1e-6, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
