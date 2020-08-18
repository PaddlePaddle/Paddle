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

from __future__ import print_function
import paddle
import unittest
import numpy as np
import paddle.fluid.core as core
from op_test import OpTest, skip_check_grad_ci
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard


class TestDropoutOp(OpTest):
    def setUp(self):
        self.op_type = "dropout"
        self.inputs = {'X': np.random.random((32, 64)).astype("float32")}
        self.attrs = {'dropout_prob': 0.0, 'fix_seed': True, 'is_test': False}
        self.outputs = {
            'Out': self.inputs['X'],
            'Mask': np.ones((32, 64)).astype('uint8')
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X'], 'Out')


class TestDropoutOp2(TestDropoutOp):
    def setUp(self):
        self.op_type = "dropout"
        self.inputs = {'X': np.random.random((32, 64)).astype("float32")}
        self.attrs = {'dropout_prob': 1.0, 'fix_seed': True, 'is_test': False}
        self.outputs = {
            'Out': np.zeros((32, 64)).astype('float32'),
            'Mask': np.zeros((32, 64)).astype('uint8')
        }


class TestDropoutOp3(TestDropoutOp):
    def setUp(self):
        self.op_type = "dropout"
        self.inputs = {'X': np.random.random((32, 64, 2)).astype("float32")}
        self.attrs = {'dropout_prob': 0.0, 'fix_seed': True, 'is_test': False}
        self.outputs = {
            'Out': self.inputs['X'],
            'Mask': np.ones((32, 64, 2)).astype('uint8')
        }


@skip_check_grad_ci(reason="For inference, check_grad is not required.")
class TestDropoutOp4(OpTest):
    def setUp(self):
        self.op_type = "dropout"
        self.inputs = {'X': np.random.random((32, 64)).astype("float32")}
        self.attrs = {'dropout_prob': 0.35, 'fix_seed': True, 'is_test': True}
        self.outputs = {
            'Out': self.inputs['X'] * (1.0 - self.attrs['dropout_prob'])
        }

    def test_check_output(self):
        self.check_output()


@skip_check_grad_ci(reason="For inference, check_grad is not required.")
class TestDropoutOp5(OpTest):
    def setUp(self):
        self.op_type = "dropout"
        self.inputs = {'X': np.random.random((32, 64, 3)).astype("float32")}
        self.attrs = {'dropout_prob': 0.75, 'is_test': True}
        self.outputs = {
            'Out': self.inputs['X'] * (1.0 - self.attrs['dropout_prob'])
        }

    def test_check_output(self):
        self.check_output()


class TestDropoutOp6(TestDropoutOp):
    def setUp(self):
        self.op_type = "dropout"
        self.inputs = {'X': np.random.random((32, 64)).astype("float32")}
        self.attrs = {
            'dropout_prob': 1.0,
            'fix_seed': True,
            'is_test': False,
            'dropout_implementation': 'upscale_in_train'
        }
        self.outputs = {
            'Out': np.zeros((32, 64)).astype('float32'),
            'Mask': np.zeros((32, 64)).astype('uint8')
        }


class TestDropoutOp7(TestDropoutOp):
    def setUp(self):
        self.op_type = "dropout"
        self.inputs = {'X': np.random.random((32, 64, 2)).astype("float32")}
        self.attrs = {
            'dropout_prob': 0.0,
            'fix_seed': True,
            'is_test': False,
            'dropout_implementation': 'upscale_in_train'
        }
        self.outputs = {
            'Out': self.inputs['X'],
            'Mask': np.ones((32, 64, 2)).astype('uint8')
        }


@skip_check_grad_ci(reason="For inference, check_grad is not required.")
class TestDropoutOp8(OpTest):
    def setUp(self):
        self.op_type = "dropout"
        self.inputs = {'X': np.random.random((32, 64)).astype("float32")}
        self.attrs = {
            'dropout_prob': 0.35,
            'fix_seed': True,
            'is_test': True,
            'dropout_implementation': 'upscale_in_train'
        }
        self.outputs = {'Out': self.inputs['X']}

    def test_check_output(self):
        self.check_output()


@skip_check_grad_ci(reason="For inference, check_grad is not required.")
class TestDropoutOp9(OpTest):
    def setUp(self):
        self.op_type = "dropout"
        self.inputs = {'X': np.random.random((32, 64, 3)).astype("float32")}
        self.attrs = {
            'dropout_prob': 0.75,
            'is_test': True,
            'dropout_implementation': 'upscale_in_train'
        }
        self.outputs = {'Out': self.inputs['X']}

    def test_check_output(self):
        self.check_output()


class TestDropoutOpWithSeed(OpTest):
    def setUp(self):
        self.op_type = "dropout"
        self.inputs = {
            "X": np.random.random((32, 64)).astype("float32"),
            "Seed": np.asarray(
                [125], dtype="int32")
        }
        self.attrs = {'dropout_prob': 0.0, }
        self.outputs = {
            'Out': self.inputs['X'],
            'Mask': np.ones((32, 64)).astype('uint8')
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X'], 'Out', max_relative_error=0.05)


@unittest.skipIf(
    not core.is_compiled_with_cuda() or not core.op_support_gpu("dropout"),
    "core is not compiled with CUDA or core is not support dropout")
@skip_check_grad_ci(reason="For inference, check_grad is not required.")
class TestFP16DropoutOp(OpTest):
    def setUp(self):
        self.op_type = "dropout"
        self.init_test_case()

        x = np.random.random(self.input_size).astype("float16")
        out = x * (1.0 - self.prob)
        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.attrs = {
            'dropout_prob': self.prob,
            'fix_seed': self.fix_seed,
            'is_test': True
        }
        self.outputs = {'Out': out}

    def init_test_case(self):
        self.input_size = [32, 64]
        self.prob = 0.35
        self.fix_seed = True

    def test_check_output(self):
        self.check_output_with_place(core.CUDAPlace(0), atol=1e-3)


@unittest.skipIf(
    not core.is_compiled_with_cuda() or not core.op_support_gpu("dropout"),
    "core is not compiled with CUDA or core is not support dropout")
@skip_check_grad_ci(reason="For inference, check_grad is not required.")
class TestFP16DropoutOp2(TestFP16DropoutOp):
    def init_test_case(self):
        self.input_size = [32, 64, 3]
        self.prob = 0.75
        self.fix_seed = False


class TestDropoutOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):

            def test_Variable():
                # the input of dropout must be Variable.
                x1 = fluid.create_lod_tensor(
                    np.array([-1, 3, 5, 5]), [[1, 1, 1, 1]], fluid.CPUPlace())
                fluid.layers.dropout(x1, dropout_prob=0.5)

            self.assertRaises(TypeError, test_Variable)

            def test_dtype():
                # the input dtype of dropout must be float16 or float32 or float64
                # float16 only can be set on GPU place
                x2 = fluid.layers.data(
                    name='x2', shape=[3, 4, 5, 6], dtype="int32")
                fluid.layers.dropout(x2, dropout_prob=0.5)

            self.assertRaises(TypeError, test_dtype)


class TestAlphaDropoutFAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(fluid.CUDAPlace(0))

    def check_static_result(self, place):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            input = fluid.data(name="input", shape=[40, 40], dtype="float32")
            res1 = paddle.nn.functional.alpha_dropout(x=input, p=0.)
            res2 = paddle.nn.functional.alpha_dropout(
                x=input, p=0., training=False)

            in_np = np.random.random([40, 40]).astype("float32")
            res_np = in_np

            exe = fluid.Executor(place)
            res_list = [res1, res2]
            for res in res_list:
                fetches = exe.run(fluid.default_main_program(),
                                  feed={"input": in_np},
                                  fetch_list=[res])
                self.assertTrue(np.allclose(fetches[0], res_np))

    def test_static(self):
        for place in self.places:
            self.check_static_result(place=place)

    def test_dygraph(self):
        for place in self.places:
            with fluid.dygraph.guard(place):
                in_np = np.random.random([40, 40]).astype("float32")
                res_np = in_np
                input = fluid.dygraph.to_variable(in_np)

                res1 = paddle.nn.functional.alpha_dropout(x=input, p=0.)
                res2 = paddle.nn.functional.alpha_dropout(
                    x=input, p=0., training=False)

            res_list = [res1, res2]
            for res in res_list:
                self.assertTrue(np.allclose(res.numpy(), res_np))


class TestAlphaDropoutFAPIError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):

            def test_Variable():
                # the input of dropout must be Variable.
                x1 = fluid.create_lod_tensor(
                    np.array([-1, 3, 5, 5]), [[1, 1, 1, 1]], fluid.CPUPlace())
                paddle.nn.functional.alpha_dropout(x1, p=0.5)

            self.assertRaises(TypeError, test_Variable)

            def test_dtype():
                # the input dtype of dropout must be float32 or float64
                xr = fluid.data(name='xr', shape=[3, 4, 5, 6], dtype="int32")
                paddle.nn.functional.alpha_dropout(xr)

            self.assertRaises(TypeError, test_dtype)

            def test_pdtype():
                # p should be int or float
                x2 = fluid.data(name='x2', shape=[3, 4, 5, 6], dtype="float32")
                paddle.nn.functional.alpha_dropout(x2, p='0.5')

            self.assertRaises(TypeError, test_pdtype)

            def test_pvalue():
                # p should be 0.<=p<=1.
                x2 = fluid.data(name='x2', shape=[3, 4, 5, 6], dtype="float32")
                paddle.nn.functional.alpha_dropout(x2, p=1.2)

            self.assertRaises(ValueError, test_pvalue)


class TestAlphaDropoutCAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(fluid.CUDAPlace(0))

    def test_dygraph(self):
        for place in self.places:
            with fluid.dygraph.guard(place):
                input_np = np.random.random([40, 40]).astype("float32")
                result_np = input_np
                input = fluid.dygraph.to_variable(input_np)
                m = paddle.nn.AlphaDropout(p=0.)
                m.eval()
                result = m(input)
                self.assertTrue(np.allclose(result.numpy(), result_np))


if __name__ == '__main__':
    unittest.main()
