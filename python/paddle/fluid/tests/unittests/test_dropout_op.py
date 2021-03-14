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

import unittest
import numpy as np
import paddle.fluid.core as core
from op_test import OpTest, skip_check_grad_ci
import paddle
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


class TestDropoutOpInput1d(OpTest):
    def setUp(self):
        self.op_type = "dropout"
        self.inputs = {'X': np.random.random((2000, )).astype("float32")}
        self.attrs = {'dropout_prob': 0.0, 'fix_seed': True, 'is_test': False}
        self.outputs = {
            'Out': self.inputs['X'],
            'Mask': np.ones((2000)).astype('uint8')
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


class TestDropoutFAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(fluid.CUDAPlace(0))

    def check_static_result(self, place):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            input = fluid.data(name="input", shape=[40, 40], dtype="float32")
            res1 = paddle.nn.functional.dropout(x=input, p=0., training=False)
            res2 = paddle.nn.functional.dropout(
                x=input, p=0., axis=0, training=True, mode='upscale_in_train')
            res3 = paddle.nn.functional.dropout(
                x=input, p=0., axis=0, training=True, mode='downscale_in_infer')
            res4 = paddle.nn.functional.dropout(
                x=input, p=0., axis=0, training=False, mode='upscale_in_train')
            res5 = paddle.nn.functional.dropout(
                x=input,
                p=0.,
                axis=0,
                training=False,
                mode='downscale_in_infer')
            res6 = paddle.nn.functional.dropout(
                x=input,
                p=0.,
                axis=[0, 1],
                training=True,
                mode='upscale_in_train')
            res7 = paddle.nn.functional.dropout(
                x=input,
                p=0.,
                axis=[0, 1],
                training=True,
                mode='downscale_in_infer')
            res8 = paddle.nn.functional.dropout(
                x=input,
                p=0.,
                axis=[0, 1],
                training=False,
                mode='upscale_in_train')
            res9 = paddle.nn.functional.dropout(
                x=input,
                p=0.,
                axis=[0, 1],
                training=False,
                mode='downscale_in_infer')
            res10 = paddle.nn.functional.dropout(x=input, p=1., training=True)
            res11 = paddle.fluid.layers.dropout(x=input, dropout_prob=0.)

            in_np = np.random.random([40, 40]).astype("float32")
            res_np = in_np
            res_np2 = np.zeros_like(in_np)

            exe = fluid.Executor(place)
            res_list = [
                res1, res2, res3, res4, res5, res6, res7, res8, res9, res11
            ]
            for res in res_list:
                fetches = exe.run(fluid.default_main_program(),
                                  feed={"input": in_np},
                                  fetch_list=[res])
                self.assertTrue(np.allclose(fetches[0], res_np))
            fetches2 = exe.run(fluid.default_main_program(),
                               feed={"input": in_np},
                               fetch_list=[res10])
            self.assertTrue(np.allclose(fetches2[0], res_np2))

    def test_static(self):
        for place in self.places:
            self.check_static_result(place=place)

    def test_dygraph(self):
        for place in self.places:
            with fluid.dygraph.guard(place):
                in_np = np.random.random([40, 40]).astype("float32")
                res_np = in_np
                res_np2 = np.zeros_like(in_np)
                input = fluid.dygraph.to_variable(in_np)

                res1 = paddle.nn.functional.dropout(
                    x=input, p=0., training=False)
                res2 = paddle.nn.functional.dropout(
                    x=input,
                    p=0.,
                    axis=0,
                    training=True,
                    mode='upscale_in_train')
                res3 = paddle.nn.functional.dropout(
                    x=input,
                    p=0.,
                    axis=0,
                    training=True,
                    mode='downscale_in_infer')
                res4 = paddle.nn.functional.dropout(
                    x=input,
                    p=0.,
                    axis=0,
                    training=False,
                    mode='upscale_in_train')
                res5 = paddle.nn.functional.dropout(
                    x=input,
                    p=0.,
                    axis=0,
                    training=False,
                    mode='downscale_in_infer')
                res6 = paddle.nn.functional.dropout(
                    x=input,
                    p=0.,
                    axis=[0, 1],
                    training=True,
                    mode='upscale_in_train')
                res7 = paddle.nn.functional.dropout(
                    x=input,
                    p=0.,
                    axis=[0, 1],
                    training=True,
                    mode='downscale_in_infer')
                res8 = paddle.nn.functional.dropout(
                    x=input,
                    p=0.,
                    axis=[0, 1],
                    training=False,
                    mode='upscale_in_train')
                res9 = paddle.nn.functional.dropout(
                    x=input,
                    p=0.,
                    axis=[0, 1],
                    training=False,
                    mode='downscale_in_infer')
                res10 = paddle.nn.functional.dropout(
                    x=input, p=1., training=True)
                dropout = paddle.fluid.dygraph.Dropout(p=0, )
                res11 = dropout(input)

            res_list = [
                res1, res2, res3, res4, res5, res6, res7, res8, res9, res11
            ]
            for res in res_list:
                self.assertTrue(np.allclose(res.numpy(), res_np))
            self.assertTrue(np.allclose(res10.numpy(), res_np2))


class TestDropoutFAPIError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):

            def test_Variable():
                # the input of dropout must be Variable.
                x1 = fluid.create_lod_tensor(
                    np.array([-1, 3, 5, 5]), [[1, 1, 1, 1]], fluid.CPUPlace())
                paddle.nn.functional.dropout(x1, p=0.5)

            self.assertRaises(TypeError, test_Variable)

            def test_Variable2():
                # the input of dropout must be Variable.
                x1 = fluid.create_lod_tensor(
                    np.array([-1, 3, 5, 5]), [[1, 1, 1, 1]], fluid.CPUPlace())
                paddle.nn.functional.dropout(x1, p=0.5, axis=0)

            self.assertRaises(TypeError, test_Variable2)

            def test_dtype():
                # the input dtype of dropout must be float32 or float64
                # float16 only can be set on GPU place
                xr = fluid.data(name='xr', shape=[3, 4, 5, 6], dtype="int32")
                paddle.nn.functional.dropout(xr, p=0.5)

            self.assertRaises(TypeError, test_dtype)

            def test_pdtype():
                # p should be int or float
                x2 = fluid.data(name='x2', shape=[3, 4, 5, 6], dtype="float32")
                paddle.nn.functional.dropout(x2, p='0.5')

            self.assertRaises(TypeError, test_pdtype)

            def test_pvalue():
                # p should be 0.<=p<=1.
                x2 = fluid.data(name='x2', shape=[3, 4, 5, 6], dtype="float32")
                paddle.nn.functional.dropout(x2, p=1.2)

            self.assertRaises(ValueError, test_pvalue)

            def test_mode():
                # mode should be 'downscale_in_infer' or 'upscale_in_train'
                x2 = fluid.data(name='x2', shape=[3, 4, 5, 6], dtype="float32")
                paddle.nn.functional.dropout(x2, mode='abc')

            self.assertRaises(ValueError, test_mode)

            def test_axis():
                # axis should be int or list
                x2 = fluid.data(name='x2', shape=[3, 4, 5, 6], dtype="float32")
                paddle.nn.functional.dropout(x2, axis=1.2)

            self.assertRaises(TypeError, test_axis)

            def test_axis_max():
                # maximum of axis should less than dimensions of x
                x2 = fluid.data(name='x2', shape=[3, 4, 5, 6], dtype="float32")
                paddle.nn.functional.dropout(x2, axis=[0, 5])

            self.assertRaises(ValueError, test_axis_max)

            def test_axis_min():
                # minimum of axis should greater equal than 0
                x2 = fluid.data(name='x2', shape=[3, 4, 5, 6], dtype="float32")
                paddle.nn.functional.dropout(x2, axis=[0, -1])

            self.assertRaises(ValueError, test_axis_min)

            def test_axis_len():
                # length of axis should not greater than dimensions of x
                x2 = fluid.data(name='x2', shape=[3, 4, 5, 6], dtype="float32")
                paddle.nn.functional.dropout(x2, axis=[0, 1, 2, 3, 4])

            self.assertRaises(ValueError, test_axis_len)


class TestDropoutCAPI(unittest.TestCase):
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
                m = paddle.nn.Dropout(p=0.)
                m.eval()
                result = m(input)
                self.assertTrue(np.allclose(result.numpy(), result_np))


class TestDropout2DFAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(fluid.CUDAPlace(0))

    def check_static_result(self, place):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            input = fluid.data(
                name="input", shape=[2, 3, 4, 5], dtype="float32")
            res1 = paddle.nn.functional.dropout2d(
                x=input, p=0., training=False, data_format='NCHW')
            res2 = paddle.nn.functional.dropout2d(
                x=input, p=0., training=False, data_format='NHWC')

            in_np = np.random.random([2, 3, 4, 5]).astype("float32")
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
                in_np = np.random.random([2, 3, 4, 5]).astype("float32")
                res_np = in_np
                input = fluid.dygraph.to_variable(in_np)

                res1 = paddle.nn.functional.dropout2d(
                    x=input, p=0., training=False, data_format='NCHW')
                res2 = paddle.nn.functional.dropout2d(
                    x=input, p=0., training=False, data_format='NHWC')

            res_list = [res1, res2]
            for res in res_list:
                self.assertTrue(np.allclose(res.numpy(), res_np))


class TestDropout2DFAPIError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):

            def test_xdim():
                # dimentions of x should be 4
                x = fluid.data(name='x1', shape=[2, 3, 4, 5, 6], dtype="int32")
                paddle.nn.functional.dropout2d(x)

            self.assertRaises(ValueError, test_xdim)

            def test_dataformat():
                # data_format should be 'NCHW' or 'NHWC'
                x = fluid.data(name='x2', shape=[2, 3, 4, 5], dtype="int32")
                paddle.nn.functional.dropout2d(x, data_format='CNHW')

            self.assertRaises(ValueError, test_dataformat)


class TestDropout2DCAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(fluid.CUDAPlace(0))

    def test_dygraph(self):
        for place in self.places:
            with fluid.dygraph.guard(place):
                input_np = np.random.random([2, 3, 4, 5]).astype("float32")
                result_np = input_np
                input = fluid.dygraph.to_variable(input_np)
                m = paddle.nn.Dropout2D(p=0.)
                m.eval()
                result = m(input)
                self.assertTrue(np.allclose(result.numpy(), result_np))


class TestDropout3DFAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(fluid.CUDAPlace(0))

    def check_static_result(self, place):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            input = fluid.data(
                name="input", shape=[2, 3, 4, 5, 6], dtype="float32")
            res1 = paddle.nn.functional.dropout3d(
                x=input, p=0., training=False, data_format='NCDHW')
            res2 = paddle.nn.functional.dropout3d(
                x=input, p=0., training=False, data_format='NDHWC')

            in_np = np.random.random([2, 3, 4, 5, 6]).astype("float32")
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
                in_np = np.random.random([2, 3, 4, 5, 6]).astype("float32")
                res_np = in_np
                input = fluid.dygraph.to_variable(in_np)

                res1 = paddle.nn.functional.dropout3d(
                    x=input, p=0., training=False, data_format='NCDHW')
                res2 = paddle.nn.functional.dropout3d(
                    x=input, p=0., training=False, data_format='NDHWC')

            res_list = [res1, res2]
            for res in res_list:
                self.assertTrue(np.allclose(res.numpy(), res_np))


class TestDropout3DFAPIError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):

            def test_xdim():
                # dimentions of x should be 5
                x = fluid.data(name='x1', shape=[2, 3, 4, 5], dtype="int32")
                paddle.nn.functional.dropout3d(x)

            self.assertRaises(ValueError, test_xdim)

            def test_dataformat():
                # data_format should be 'NCDHW' or 'NDHWC'
                x = fluid.data(name='x2', shape=[2, 3, 4, 5, 6], dtype="int32")
                paddle.nn.functional.dropout3d(x, data_format='CNDHW')

            self.assertRaises(ValueError, test_dataformat)


class TestDropout3DCAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(fluid.CUDAPlace(0))

    def test_dygraph(self):
        for place in self.places:
            with fluid.dygraph.guard(place):
                input_np = np.random.random([2, 3, 4, 5, 6]).astype("float32")
                result_np = input_np
                input = fluid.dygraph.to_variable(input_np)
                m = paddle.nn.Dropout3D(p=0.)
                m.eval()
                result = m(input)
                self.assertTrue(np.allclose(result.numpy(), result_np))


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
            res3 = paddle.nn.functional.alpha_dropout(x=input, p=1.)

            in_np = np.random.random([40, 40]).astype("float32")
            res_np = in_np
            res_np3 = np.zeros_like(in_np)

            exe = fluid.Executor(place)
            res_list = [res1, res2]
            for res in res_list:
                fetches = exe.run(fluid.default_main_program(),
                                  feed={"input": in_np},
                                  fetch_list=[res])
                self.assertTrue(np.allclose(fetches[0], res_np))
            fetches = exe.run(fluid.default_main_program(),
                              feed={"input": in_np},
                              fetch_list=[res3])
            self.assertTrue(np.allclose(fetches[0], res_np3))

    def test_static(self):
        for place in self.places:
            self.check_static_result(place=place)

    def test_dygraph(self):
        for place in self.places:
            with fluid.dygraph.guard(place):
                in_np = np.random.random([40, 40]).astype("float32")
                res_np = in_np
                res_np3 = np.zeros_like(in_np)
                input = fluid.dygraph.to_variable(in_np)

                res1 = paddle.nn.functional.alpha_dropout(x=input, p=0.)
                res2 = paddle.nn.functional.alpha_dropout(
                    x=input, p=0., training=False)
                res3 = paddle.nn.functional.alpha_dropout(x=input, p=1.)

            res_list = [res1, res2]
            for res in res_list:
                self.assertTrue(np.allclose(res.numpy(), res_np))
            self.assertTrue(np.allclose(res3.numpy(), res_np3))


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
