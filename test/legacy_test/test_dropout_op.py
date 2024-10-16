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
import parameterized as param
from op_test import OpTest, convert_float_to_uint16, skip_check_grad_ci
from utils import static_guard

import paddle
from paddle import base, static
from paddle.autograd.ir_backward import grad
from paddle.base import Program, Scope, core, program_guard
from paddle.base.executor import scope_guard
from paddle.decomposition import decompose
from paddle.incubate.autograd import primapi


def dropout_wrapper(
    X,
    Seed=None,
    dropout_prob=0.5,
    is_test=False,
    dropout_implementation="downgrade_in_infer",
    seed=0,
    fix_seed=False,
):
    return paddle._C_ops.dropout(
        X,
        Seed,
        dropout_prob,
        is_test,
        dropout_implementation,
        seed,
        fix_seed,
    )


def prim_dropout_wrapper(
    x,
    Seed=None,
    dropout_prob=0.5,
    is_test=False,
    dropout_implementation='upscale_in_train',
    seed=None,
    fix_seed=None,
):
    return paddle.nn.functional.dropout(
        x,
        p=dropout_prob,
        axis=None,
        training=not is_test,
        mode=dropout_implementation,
    )


class TestDropoutOp(OpTest):
    def setUp(self):
        self.op_type = "dropout"
        self.prim_op_type = "comp"
        self.python_api = dropout_wrapper
        self.public_python_api = prim_dropout_wrapper
        self.inputs = {'X': np.random.random((32, 64)).astype("float32")}
        self.attrs = {'dropout_prob': 0.0, 'fix_seed': True, 'is_test': False}
        self.outputs = {
            'Out': self.inputs['X'],
            'Mask': np.ones((32, 64)).astype('uint8'),
        }
        self.python_out_sig = [
            "Out"
        ]  # python out sig is customized output signature.
        # Because prim op compare res with dygraph
        # when p = 0 dropout api return x,in dygraph mode x_grad = out_grad,
        # but in static mode x_grad = []
        self.enable_check_static_comp = False

    def test_check_output(self):
        self.check_output(check_prim=True, check_prim_pir=True, check_pir=True)

    def test_check_grad_normal(self):
        # Now in dy2st mode x_grad = [], so set check_prim=False
        self.check_grad(['X'], 'Out', check_prim=False, check_pir=True)


class TestDropoutOp_ZeroDim(TestDropoutOp):
    def setUp(self):
        self.op_type = "dropout"
        self.prim_op_type = "comp"
        self.python_api = dropout_wrapper
        self.public_python_api = prim_dropout_wrapper
        self.inputs = {'X': np.random.random(()).astype("float32")}
        self.attrs = {'dropout_prob': 0.0, 'fix_seed': True, 'is_test': False}
        self.outputs = {
            'Out': self.inputs['X'],
            'Mask': np.ones(()).astype('uint8'),
        }
        # Because prim op compare res with dygraph
        # when p = 0 dropout api return x,in dygraph mode x_grad = out_grad,
        # but in static mode x_grad = []
        self.enable_check_static_comp = False
        self.python_out_sig = [
            "Out"
        ]  # python out sig is customized output signature.


class TestDropoutOpInput1d(OpTest):
    def setUp(self):
        self.op_type = "dropout"
        self.python_api = dropout_wrapper
        self.public_python_api = prim_dropout_wrapper
        self.prim_op_type = "comp"
        self.inputs = {'X': np.random.random((2000,)).astype("float32")}
        self.attrs = {'dropout_prob': 0.0, 'fix_seed': True, 'is_test': False}
        self.outputs = {
            'Out': self.inputs['X'],
            'Mask': np.ones(2000).astype('uint8'),
        }
        self.python_out_sig = [
            "Out"
        ]  # python out sig is customized output signature.
        # Because prim op compare res with dygraph
        # when p = 0 dropout api return x,in dygraph mode x_grad = out_grad,
        # but in static mode x_grad = []
        self.enable_check_static_comp = False

    def test_check_output(self):
        self.check_output(check_prim=True, check_prim_pir=True, check_pir=True)

    def test_check_grad_normal(self):
        # Now in dy2st mode x_grad = [], so set check_prim=False
        self.check_grad(['X'], 'Out', check_prim=False, check_pir=True)


class TestDropoutOp2(TestDropoutOp):
    def setUp(self):
        self.op_type = "dropout"
        self.python_api = dropout_wrapper
        self.public_python_api = prim_dropout_wrapper
        self.prim_op_type = "comp"
        self.inputs = {'X': np.random.random((32, 64)).astype("float32")}
        self.attrs = {'dropout_prob': 1.0, 'fix_seed': True, 'is_test': False}
        self.outputs = {
            'Out': np.zeros((32, 64)).astype('float32'),
            'Mask': np.zeros((32, 64)).astype('uint8'),
        }
        self.python_out_sig = [
            "Out"
        ]  # python out sig is customized output signature.


class TestDropoutOp2_ZeroDim(TestDropoutOp2):
    def setUp(self):
        self.op_type = "dropout"
        self.python_api = dropout_wrapper
        self.public_python_api = prim_dropout_wrapper
        self.prim_op_type = "comp"
        self.inputs = {'X': np.random.random(()).astype("float32")}
        self.attrs = {'dropout_prob': 1.0, 'fix_seed': True, 'is_test': False}
        self.outputs = {
            'Out': np.zeros(()).astype('float32'),
            'Mask': np.zeros(()).astype('uint8'),
        }
        self.python_out_sig = [
            "Out"
        ]  # python out sig is customized output signature.


class TestDropoutOp3(TestDropoutOp):
    def setUp(self):
        self.op_type = "dropout"
        self.python_api = dropout_wrapper
        self.public_python_api = prim_dropout_wrapper
        self.prim_op_type = "comp"
        self.inputs = {'X': np.random.random((32, 64, 2)).astype("float32")}
        self.attrs = {'dropout_prob': 0.0, 'fix_seed': True, 'is_test': False}
        self.outputs = {
            'Out': self.inputs['X'],
            'Mask': np.ones((32, 64, 2)).astype('uint8'),
        }
        # Because prim op compare res with dygraph
        # when p = 0 dropout api return x,in dygraph mode x_grad = out_grad,
        # but in static mode x_grad = []
        self.enable_check_static_comp = False
        self.python_out_sig = [
            "Out"
        ]  # python out sig is customized output signature.


@skip_check_grad_ci(reason="For inference, check_grad is not required.")
class TestDropoutOp4(OpTest):
    def setUp(self):
        self.op_type = "dropout"
        self.python_api = dropout_wrapper
        self.public_python_api = prim_dropout_wrapper
        self.prim_op_type = "comp"
        self.inputs = {'X': np.random.random((32, 64)).astype("float32")}
        self.attrs = {'dropout_prob': 0.35, 'fix_seed': True, 'is_test': True}
        self.outputs = {
            'Out': self.inputs['X'] * (1.0 - self.attrs['dropout_prob'])
        }
        self.python_out_sig = [
            "Out"
        ]  # python out sig is customized output signature.

    def test_check_output(self):
        self.check_output(check_prim=True, check_prim_pir=True, check_pir=True)


@skip_check_grad_ci(reason="For inference, check_grad is not required.")
class TestDropoutOp5(OpTest):
    def setUp(self):
        self.op_type = "dropout"
        self.python_api = dropout_wrapper
        self.public_python_api = prim_dropout_wrapper
        self.prim_op_type = "comp"
        self.inputs = {'X': np.random.random((32, 64, 3)).astype("float32")}
        self.attrs = {'dropout_prob': 0.75, 'is_test': True}
        self.outputs = {
            'Out': self.inputs['X'] * (1.0 - self.attrs['dropout_prob'])
        }
        self.python_out_sig = [
            "Out"
        ]  # python out sig is customized output signature.

    def test_check_output(self):
        self.check_output(check_prim=True, check_prim_pir=True, check_pir=True)


class TestDropoutOp6(TestDropoutOp):
    def setUp(self):
        self.op_type = "dropout"
        self.python_api = dropout_wrapper
        self.public_python_api = prim_dropout_wrapper
        self.prim_op_type = "comp"
        self.inputs = {'X': np.random.random((32, 64)).astype("float32")}
        self.attrs = {
            'dropout_prob': 1.0,
            'fix_seed': True,
            'is_test': False,
            'dropout_implementation': 'upscale_in_train',
        }
        self.outputs = {
            'Out': np.zeros((32, 64)).astype('float32'),
            'Mask': np.zeros((32, 64)).astype('uint8'),
        }
        self.python_out_sig = [
            "Out"
        ]  # python out sig is customized output signature.


class TestDropoutOp7(TestDropoutOp):
    def setUp(self):
        self.op_type = "dropout"
        self.python_api = dropout_wrapper
        self.public_python_api = prim_dropout_wrapper
        self.prim_op_type = "comp"
        self.inputs = {'X': np.random.random((32, 64, 2)).astype("float32")}
        self.attrs = {
            'dropout_prob': 0.0,
            'fix_seed': True,
            'is_test': False,
            'dropout_implementation': 'upscale_in_train',
        }
        self.outputs = {
            'Out': self.inputs['X'],
            'Mask': np.ones((32, 64, 2)).astype('uint8'),
        }
        # Because prim op compare res with dygraph
        # when p = 0 dropout api return x,in dygraph mode x_grad = out_grad,
        # but in static mode x_grad = []
        self.enable_check_static_comp = False
        self.python_out_sig = [
            "Out"
        ]  # python out sig is customized output signature.


@skip_check_grad_ci(reason="For inference, check_grad is not required.")
class TestDropoutOp8(OpTest):
    def setUp(self):
        self.op_type = "dropout"
        self.python_api = dropout_wrapper
        self.public_python_api = prim_dropout_wrapper
        self.prim_op_type = "comp"
        self.inputs = {'X': np.random.random((32, 64)).astype("float32")}
        self.attrs = {
            'dropout_prob': 0.35,
            'fix_seed': True,
            'is_test': True,
            'dropout_implementation': 'upscale_in_train',
        }
        self.outputs = {'Out': self.inputs['X']}
        self.python_out_sig = [
            "Out"
        ]  # python out sig is customized output signature.

    def test_check_output(self):
        self.check_output(check_prim=True, check_prim_pir=True, check_pir=True)


@skip_check_grad_ci(reason="For inference, check_grad is not required.")
class TestDropoutOp9(OpTest):
    def setUp(self):
        self.op_type = "dropout"
        self.python_api = dropout_wrapper
        self.public_python_api = prim_dropout_wrapper
        self.prim_op_type = "comp"
        self.inputs = {'X': np.random.random((32, 64, 3)).astype("float32")}
        self.attrs = {
            'dropout_prob': 0.75,
            'is_test': True,
            'dropout_implementation': 'upscale_in_train',
        }
        self.outputs = {'Out': self.inputs['X']}
        self.python_out_sig = [
            "Out"
        ]  # python out sig is customized output signature.

    def test_check_output(self):
        self.check_output(check_prim=True, check_prim_pir=True, check_pir=True)


class TestDropoutOpWithSeed(OpTest):
    def setUp(self):
        self.op_type = "dropout"
        self.python_api = dropout_wrapper
        self.public_python_api = prim_dropout_wrapper
        self.prim_op_type = "comp"
        self.inputs = {
            "X": np.random.random((32, 64)).astype("float32"),
            "Seed": np.asarray([125], dtype="int32"),
        }
        self.attrs = {
            'dropout_prob': 0.0,
        }
        self.outputs = {
            'Out': self.inputs['X'],
            'Mask': np.ones((32, 64)).astype('uint8'),
        }
        self.python_out_sig = [
            "Out"
        ]  # python out sig is customized output signature.
        # Because prim op compare res with dygraph
        # when p = 0 dropout api return x,in dygraph mode x_grad = out_grad,
        # but in static mode x_grad = []
        self.enable_check_static_comp = False

    def test_check_output(self):
        # ir backward don't support of variable derivation of itself
        self.check_output(check_prim=True, check_prim_pir=False, check_pir=True)

    def test_check_grad_normal(self):
        # Now in dy2st mode x_grad = [], so set check_prim=False
        self.check_grad(
            ['X'],
            'Out',
            max_relative_error=0.05,
            check_prim=False,
            check_pir=True,
        )


@unittest.skipIf(
    not core.is_compiled_with_cuda() or not core.op_support_gpu("dropout"),
    "core is not compiled with CUDA or core is not support dropout",
)
@skip_check_grad_ci(reason="For inference, check_grad is not required.")
class TestFP16DropoutOp(OpTest):
    def setUp(self):
        self.op_type = "dropout"
        self.python_api = dropout_wrapper
        self.public_python_api = prim_dropout_wrapper
        self.prim_op_type = "comp"
        self.init_test_case()

        x = np.random.random(self.input_size).astype("float16")
        out = x * (1.0 - self.prob)
        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.attrs = {
            'dropout_prob': self.prob,
            'fix_seed': self.fix_seed,
            'is_test': True,
        }
        self.outputs = {'Out': out}
        self.python_out_sig = [
            "Out"
        ]  # python out sig is customized output signature.

    def init_test_case(self):
        self.input_size = [32, 64]
        self.prob = 0.35
        self.fix_seed = True

    def test_check_output(self):
        self.check_output_with_place(
            core.CUDAPlace(0),
            atol=1e-3,
            check_prim=True,
            check_prim_pir=True,
            check_pir=True,
        )

    def test_check_grad_normal(self):
        self.check_grad(['X'], 'Out', check_pir=True)


@unittest.skipIf(
    not core.is_compiled_with_cuda() or not core.op_support_gpu("dropout"),
    "core is not compiled with CUDA or core is not support dropout",
)
@skip_check_grad_ci(reason="For inference, check_grad is not required.")
class TestFP16DropoutOp2(TestFP16DropoutOp):
    def init_test_case(self):
        self.input_size = [32, 64, 3]
        self.prob = 0.75
        self.fix_seed = False


class TestBF16DropoutOp(OpTest):
    def setUp(self):
        self.op_type = "dropout"
        self.python_api = dropout_wrapper
        self.public_python_api = prim_dropout_wrapper
        self.prim_op_type = "comp"
        self.dtype = np.uint16
        self.enable_cinn = False

        x = np.random.random((32, 64)).astype("float32")
        self.inputs = {'X': convert_float_to_uint16(x)}
        self.attrs = {'dropout_prob': 1.0, 'fix_seed': True, 'is_test': False}
        self.outputs = {
            'Out': convert_float_to_uint16(
                np.zeros((32, 64)).astype('float32')
            ),
            'Mask': np.zeros((32, 64)).astype('uint8'),
        }
        self.python_out_sig = [
            "Out"
        ]  # python out sig is customized output signature.

    def test_check_output(self):
        self.check_output(check_prim=True, check_prim_pir=True, check_pir=True)

    def test_check_grad_normal(self):
        self.check_grad(
            ['X'],
            'Out',
            check_prim=True,
            check_prim_pir=True,
            check_pir=True,
        )


class TestDropoutOpWithSeedOnCPUPlace(unittest.TestCase):
    def test_seed_cpu_place(self):
        paddle.enable_static()
        main_program = Program()
        with program_guard(main_program):
            seed_input_name = "tensor@SeedInput"
            x_var_name = "tensor@X"
            x_out_var = "tensor@XOut"

            mask_var_name = "tensor@Mask"
            seed_input_var = main_program.global_block().create_var(
                name=seed_input_name,
                shape=[1],
                dtype='int32',
                persistable=False,
                stop_gradient=True,
            )
            x_out_var = main_program.global_block().create_var(
                name=x_out_var,
                shape=[40, 40],
                dtype='float32',
                persistable=False,
                stop_gradient=True,
            )
            x_var = main_program.global_block().create_var(
                name=x_var_name,
                shape=[40, 40],
                dtype='float32',
                persistable=False,
                stop_gradient=True,
            )
            mask_var = main_program.global_block().create_var(
                name=mask_var_name,
                shape=[1],
                dtype='int',
                persistable=False,
                stop_gradient=True,
            )

            main_program.global_block().append_op(
                type="fill_constant",
                outputs={"Out": x_var_name},
                attrs={
                    "shape": [40, 40],
                    "dtype": x_var.dtype,
                    "value": 1.0,
                    "place_type": 0,
                },
            )
            main_program.global_block().append_op(
                type='seed',
                inputs={},
                outputs={'Out': seed_input_var},
                attrs={'seed': 1, 'force_cpu': True},
            )
            main_program.global_block().append_op(
                type='dropout',
                inputs={'X': x_var, 'Seed': seed_input_var},
                attrs={'dropout_prob': 0.0},
                outputs={'Out': x_out_var, 'Mask': mask_var},
            )
            place = base.CPUPlace()
            if core.is_compiled_with_cuda():
                place = base.CUDAPlace(0)
            exe = base.Executor(place)
            x_out, mask_out = exe.run(
                main_program,
                feed={},
                fetch_list=[x_out_var.name, mask_var.name],
            )
            x_in_np = np.ones([40, 40]).astype("float32")
            np.testing.assert_allclose(x_out, x_in_np, rtol=1e-05)


class TestDropoutOpError(unittest.TestCase):

    def test_errors(self):
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            paddle.enable_static()

            def test_Variable():
                # the input of dropout must be Variable.
                x1 = base.create_lod_tensor(
                    np.array([-1, 3, 5, 5]), [[1, 1, 1, 1]], base.CPUPlace()
                )
                paddle.nn.functional.dropout(x1, p=0.5)

            self.assertRaises(TypeError, test_Variable)

            def test_dtype():
                # the input dtype of dropout must be float16 or float32 or float64
                # float16 only can be set on GPU place
                x2 = paddle.static.data(
                    name='x2', shape=[-1, 3, 4, 5, 6], dtype="int32"
                )
                paddle.nn.functional.dropout(x2, p=0.5)

            self.assertRaises(TypeError, test_dtype)


class TestDropoutFAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            self.places.append(base.CUDAPlace(0))

    def check_static_result(self, place):
        paddle.enable_static()
        main_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog):
            input = paddle.static.data(
                name="input", shape=[-1, -1], dtype="float32"
            )
            res1 = paddle.nn.functional.dropout(x=input, p=0.0, training=False)
            res2 = paddle.nn.functional.dropout(
                x=input, p=0.0, axis=0, training=True, mode='upscale_in_train'
            )
            res3 = paddle.nn.functional.dropout(
                x=input, p=0.0, axis=0, training=True, mode='downscale_in_infer'
            )
            res4 = paddle.nn.functional.dropout(
                x=input, p=0.0, axis=0, training=False, mode='upscale_in_train'
            )
            res5 = paddle.nn.functional.dropout(
                x=input,
                p=0.0,
                axis=0,
                training=False,
                mode='downscale_in_infer',
            )
            res6 = paddle.nn.functional.dropout(
                x=input,
                p=0.0,
                axis=[0, 1],
                training=True,
                mode='upscale_in_train',
            )
            res7 = paddle.nn.functional.dropout(
                x=input,
                p=0.0,
                axis=[0, 1],
                training=True,
                mode='downscale_in_infer',
            )
            res8 = paddle.nn.functional.dropout(
                x=input,
                p=0.0,
                axis=[0, 1],
                training=False,
                mode='upscale_in_train',
            )
            res9 = paddle.nn.functional.dropout(
                x=input,
                p=0.0,
                axis=[0, 1],
                training=False,
                mode='downscale_in_infer',
            )
            res11 = paddle.nn.functional.dropout(x=input, p=0.0)
            res12 = paddle.nn.functional.dropout(
                x=input,
                p=0.0,
                axis=(0, 1),
                training=False,
                mode='upscale_in_train',
            )

            in_np = np.ones([40, 40]).astype("float32")
            res_np = in_np

            exe = base.Executor(place)
            res_list = [
                res1,
                res2,
                res3,
                res4,
                res5,
                res6,
                res7,
                res8,
                res9,
                res11,
                res12,
            ]
            for res in res_list:
                fetches = exe.run(
                    main_prog,
                    feed={"input": in_np},
                    fetch_list=[res],
                )
                np.testing.assert_allclose(fetches[0], res_np, rtol=1e-05)

    def check_static_result2(self, place):
        paddle.enable_static()
        main_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog):
            input = paddle.static.data(
                name="input", shape=[-1, -1], dtype="float32"
            )
            res10 = paddle.nn.functional.dropout(x=input, p=1.0, training=True)
            res13 = paddle.nn.functional.dropout(
                x=input, p=0.7, axis=1, training=True, mode='upscale_in_train'
            )
            in_np = np.ones([40, 40]).astype("float32")
            res_np2 = np.zeros_like(in_np)

            exe = base.Executor(place)
            fetches2 = exe.run(
                main_prog,
                feed={"input": in_np},
                fetch_list=[res10, res13],
            )
            np.testing.assert_allclose(fetches2[0], res_np2, rtol=1e-05)

    def test_static(self):
        for place in self.places:
            self.check_static_result(place=place)
            self.check_static_result2(place=place)

    def test_dygraph(self):
        for place in self.places:
            with base.dygraph.guard(place):
                in_np = np.random.random([40, 40]).astype("float32")
                res_np = in_np
                res_np2 = np.zeros_like(in_np)
                input = paddle.to_tensor(in_np)

                res1 = paddle.nn.functional.dropout(
                    x=input, p=0.0, training=False
                )
                res2 = paddle.nn.functional.dropout(
                    x=input,
                    p=0.0,
                    axis=0,
                    training=True,
                    mode='upscale_in_train',
                )
                res3 = paddle.nn.functional.dropout(
                    x=input,
                    p=0.0,
                    axis=0,
                    training=True,
                    mode='downscale_in_infer',
                )
                res4 = paddle.nn.functional.dropout(
                    x=input,
                    p=0.0,
                    axis=0,
                    training=False,
                    mode='upscale_in_train',
                )
                res5 = paddle.nn.functional.dropout(
                    x=input,
                    p=0.0,
                    axis=0,
                    training=False,
                    mode='downscale_in_infer',
                )
                res6 = paddle.nn.functional.dropout(
                    x=input,
                    p=0.0,
                    axis=[0, 1],
                    training=True,
                    mode='upscale_in_train',
                )
                res7 = paddle.nn.functional.dropout(
                    x=input,
                    p=0.0,
                    axis=[0, 1],
                    training=True,
                    mode='downscale_in_infer',
                )
                res8 = paddle.nn.functional.dropout(
                    x=input,
                    p=0.0,
                    axis=[0, 1],
                    training=False,
                    mode='upscale_in_train',
                )
                res9 = paddle.nn.functional.dropout(
                    x=input,
                    p=0.0,
                    axis=[0, 1],
                    training=False,
                    mode='downscale_in_infer',
                )
                res10 = paddle.nn.functional.dropout(
                    x=input, p=1.0, training=True
                )
                dropout = paddle.nn.Dropout(
                    p=0,
                )
                res11 = dropout(input)
                res12 = paddle.nn.functional.dropout(
                    x=input,
                    p=0.0,
                    axis=(0, 1),
                    training=False,
                    mode='upscale_in_train',
                )
                res13 = paddle.nn.functional.dropout(
                    x=input,
                    p=0.5,
                    axis=1,
                    training=True,
                    mode='upscale_in_train',
                )

            res_list = [
                res1,
                res2,
                res3,
                res4,
                res5,
                res6,
                res7,
                res8,
                res9,
                res11,
                res12,
            ]
            for res in res_list:
                np.testing.assert_allclose(res.numpy(), res_np, rtol=1e-05)
            np.testing.assert_allclose(res10.numpy(), res_np2, rtol=1e-05)


class TestDropoutFAPIError(unittest.TestCase):

    def test_errors(self):
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):

            def test_Variable():
                # the input of dropout must be Variable.
                x1 = base.create_lod_tensor(
                    np.array([-1, 3, 5, 5]), [[1, 1, 1, 1]], base.CPUPlace()
                )
                paddle.nn.functional.dropout(x1, p=0.5)

            self.assertRaises(TypeError, test_Variable)

            def test_Variable2():
                # the input of dropout must be Variable.
                x1 = base.create_lod_tensor(
                    np.array([-1, 3, 5, 5]), [[1, 1, 1, 1]], base.CPUPlace()
                )
                paddle.nn.functional.dropout(x1, p=0.5, axis=0)

            self.assertRaises(TypeError, test_Variable2)

            def test_dtype():
                # the input dtype of dropout must be float32 or float64
                # float16 only can be set on GPU place
                xr = paddle.static.data(
                    name='xr', shape=[3, 4, 5, 6], dtype="int32"
                )
                paddle.nn.functional.dropout(xr, p=0.5)

            self.assertRaises(TypeError, test_dtype)

    def test_errors2(self):
        paddle.enable_static()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, startup_prog):

            def test_pdtype():
                # p should be int or float
                x2 = paddle.static.data(
                    name='x2', shape=[3, 4, 5, 6], dtype="float32"
                )
                paddle.nn.functional.dropout(x2, p='0.5')

            self.assertRaises(TypeError, test_pdtype)

            def test_pvalue():
                # p should be 0.<=p<=1.
                x2 = paddle.static.data(
                    name='x2', shape=[3, 4, 5, 6], dtype="float32"
                )
                paddle.nn.functional.dropout(x2, p=1.2)

            self.assertRaises(ValueError, test_pvalue)

            def test_mode():
                # mode should be 'downscale_in_infer' or 'upscale_in_train'
                x2 = paddle.static.data(
                    name='x2', shape=[3, 4, 5, 6], dtype="float32"
                )
                paddle.nn.functional.dropout(x2, mode='abc')

            self.assertRaises(ValueError, test_mode)

            def test_axis():
                # axis should be int or list
                x2 = paddle.static.data(
                    name='x2', shape=[3, 4, 5, 6], dtype="float32"
                )
                paddle.nn.functional.dropout(x2, axis=1.2)

            self.assertRaises(TypeError, test_axis)

            def test_axis_max():
                # maximum of axis should less than dimensions of x
                x2 = paddle.static.data(
                    name='x2', shape=[3, 4, 5, 6], dtype="float32"
                )
                paddle.nn.functional.dropout(x2, axis=[0, 5])

            self.assertRaises(ValueError, test_axis_max)

            def test_axis_min():
                # minimum of axis should greater equal than 0
                x2 = paddle.static.data(
                    name='x2', shape=[3, 4, 5, 6], dtype="float32"
                )
                paddle.nn.functional.dropout(x2, axis=[0, -1])

            self.assertRaises(ValueError, test_axis_min)

            def test_axis_len():
                # length of axis should not greater than dimensions of x
                x2 = paddle.static.data(
                    name='x2', shape=[3, 4, 5, 6], dtype="float32"
                )
                paddle.nn.functional.dropout(x2, axis=[0, 1, 2, 3, 4])

            self.assertRaises(ValueError, test_axis_len)


class TestDropoutCAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            self.places.append(base.CUDAPlace(0))

    def test_dygraph(self):
        for place in self.places:
            with base.dygraph.guard(place):
                input_np = np.random.random([40, 40]).astype("float32")
                result_np = input_np
                input = paddle.to_tensor(input_np)
                m = paddle.nn.Dropout(p=0.0)
                m.eval()
                result = m(input)
                np.testing.assert_allclose(
                    result.numpy(), result_np, rtol=1e-05
                )


class TestDropout2DFAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            self.places.append(base.CUDAPlace(0))

    def check_static_result(self, place):
        paddle.enable_static()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, startup_prog):
            input = paddle.static.data(
                name="input", shape=[2, 3, 4, 5], dtype="float32"
            )
            res1 = paddle.nn.functional.dropout2d(
                x=input, p=0.0, training=False, data_format='NCHW'
            )
            res2 = paddle.nn.functional.dropout2d(
                x=input, p=0.0, training=False, data_format='NHWC'
            )

            in_np = np.random.random([2, 3, 4, 5]).astype("float32")
            res_np = in_np

            exe = base.Executor(place)
            res_list = [res1, res2]
            for res in res_list:
                fetches = exe.run(
                    main_prog,
                    feed={"input": in_np},
                    fetch_list=[res],
                )
                np.testing.assert_allclose(fetches[0], res_np, rtol=1e-05)

    def test_static(self):
        for place in self.places:
            self.check_static_result(place=place)

    def test_dygraph(self):
        for place in self.places:
            with base.dygraph.guard(place):
                in_np = np.random.random([2, 3, 4, 5]).astype("float32")
                res_np = in_np
                input = paddle.to_tensor(in_np)

                res1 = paddle.nn.functional.dropout2d(
                    x=input, p=0.0, training=False, data_format='NCHW'
                )
                res2 = paddle.nn.functional.dropout2d(
                    x=input, p=0.0, training=False, data_format='NHWC'
                )

            res_list = [res1, res2]
            for res in res_list:
                np.testing.assert_allclose(res.numpy(), res_np, rtol=1e-05)


class TestDropout2DFAPIError(unittest.TestCase):

    def test_errors(self):
        paddle.enable_static()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, startup_prog):

            def test_xdim():
                # dimensions of x should be 4
                x = paddle.static.data(
                    name='x1', shape=[2, 3, 4, 5, 6], dtype="int32"
                )
                paddle.nn.functional.dropout2d(x)

            self.assertRaises(ValueError, test_xdim)

            def test_dataformat():
                # data_format should be 'NCHW' or 'NHWC'
                x = paddle.static.data(
                    name='x2', shape=[2, 3, 4, 5], dtype="int32"
                )
                paddle.nn.functional.dropout2d(x, data_format='CNHW')

            self.assertRaises(ValueError, test_dataformat)


class TestDropout2DCAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            self.places.append(base.CUDAPlace(0))

    def test_dygraph(self):
        for place in self.places:
            with base.dygraph.guard(place):
                input_np = np.random.random([2, 3, 4, 5]).astype("float32")
                result_np = input_np
                input = paddle.to_tensor(input_np)
                m = paddle.nn.Dropout2D(p=0.0)
                m.eval()
                result = m(input)
                np.testing.assert_allclose(
                    result.numpy(), result_np, rtol=1e-05
                )

    def test_static_fp16_with_gpu(self):
        if paddle.base.core.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
            paddle.enable_static()
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                input = paddle.static.data(
                    name="input", shape=[2, 3, 4, 5], dtype="float16"
                )

                m = paddle.nn.Dropout2D(p=0.5)
                res1 = m(input)

                in_np = np.random.random([2, 3, 4, 5]).astype("float16")
                res_np = in_np

                exe = paddle.static.Executor(place)
                fetches = exe.run(
                    paddle.static.default_main_program(),
                    feed={"input": in_np},
                    fetch_list=[res1],
                )


class TestDropout3DFAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            self.places.append(base.CUDAPlace(0))

    def check_static_result(self, place):
        paddle.enable_static()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, startup_prog):
            input = paddle.static.data(
                name="input", shape=[2, 3, 4, 5, 6], dtype="float32"
            )
            res1 = paddle.nn.functional.dropout3d(
                x=input, p=0.0, training=False, data_format='NCDHW'
            )
            res2 = paddle.nn.functional.dropout3d(
                x=input, p=0.0, training=False, data_format='NDHWC'
            )

            in_np = np.random.random([2, 3, 4, 5, 6]).astype("float32")
            res_np = in_np

            exe = base.Executor(place)
            res_list = [res1, res2]
            for res in res_list:
                fetches = exe.run(
                    main_prog,
                    feed={"input": in_np},
                    fetch_list=[res],
                )
                np.testing.assert_allclose(fetches[0], res_np, rtol=1e-05)

    def test_static(self):
        for place in self.places:
            self.check_static_result(place=place)

    def test_dygraph(self):
        for place in self.places:
            with base.dygraph.guard(place):
                in_np = np.random.random([2, 3, 4, 5, 6]).astype("float32")
                res_np = in_np
                input = paddle.to_tensor(in_np)

                res1 = paddle.nn.functional.dropout3d(
                    x=input, p=0.0, training=False, data_format='NCDHW'
                )
                res2 = paddle.nn.functional.dropout3d(
                    x=input, p=0.0, training=False, data_format='NDHWC'
                )

            res_list = [res1, res2]
            for res in res_list:
                np.testing.assert_allclose(res.numpy(), res_np, rtol=1e-05)


class TestDropout3DFAPIError(unittest.TestCase):

    def test_errors(self):
        paddle.enable_static()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, startup_prog):

            def test_xdim():
                # dimensions of x should be 5
                x = paddle.static.data(
                    name='x1', shape=[2, 3, 4, 5], dtype="int32"
                )
                paddle.nn.functional.dropout3d(x)

            self.assertRaises(ValueError, test_xdim)

            def test_dataformat():
                # data_format should be 'NCDHW' or 'NDHWC'
                x = paddle.static.data(
                    name='x2', shape=[2, 3, 4, 5, 6], dtype="int32"
                )
                paddle.nn.functional.dropout3d(x, data_format='CNDHW')

            self.assertRaises(ValueError, test_dataformat)


class TestDropout3DCAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            self.places.append(base.CUDAPlace(0))

    def test_dygraph(self):
        for place in self.places:
            with base.dygraph.guard(place):
                input_np = np.random.random([2, 3, 4, 5, 6]).astype("float32")
                result_np = input_np
                input = paddle.to_tensor(input_np)
                m = paddle.nn.Dropout3D(p=0.0)
                m.eval()
                result = m(input)
                np.testing.assert_allclose(
                    result.numpy(), result_np, rtol=1e-05
                )


class TestAlphaDropoutFAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            self.places.append(base.CUDAPlace(0))

    def check_static_result(self, place):
        paddle.enable_static()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, startup_prog):
            input = paddle.static.data(
                name="input", shape=[40, 40], dtype="float32"
            )
            res1 = paddle.nn.functional.alpha_dropout(x=input, p=0.0)
            res2 = paddle.nn.functional.alpha_dropout(
                x=input, p=0.0, training=False
            )
            res3 = paddle.nn.functional.alpha_dropout(x=input, p=1.0)

            in_np = np.random.random([40, 40]).astype("float32")
            res_np = in_np
            res_np3 = np.zeros_like(in_np)

            exe = base.Executor(place)

            fetches = exe.run(
                main_prog,
                feed={"input": in_np},
                fetch_list=[res1, res2, res3],
            )
            np.testing.assert_allclose(fetches[0], res_np, rtol=1e-05)
            np.testing.assert_allclose(fetches[1], res_np, rtol=1e-05)
            np.testing.assert_allclose(fetches[2], res_np3, rtol=1e-05)

    def test_static(self):
        for place in self.places:
            self.check_static_result(place=place)

    def test_dygraph(self):
        for place in self.places:
            with base.dygraph.guard(place):
                in_np = np.random.random([40, 40]).astype("float32")
                res_np = in_np
                res_np3 = np.zeros_like(in_np)
                input = paddle.to_tensor(in_np)

                res1 = paddle.nn.functional.alpha_dropout(x=input, p=0.0)
                res2 = paddle.nn.functional.alpha_dropout(
                    x=input, p=0.0, training=False
                )
                res3 = paddle.nn.functional.alpha_dropout(x=input, p=1.0)

            res_list = [res1, res2]
            for res in res_list:
                np.testing.assert_allclose(res.numpy(), res_np, rtol=1e-05)
            np.testing.assert_allclose(res3.numpy(), res_np3, rtol=1e-05)


class TestAlphaDropoutFAPIError(unittest.TestCase):

    def test_errors(self):
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):

            def test_Variable():
                # the input of dropout must be Variable.
                x1 = base.create_lod_tensor(
                    np.array([-1, 3, 5, 5]), [[1, 1, 1, 1]], base.CPUPlace()
                )
                paddle.nn.functional.alpha_dropout(x1, p=0.5)

            self.assertRaises(TypeError, test_Variable)

    def test_errors2(self):
        paddle.enable_static()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, startup_prog):

            def test_dtype():
                # the input dtype of dropout must be float32 or float64
                xr = paddle.static.data(
                    name='xr', shape=[3, 4, 5, 6], dtype="int32"
                )
                paddle.nn.functional.alpha_dropout(xr)

            self.assertRaises(TypeError, test_dtype)

            def test_pdtype():
                # p should be int or float
                x2 = paddle.static.data(
                    name='x2', shape=[3, 4, 5, 6], dtype="float32"
                )
                paddle.nn.functional.alpha_dropout(x2, p='0.5')

            self.assertRaises(TypeError, test_pdtype)

            def test_pvalue():
                # p should be 0.<=p<=1.
                x2 = paddle.static.data(
                    name='x2', shape=[3, 4, 5, 6], dtype="float32"
                )
                paddle.nn.functional.alpha_dropout(x2, p=1.2)

            self.assertRaises(ValueError, test_pvalue)


class TestAlphaDropoutCAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            self.places.append(base.CUDAPlace(0))

    def test_dygraph(self):
        for place in self.places:
            with base.dygraph.guard(place):
                input_np = np.random.random([40, 40]).astype("float32")
                result_np = input_np
                input = paddle.to_tensor(input_np)
                m = paddle.nn.AlphaDropout(p=0.0)
                m.eval()
                result = m(input)
                np.testing.assert_allclose(
                    result.numpy(), result_np, rtol=1e-05
                )

    def test_static_fp16_gpu(self):
        if paddle.base.core.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                input = np.random.random([2, 3]).astype("float16")

                x = paddle.static.data(name="x", shape=[2, 3], dtype="float16")

                m = paddle.nn.AlphaDropout(p=0.0)
                y = m(x)

                exe = paddle.static.Executor(place)
                res = exe.run(
                    paddle.static.default_main_program(),
                    feed={
                        "x": input,
                    },
                    fetch_list=[y],
                )

                np.testing.assert_allclose(res[0], input, rtol=1e-05)


class TestDropoutWithDeterminateSeedGenerator(unittest.TestCase):
    def setUp(self):
        paddle.framework.random.set_random_seed_generator('seed0', 123)
        paddle.framework.random.set_random_seed_generator('seed1', 123)
        rng0 = paddle.framework.random.get_random_seed_generator('seed0')
        rng1 = paddle.framework.random.get_random_seed_generator('seed1')
        self.places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not paddle.is_compiled_with_cuda()
        ):
            self.places.append(paddle.CPUPlace())
        if paddle.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def check_static_result(self, place):
        from paddle.distributed.fleet.meta_parallel.parallel_layers.random import (
            dropout,
        )

        with static.program_guard(static.Program(), static.Program()):
            input = static.data(name="input", shape=[40, 40], dtype="float32")
            res1 = dropout(
                input,
                p=0.3,
                training=True,
                mode='upscale_in_train',
                rng_name='seed0',
            )
            res2 = dropout(
                input,
                p=0.3,
                training=True,
                mode='upscale_in_train',
                rng_name='seed1',
            )
            res3 = dropout(input, p=0.3)

            in_np = np.random.random([40, 40]).astype("float32")

            exe = static.Executor(place)
            res_list = [res1, res2]
            for i in range(2):
                out1, out2 = exe.run(
                    static.default_main_program(),
                    feed={"input": in_np},
                    fetch_list=res_list,
                )
                np.testing.assert_allclose(out1, out2, rtol=1e-05)

    def test_static(self):
        for place in self.places:
            self.check_static_result(place=place)


class TestDropoutBackward(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            self.places.append(base.CUDAPlace(0))

    def cal_grad_upscale_train(self, mask, prob):
        return mask.astype("float32") / (1 - prob)

    def cal_grad_downscale_in_infer(self, mask):
        return mask.astype("float32")


class TestDropOutWithProbTensor(unittest.TestCase):
    def setUp(self):
        self.init_info()
        self.input = np.random.random(self.shape).astype("float32")
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def init_info(self):
        self.shape = [10, 10]
        self.api = paddle.nn.functional.dropout

    def api_case(self, x):
        p = paddle.assign([0.5])
        out = self.api(x=x, p=p, training=True)
        return out

    def run_static(self, x):
        paddle.seed(2022)
        paddle.enable_static()
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            input = paddle.static.data(shape=x.shape, name='x', dtype='float32')
            out = self.api_case(input)
            sgd = paddle.optimizer.SGD(learning_rate=0.1)
            sgd.minimize(paddle.mean(out))

            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'x': x}, fetch_list=[out])

        return res[0]

    def run_dygraph(self, x):
        paddle.seed(2022)
        with base.dygraph.guard(self.place):
            out = self.api_case(paddle.to_tensor(x))
        return out

    def test_p_tensor(self):
        static_res = self.run_static(self.input)
        dygraph_res = self.run_dygraph(self.input)
        np.testing.assert_array_equal(static_res, dygraph_res)


class TestDropOut2DWithProbTensor(TestDropOutWithProbTensor):
    def init_info(self):
        self.shape = [2, 3, 10, 10]
        self.api = paddle.nn.functional.dropout2d


class TestDropOut3DWithProbTensor(TestDropOutWithProbTensor):
    def init_info(self):
        self.shape = [2, 3, 8, 8, 8]
        self.api = paddle.nn.functional.dropout3d


class TestRandomValue(unittest.TestCase):
    def test_fixed_random_number(self):
        # Test GPU Fixed random number, which is generated by 'curandStatePhilox4_32_10_t'
        if not paddle.is_compiled_with_cuda():
            return

        # Different GPU generate different random value. Only test V100 here.
        if "V100" not in paddle.device.cuda.get_device_name():
            return

        print("Test Fixed Random number on V100 GPU------>")
        paddle.disable_static()
        paddle.set_device('gpu')
        paddle.seed(100)

        x = paddle.rand([32, 1024, 1024], dtype='float32')
        out = paddle.nn.functional.dropout(x, 0.25).numpy()
        index0, index1, index2 = np.nonzero(out)
        self.assertEqual(np.sum(index0), 390094540)
        self.assertEqual(np.sum(index1), 12871475125)
        self.assertEqual(np.sum(index2), 12872777397)
        self.assertEqual(np.sum(out), 16778744.0)
        expect = [
            0.6914956,
            0.5294584,
            0.19032137,
            0.6996228,
            0.3338527,
            0.8442094,
            0.96965003,
            1.1726775,
            0.0,
            0.28037727,
        ]
        np.testing.assert_allclose(out[10, 100, 500:510], expect, rtol=1e-05)

        x = paddle.rand([32, 1024, 1024], dtype='float64')
        out = paddle.nn.functional.dropout(x).numpy()
        index0, index1, index2 = np.nonzero(out)
        self.assertEqual(np.sum(index0), 260065137)
        self.assertEqual(np.sum(index1), 8582636095)
        self.assertEqual(np.sum(index2), 8582219962)
        self.assertEqual(np.sum(out), 16778396.563660286)
        expect = [
            1.28587354,
            0.15563703,
            0.0,
            0.28799703,
            0.0,
            0.0,
            0.0,
            0.54964,
            0.51355682,
            0.33818988,
        ]
        np.testing.assert_allclose(out[20, 100, 500:510], expect, rtol=1e-05)

        x = paddle.ones([32, 1024, 1024], dtype='float16')
        out = paddle.nn.functional.dropout(x, 0.75).numpy()
        index0, index1, index2 = np.nonzero(out)
        self.assertEqual(np.sum(index0), 130086900)
        self.assertEqual(np.sum(index1), 4291190105)
        self.assertEqual(np.sum(index2), 4292243807)
        expect = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 4.0]
        np.testing.assert_allclose(out[0, 100, 500:510], expect, rtol=1e-05)

        paddle.enable_static()


places = []
if (
    os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
    in ['1', 'true', 'on']
    or not paddle.is_compiled_with_cuda()
):
    places.append(paddle.CPUPlace())
if paddle.is_compiled_with_cuda():
    places.append(paddle.CUDAPlace(0))


class PrimNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x,
        p=0.5,
        axis=None,
        training=True,
        mode="upscale_in_train",
    ):
        y = paddle.assign(x)
        out = paddle.nn.functional.dropout(
            x=y, p=p, axis=axis, training=training, mode=mode
        )
        return out


def apply_to_static(net, use_cinn):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(
        net, build_strategy=build_strategy, full_graph=True
    )


@param.parameterized_class(
    ('name', 'x', 'p', 'is_test', 'mode', 'seed', 'dtype', 'places'),
    (
        (
            'fp32',
            np.ones(100000),
            0.3,
            False,
            'upscale_in_train',
            1002,
            'float32',
            places,
        ),
        (
            'bfp16',
            np.ones(100000),
            0.3,
            False,
            'upscale_in_train',
            1002,
            'bfloat16',
            places,
        ),
        (
            'fp64',
            np.ones(100000),
            0.7,
            False,
            'upscale_in_train',
            9999,
            'float64',
            places,
        ),
        (
            'is_test=True',
            np.ones(100000),
            0.5,
            True,
            'upscale_in_train',
            1002,
            'float32',
            places,
        ),
        (
            'p=1.0',
            np.ones(100000),
            1.0,
            True,
            'upscale_in_train',
            1002,
            'float32',
            places,
        ),
        (
            'p=1.0,dtype=bfp16',
            np.ones(100000),
            1.0,
            True,
            'upscale_in_train',
            1002,
            'bfloat16',
            places,
        ),
        (
            'p=1.0,test=False',
            np.ones(100000),
            1.0,
            False,
            'upscale_in_train',
            1002,
            'float32',
            places,
        ),
        (
            'p=1.0,test=False,dtype=bfp16',
            np.ones(100000),
            1.0,
            False,
            'upscale_in_train',
            1002,
            'bfloat16',
            places,
        ),
        (
            'p=0.0',
            np.ones(100000),
            0,
            True,
            'upscale_in_train',
            1002,
            'float32',
            places,
        ),
        (
            'p=0.0,dtype=bfp16',
            np.ones(100000),
            0,
            True,
            'upscale_in_train',
            1002,
            'bfloat16',
            places,
        ),
        (
            'downgrade_train',
            np.ones(100000),
            0.5,
            False,
            'downscale_in_infer',
            1002,
            'float32',
            places,
        ),
        (
            'downgrade_train,dtype=bfp16',
            np.ones(100000),
            0.5,
            False,
            'downscale_in_infer',
            1002,
            'bfloat16',
            places,
        ),
        (
            'fp32_cpu',
            np.ones(100000),
            0.6,
            False,
            'upscale_in_train',
            9899,
            'float64',
            [paddle.CPUPlace()],
        ),
        (
            'fp64_cpu',
            np.ones(100000),
            0.6,
            False,
            'upscale_in_train',
            9899,
            'float64',
            [paddle.CPUPlace()],
        ),
        (
            'downgrade_train_cpu',
            np.ones(100000),
            0.5,
            False,
            'downscale_in_infer',
            1002,
            'float32',
            [paddle.CPUPlace()],
        ),
    ),
)
class TestCompositeDropout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x = (
            cls.x.astype(cls.dtype)
            if cls.dtype != "bfloat16"
            else cls.x.astype("float32")
        )
        core._set_prim_all_enabled(True)

    @classmethod
    def tearDownClass(cls):
        core._set_prim_all_enabled(False)

    def setUp(self):
        paddle.seed(self.seed)
        self.fwd_desire = []
        self.rev_desire = []
        for place in self.places:
            fwd_desire, rev_desire = self.get_eager_desire(place)
            self.fwd_desire.append(fwd_desire.numpy())
            self.rev_desire.append(rev_desire.numpy())

    def get_eager_desire(self, place):
        paddle.disable_static()
        paddle.seed(self.seed)
        if isinstance(place, base.CPUPlace):
            paddle.set_device("cpu")
        if isinstance(place, base.CUDAPlace):
            paddle.set_device("gpu")
        core.set_prim_eager_enabled(False)
        input_ = paddle.to_tensor(
            data=self.x,
            dtype=self.dtype if self.dtype != "bfloat16" else "float32",
            place=place,
            stop_gradient=False,
        )
        output = paddle.nn.functional.dropout(
            input_, self.p, training=(not self.is_test), mode=self.mode
        )
        grad = paddle.grad(output, input_)
        if self.dtype == "bfloat16":
            output = paddle.cast(output, "float32")
            grad[0] = paddle.cast(grad[0], "float32")
        return output, grad[0]

    def test_static_comp(self):
        fwd_actual = []
        rev_actual = []
        mps = []
        with static_guard():
            for place in self.places:
                paddle.seed(self.seed)
                mp, sp = paddle.static.Program(), paddle.static.Program()
                with paddle.static.program_guard(mp, sp):
                    input_ = paddle.static.data(
                        'x',
                        shape=self.x.shape,
                        dtype=(
                            self.x.dtype
                            if self.dtype != "bfloat16"
                            else "float32"
                        ),
                    )
                    input_.stop_gradient = False
                    y = paddle.assign(input_)
                    output = paddle.nn.functional.dropout(
                        y,
                        self.p,
                        training=(not self.is_test),
                        mode=self.mode,
                    )
                    if core._is_fwd_prim_enabled():
                        primapi.to_prim(mp.blocks)
                    grad = paddle.static.gradients(output, input_)[0]
                    if self.dtype == "bfloat16":
                        output = paddle.cast(output, "float32")
                        grad = paddle.cast(grad, "float32")
                exe = paddle.static.Executor(place)
                exe.run(sp)
                fwd, rev = exe.run(
                    mp, feed={input_.name: self.x}, fetch_list=[output, grad]
                )
                fwd_actual.append(fwd)
                rev_actual.append(rev)
                mps.append(mp)
        for i in range(len(self.places)):
            self.assertTrue(
                'dropout' not in [op.type for op in mps[i].block(0).ops]
            )
            np.testing.assert_allclose(
                self.fwd_desire[i].sum(),
                fwd_actual[i].sum(),
                rtol=2e-2,  # mean of uniform distribution, scale for avoid random failed
                atol=0,
            )
            np.testing.assert_allclose(
                self.rev_desire[i].sum(),
                rev_actual[i].sum(),
                rtol=2e-2,  # mean of uniform distribution, scale for avoid random failed
                atol=0,
            )

    def test_jit_comp(self):
        fwd_actual = []
        rev_actual = []
        paddle.disable_static()
        for place in self.places:
            if isinstance(place, base.CPUPlace):
                paddle.set_device("cpu")
            if isinstance(place, base.CUDAPlace):
                paddle.set_device("gpu")
            paddle.seed(self.seed)
            input_ = paddle.to_tensor(
                data=self.x,
                dtype=self.dtype if self.dtype != "bfloat16" else "float32",
                place=place,
                stop_gradient=False,
            )
            net = PrimNet()
            net = apply_to_static(net, False)
            output = net(
                input_, self.p, training=(not self.is_test), mode=self.mode
            )
            grad = paddle.grad(output, input_)
            if self.dtype == "bfloat16":
                output = paddle.cast(output, "float32")
                grad[0] = paddle.cast(grad[0], "float32")
            fwd_actual.append(output.numpy())
            rev_actual.append(grad[0].numpy())
        for i in range(len(self.places)):
            np.testing.assert_allclose(
                self.fwd_desire[i].sum(),
                fwd_actual[i].sum(),
                rtol=2e-2,  # mean of uniform distribution, scale for avoid random failed
                atol=0,
            )
            np.testing.assert_allclose(
                self.rev_desire[i].sum(),
                rev_actual[i].sum(),
                rtol=2e-2,  # mean of uniform distribution, scale for avoid random failed
                atol=0,
            )

    def test_jit_comp_with_cinn(self):
        fwd_actual = []
        rev_actual = []
        paddle.disable_static()
        for place in self.places:
            if not isinstance(place, base.CUDAPlace):
                continue
            paddle.set_device("gpu")
            paddle.seed(self.seed)
            input_ = paddle.to_tensor(
                data=self.x,
                dtype=self.dtype if self.dtype != "bfloat16" else "float32",
                place=place,
                stop_gradient=False,
            )
            net = PrimNet()
            net = apply_to_static(net, True)
            output = net(
                input_, self.p, training=(not self.is_test), mode=self.mode
            )
            grad = paddle.grad(output, input_)
            if self.dtype == "bfloat16":
                output = paddle.cast(output, "float32")
                grad[0] = paddle.cast(grad[0], "float32")
            fwd_actual.append(output.numpy())
            rev_actual.append(grad[0].numpy())
        i = 0
        for place in self.places:
            if not isinstance(self.places[i], base.CUDAPlace):
                continue
            np.testing.assert_allclose(
                self.fwd_desire[i].sum(),
                fwd_actual[i].sum(),
                rtol=2e-2,  # mean of uniform distribution, scale for avoid random failed
                atol=0,
            )
            np.testing.assert_allclose(
                self.rev_desire[i].sum(),
                rev_actual[i].sum(),
                rtol=2e-2,  # mean of uniform distribution, scale for avoid random failed
                atol=0,
            )
            i += 1


@param.parameterized_class(
    ('name', 'x', 'p', 'is_test', 'mode', 'seed', 'dtype', 'places'),
    (
        (
            'fp32',
            np.ones(100000),
            0.3,
            False,
            'upscale_in_train',
            1002,
            'float32',
            places,
        ),
        (
            'bfp16',
            np.ones(100000),
            0.3,
            False,
            'upscale_in_train',
            1002,
            'bfloat16',
            places,
        ),
        (
            'fp64',
            np.ones(100000),
            0.7,
            False,
            'upscale_in_train',
            9999,
            'float64',
            places,
        ),
        (
            'is_test=True',
            np.ones(100000),
            0.5,
            True,
            'upscale_in_train',
            1002,
            'float32',
            places,
        ),
        (
            'p=1.0',
            np.ones(100000),
            1.0,
            True,
            'upscale_in_train',
            1002,
            'float32',
            places,
        ),
        (
            'p=1.0,dtype=bfp16',
            np.ones(100000),
            1.0,
            True,
            'upscale_in_train',
            1002,
            'bfloat16',
            places,
        ),
        (
            'p=1.0,test=False',
            np.ones(100000),
            1.0,
            False,
            'upscale_in_train',
            1002,
            'float32',
            places,
        ),
        (
            'p=1.0,test=False,dtype=bfp16',
            np.ones(100000),
            1.0,
            False,
            'upscale_in_train',
            1002,
            'bfloat16',
            places,
        ),
        (
            'p=0.0',
            np.ones(100000),
            0,
            True,
            'upscale_in_train',
            1002,
            'float32',
            places,
        ),
        (
            'p=0.0,dtype=bfp16',
            np.ones(100000),
            0,
            True,
            'upscale_in_train',
            1002,
            'bfloat16',
            places,
        ),
        (
            'downgrade_train',
            np.ones(100000),
            0.5,
            False,
            'downscale_in_infer',
            1002,
            'float32',
            places,
        ),
        (
            'downgrade_train,dtype=bfp16',
            np.ones(100000),
            0.5,
            False,
            'downscale_in_infer',
            1002,
            'bfloat16',
            places,
        ),
        (
            'fp32_cpu',
            np.ones(100000),
            0.6,
            False,
            'upscale_in_train',
            9899,
            'float64',
            [paddle.CPUPlace()],
        ),
        (
            'fp64_cpu',
            np.ones(100000),
            0.6,
            False,
            'upscale_in_train',
            9899,
            'float64',
            [paddle.CPUPlace()],
        ),
        (
            'downgrade_train_cpu',
            np.ones(100000),
            0.5,
            False,
            'downscale_in_infer',
            1002,
            'float32',
            [paddle.CPUPlace()],
        ),
    ),
)
class TestPirCompositeDropout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x = (
            cls.x.astype(cls.dtype)
            if cls.dtype != "bfloat16"
            else cls.x.astype("float32")
        )
        core._set_prim_all_enabled(True)

    @classmethod
    def tearDownClass(cls):
        core._set_prim_all_enabled(False)

    def setUp(self):
        paddle.seed(self.seed)
        self.fwd_desire = []
        self.rev_desire = []
        for place in self.places:
            fwd_desire, rev_desire = self.get_eager_desire(place)
            self.fwd_desire.append(fwd_desire.numpy())
            self.rev_desire.append(rev_desire.numpy())

    def get_eager_desire(self, place):
        paddle.disable_static()
        paddle.seed(self.seed)
        if isinstance(place, base.CPUPlace):
            paddle.set_device("cpu")
        if isinstance(place, base.CUDAPlace):
            paddle.set_device("gpu")
        core.set_prim_eager_enabled(False)
        input_ = paddle.to_tensor(
            data=self.x,
            dtype=self.dtype if self.dtype != "bfloat16" else "float32",
            place=place,
            stop_gradient=False,
        )
        output = paddle.nn.functional.dropout(
            input_, self.p, training=(not self.is_test), mode=self.mode
        )
        grad = paddle.grad(output, input_)
        if self.dtype == "bfloat16":
            output = paddle.cast(output, "float32")
            grad[0] = paddle.cast(grad[0], "float32")
        return output, grad[0]

    def test_static_comp(self):
        fwd_actual = []
        rev_actual = []
        mps = []
        for place in self.places:
            with paddle.pir_utils.IrGuard(), static_guard(), scope_guard(
                Scope()
            ):
                core._set_prim_backward_enabled(True)
                core._set_prim_forward_enabled(False)

                paddle.seed(self.seed)
                sp, mp = (
                    paddle.static.Program(),
                    paddle.static.Program(),
                )
                with paddle.static.program_guard(mp, sp):
                    input_ = paddle.static.data(
                        'x',
                        shape=self.x.shape,
                        dtype=(
                            self.x.dtype
                            if self.dtype != "bfloat16"
                            else "float32"
                        ),
                    )
                    input_.stop_gradient = False
                    output = paddle.nn.functional.dropout(
                        input_,
                        self.p,
                        training=(not self.is_test),
                        mode=self.mode,
                    )
                    [output] = decompose(
                        mp, [output]
                    )  # decompose backward, custom vjp
                    gradient = grad(output, input_)[0]
                    self.assertTrue(
                        'pd_op.dropout_grad'
                        not in [op.name() for op in mp.global_block().ops]
                    )

                    core._set_prim_forward_enabled(True)
                    [output] = decompose(
                        mp, [output], whitelist={"pd_op.dropout"}
                    )  # decompose forward
                    self.assertTrue(
                        'pd_op.dropout'
                        not in [op.name() for op in mp.global_block().ops]
                    )

                    if self.dtype == "bfloat16":
                        output = paddle.cast(output, "float32")
                        gradient = paddle.cast(gradient, "float32")

                exe = paddle.static.Executor(place)
                exe.run(sp)
                fwd, rev = exe.run(
                    mp, feed={'x': self.x}, fetch_list=[output, gradient]
                )
                fwd_actual.append(fwd)
                rev_actual.append(rev)
                mps.append(mp)
                core._set_prim_backward_enabled(False)
                core._set_prim_forward_enabled(False)

        for i in range(len(self.places)):
            np.testing.assert_allclose(
                self.fwd_desire[i].sum(),
                fwd_actual[i].sum(),
                rtol=2e-2,  # mean of uniform distribution, scale for avoid random failed
                atol=0,
            )
            np.testing.assert_allclose(
                self.rev_desire[i].sum(),
                rev_actual[i].sum(),
                rtol=2e-2,  # mean of uniform distribution, scale for avoid random failed
                atol=0,
            )


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
