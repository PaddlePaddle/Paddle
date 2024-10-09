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

import gradient_checker
import numpy as np
from decorator_helper import prog_scope
from op_test import OpTest, convert_float_to_uint16
from utils import static_guard

import paddle
from paddle import base
from paddle.base import Program, core, program_guard
from paddle.framework import in_pir_mode


# Situation 1: shape is a list(without tensor)
class TestExpandV2OpRank1(OpTest):
    def setUp(self):
        self.op_type = "expand_v2"
        self.prim_op_type = "prim"
        self.init_data()
        self.python_api = paddle.expand
        self.public_python_api = paddle.expand
        self.inputs = {'X': np.random.random(self.ori_shape).astype("float64")}
        self.attrs = {'shape': self.shape}
        output = np.tile(self.inputs['X'], self.expand_times)
        self.outputs = {'Out': output}
        self.if_enable_cinn()

    def init_data(self):
        self.ori_shape = [100]
        self.shape = [100]
        self.expand_times = [1]

    def if_enable_cinn(self):
        pass

    def test_check_output(self):
        self.check_output(check_cinn=True, check_pir=True)

    def test_check_grad(self):
        self.check_grad(
            ['X'],
            'Out',
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
        )


class TestExpandV2OpRank1_ZeroDim1(TestExpandV2OpRank1):
    def init_data(self):
        self.ori_shape = []
        self.shape = [10]
        self.expand_times = [10]

    def if_enable_cinn(self):
        self.enable_cinn = False


class TestExpandV2OpRank1_ZeroDim2(TestExpandV2OpRank1):
    def init_data(self):
        self.ori_shape = []
        self.shape = []
        self.expand_times = []

    def if_enable_cinn(self):
        pass


class TestExpandV2OpRank2_DimExpanding(TestExpandV2OpRank1):
    def init_data(self):
        self.ori_shape = [120]
        self.shape = [2, 120]
        self.expand_times = [2, 1]


class TestExpandV2OpRank2(TestExpandV2OpRank1):
    def init_data(self):
        self.ori_shape = [1, 140]
        self.shape = [12, 140]
        self.expand_times = [12, 1]


class TestExpandV2OpRank3_Corner(TestExpandV2OpRank1):
    def init_data(self):
        self.ori_shape = (2, 10, 5)
        self.shape = (2, 10, 5)
        self.expand_times = (1, 1, 1)


class TestExpandV2OpRank4(TestExpandV2OpRank1):
    def init_data(self):
        self.ori_shape = (2, 4, 5, 7)
        self.shape = (-1, -1, -1, -1)
        self.expand_times = (1, 1, 1, 1)


class TestExpandV2OpRank5(TestExpandV2OpRank1):
    def init_data(self):
        self.ori_shape = [5, 2, 1, 4, 5]
        self.shape = [5, 2, 3, 4, 5]
        self.expand_times = [1, 1, 3, 1, 1]


class TestExpandV2OpRank5_Corner(TestExpandV2OpRank1):
    def init_data(self):
        self.ori_shape = [5, 2, 3, 4, 5]
        self.shape = [5, 2, 3, 4, 5]
        self.expand_times = [1, 1, 1, 1, 1]


class TestExpandV2OpRank5_ZeroDim(TestExpandV2OpRank1):
    def init_data(self):
        self.ori_shape = []
        self.shape = [5, 2, 3, 4, 5]
        self.expand_times = [5, 2, 3, 4, 5]

    def if_enable_cinn(self):
        self.enable_cinn = False


class TestExpandV2OpRank6(TestExpandV2OpRank1):
    def init_data(self):
        self.ori_shape = [1, 2, 1, 4, 5, 6]
        self.shape = [1, 2, 3, 4, 5, 6]
        self.expand_times = [1, 1, 3, 1, 1, 1]


class TestExpandV2OpRank6_Corner(TestExpandV2OpRank1):
    def init_data(self):
        self.ori_shape = [1, 2, 3, 4, 5, 6]
        self.shape = [1, 2, 3, 4, 5, 6]
        self.expand_times = [1, 1, 1, 1, 1, 1]


class TestExpandV2OpRank6_ZeroDim(TestExpandV2OpRank1):
    def init_data(self):
        self.ori_shape = []
        self.shape = [1, 2, 3, 4, 5, 6]
        self.expand_times = [1, 2, 3, 4, 5, 6]

    def if_enable_cinn(self):
        self.enable_cinn = False


class TestExpandV2OpRank7(TestExpandV2OpRank1):
    def init_data(self):
        self.ori_shape = [5, 2, 1, 4, 5, 6, 7]
        self.shape = [5, 2, 3, 4, 5, 6, 7]
        self.expand_times = [1, 1, 3, 1, 1, 1, 1]


class TestExpandV2OpRank7_Corner(TestExpandV2OpRank1):
    def init_data(self):
        self.ori_shape = [1, 2, 3, 4, 5, 2, 2]
        self.shape = [1, 2, 3, 4, 5, 2, 2]
        self.expand_times = [1, 1, 1, 1, 1, 1, 1]


class TestExpandV2OpRank7_ZeroDim(TestExpandV2OpRank1):
    def init_data(self):
        self.ori_shape = []
        self.shape = [1, 2, 3, 4, 5, 6, 7]
        self.expand_times = [1, 2, 3, 4, 5, 6, 7]

    def if_enable_cinn(self):
        self.enable_cinn = False


class TestExpandV2OpRank8(TestExpandV2OpRank1):
    def init_data(self):
        self.ori_shape = [1, 2, 1, 4, 5, 6, 7, 8]
        self.shape = [1, 2, 3, 4, 5, 6, 7, 8]
        self.expand_times = [1, 1, 3, 1, 1, 1, 1, 1]


class TestExpandV2OpRank8_Corner(TestExpandV2OpRank1):
    def init_data(self):
        self.ori_shape = [1, 2, 3, 4, 5, 2, 2, 2]
        self.shape = [1, 2, 3, 4, 5, 2, 2, 2]
        self.expand_times = [1, 1, 1, 1, 1, 1, 1, 1]

    def test_check_grad(self):
        self.check_grad(
            ['X'],
            'Out',
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
            numeric_grad_delta=1e-5,
            max_relative_error=2e-7,  # need slightly larger than 1e-7.
        )


class TestExpandV2OpRank8_ZeroDim(TestExpandV2OpRank1):
    def init_data(self):
        self.ori_shape = []
        self.shape = [1, 2, 3, 4, 5, 6, 7, 8]
        self.expand_times = [1, 2, 3, 4, 5, 6, 7, 8]


# Situation 2: shape is a list(with tensor)
class TestExpandV2OpRank1_tensor_attr(OpTest):
    def setUp(self):
        self.op_type = "expand_v2"
        self.prim_op_type = "prim"
        self.python_api = paddle.expand
        self.public_python_api = paddle.expand
        self.init_data()
        expand_shapes_tensor = []
        for index, ele in enumerate(self.expand_shape):
            expand_shapes_tensor.append(
                ("x" + str(index), np.ones(1).astype('int32') * ele)
            )

        self.inputs = {
            'X': np.random.random(self.ori_shape).astype("float64"),
            'expand_shapes_tensor': expand_shapes_tensor,
        }
        self.attrs = {"shape": self.infer_expand_shape}
        output = np.tile(self.inputs['X'], self.expand_times)
        self.outputs = {'Out': output}

    def init_data(self):
        self.ori_shape = [100]
        self.expand_times = [1]
        self.expand_shape = [100]
        self.infer_expand_shape = [-1]

    def test_check_output(self):
        self.check_output(
            check_cinn=True, check_pir=True, check_symbol_infer=False
        )

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_cinn=True, check_pir=True)


class TestExpandV2OpRank2_Corner_tensor_attr(TestExpandV2OpRank1_tensor_attr):
    def init_data(self):
        self.ori_shape = [12, 14]
        self.expand_times = [1, 1]
        self.expand_shape = [12, 14]
        self.infer_expand_shape = [12, -1]


# Situation 3: shape is a tensor
class TestExpandV2OpRank1_tensor(OpTest):
    def setUp(self):
        self.op_type = "expand_v2"
        self.prim_op_type = "prim"
        self.python_api = paddle.expand
        self.public_python_api = paddle.expand
        self.init_data()

        self.inputs = {
            'X': np.random.random(self.ori_shape).astype("float64"),
            'Shape': np.array(self.expand_shape).astype("int32"),
        }
        self.attrs = {}
        output = np.tile(self.inputs['X'], self.expand_times)
        self.outputs = {'Out': output}

    def init_data(self):
        self.ori_shape = [100]
        self.expand_times = [2, 1]
        self.expand_shape = [2, 100]

    def test_check_output(self):
        self.check_output(
            check_cinn=True, check_pir=True, check_symbol_infer=False
        )

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_cinn=True, check_pir=True)


# Situation 4: input x is Integer
class TestExpandV2OpInteger(OpTest):
    def setUp(self):
        self.op_type = "expand_v2"
        self.prim_op_type = "prim"
        self.python_api = paddle.expand
        self.public_python_api = paddle.expand
        self.inputs = {
            'X': np.random.randint(10, size=(2, 4, 5)).astype("int32")
        }
        self.attrs = {'shape': [2, 4, 5]}
        output = np.tile(self.inputs['X'], (1, 1, 1))
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output(check_cinn=True, check_pir=True)


#  Situation 5: input x is Bool
class TestExpandV2OpBoolean(OpTest):
    def setUp(self):
        self.op_type = "expand_v2"
        self.prim_op_type = "prim"
        self.python_api = paddle.expand
        self.public_python_api = paddle.expand
        self.inputs = {'X': np.random.randint(2, size=(2, 4, 5)).astype("bool")}
        self.attrs = {'shape': [2, 4, 5]}
        output = np.tile(self.inputs['X'], (1, 1, 1))
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output(check_cinn=True, check_pir=True)


#  Situation 6: input x is Integer
class TestExpandV2OpInt64_t(OpTest):
    def setUp(self):
        self.op_type = "expand_v2"
        self.prim_op_type = "prim"
        self.python_api = paddle.expand
        self.public_python_api = paddle.expand
        self.inputs = {
            'X': np.random.randint(10, size=(2, 4, 5)).astype("int64")
        }
        self.attrs = {'shape': [2, 4, 5]}
        output = np.tile(self.inputs['X'], (1, 1, 1))
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output(check_cinn=True, check_pir=True)


#  Situation 7: input x is Float16
class TestExpandV2FP16Op(OpTest):
    def setUp(self):
        self.op_type = "expand_v2"
        self.prim_op_type = "prim"
        self.dtype = np.float16
        self.python_api = paddle.expand
        self.public_python_api = paddle.expand
        self.inputs = {
            'X': np.random.randint(10, size=(8, 8, 5)).astype(self.dtype)
        }
        self.attrs = {'shape': [8, 8, 5]}
        output = np.tile(self.inputs['X'], (1, 1, 1))
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output(check_cinn=True)

    def test_check_grad(self):
        self.check_grad(
            ['X'],
            'Out',
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
        )


#  Situation 8: input x is BF16
@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support the bfloat16",
)
class TestExpandV2BF16Op(OpTest):
    def setUp(self):
        self.op_type = "expand_v2"
        self.prim_op_type = "prim"
        self.dtype = np.uint16
        self.python_api = paddle.expand
        self.public_python_api = paddle.expand
        x = np.random.randint(10, size=(8, 8, 5)).astype(np.float32)
        self.inputs = {'X': convert_float_to_uint16(x)}
        self.attrs = {'shape': [8, 8, 5]}
        output = np.tile(x, (1, 1, 1)).astype(np.float32)
        self.outputs = {'Out': convert_float_to_uint16(output)}

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, check_cinn=True, check_pir=True)

    def test_check_grad(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
            place,
            ['X'],
            'Out',
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
        )


class TestExpandV2Error(unittest.TestCase):

    def test_errors(self):
        with static_guard():
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                shape = [2, 2]
                if not in_pir_mode():
                    x1 = base.create_lod_tensor(
                        np.array([[-1]]), [[1]], base.CPUPlace()
                    )
                    self.assertRaises(
                        TypeError, paddle.tensor.expand, x1, shape
                    )
                x2 = paddle.static.data(name='x2', shape=[-1, 4], dtype="bool")
                x2.stop_gradient = False
                self.assertRaises(ValueError, paddle.tensor.expand, x2, shape)
                x2.stop_gradient = True
                self.assertRaises(TypeError, paddle.tensor.expand, x2, 1)


# Test python API
class TestExpandV2API(unittest.TestCase):

    def test_api(self):
        with paddle.static.program_guard(paddle.static.Program()):
            input = np.random.random([12, 14]).astype("float32")
            x = paddle.static.data(name='x', shape=[12, 14], dtype="float32")

            positive_2 = paddle.tensor.fill_constant([1], "int32", 12)
            expand_shape = paddle.static.data(
                name="expand_shape",
                shape=[2],
                dtype="int32",
            )

            out_1 = paddle.expand(x, shape=[12, 14])
            out_2 = paddle.expand(x, shape=[positive_2, 14])
            out_3 = paddle.expand(x, shape=expand_shape)

            exe = base.Executor(place=base.CPUPlace())
            res_1, res_2, res_3 = exe.run(
                paddle.static.default_main_program(),
                feed={
                    "x": input,
                    "expand_shape": np.array([12, 14]).astype("int32"),
                },
                fetch_list=[out_1, out_2, out_3],
            )
            np.testing.assert_array_equal(res_1, np.tile(input, (1, 1)))
            np.testing.assert_array_equal(res_2, np.tile(input, (1, 1)))
            np.testing.assert_array_equal(res_3, np.tile(input, (1, 1)))


class TestExpandInferShape(unittest.TestCase):
    def test_shape_with_var(self):
        with program_guard(Program(), Program()):
            x = paddle.static.data(shape=[-1, 1, 3], name='x')
            fake_var = paddle.randn([2, 3])
            target_shape = [
                -1,
                paddle.shape(fake_var)[0],
                paddle.shape(fake_var)[1],
            ]
            out = paddle.expand(x, shape=target_shape)
            self.assertListEqual(list(out.shape), [-1, -1, -1])


# Test python Dygraph API
class TestExpandV2DygraphAPI(unittest.TestCase):
    def test_expand_times_is_tensor(self):
        with paddle.base.dygraph.guard():
            paddle.seed(1)
            a = paddle.rand([2, 5])
            expand_1 = paddle.expand(a, shape=[2, 5])
            np_array = np.array([2, 5])
            expand_2 = paddle.expand(a, shape=np_array)
            np.testing.assert_array_equal(expand_1.numpy(), expand_2.numpy())


class TestExpandDoubleGradCheck(unittest.TestCase):
    def expand_wrapper(self, x):
        return paddle.expand(x[0], [2, 3])

    @prog_scope()
    def func(self, place):
        # the shape of input variable should be clearly specified, not include -1.
        eps = 0.005
        dtype = np.float32

        data = paddle.static.data('data', [2, 3], dtype)
        data.persistable = True
        out = paddle.expand(data, [2, 3])
        data_arr = np.random.uniform(-1, 1, data.shape).astype(dtype)

        gradient_checker.double_grad_check(
            [data], out, x_init=[data_arr], place=place, eps=eps
        )
        gradient_checker.double_grad_check_for_dygraph(
            self.expand_wrapper, [data], out, x_init=[data_arr], place=place
        )

    def test_grad(self):
        paddle.enable_static()
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestExpandTripleGradCheck(unittest.TestCase):
    def expand_wrapper(self, x):
        return paddle.expand(x[0], [2, 3])

    @prog_scope()
    def func(self, place):
        # the shape of input variable should be clearly specified, not include -1.
        eps = 0.005
        dtype = np.float32

        data = paddle.static.data('data', [2, 3], dtype)
        data.persistable = True
        out = paddle.expand(data, [2, 3])
        data_arr = np.random.uniform(-1, 1, data.shape).astype(dtype)

        gradient_checker.triple_grad_check(
            [data], out, x_init=[data_arr], place=place, eps=eps
        )
        gradient_checker.triple_grad_check_for_dygraph(
            self.expand_wrapper, [data], out, x_init=[data_arr], place=place
        )

    def test_grad(self):
        paddle.enable_static()
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)


# Situation 9: comp case, shape is a list(without tensor)
class TestExpandV2CompOpRank1(OpTest):
    def setUp(self):
        self.op_type = "expand_v2"
        self.prim_op_type = "comp"
        self.init_data()
        self.python_api = paddle.expand
        self.public_python_api = paddle.expand
        self.inputs = {'X': np.random.random(self.ori_shape).astype("float64")}
        self.attrs = {'shape': self.shape}
        output = np.tile(self.inputs['X'], self.expand_times)
        self.outputs = {'Out': output}
        self.enable_cinn = True

    def init_data(self):
        self.ori_shape = [100]
        self.shape = [100]
        self.expand_times = [1]

    def test_check_output(self):
        self.check_output(check_prim=True)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_prim=True, check_prim_pir=True)


class TestExpandV2OpCompRank2_DimExpanding(TestExpandV2CompOpRank1):
    def init_data(self):
        self.ori_shape = [120]
        self.shape = [2, 120]
        self.expand_times = [2, 1]


class TestExpandV2CompOpRank2(TestExpandV2CompOpRank1):
    def init_data(self):
        self.ori_shape = [1, 140]
        self.shape = [12, 140]
        self.expand_times = [12, 1]


class TestExpandV2CompOpRank3_Corner(TestExpandV2CompOpRank1):
    def init_data(self):
        self.ori_shape = (2, 10, 5)
        self.shape = (2, 10, 5)
        self.expand_times = (1, 1, 1)


class TestExpandV2CompOpRank4(TestExpandV2CompOpRank1):
    def init_data(self):
        self.ori_shape = (2, 4, 5, 7)
        self.shape = (-1, -1, -1, -1)
        self.expand_times = (1, 1, 1, 1)


# Situation 10: comp case, input x is Integer
class TestExpandV2CompOpInteger(OpTest):
    def setUp(self):
        self.op_type = "expand_v2"
        self.prim_op_type = "comp"
        self.python_api = paddle.expand
        self.public_python_api = paddle.expand
        self.inputs = {
            'X': np.random.randint(10, size=(2, 4, 5)).astype("int32")
        }
        self.attrs = {'shape': [2, 4, 5]}
        output = np.tile(self.inputs['X'], (1, 1, 1))
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output(check_prim=True)


#  Situation 11: comp case, input x is Bool
class TestExpandV2CompOpBoolean(OpTest):
    def setUp(self):
        self.op_type = "expand_v2"
        self.prim_op_type = "comp"
        self.python_api = paddle.expand
        self.public_python_api = paddle.expand
        self.inputs = {'X': np.random.randint(2, size=(2, 4, 5)).astype("bool")}
        self.attrs = {'shape': [2, 4, 5]}
        output = np.tile(self.inputs['X'], (1, 1, 1))
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output(check_prim=True)


#  Situation 12: comp case, input x is Integer
class TestExpandV2CompOpInt64_t(OpTest):
    def setUp(self):
        self.op_type = "expand_v2"
        self.prim_op_type = "comp"
        self.python_api = paddle.expand
        self.public_python_api = paddle.expand
        self.inputs = {
            'X': np.random.randint(10, size=(2, 4, 5)).astype("int64")
        }
        self.attrs = {'shape': [2, 4, 5]}
        output = np.tile(self.inputs['X'], (1, 1, 1))
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output(check_prim=True)


class TestExpandPirValueListShape(unittest.TestCase):

    def test_value_list_shape1(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data('x', [1, 1])
                shape = [2, paddle.full([], 4)]
                out = paddle.expand(x, shape)
                np.testing.assert_array_equal(tuple(out.shape), (2, -1))

    def test_value_list_shape2(self):
        with static_guard():
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data('x', [1, 1, -1, -1], 'float32')
                shape1 = paddle.static.data('shape1', [], 'int32')
                x = paddle.expand(x, shape=[shape1, 1, -1, -1])
                np.testing.assert_equal(tuple(x.shape), (-1, 1, -1, -1))


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
