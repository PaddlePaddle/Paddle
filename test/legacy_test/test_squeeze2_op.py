#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest, convert_float_to_uint16
from utils import dygraph_guard, static_guard

import paddle
from paddle.base import core

paddle.enable_static()


# Correct: General.
class TestSqueezeOp(OpTest):
    def setUp(self):
        self.op_type = "squeeze2"
        self.prim_op_type = "prim"
        self.python_api = paddle.squeeze
        self.public_python_api = paddle.squeeze
        self.python_out_sig = [
            "Out"
        ]  # python out sig is customized output signature.
        self.init_test_case()
        self.init_dtype()
        self.if_enable_cinn()
        x = np.random.random(self.ori_shape).astype("float64")
        xshape = np.random.random(self.ori_shape).astype("float64")
        if hasattr(self, "dtype") and self.dtype == np.uint16:
            x = convert_float_to_uint16(x.astype(np.float32))
            xshape = convert_float_to_uint16(xshape.astype(np.float32))
        self.inputs = {"X": x}
        self.init_attrs()
        self.outputs = {
            "Out": self.inputs["X"].reshape(self.new_shape),
            "XShape": xshape,
        }

    def if_enable_cinn(self):
        pass

    def test_check_output(self):
        self.check_output(
            no_check_set=['XShape'],
            check_pir=True,
            check_prim_pir=True,
        )

    def test_check_grad(self):
        self.check_grad(
            ["X"],
            "Out",
            check_pir=True,
            check_prim_pir=True,
        )

    def init_dtype(self):
        self.dtype = np.float64

    def init_test_case(self):
        self.ori_shape = (1, 3, 1, 40)
        self.axes = (0, 2)
        self.new_shape = (3, 40)

    def init_attrs(self):
        self.attrs = {"axes": self.axes}


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and do not support bfloat16",
)
class TestSqueezeOpBF16OP(TestSqueezeOp):
    def init_dtype(self):
        self.dtype = np.uint16


# Correct: There is mins axis.
class TestSqueezeOp1(TestSqueezeOp):
    def init_test_case(self):
        self.ori_shape = (1, 20, 1, 5)
        self.axes = (0, -2)
        self.new_shape = (20, 5)


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and do not support bfloat16",
)
class TestSqueezeOp1BF16Op(TestSqueezeOp):
    def init_dtype(self):
        self.dtype = np.uint16


class TestSqueezeOp_ZeroDim1(TestSqueezeOp):
    def init_test_case(self):
        self.ori_shape = ()
        self.axes = (0,)
        self.new_shape = ()


class TestSqueezeOp_ZeroDim2(TestSqueezeOp):
    def init_test_case(self):
        self.ori_shape = (1, 1, 1)
        self.axes = (0, 1, 2)
        self.new_shape = ()


# Correct: No axes input.
class TestSqueezeOp2(TestSqueezeOp):
    def setUp(self):
        self.op_type = "squeeze2"
        self.prim_op_type = "comp"
        self.python_api = paddle.squeeze
        self.public_python_api = paddle.squeeze
        self.python_out_sig = [
            "Out"
        ]  # python out sig is customized output signature.
        self.init_test_case()
        self.init_dtype()
        self.if_enable_cinn()
        x = np.random.random(self.ori_shape).astype("float64")
        xshape = np.random.random(self.ori_shape).astype("float64")
        if hasattr(self, "dtype") and self.dtype == np.uint16:
            x = convert_float_to_uint16(x.astype(np.float32))
            xshape = convert_float_to_uint16(xshape.astype(np.float32))
        self.inputs = {"X": x}
        self.init_attrs()
        self.outputs = {
            "Out": self.inputs["X"].reshape(self.new_shape),
            "XShape": xshape,
        }

    def if_enable_cinn(self):
        pass

    def init_dtype(self):
        self.dtype = np.float64

    def init_test_case(self):
        self.ori_shape = (1, 20, 1, 5)
        self.axes = ()
        self.new_shape = (20, 5)


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and do not support bfloat16",
)
class TestSqueezeOp2BF16Op(TestSqueezeOp):
    def init_dtype(self):
        self.dtype = np.uint16


# Correct: Just part of axes be squeezed.
class TestSqueezeOp3(TestSqueezeOp):
    def init_test_case(self):
        self.ori_shape = (6, 1, 5, 1, 4, 1)
        self.axes = (1, -1)
        self.new_shape = (6, 5, 1, 4)


# Correct: Just not change shape.
class TestSqueezeOp4(TestSqueezeOp):
    def init_test_case(self):
        self.ori_shape = (3, 1, 5, 2)
        self.axes = (2, 3)
        self.new_shape = (3, 1, 5, 2)


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and do not support bfloat16",
)
class TestSqueezeOp3BF16Op(TestSqueezeOp):
    def init_dtype(self):
        self.dtype = np.uint16


# test api
class TestSqueezeAPI(unittest.TestCase):
    def setUp(self):
        self.executed_api()

    def executed_api(self):
        self.squeeze = paddle.squeeze

    def test_api(self):
        paddle.disable_static()
        input_data = np.random.random([3, 2, 1]).astype("float32")
        x = paddle.to_tensor(input_data)
        out = self.squeeze(x, axis=2)
        out.backward()

        self.assertEqual(out.shape, [3, 2])

        paddle.enable_static()

    def test_error(self):
        def test_axes_type():
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x2 = paddle.static.data(
                    name="x2", shape=[2, 1, 25], dtype="int32"
                )
                self.squeeze(x2, axis=2.1)

        self.assertRaises(TypeError, test_axes_type)


class TestSqueezeInplaceAPI(TestSqueezeAPI):
    def executed_api(self):
        self.squeeze = paddle.squeeze_


class TestSqueezeAPI_ZeroSize(unittest.TestCase):
    def setUp(self):
        self.executed_api()

    def executed_api(self):
        self.squeeze = paddle.squeeze

    def test_api(self):
        paddle.disable_static()
        input_data = np.random.random([3, 2, 1]).astype("float32")
        x = paddle.to_tensor(input_data)
        x.stop_gradient = False
        # axis set to 0-size
        out = self.squeeze(x, axis=paddle.to_tensor([], dtype=paddle.int32))
        np.testing.assert_allclose(out.numpy(), x.numpy())

        out.backward()
        np.testing.assert_allclose(x.grad.shape, x.shape)
        paddle.enable_static()


class TestSqueezeCompatibility(unittest.TestCase):
    def setUp(self):
        self.places = [paddle.CPUPlace()]
        if paddle.base.core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))
        self.func = paddle.squeeze
        self.init_data()
        self.init_case()

    def init_data(self):
        self.shape = [5, 1, 6]
        self.dtype = 'float32'
        self.axis = 1
        self.np_input = np.random.rand(*self.shape).astype(self.dtype)
        self.np_out = np.squeeze(self.np_input, axis=self.axis)

    def init_case(self):
        params = [['x', 'input'], ['axis', 'dim']]  # param1  # param2

        # Generate all valid combinations
        def generate_cases(param_groups, case_list):
            from itertools import product

            for combo in product(*[[None, *names] for names in param_groups]):
                args = ['pos' if p is None else 'kw' for p in combo]
                if args == sorted(args, key=lambda x: x != 'pos'):
                    case_list.append(combo)

        # paddle.squeeze()
        self.test_cases = []
        generate_cases(params, self.test_cases)
        # x.squeeze()
        self.tensor_test_cases = []
        generate_cases(params[1:], self.tensor_test_cases)

    def _build_args_kwargs(self, param_names, params):
        args = []
        kwargs = {}
        for name, param in zip(param_names, params):
            if name is None:
                args.append(param)
            else:
                kwargs[name] = param
        return args, kwargs

    def test_dygraph_compatibility(self):
        with dygraph_guard():
            for place in self.places:
                paddle.device.set_device(place)
                x = paddle.to_tensor(self.np_input)
                # paddle.
                for param_names in self.test_cases:
                    args, kwargs = self._build_args_kwargs(
                        param_names, (x, self.axis)
                    )
                    out = self.func(*args, **kwargs)
                    np.testing.assert_array_equal(self.np_out, out.numpy())
                # paddle.Tensor.
                for param_names in self.tensor_test_cases:
                    args, kwargs = self._build_args_kwargs(
                        param_names, (self.axis,)
                    )
                    out = x.squeeze(*args, **kwargs)
                    np.testing.assert_array_equal(self.np_out, out.numpy())

    def test_static_compatibility(self):
        with static_guard():
            for place in self.places:
                main = paddle.static.Program()
                startup = paddle.static.Program()
                with paddle.base.program_guard(main, startup):
                    x = paddle.static.data(
                        name="x", shape=self.shape, dtype=self.dtype
                    )
                    # paddle.
                    for param_names in self.test_cases:
                        args, kwargs = self._build_args_kwargs(
                            param_names, (x, self.axis)
                        )
                        out = self.func(*args, **kwargs)

                        exe = paddle.base.Executor(place)
                        fetches = exe.run(
                            main,
                            feed={"x": self.np_input},
                            fetch_list=[out],
                        )
                        np.testing.assert_array_equal(self.np_out, fetches[0])
                    # paddle.Tensor.
                    for param_names in self.tensor_test_cases:
                        args, kwargs = self._build_args_kwargs(
                            param_names, (self.axis,)
                        )

                        out = x.squeeze(*args, **kwargs)

                        exe = paddle.base.Executor(place)
                        fetches = exe.run(
                            main,
                            feed={"x": self.np_input},
                            fetch_list=[out],
                        )
                        np.testing.assert_array_equal(self.np_out, fetches[0])


if __name__ == "__main__":
    unittest.main()
