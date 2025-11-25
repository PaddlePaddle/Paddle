# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import (
    OpTest,
    convert_float_to_uint16,
    get_device_place,
    get_devices,
    is_custom_device,
)

import paddle
from paddle.base import core


def compute_index_add_ref(
    axis, x_shape, x_np, add_value_shape, add_value_np, index_size, index_np
):
    if axis < 0:
        axis = axis + len(x_shape)
    if axis != 0:
        outer_loop = np.prod(x_shape[:axis]).astype(int)
        x_reshape = [outer_loop, *x_shape[axis:]]
        x_np_reshape = np.reshape(x_np, tuple(x_reshape))

        add_value_reshape = [
            np.prod(add_value_shape[:axis]).astype(int),
            *add_value_shape[axis:],
        ]

        add_value_np_reshape = np.reshape(
            add_value_np, tuple(add_value_reshape)
        )
    else:
        x_np_reshape = x_np
        add_value_np_reshape = add_value_np
    out_np = x_np_reshape.copy()

    if axis != 0:
        for i in range(outer_loop):
            for j in range(index_size):
                out_np[i, index_np[j]] += add_value_np_reshape[i, j]
    else:
        for j in range(index_size):
            out_np[index_np[j]] += add_value_np_reshape[j]
    ref_out = np.reshape(out_np, x_shape)
    return ref_out


def raw_index_add(x, index, value, axis):
    return paddle.index_add(x, index, axis, value)


class TestIndexAddOp(OpTest):
    def setUp(self):
        self.python_api = raw_index_add
        self.op_type = "index_add"
        self.prim_op_type = "prim"
        self.public_python_api = raw_index_add
        self.init_dtype_type()
        index_np = np.random.randint(
            low=-self.x_shape[self.axis],
            high=self.x_shape[self.axis],
            size=self.index_size,
        )
        x_np = np.random.random(self.x_shape).astype(self.x_type)
        add_value_np = np.random.random(self.add_value_shape).astype(
            self.x_type
        )

        self.inputs = {'X': x_np, 'Index': index_np, 'AddValue': add_value_np}
        self.attrs = {'axis': self.axis}
        out = compute_index_add_ref(
            self.axis,
            self.x_shape,
            x_np,
            self.add_value_shape,
            add_value_np,
            self.index_size,
            index_np,
        )
        self.outputs = {'Out': out}

    def init_dtype_type(self):
        self.axis = 0
        self.x_type = np.float64
        self.index_type = np.int64
        self.x_shape = (101, 3)
        self.index_size = 3
        self.add_value_shape = (3, 3)

    def test_check_output(self):
        self.check_output(atol=1e-2, check_pir=True, check_prim_pir=True)

    def test_check_grad_normal(self):
        self.check_grad(
            ['X', 'AddValue'], 'Out', check_pir=True, check_prim_pir=True
        )


class TestIndexAddFP16Op(TestIndexAddOp):
    def init_dtype_type(self):
        self.axis = 0
        self.x_type = np.float16
        self.index_type = np.int64
        self.x_shape = (101, 3)
        self.index_size = 3
        self.add_value_shape = (3, 3)
        self.dtype = np.float16


@unittest.skipIf(
    not (core.is_compiled_with_cuda() or is_custom_device())
    or not core.is_bfloat16_supported(get_device_place()),
    "core is not compiled with CUDA or not support bfloat16",
)
class TestIndexAddBF16Op(OpTest):
    def setUp(self):
        self.python_api = raw_index_add
        self.op_type = "index_add"
        self.prim_op_type = "prim"
        self.public_python_api = raw_index_add
        self.init_dtype_type()
        index_np = np.random.randint(
            low=-self.x_shape[self.axis],
            high=self.x_shape[self.axis],
            size=self.index_size,
        )
        x_np = np.random.random(self.x_shape).astype(self.x_type)
        add_value_np = np.random.random(self.add_value_shape).astype(
            self.x_type
        )

        self.inputs = {
            'X': convert_float_to_uint16(x_np),
            'Index': index_np,
            'AddValue': convert_float_to_uint16(add_value_np),
        }
        self.attrs = {'axis': self.axis}
        out = compute_index_add_ref(
            self.axis,
            self.x_shape,
            x_np,
            self.add_value_shape,
            add_value_np,
            self.index_size,
            index_np,
        )
        self.outputs = {'Out': convert_float_to_uint16(out)}
        self.place = get_device_place()

    def init_dtype_type(self):
        self.axis = 0
        self.x_type = np.float32
        self.index_type = np.int64
        self.x_shape = (101, 3)
        self.index_size = 3
        self.add_value_shape = (3, 3)
        self.dtype = np.uint16

    def test_check_output(self):
        self.check_output_with_place(self.place, check_pir=True)

    def test_check_grad_normal(self):
        self.check_grad_with_place(
            self.place,
            ['X', 'AddValue'],
            'Out',
            check_pir=True,
            check_prim_pir=True,
        )


class TestIndexAddAPI(unittest.TestCase):
    def setUp(self):
        self.setType()
        self.setPlace()
        self.config()
        self.check_backward = True
        self.generate_input_data()

        self.index_shape = (self.index_size,)

        self.rtol = 1e-5
        self.atol = 1e-2
        if self.x_type is np.float16:
            self.atol = 1e-1

    def setType(self):
        self.x_type = np.float32
        self.index_type = np.int32

    def setPlace(self):
        self.place = get_devices()

    def config(self):
        self.axis = 0
        self.x_shape = (100, 5)
        self.index_size = 20
        self.add_value_shape = (20, 5)

    def generate_input_data(self):
        axis = self.axis
        if self.axis < 0:
            axis = self.axis + len(self.x_shape)

        self.x_np = np.random.random(self.x_shape).astype(self.x_type)
        self.add_value_np = np.random.random(self.add_value_shape).astype(
            self.x_type
        )
        self.index_np = np.random.randint(
            low=-self.x_shape[axis],
            high=self.x_shape[axis],
            size=self.index_size,
        ).astype(self.index_type)
        if self.check_backward:
            self.dout_np = np.random.random(self.x_shape).astype(self.x_type)

    def compute_index_add_backward_ref(self):
        axis = self.axis
        if self.axis < 0:
            axis = self.axis + len(self.x_shape)

        x_grad = self.dout_np

        dout_tensor = paddle.to_tensor(self.dout_np)
        index = paddle.to_tensor(self.index_np)
        add_value_grad = paddle.index_select(dout_tensor, index, axis)

        return x_grad, add_value_grad.numpy()

    def run_imperative(self, device):
        input_tensor = paddle.to_tensor(
            self.x_np, stop_gradient=False, place=device
        )
        index = paddle.to_tensor(self.index_np, place=device)
        add_value = paddle.to_tensor(
            self.add_value_np, stop_gradient=False, place=device
        )

        out = paddle.index_add(input_tensor, index, self.axis, add_value)
        ref_out = compute_index_add_ref(
            self.axis,
            self.x_shape,
            self.x_np,
            self.add_value_shape,
            self.add_value_np,
            self.index_size,
            self.index_np,
        )
        np.testing.assert_allclose(
            ref_out, out.numpy(), rtol=self.rtol, atol=self.atol
        )

        if self.check_backward:
            dout_tensor = paddle.to_tensor(self.dout_np)
            (input_tensor_grad,) = paddle.autograd.grad(
                [out], [input_tensor], dout_tensor
            )
            (add_value_grad,) = paddle.autograd.grad(
                [out], [add_value], dout_tensor
            )

            (
                ref_x_grad,
                ref_add_value_grad,
            ) = self.compute_index_add_backward_ref()
            np.testing.assert_allclose(
                ref_x_grad,
                input_tensor_grad.numpy(),
                rtol=self.rtol,
                atol=self.atol,
            )
            np.testing.assert_allclose(
                ref_add_value_grad,
                add_value_grad.numpy(),
                rtol=self.rtol,
                atol=self.atol,
            )

    def run_static(self, device):
        x = paddle.static.data(name='X', shape=self.x_shape, dtype=self.x_type)
        index = paddle.static.data(
            name='Index', shape=self.index_shape, dtype=self.index_type
        )
        add_value = paddle.static.data(
            name='AddValue', shape=self.add_value_shape, dtype=self.x_type
        )

        out = paddle.index_add(x, index, self.axis, add_value)

        if device == "cpu":
            place = paddle.CPUPlace()
        elif device == "gpu" or is_custom_device():
            place = get_device_place()
        else:
            raise TypeError(
                "paddle.index_add api only support cpu and gpu device now."
            )

        exe = paddle.static.Executor(place)
        exe.run(paddle.static.default_startup_program())

        res = exe.run(
            paddle.static.default_main_program(),
            feed={
                "X": self.x_np,
                "Index": self.index_np,
                "AddValue": self.add_value_np,
            },
            fetch_list=[out],
            return_numpy=False,
        )
        return res

    def test_static(self):
        paddle.enable_static()
        for device in self.place:
            with paddle.static.program_guard(paddle.static.Program()):
                out = self.run_static(device)
            ref_out = compute_index_add_ref(
                self.axis,
                self.x_shape,
                self.x_np,
                self.add_value_shape,
                self.add_value_np,
                self.index_size,
                self.index_np,
            )
            np.testing.assert_allclose(
                ref_out, np.array(out[0]), rtol=self.rtol, atol=self.atol
            )

    def test_dynamic(self):
        paddle.disable_static()
        for device in self.place:
            self.run_imperative(device)


class TestIndexAddAPIMoreType(TestIndexAddAPI):
    def setType(self):
        self.x_type = np.float64
        self.index_type = np.int64


class TestIndexAddAPICase2(TestIndexAddAPI):
    def config(self):
        self.axis = 1
        self.x_shape = (100, 100, 5)
        self.index_size = 20
        self.add_value_shape = (100, 20, 5)


class TestIndexAddAPICase3(TestIndexAddAPI):
    def config(self):
        self.axis = 2
        self.x_shape = (100, 100, 25)
        self.index_size = 20
        self.add_value_shape = (100, 100, 20)


class TestIndexAddAPICase4(TestIndexAddAPI):
    def config(self):
        self.axis = 0
        self.x_shape = (10,)
        self.index_size = 4
        self.add_value_shape = (4,)


class TestIndexAddAPICase5(TestIndexAddAPI):
    def config(self):
        self.axis = -1
        self.x_shape = (10, 10)
        self.index_size = 4
        self.add_value_shape = (10, 4)


# class TestIndexAddAPIError(unittest.TestCase):

#     def test_errors(self):
#         paddle.enable_static()
#         with paddle.static.program_guard(paddle.static.Program(),
#                                          paddle.static.Program()):

#             def test_add_value_shape():
#                 axis = 0
#                 x = paddle.static.data(name='X',
#                                        shape=[10, 10],
#                                        dtype="float64")
#                 index = paddle.static.data(name='Index',
#                                            shape=[4],
#                                            dtype="int32")
#                 add_value = paddle.static.data(name='AddValue',
#                                                shape=[4, 3],
#                                                dtype="float64")
#                 out = paddle.index_add(x, index, axis, add_value)

#             self.assertRaises(ValueError, test_add_value_shape)

#             def test_index_dtype():
#                 axis = 0
#                 x = paddle.static.data(name='X1',
#                                        shape=[10, 10],
#                                        dtype="float64")
#                 index = paddle.static.data(name='Index1',
#                                            shape=[4],
#                                            dtype="float32")
#                 add_value = paddle.static.data(name='AddValue1',
#                                                shape=[4, 10],
#                                                dtype="float64")
#                 out = paddle.index_add(x, index, axis, add_value)

#             self.assertRaises(TypeError, test_index_dtype)

#             def test_index_shape():
#                 axis = 0
#                 x = paddle.static.data(name='X2',
#                                        shape=[10, 10],
#                                        dtype="float64")
#                 index = paddle.static.data(name='Index2',
#                                            shape=[4, 3],
#                                            dtype="int32")
#                 add_value = paddle.static.data(name='AddValue2',
#                                                shape=[4, 10],
#                                                dtype="float64")
#                 out = paddle.index_add(x, index, axis, add_value)

#             self.assertRaises(ValueError, test_index_shape)

#             def test_axis_value():
#                 axis = 3
#                 x = paddle.static.data(name='X3',
#                                        shape=[10, 10],
#                                        dtype="float64")
#                 index = paddle.static.data(name='Index3',
#                                            shape=[4],
#                                            dtype="int32")
#                 add_value = paddle.static.data(name='AddValue3',
#                                                shape=[4, 10],
#                                                dtype="float64")
#                 out = paddle.index_add(x, index, axis, add_value)

#             self.assertRaises(ValueError, test_axis_value)

#             def test_add_value_broadcast():
#                 axis = 0
#                 x = paddle.static.data(name='X4',
#                                        shape=[10, 10],
#                                        dtype="float64")
#                 index = paddle.static.data(name='Index4',
#                                            shape=[4],
#                                            dtype="int32")
#                 add_value = paddle.static.data(name='AddValue4',
#                                                shape=[4],
#                                                dtype="float64")
#                 out = paddle.index_add(x, index, axis, add_value)

#             self.assertRaises(ValueError, test_add_value_broadcast)


class TestIndexAddOp_ZeroSize(OpTest):
    def setUp(self):
        self.python_api = raw_index_add
        self.op_type = "index_add"
        self.prim_op_type = "prim"
        self.public_python_api = raw_index_add
        self.init_dtype_type()
        index_np = np.random.randint(
            low=-self.x_shape[self.axis],
            high=self.x_shape[self.axis],
            size=self.index_size,
        )
        x_np = np.random.random(self.x_shape).astype(self.x_type)
        add_value_np = np.random.random(self.add_value_shape).astype(
            self.x_type
        )

        self.inputs = {'X': x_np, 'Index': index_np, 'AddValue': add_value_np}
        self.attrs = {'axis': self.axis}
        out = compute_index_add_ref(
            self.axis,
            self.x_shape,
            x_np,
            self.add_value_shape,
            add_value_np,
            self.index_size,
            index_np,
        )
        self.outputs = {'Out': out}

    def init_dtype_type(self):
        self.axis = 0
        self.x_type = np.float64
        self.index_type = np.int64
        self.x_shape = (101, 0)
        self.index_size = 3
        self.add_value_shape = (3, 0)

    def test_check_output(self):
        self.check_output(atol=1e-2, check_pir=True)

    def test_check_grad_normal(self):
        self.check_grad(
            ['X', 'AddValue'], 'Out', check_pir=True, check_prim_pir=True
        )


class TestIndexAdd_ZeroSize2(OpTest):
    def setUp(self):
        self.python_api = raw_index_add
        self.op_type = "index_add"
        self.prim_op_type = "prim"
        self.public_python_api = raw_index_add
        self.init_dtype_type()
        index_np = np.array([], dtype=self.index_type)
        x_np = np.random.random(self.x_shape).astype(self.x_type)
        add_value_np = np.random.random(self.add_value_shape).astype(
            self.x_type
        )

        self.inputs = {'X': x_np, 'Index': index_np, 'AddValue': add_value_np}
        self.attrs = {'axis': self.axis}
        out = x_np.copy()
        self.outputs = {'Out': out}

    def init_dtype_type(self):
        self.x_type = np.float32
        self.index_type = np.int32
        self.x_shape = (10,)
        self.index_size = 0
        self.axis = 0
        self.add_value_shape = (0,)

    def test_check_output(self):
        self.check_output(atol=1e-2, check_pir=True)

    def test_check_grad_normal(self):
        self.check_grad(
            ['X', 'AddValue'], 'Out', check_pir=True, check_prim_pir=True
        )


def get_places():
    places = []
    if paddle.base.is_compiled_with_cuda() or is_custom_device():
        places.append(get_device_place())
    places.append(paddle.CPUPlace())
    return places


class TestIndexAddAPI_Compatibility(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        self.places = get_places()
        self.shape = [10, 20]
        self.index_shape = [5]
        self.axis = 1
        self.dtype = "float32"
        self.value_shape = list(self.shape)
        self.value_shape[self.axis] = self.index_shape[0]
        self.init_data()

    def init_data(self):
        self.np_input = np.random.rand(*self.shape).astype(self.dtype)
        self.np_index = np.random.randint(
            0, self.shape[self.axis], self.index_shape
        ).astype("int64")
        self.np_value = np.random.rand(*self.value_shape).astype(self.dtype)

    def get_ref_out(self, alpha=1.0):
        ref_out = np.copy(self.np_input)
        idx = [slice(None)] * len(self.shape)
        idx[self.axis] = self.np_index
        np.add.at(ref_out, tuple(idx), self.np_value * alpha)
        return ref_out

    def test_dygraph_Compatibility(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.np_input)
        index = paddle.to_tensor(self.np_index)
        value = paddle.to_tensor(self.np_value)
        paddle_dygraph_out = []

        ref_out = self.get_ref_out()
        # 1. Position args (Paddle style: x, index, axis, value)
        out1 = paddle.index_add(x, index, self.axis, value)
        paddle_dygraph_out.append(out1)
        # 2. Key words args (kwargs) for paddle
        out2 = paddle.index_add(x=x, index=index, axis=self.axis, value=value)
        paddle_dygraph_out.append(out2)
        # 3. Key words args (kwargs) for torch
        out3 = paddle.index_add(
            input=x, dim=self.axis, index=index, source=value
        )
        paddle_dygraph_out.append(out3)
        # 4. PyTorch positional args order: (input, dim, index, source)
        out4 = paddle.index_add(x, self.axis, index, value)
        paddle_dygraph_out.append(out4)
        # 5. Tensor method args (Paddle style)
        out5 = x.index_add(index, self.axis, value)
        paddle_dygraph_out.append(out5)
        # 6. Tensor method kwargs (PyTorch style)
        out6 = x.index_add(dim=self.axis, index=index, source=value)
        paddle_dygraph_out.append(out6)
        # 7. Test 'out' parameter
        out7 = paddle.empty_like(x)
        paddle.index_add(
            input=x, dim=self.axis, index=index, source=value, out=out7
        )
        paddle_dygraph_out.append(out7)
        # 8. Test 'alpha' parameter
        alpha = 2.0
        out8 = paddle.index_add(
            input=x, dim=self.axis, index=index, source=value, alpha=alpha
        )
        out9 = paddle.index_add_(
            input=x, dim=self.axis, index=index, source=value, alpha=alpha
        )
        ref_out_alpha = self.get_ref_out(alpha=alpha)

        for out in paddle_dygraph_out:
            np.testing.assert_allclose(ref_out, out.numpy(), rtol=1e-05)
        np.testing.assert_allclose(ref_out_alpha, out8.numpy(), rtol=1e-05)
        np.testing.assert_allclose(ref_out_alpha, out9.numpy(), rtol=1e-05)
        paddle.enable_static()

    def test_static_Compatibility(self):
        paddle.enable_static()
        main = paddle.static.Program()
        startup = paddle.static.Program()
        with paddle.base.program_guard(main, startup):
            x = paddle.static.data(name="x", shape=self.shape, dtype=self.dtype)
            index = paddle.static.data(
                name="index", shape=self.index_shape, dtype="int64"
            )
            value = paddle.static.data(
                name="value", shape=self.value_shape, dtype=self.dtype
            )
            # 1. Position args (Paddle style: x, index, axis, value)
            out1 = paddle.index_add(x, index, self.axis, value)
            # 2. Key words args (kwargs) for paddle
            out2 = paddle.index_add(
                x=x, index=index, axis=self.axis, value=value
            )
            # 3. Key words args (kwargs) for torch
            out3 = paddle.index_add(
                input=x, dim=self.axis, index=index, source=value
            )
            # 4. PyTorch positional args order: (input, dim, index, source)
            out4 = paddle.index_add(x, self.axis, index, value)
            # 5. Tensor method args (Paddle style)
            out5 = x.index_add(index, self.axis, value)
            # 6. Tensor method kwargs (PyTorch style)
            out6 = x.index_add(dim=self.axis, index=index, source=value)
            # 7. Test 'alpha' parameter
            alpha = 2.0
            out7 = paddle.index_add(
                input=x, dim=self.axis, index=index, source=value, alpha=alpha
            )
            ref_out = self.get_ref_out()
            ref_out_alpha = self.get_ref_out(alpha=alpha)

            fetch_list = [
                out1,
                out2,
                out3,
                out4,
                out5,
                out6,
                out7,
            ]
            feed_dict = {
                "x": self.np_input,
                "index": self.np_index,
                "value": self.np_value,
            }

            for place in self.places:
                exe = paddle.base.Executor(place)
                fetches = exe.run(
                    main,
                    feed=feed_dict,
                    fetch_list=fetch_list,
                )
                for out in fetches[:-1]:
                    np.testing.assert_allclose(out, ref_out, rtol=1e-05)
                np.testing.assert_allclose(
                    fetches[-1], ref_out_alpha, rtol=1e-05
                )


if __name__ == '__main__':
    unittest.main()
