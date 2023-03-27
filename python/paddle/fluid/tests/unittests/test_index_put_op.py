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

import paddle
from paddle import _C_ops


def compute_index_put_ref(x_np, indices_np, value_np):
    print(indices_np)
    print(x_np[indices_np].shape)
    x_np[indices_np] = value_np
    return x_np


def raw_index_put(x, indices, value):
    return _C_ops.index_put(x, indices, value)


def gen_indices_np(x_shape, indices_shapes, index_type):
    indices = []
    if index_type == np.bool:
        indices = np.zeros(indices_shapes[0], dtype=np.bool)
        indices.flatten()
        for i in range(len(indices)):
            indices[i] = (i & 1) == 0
        indices = indices.reshape(indices_shapes[0])

    else:
        for i in range(len(indices_shapes)):
            index_np = np.random.randint(
                low=0, high=x_shape[i], size=indices_shapes[i], dtype=index_type
            )
            indices.append(index_np)
    return indices


class TestIndexPutOp(unittest.TestCase):
    def setUp(self):
        self.init_dtype_type()
        self.x_np = np.random.random(self.x_shape).astype(self.dtype_np)
        self.value_np = np.random.random(self.value_shape).astype(self.dtype_np)
        self.indices_np = gen_indices_np(
            self.x_shape, self.indices_shapes, self.index_type_np
        )

        self.x_pd = paddle.to_tensor(self.x_np, dtype=self.dtype_pd)
        self.value_pd = paddle.to_tensor(self.value_np, dtype=self.dtype_pd)
        self.indices_pd = [
            paddle.to_tensor(indice, dtype=self.index_type_pd)
            for indice in self.indices_np
        ]

    def init_dtype_type(self):
        self.dtype_np = np.float64
        self.index_type_np = np.int64
        self.x_shape = (100, 110)
        self.indices_shapes = ((21,), (21,))
        self.value_shape = (21,)
        self.dtype_pd = paddle.float64
        self.index_type_pd = paddle.int64
        self.dtype_pd = paddle.float64

    def test_forward(self):
        ref_res = compute_index_put_ref(
            self.x_np, self.indices_np, self.value_np
        )
        pd_res = raw_index_put(self.x_pd, self.indices_pd, self.value_pd)
        print("here executino!")
        print(ref_res.shape)
        print(pd_res.shape)
        np.testing.assert_allclose(ref_res, pd_res.numpy(), atol=1e-7)

    def test_backward(self):
        value = paddle.ones(shape=[4], dtype=self.dtype_pd)
        x = paddle.ones(shape=[16, 21], dtype=self.dtype_pd)
        ix1 = paddle.to_tensor([0, 1, 2, 3], dtype=self.index_type_pd)
        ix2 = paddle.to_tensor([0, 1, 2, 3], dtype=self.index_type_pd)
        value.stop_gradient = False
        x[ix1, ix2] = value

        dvalue = paddle.grad(
            outputs=[x], inputs=[value], create_graph=False, retain_graph=True
        )[0]

        np.testing.assert_allclose(
            np.array([1.0, 1.0, 1.0, 1.0], dtype=self.dtype_np),
            dvalue.numpy(),
            atol=1e-7,
        )

    def test_backward1(self):
        value = paddle.ones(shape=[1], dtype=self.dtype_pd)
        x = paddle.ones(shape=[16, 21], dtype=self.dtype_pd)
        ix1 = paddle.to_tensor([0, 1, 2, 3], dtype=self.index_type_pd)
        ix2 = paddle.to_tensor([0, 1, 2, 3], dtype=self.index_type_pd)
        value.stop_gradient = False
        x[ix1, ix2] = value

        dvalue = paddle.grad(
            outputs=[x], inputs=[value], create_graph=False, retain_graph=True
        )[0]

        np.testing.assert_allclose(
            np.array([4.0], dtype=self.dtype_np), dvalue.numpy(), atol=1e-7
        )


class TestIndexPutOpFloat32(TestIndexPutOp):
    def init_dtype_type(self):
        self.dtype_np = np.float32
        self.index_type_np = np.int64
        self.x_shape = (100, 110)
        self.indices_shapes = ((21,), (21,))
        self.value_shape = (21,)
        self.dtype_pd = paddle.float32
        self.index_type_pd = paddle.int64
        self.dtype_pd = paddle.float32


class TestIndexPutOpFloat16(TestIndexPutOp):
    def init_dtype_type(self):
        self.dtype_np = np.float16
        self.index_type_np = np.int64
        self.x_shape = (100, 110)
        self.indices_shapes = ((21,), (21,))
        self.value_shape = (21,)
        self.dtype_pd = paddle.float16
        self.index_type_pd = paddle.int64
        self.dtype_pd = paddle.float16


class TestIndexPutOpInt32(TestIndexPutOp):
    def init_dtype_type(self):
        self.dtype_np = np.int32
        self.index_type_np = np.int64
        self.x_shape = (100, 110)
        self.indices_shapes = ((21,), (21,))
        self.value_shape = (21,)
        self.dtype_pd = paddle.int32
        self.index_type_pd = paddle.int64
        self.dtype_pd = paddle.int32


class TestIndexPutOpInt64(TestIndexPutOp):
    def init_dtype_type(self):
        self.dtype_np = np.int64
        self.index_type_np = np.int64
        self.x_shape = (100, 110)
        self.indices_shapes = ((21,), (21,))
        self.value_shape = (21,)
        self.dtype_pd = paddle.int64
        self.index_type_pd = paddle.int64
        self.dtype_pd = paddle.int64


class TestIndexPutOpBool(TestIndexPutOp):
    def init_dtype_type(self):
        self.dtype_np = np.bool
        self.index_type_np = np.int64
        self.x_shape = (100, 110)
        self.indices_shapes = ((21,), (21,))
        self.value_shape = (21,)
        self.dtype_pd = paddle.bool
        self.index_type_pd = paddle.int64
        self.dtype_pd = paddle.bool


if __name__ == '__main__':
    unittest.main()


# 下面都做了，还没加反向，有点复杂
# TODO different type test
# TODO less dims
# TODO different dims to bd
# TODO int or bool index tensor


# def compute_index_put_ref(x_np, value_np, indices_np, accumulate):
#     if accumulate:
#         x_np[indices_np] += value_np
#         return x_np
#     else:
#         x_np[indices_np] = value_np
#         return x_np


# def raw_index_put(x, indices, value, accumulate):
#     return paddle.index_put(x, indices, value, accumulate)


# def gen_indices_np(x_shape, indices_shapes, index_type):
#     indices = []
#     if index_type == np.bool:
#         indices = np.zeros(indices_shapes[0], dtype=np.bool)
#         indices.flatten()
#         for i in range(len(indices)):
#             indices[i] = (i & 1) == 0
#         indices = indices.reshape(indices_shapes[0])

#     else:
#         for i in range(len(indices_shapes)):
#             index_np = np.random.randint(
#                 low=0, high=x_shape[i], size=indices_shapes[i], dtype=index_type
#             )
#         indices.append(index_np)
#     return indices


# class TestIndexPutOp(OpTest):
#     def setUp(self):
#         self.python_api = raw_index_put
#         self.op_type = "index_put"
#         self.init_dtype_type()
#         x_np = np.random.random(self.x_shape).astype(self.x_type)
#         value_np = np.random.random(self.value_shape).astype(self.x_type)
#         indices_np = gen_indices_np(
#             self.x_shape, self.indices_shapes, self.index_type
#         )

#         self.inputs = {'X': x_np, 'Indices': indices_np, 'Value': value_np}

#         out = compute_index_put_ref(x_np, value_np, indices_np)
#         self.outputs = {'Out': out}

#     def init_dtype_type(self):
#         self.x_type = np.float64
#         self.index_type = np.int64
#         self.x_shape = (100, 110)
#         self.indices_shapes = ((21,), (21,))
#         self.value_shape = (21,)

#     def test_check_output(self):
#         self.check_output(check_eager=True, atol=1e-2)

#     def test_check_grad_normal(self):
#         self.check_grad(['X', 'Indices', 'Value'], 'Out', check_eager=True)


# class TestIndexPutOpFloat32(TestIndexPutOp):
#     def init_dtype_type(self):
#         self.x_type = np.float32
#         self.index_type = np.int64
#         self.x_shape = (100, 110)
#         self.indices_shapes = ((21,), (21,))
#         self.value_shape = (21,)


# class TestIndexPutOpFloat16(TestIndexPutOp):
#     def init_dtype_type(self):
#         self.x_type = np.float16
#         self.index_type = np.int64
#         self.x_shape = (100, 110)
#         self.indices_shapes = ((21,), (21,))
#         self.value_shape = (21,)


# class TestIndexPutOpInt32(TestIndexPutOp):
#     def init_dtype_type(self):
#         self.x_type = np.int32
#         self.index_type = np.int64
#         self.x_shape = (100, 110)
#         self.indices_shapes = ((21,), (21,))
#         self.value_shape = (21,)


# class TestIndexPutOpInt64(TestIndexPutOp):
#     def init_dtype_type(self):
#         self.x_type = np.int64
#         self.index_type = np.int64
#         self.x_shape = (100, 110)
#         self.indices_shapes = ((21,), (21,))
#         self.value_shape = (21,)


# # set_up 生成可能要改
# class TestIndexPutOpBool(TestIndexPutOp):
#     def init_dtype_type(self):
#         self.x_type = np.Bool
#         self.index_type = np.int64
#         self.x_shape = (100, 110)
#         self.indices_shapes = ((21,), (21,))
#         self.value_shape = (21,)


# class TestIndexPutAPI(unittest.TestCase):
#     def setUp(self):
#         self.setType()
#         self.setPlace()
#         self.config()
#         self.check_backward = False
#         self.generate_input_data()

#         self.rtol = 1e-5
#         self.atol = 1e-2
#         if self.x_type is np.float16:
#             self.atol = 1e-1

#     def setType(self):
#         self.x_type = np.float32
#         self.index_type = np.int32

#     def setPlace(self):
#         self.place = ['cpu']
#         if paddle.is_compiled_with_cuda():
#             self.place.append('gpu')

#     def config(self):
#         self.x_shape = (100, 5)
#         self.indices_shape = 20
#         self.value_shape = (20, 5)
#         self.accumulate = False

#     def generate_input_data(self):
#         self.x_np = np.random.random(self.x_shape).astype(self.x_type)
#         self.add_value_np = np.random.random(self.value_shape).astype(
#             self.x_type
#         )
#         self.indices_np = gen_indices_np(
#             self.x_shape, self.indices_shape, self.index_type
#         )

#         if self.check_backward:
#             self.dout_np = np.random.random(self.x_shape).astype(self.x_type)

#     # this API backward is complicated
#     def compute_index_put_backward_ref(self):
#         a = 1
#         return a
#         # TODO: implement this

#     def run_imperative(self, device):
#         paddle.device.set_device(device)
#         input_tensor = paddle.to_tensor(self.x_np, stop_gradient=False)
#         indices = []
#         for ele in self.indices_np:
#             indices.append(paddle.to_tensor(ele))
#         indices = tuple(indices)
#         value = paddle.to_tensor(self.value_np, stop_gradient=False)

#         out = paddle.index_put(input_tensor, indices, value, self.accumulate)
#         ref_out = compute_index_put_ref(
#             self.x_np, self.value_np, self.indices_np, self.accumulate
#         )
#         np.testing.assert_allclose(
#             ref_out, out.numpy(), rtol=self.rtol, atol=self.atol
#         )

#         # if self.check_backward:
#         #     dout_tensor = paddle.to_tensor(self.dout_np)
#         #     paddle.autograd.backward([out], [dout_tensor], retain_graph=True)
#         #     (
#         #         ref_x_grad,
#         #         ref_add_value_grad,
#         #     ) = self.compute_index_add_backward_ref()
#         #     np.testing.assert_allclose(
#         #         ref_x_grad,
#         #         input_tensor.grad.numpy(),
#         #         rtol=self.rtol,
#         #         atol=self.atol,
#         #     )
#         #     np.testing.assert_allclose(
#         #         ref_add_value_grad,
#         #         add_value.grad.numpy(),
#         #         rtol=self.rtol,
#         #         atol=self.atol,
#         #     )

#     def run_static(self, device):
#         x = paddle.static.data(name='X', shape=self.x_shape, dtype=self.x_type)
#         indices = []
#         for shape in self.indices_shape:
#             index = paddle.static.data(
#                 name='Index', shape=shape, dtype=self.index_type
#             )
#             indices.append(index)
#         indices = tuple(indices)
#         value = paddle.static.data(
#             name='Value', shape=self.value_shape, dtype=self.x_type
#         )

#         out = paddle.index_put(x, indices, value)

#         if device == "cpu":
#             place = paddle.CPUPlace()
#         elif device == "gpu":
#             place = paddle.CUDAPlace(0)
#         else:
#             raise TypeError(
#                 "paddle.index_put api only support cpu and gpu device now."
#             )

#         exe = paddle.static.Executor(place)
#         exe.run(paddle.static.default_startup_program())

#         res = exe.run(
#             paddle.static.default_main_program(),
#             feed={
#                 "X": self.x_np,
#                 "Indices": self.indices_np,
#                 "Value": self.value_np,
#                 "Accumulate": self.accumulate,
#             },
#             fetch_list=[out.name],
#             return_numpy=False,
#         )
#         return res

#     def test_static(self):
#         paddle.enable_static()
#         for device in self.place:
#             with paddle.static.program_guard(Program()):
#                 out = self.run_static(device)
#             ref_out = compute_index_put_ref(
#                 self.x_np, self.value_np, self.indices_np, self.accumulate
#             )
#             np.testing.assert_allclose(
#                 ref_out, np.array(out[0]), rtol=self.rtol, atol=self.atol
#             )

#     def test_dynamic(self):
#         paddle.disable_static()
#         for device in self.place:
#             self.run_imperative(device)


# # class TestIndexAddAPIMoreType(TestIndexAddAPI):
# #     def setType(self):
# #         self.x_type = np.float64
# #         self.index_type = np.int64

# # Bool Index Test
# class TestIndexAddAPICase2(TestIndexPutAPI):
#     def setType(self):
#         self.x_type = np.float32
#         self.index_type = np.bool

#     def config(self):
#         self.x_shape = (20, 24)
#         self.index_type = np.bool
#         self.value_shape = (240,)
#         self.indices_shape = (20, 24)


# # Int Index Test
# class TestIndexAddAPICase3(TestIndexPutAPI):
#     def setType(self):
#         self.x_type = np.float32
#         self.index_type = np.int32

#     def config(self):
#         self.x_shape = (20, 24, 42)
#         self.index_type = np.int32
#         self.value_shape = (8, 8, 42)
#         self.indices_shape = ((8, 8), (1, 8))


# # Int64 Index Test
# class TestIndexAddAPICase4(TestIndexPutAPI):
#     def setType(self):
#         self.x_type = np.float32
#         self.index_type = np.int64

#     def config(self):
#         self.x_shape = (20, 24, 42)
#         self.index_type = np.int64
#         self.value_shape = (8, 8, 42)
#         self.indices_shape = ((8, 8), (1, 8))


# # accumulate is True and index_type is Bool
# class TestIndexAddAPICase5(TestIndexPutAPI):
#     def setType(self):
#         self.x_type = np.float32
#         self.index_type = np.bool

#     def config(self):
#         self.x_shape = (20, 24)
#         self.index_type = np.bool
#         self.value_shape = (240,)
#         self.indices_shape = (20, 24)
#         self.accumulate = True


# # accumulate is True and index_type is int64
# class TestIndexAddAPICase6(TestIndexPutAPI):
#     def setType(self):
#         self.x_type = np.float32
#         self.index_type = np.int64

#     def config(self):
#         self.x_shape = (20, 24, 42)
#         self.index_type = np.int64
#         self.value_shape = (8, 8, 8)
#         self.indices_shape = ((8, 8), (1, 8), (1,))
#         self.accumulate = True


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
