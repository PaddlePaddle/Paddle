# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np

import paddle


def numpy_unflatten(x, shape, axis):
    if isinstance(shape, list) or isinstance(shape, tuple):
        if len(shape) == 0:
            raise ValueError("The input for shape cannot be empty.")
        if isinstance(shape, list) or isinstance(shape, tuple):
            if np.min(shape) < -1:
                raise ValueError(f"invalid shape dimension {np.min(shape)}.")
            if shape.count(-1) > 1:
                raise ValueError("The shape can contain only one -1.")
            elif shape.count(-1) == 1:
                list(shape)[shape.index(-1)] = x.shape[axis] / abs(
                    np.prod(shape)
                )
            else:
                sizes = np.prod(shape)
                if sizes != x.shape[axis]:
                    raise ValueError(
                        "The product of the elements in shape{} is not equal to {}.".format(
                            shape, x.shape[axis]
                        )
                    )
    else:
        raise TypeError(
            "The data type of x should be one of ['List', 'Tuple', 'Tensor'], but got {}".format(
                type(shape)
            )
        )
    length = len(x.shape)
    if axis < 0:
        axis = axis + length
    new_shape = x.shape[:axis] + tuple(shape) + x.shape[axis + 1 :]
    x = x.reshape(new_shape)
    return x


class TestUnflattenAPI(unittest.TestCase):
    def set_args(self):
        self.x = np.random.rand(4, 6, 16)
        self.shape = (2, 2)
        self.axis = 0
        self.shape_is_tensor = False

    def get_output(self):
        self.output = self.ref_api(self.x, self.shape, self.axis)

    def set_api(self):
        self.ref_api = numpy_unflatten
        self.paddle_api = paddle.unflatten

    def setUp(self):
        self.set_api()
        self.set_args()
        self.get_output()
        self.places = [paddle.CPUPlace()]
        if paddle.device.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def func_dygraph(self):
        for place in self.places:
            paddle.disable_static()
            x = paddle.to_tensor(self.x, place=place)
            if self.shape_is_tensor:
                shape = paddle.to_tensor(self.shape, 'int32')
            else:
                shape = self.shape
            out = self.paddle_api(x=x, shape=shape, axis=self.axis)
            np.testing.assert_allclose(out, self.output, rtol=1e-05)

    def test_dygraph(self):
        self.setUp()
        self.func_dygraph()

    def test_static(self):
        paddle.enable_static()
        places = [paddle.CPUPlace()]
        if paddle.device.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))
        for place in places:
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x = paddle.static.data(
                    name="x", shape=self.x.shape, dtype=self.x.dtype
                )
                if self.shape_is_tensor:
                    shape = np.array(self.shape)
                    shape = paddle.static.data(
                        name='shape', shape=shape.shape, dtype=shape.dtype
                    )
                else:
                    shape = self.shape
                exe = paddle.static.Executor(place)
                out = self.paddle_api(x=x, shape=shape, axis=self.axis)
                fetches = exe.run(
                    paddle.static.default_main_program(),
                    feed={
                        "x": self.x,
                        "shape": self.shape,
                        "axis": self.axis,
                    },
                    fetch_list=[out],
                )

                np.testing.assert_allclose(fetches[0], self.output, rtol=1e-05)


class TestUnflattenXInt16(TestUnflattenAPI):
    def set_args(self):
        self.x = np.random.rand(4, 6, 16).astype('int16')
        self.shape = (2, 2)
        self.axis = 0
        self.shape_is_tensor = False


class TestUnflattenXInt32(TestUnflattenAPI):
    def set_args(self):
        self.x = np.random.rand(4, 6, 16).astype('int32')
        self.shape = (2, 2)
        self.axis = 0
        self.shape_is_tensor = False


class TestUnflattenXInt64(TestUnflattenAPI):
    def set_args(self):
        self.x = np.random.rand(4, 6, 16).astype('int64')
        self.shape = (2, 2)
        self.axis = 0
        self.shape_is_tensor = False


class TestUnflattenXFloat16(TestUnflattenAPI):
    def set_args(self):
        self.x = np.random.rand(4, 6, 16).astype('float16')
        self.shape = (2, 2)
        self.axis = 0
        self.shape_is_tensor = False


class TestUnflattenXFloat32(TestUnflattenAPI):
    def set_args(self):
        self.x = np.random.rand(4, 6, 16).astype('float32')
        self.shape = (2, 2)
        self.axis = 0
        self.shape_is_tensor = False


class TestUnflattenXFloat64(TestUnflattenAPI):
    def set_args(self):
        self.x = np.random.rand(4, 6, 16).astype('float64')
        self.shape = (2, 2)
        self.axis = 0
        self.shape_is_tensor = False


class TestUnflattenXbool(TestUnflattenAPI):
    def set_args(self):
        self.x = np.random.rand(4, 6, 16).astype('bool')
        self.shape = (2, 2)
        self.axis = 0
        self.shape_is_tensor = False


# shape 的数据类型和边界情况
class TestUnflattenShapeLIST1(TestUnflattenAPI):
    def set_args(self):
        self.x = np.random.rand(4, 6, 16).astype('float32')
        self.shape = [2, 2]
        self.axis = 0
        self.shape_is_tensor = False


class TestUnflattenShapeLIST2(TestUnflattenAPI):
    def set_args(self):
        self.x = np.random.rand(4, 6, 16).astype('float32')
        self.shape = [-1, 2]
        self.axis = -1
        self.shape_is_tensor = False


class TestUnflattenShapeLIST3(TestUnflattenAPI):
    def set_args(self):
        self.x = np.random.rand(4, 6, 16).astype('float32')
        self.shape = [
            -1,
        ]
        self.axis = 0
        self.shape_is_tensor = False


class TestUnflattenTupleShape1(TestUnflattenAPI):
    def set_args(self):
        self.x = np.random.rand(4, 6, 16).astype('float32')
        self.shape = (2, 2)
        self.axis = 0
        self.shape_is_tensor = False


class TestUnflattenTupleShape2(TestUnflattenAPI):
    def set_args(self):
        self.x = np.random.rand(4, 6, 16).astype('float32')
        self.shape = (-1, 2)
        self.axis = 0
        self.shape_is_tensor = False


class TestUnflattenTupleShape3(TestUnflattenAPI):
    def set_args(self):
        self.x = np.random.rand(4, 6, 16).astype('float32')
        self.shape = (-1,)
        self.axis = 0
        self.shape_is_tensor = False


# paddle.prod 不支持 int16
# class TestUnflattenShapeTensorInt16(TestUnflattenAPI):
#     def set_args(self):
#         self.x = np.random.rand(4, 6, 16).astype('float32')
#         self.shape = np.array((2, 8)).astype('int16')
#         self.axis = -1
#         self.shape_is_tensor = True


class TestUnflattenShapeTensorInt32(TestUnflattenAPI):
    def set_args(self):
        self.x = np.random.rand(4, 6, 16).astype('float32')
        self.shape = tuple(np.array((-1, 4)).astype('int32'))
        self.axis = 0
        self.shape_is_tensor = True


# class TestUnflattenShapeTensorInt64(TestUnflattenAPI):
#     def set_args(self):
#         self.x = np.random.rand(4, 6, 16).astype('float32')
#         self.shape = list(np.array([-1, 2]).astype('int64'))
#         self.axis = 1
#         self.shape_is_tensor = True


# axis 的取值
class TestUnflattenAxis1(TestUnflattenAPI):
    def set_args(self):
        self.x = np.random.rand(4, 6, 16).astype('float32')
        self.shape = (2, 3)
        self.axis = 1
        self.shape_is_tensor = False


class TestUnflattenAxis2(TestUnflattenAPI):
    def set_args(self):
        self.x = np.random.rand(4, 6, 16).astype('float32')
        self.shape = (2, 8)
        self.axis = -1
        self.shape_is_tensor = False


# Test for the types supported by dynamic graphs but not supported by static graphs
class TestUnflattenOnlyDynamic(unittest.TestCase):
    def set_args(self):
        self.x = np.random.rand(4, 6, 16)
        self.shape = (2, 2)
        self.axis = 0
        self.shape_is_tensor = False

    def get_output(self):
        self.output = self.ref_api(self.x, self.shape, self.axis)

    def set_api(self):
        self.ref_api = numpy_unflatten
        self.paddle_api = paddle.unflatten

    def setUp(self):
        self.set_api()
        self.set_args()
        self.get_output()
        self.places = [paddle.CPUPlace()]
        if paddle.device.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def func_dygraph(self):
        for place in self.places:
            paddle.disable_static()
            x = paddle.to_tensor(self.x, place=place)
            if self.shape_is_tensor:
                shape = paddle.to_tensor(self.shape)
            else:
                shape = self.shape
            out = self.paddle_api(x=x, shape=shape, axis=self.axis)
            np.testing.assert_allclose(out, self.output, rtol=1e-05)

    def test_dygraph(self):
        self.setUp()
        self.func_dygraph()


class TestUnflattenXUint16(TestUnflattenOnlyDynamic):
    def set_args(self):
        self.x = np.random.random((4, 6, 16)).astype('uint8')
        self.shape = (2, 2)
        self.axis = 0
        self.shape_is_tensor = False


class TestUnflattenXInt8(TestUnflattenOnlyDynamic):
    def set_args(self):
        self.x = np.random.rand(4, 6, 16).astype('int8')
        self.shape = (2, 2)
        self.axis = 0
        self.shape_is_tensor = False


class TestUnflattenXComplex64(TestUnflattenOnlyDynamic):
    def set_args(self):
        self.x = np.random.rand(4, 6, 16).astype('complex64')
        self.shape = (2, 2)
        self.axis = 0
        self.shape_is_tensor = False


class TestUnflattenXComplex128(TestUnflattenOnlyDynamic):
    def set_args(self):
        self.x = np.random.rand(4, 6, 16).astype('complex128')
        self.shape = (2, 2)
        self.axis = 0
        self.shape_is_tensor = False


# Only for bfloat16 test
class TestUnflattenXBFloat16(unittest.TestCase):
    def set_args(self):
        self.x = np.random.rand(4, 6, 16)
        self.shape = (2, 8)
        self.axis = -1

    def func_dygraph(self):
        for place in self.places:
            paddle.disable_static()
            x = paddle.to_tensor(self.x, dtype='bfloat16', place=place)
            out = self.paddle_api(x=x, shape=self.shape, axis=self.axis)

    def set_api(self):
        self.paddle_api = paddle.unflatten

    def setUp(self):
        self.set_api()
        self.set_args()
        self.places = [paddle.CPUPlace()]
        if paddle.device.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def test_dygraph(self):
        self.setUp()
        self.func_dygraph()


# Test error
class TestUnflattenError(unittest.TestCase):
    def set_api(self):
        self.paddle_api = paddle.unflatten

    def test_errors(self):
        self.set_api()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):

            def test_int_shape_input():
                x = paddle.static.data(
                    name='x',
                    shape=[4, 4],
                    dtype="float32",
                )
                shape = 1
                axis = -1
                self.paddle_api(x, shape, axis)

            self.assertRaises(TypeError, test_int_shape_input)

            def test_shape_with_int64():
                x = paddle.static.data(
                    name='x',
                    shape=[4, 4],
                    dtype="float32",
                )
                shape = list(np.array([2, 2]).astype('int64'))
                axis = -1
                self.paddle_api(x, shape, axis)

            self.assertRaises(TypeError, test_shape_with_int64)


#             # test shape is empty
#             def test_list_shape_empty():
#                 x = paddle.static.data(
#                     name='x',
#                     shape=[4, 4],
#                     dtype="float32",
#                 )
#                 shape = []
#                 axis = -1
#                 self.paddle_api(x, shape, axis)

#             self.assertRaises(ValueError, test_list_shape_empty)

#             def test_tuple_shape_empty():
#                 x = paddle.static.data(
#                     name='x',
#                     shape=[4, 4],
#                     dtype="float32",
#                 )
#                 shape = ()
#                 axis = -1
#                 self.paddle_api(x, shape, axis)

#             self.assertRaises(ValueError, test_tuple_shape_empty)

#             def test_tensor_shape_empty():
#                 x = paddle.static.data(
#                     name='x',
#                     shape=[4, 4],
#                     dtype="float32",
#                 )
#                 shape = paddle.static.data(
#                     name='shape',
#                     shape=[],
#                     dtype="int32",
#                 )
#                 axis = -1
#                 self.paddle_api(x, shape, axis)

#             self.assertRaises(ValueError, test_tensor_shape_empty)

#             # test invalid_shape_dimension
#             def test_list_invalid_shape_dimension():
#                 x = paddle.static.data(
#                     name='x',
#                     shape=[4, 4],
#                     dtype="float32",
#                 )
#                 shape = [-2, 1]
#                 axis = -1
#                 self.paddle_api(x, shape, axis)

#             self.assertRaises(ValueError, test_list_invalid_shape_dimension)

#             def test_tuple_invalid_shape_dimension():
#                 x = paddle.static.data(
#                     name='x',
#                     shape=[4, 4],
#                     dtype="float32",
#                 )
#                 shape = (-2, 1)
#                 axis = -1
#                 self.paddle_api(x, shape, axis)

#             self.assertRaises(ValueError, test_tuple_invalid_shape_dimension)

#             def test_tensor_invalid_shape_dimension():
#                 x = paddle.static.data(
#                     name='x',
#                     shape=[4, 4],
#                     dtype="float32",
#                 )
#                 shape = paddle.static.data(
#                     name='shape',
#                     shape=[],
#                     dtype="int32",
#                 )
#                 axis = -1
#                 self.paddle_api(x, shape, axis)

#             self.assertRaises(ValueError, test_tensor_invalid_shape_dimension)

#             # test shape contains multiple -1
#             def test_list_shape_contain_multiple_negative_ones():
#                 x = paddle.static.data(
#                     name='x',
#                     shape=[4, 4],
#                     dtype="float32",
#                 )
#                 shape = [-1, -1, 1]
#                 axis = -1
#                 self.paddle_api(x, shape, axis)

#             self.assertRaises(
#                 ValueError, test_list_shape_contain_multiple_negative_ones
#             )

#             def test_tuple_shape_contain_multiple_negative_ones():
#                 x = paddle.static.data(
#                     name='x',
#                     shape=[4, 4],
#                     dtype="float32",
#                 )
#                 shape = (-1, -1, 1)
#                 axis = -1
#                 self.paddle_api(x, shape, axis)

#             self.assertRaises(
#                 ValueError, test_tuple_shape_contain_multiple_negative_ones
#             )

#             def test_tensor_shape_contain_multiple_negative_ones():
#                 x = paddle.static.data(
#                     name='x',
#                     shape=[4, 4],
#                     dtype="float32",
#                 )
#                 shape = paddle.static.data(
#                     name='shape',
#                     shape=[-1, -1],
#                     dtype="int32",
#                 )
#                 axis = -1
#                 self.paddle_api(x, shape, axis)

#             self.assertRaises(
#                 ValueError, test_tensor_shape_contain_multiple_negative_ones
#             )

#             # The product of the elements in shape is not equal to the target dimension.
#             def test_list_shape_not_dim():
#                 x = paddle.static.data(
#                     name='x',
#                     shape=[4, 4],
#                     dtype="float32",
#                 )
#                 shape = [2, 4]
#                 axis = -1
#                 self.paddle_api(x, shape, axis)

#             self.assertRaises(ValueError, test_list_shape_not_dim)

#             def test_tuple_shape_not_dim():
#                 x = paddle.static.data(
#                     name='x',
#                     shape=[4, 4],
#                     dtype="float32",
#                 )
#                 shape = (2, 4)
#                 axis = -1
#                 self.paddle_api(x, shape, axis)

#             self.assertRaises(ValueError, test_tuple_shape_not_dim)

#             def test_tensor_shape_not_dim():
#                 x = paddle.static.data(
#                     name='x',
#                     shape=[4, 4],
#                     dtype="float32",
#                 )
#                 shape = paddle.static.data(
#                     name='shape',
#                     shape=[2, 4],
#                     dtype="int32",
#                 )
#                 axis = -1
#                 self.paddle_api(x, shape, axis)

#             self.assertRaises(ValueError, test_tensor_shape_not_dim)

#             # test type of unexpected input
#             def test_string_shape_input():
#                 x = paddle.static.data(
#                     name='x',
#                     shape=[4, 4],
#                     dtype="float32",
#                 )
#                 shape = ['1', '2', '3']
#                 axis = -1
#                 self.paddle_api(x, shape, axis)

#             self.assertRaises(TypeError, test_string_shape_input)

#             def test_int_shape_input():
#                 x = paddle.static.data(
#                     name='x',
#                     shape=[4, 4],
#                     dtype="float32",
#                 )
#                 shape = 1
#                 axis = -1
#                 self.paddle_api(x, shape, axis)

#             self.assertRaises(TypeError, test_int_shape_input)
class TestLayer(unittest.TestCase):
    def set_args(self):
        self.x = np.random.randn(3, 4, 4, 5).astype('float32')
        self.shape = [2, 2]
        self.axis = 1

    def setUp(self):
        self.set_args()
        self.places = [paddle.CPUPlace()]
        if paddle.device.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def test_layer(self):
        paddle.enable_static()
        places = [paddle.CPUPlace()]
        if paddle.device.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))
        for place in places:
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x = paddle.static.data(
                    name="x", shape=self.x.shape, dtype=self.x.dtype
                )
                exe = paddle.static.Executor(place)
                unflatten = paddle.nn.Unflatten(self.shape, self.axis)
                out = unflatten(x)
                static_ret = exe.run(
                    paddle.static.default_main_program(),
                    feed={
                        "x": self.x,
                        "shape": self.shape,
                        "axis": self.axis,
                    },
                    fetch_list=[out],
                )[0]
        for place in self.places:
            paddle.disable_static()
            x = paddle.to_tensor(self.x, dtype='float32', place=place)
            unflatten = paddle.nn.Unflatten(self.shape, self.axis)
            dy_ret_value = unflatten(self.x)
        np.testing.assert_array_equal(static_ret, dy_ret_value)


class TestLayerName(unittest.TestCase):
    def test_name(self):
        self.x = np.random.randn(3, 4, 4, 5).astype('float32')
        self.shape = [2, 2]
        self.axis = 1
        self.name = 'unflatten'
        unflatten = paddle.nn.Unflatten(self.shape, self.axis, self.name)
        _name = unflatten.extra_repr()


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
