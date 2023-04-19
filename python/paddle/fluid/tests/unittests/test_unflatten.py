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
    if len(shape) == 0:
        raise ValueError("The input for shape cannot be empty.")
    if isinstance(shape, list) or isinstance(shape, tuple):
        if np.min(shape) < -1:
            raise ValueError(f"invalid shape dimension {np.min(shape)}.")
        if shape.count(-1) > 1:
            raise ValueError("The shape can contain only one -1.")
        elif shape.count(-1) == 1:
            list(shape)[shape.index(-1)] = x.shape[axis] / abs(np.prod(shape))
        else:
            sizes = np.prod(shape)
            if sizes != x.shape[axis]:
                raise ValueError(
                    "The product of the elements in shape{} is not equal to {}.".format(
                        shape, x.shape[axis]
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
            out = self.paddle_api(x=x, shape=self.shape, axis=self.axis)
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

                exe = paddle.static.Executor(place)
                out = self.paddle_api(x=x, shape=self.shape, axis=self.axis)
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


# # x 的数据类型
# 注释部分为 reshape 静态图下不支持检查 x 的数据类型，暂时先注释掉
# class TestUnflattenXUint16(TestUnflattenAPI):
#     def set_args(self):
#         self.x = np.random.random((4, 6, 16)).astype('uint8')
#         self.shape = (2, 2)
#         self.axis = 0


# class TestUnflattenXInt8(TestUnflattenAPI):
#     def set_args(self):
#         self.x = np.random.rand(4, 6, 16).astype('int8')
#         self.shape = (2, 2)
#         self.axis = 0


class TestUnflattenXInt16(TestUnflattenAPI):
    def set_args(self):
        self.x = np.random.rand(4, 6, 16).astype('int16')
        self.shape = (2, 2)
        self.axis = 0


class TestUnflattenXInt32(TestUnflattenAPI):
    def set_args(self):
        self.x = np.random.rand(4, 6, 16).astype('int32')
        self.shape = (2, 2)
        self.axis = 0


class TestUnflattenXInt64(TestUnflattenAPI):
    def set_args(self):
        self.x = np.random.rand(4, 6, 16).astype('int64')
        self.shape = (2, 2)
        self.axis = 0


class TestUnflattenXFloat16(TestUnflattenAPI):
    def set_args(self):
        self.x = np.random.rand(4, 6, 16).astype('float16')
        self.shape = (2, 2)
        self.axis = 0


class TestUnflattenXFloat32(TestUnflattenAPI):
    def set_args(self):
        self.x = np.random.rand(4, 6, 16).astype('float32')
        self.shape = (2, 2)
        self.axis = 0


class TestUnflattenXFloat64(TestUnflattenAPI):
    def set_args(self):
        self.x = np.random.rand(4, 6, 16).astype('float64')
        self.shape = (2, 2)
        self.axis = 0


# class TestUnflattenXBFloat16(TestUnflattenAPI):
#     def set_args(self):
#         self.x = np.random.rand(4, 6, 16).astype('bfloat16') # numpy 不支持 bfloat16
#         self.shape = (2, 2)
#         self.axis = 0


# class TestUnflattenXComplex64(TestUnflattenAPI):
#     def set_args(self):
#         self.x = np.random.rand(4, 6, 16).astype('complex64')
#         self.shape = (2, 2)
#         self.axis = 0


# class TestUnflattenXComplex128(TestUnflattenAPI):
#     def set_args(self):
#         self.x = np.random.rand(4, 6, 16).astype('complex128')
#         self.shape = (2, 2)
#         self.axis = 0


class TestUnflattenXbool(TestUnflattenAPI):
    def set_args(self):
        self.x = np.random.rand(4, 6, 16).astype('bool')
        self.shape = (2, 2)
        self.axis = 0


# shape 的数据类型和边界情况
class TestUnflattenShapeLIST1(TestUnflattenAPI):
    def set_args(self):
        self.x = np.random.rand(4, 6, 16).astype('float32')
        self.shape = [2, 2]
        self.axis = 0


class TestUnflattenShapeLIST2(TestUnflattenAPI):
    def set_args(self):
        self.x = np.random.rand(4, 6, 16).astype('float32')
        self.shape = [-1, 2]
        self.axis = -1


class TestUnflattenShapeLIST3(TestUnflattenAPI):
    def set_args(self):
        self.x = np.random.rand(4, 6, 16).astype('float32')
        self.shape = [
            -1,
        ]
        self.axis = 0


class TestUnflattenTupleShape1(TestUnflattenAPI):
    def set_args(self):
        self.x = np.random.rand(4, 6, 16).astype('float32')
        self.shape = (2, 2)
        self.axis = 0


class TestUnflattenTupleShape2(TestUnflattenAPI):
    def set_args(self):
        self.x = np.random.rand(4, 6, 16).astype('float32')
        self.shape = (-1, 2)
        self.axis = 0


class TestUnflattenTupleShape3(TestUnflattenAPI):
    def set_args(self):
        self.x = np.random.rand(4, 6, 16).astype('float32')
        self.shape = (-1,)
        self.axis = 0


# axis 的取值
class TestUnflattenAxis1(TestUnflattenAPI):
    def set_args(self):
        self.x = np.random.rand(4, 6, 16).astype('float32')
        self.shape = (2, 3)
        self.axis = 1


class TestUnflattenAxis2(TestUnflattenAPI):
    def set_args(self):
        self.x = np.random.rand(4, 6, 16).astype('float32')
        self.shape = (2, 8)
        self.axis = -1


# Test error
# 代补充


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
