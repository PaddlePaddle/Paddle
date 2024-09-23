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

import os
import unittest

import numpy as np

import paddle


def numpy_unflatten(x, axis, shape):
    if isinstance(shape, (list, tuple)):
        if len(shape) == 0:
            raise ValueError("The input for shape cannot be empty.")
        if isinstance(shape, (list, tuple)):
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
                        f"The product of the elements in shape{shape} is not equal to {x.shape[axis]}."
                    )
    else:
        raise TypeError(
            f"The data type of x should be one of ['List', 'Tuple', 'Tensor'], but got {type(shape)}"
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
        self.axis = 0
        self.shape = (2, 2)
        self.shape_is_tensor = False

    def get_output(self):
        self.output = self.ref_api(self.x, self.axis, self.shape)

    def set_api(self):
        self.ref_api = numpy_unflatten
        self.paddle_api = paddle.unflatten

    def setUp(self):
        self.set_api()
        self.set_args()
        self.get_output()
        self.places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not paddle.device.is_compiled_with_cuda()
        ):
            self.places.append(paddle.CPUPlace())
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
            out = self.paddle_api(x=x, axis=self.axis, shape=shape)
            np.testing.assert_allclose(out, self.output, rtol=1e-05)

    def test_dygraph(self):
        self.setUp()
        self.func_dygraph()

    def test_static(self):
        paddle.enable_static()
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not paddle.device.is_compiled_with_cuda()
        ):
            places.append(paddle.CPUPlace())
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
                out = self.paddle_api(x=x, axis=self.axis, shape=shape)
                fetches = exe.run(
                    paddle.static.default_main_program(),
                    feed={
                        "x": self.x,
                        "axis": self.axis,
                        "shape": self.shape,
                    },
                    fetch_list=[out],
                )

                np.testing.assert_allclose(fetches[0], self.output, rtol=1e-05)


# check the data type of the input x
class TestUnflattenInputInt16(TestUnflattenAPI):
    def set_args(self):
        self.x = np.random.rand(4, 6, 16).astype('int16')
        self.axis = 0
        self.shape = (2, 2)
        self.shape_is_tensor = False


class TestUnflattenInputInt32(TestUnflattenAPI):
    def set_args(self):
        self.x = np.random.rand(4, 6, 16).astype('int32')
        self.axis = 0
        self.shape = (2, 2)
        self.shape_is_tensor = False


class TestUnflattenInputInt64(TestUnflattenAPI):
    def set_args(self):
        self.x = np.random.rand(4, 6, 16).astype('int64')
        self.axis = 0
        self.shape = (2, 2)
        self.shape_is_tensor = False


class TestUnflattenInputFloat16(TestUnflattenAPI):
    def set_args(self):
        self.x = np.random.rand(4, 6, 16).astype('float16')
        self.axis = 0
        self.shape = (2, 2)
        self.shape_is_tensor = False


class TestUnflattenInputFloat32(TestUnflattenAPI):
    def set_args(self):
        self.x = np.random.rand(4, 6, 16).astype('float32')
        self.axis = 0
        self.shape = (2, 2)
        self.shape_is_tensor = False


class TestUnflattenInputFloat64(TestUnflattenAPI):
    def set_args(self):
        self.x = np.random.rand(4, 6, 16).astype('float64')
        self.axis = 0
        self.shape = (2, 2)
        self.shape_is_tensor = False


class TestUnflattenInputbool(TestUnflattenAPI):
    def set_args(self):
        self.x = np.random.rand(4, 6, 16).astype('bool')
        self.axis = 0
        self.shape = (2, 2)
        self.shape_is_tensor = False


# check the data type and edge cases of shape
class TestUnflattenShapeList1(TestUnflattenAPI):
    def set_args(self):
        self.x = np.random.rand(4, 6, 16).astype('float32')
        self.axis = 0
        self.shape = [2, 2]
        self.shape_is_tensor = False


class TestUnflattenShapeList2(TestUnflattenAPI):
    def set_args(self):
        self.x = np.random.rand(4, 6, 16).astype('float32')
        self.axis = -1
        self.shape = [-1, 2]
        self.shape_is_tensor = False


class TestUnflattenShapeList3(TestUnflattenAPI):
    def set_args(self):
        self.x = np.random.rand(4, 6, 16).astype('float32')
        self.axis = 0
        self.shape = [-1]
        self.shape_is_tensor = False


class TestUnflattenTupleShape1(TestUnflattenAPI):
    def set_args(self):
        self.x = np.random.rand(4, 6, 16).astype('float32')
        self.axis = 0
        self.shape = (2, 2)
        self.shape_is_tensor = False


class TestUnflattenTupleShape2(TestUnflattenAPI):
    def set_args(self):
        self.x = np.random.rand(4, 6, 16).astype('float32')
        self.axis = 0
        self.shape = (-1, 2)
        self.shape_is_tensor = False


class TestUnflattenTupleShape3(TestUnflattenAPI):
    def set_args(self):
        self.x = np.random.rand(4, 6, 16).astype('float32')
        self.axis = 0
        self.shape = (-1,)
        self.shape_is_tensor = False


class TestUnflattenShapeTensorInt32(TestUnflattenAPI):
    def set_args(self):
        self.x = np.random.rand(4, 6, 16).astype('float32')
        self.axis = 0
        self.shape = tuple(np.array((-1, 4)).astype('int32'))
        self.shape_is_tensor = True


# check the value of axis
class TestUnflattenAxis1(TestUnflattenAPI):
    def set_args(self):
        self.x = np.random.rand(4, 6, 16).astype('float32')
        self.axis = 1
        self.shape = (2, 3)
        self.shape_is_tensor = False


class TestUnflattenAxis2(TestUnflattenAPI):
    def set_args(self):
        self.x = np.random.rand(4, 6, 16).astype('float32')
        self.axis = -1
        self.shape = (2, 8)
        self.shape_is_tensor = False


class TestLayer(unittest.TestCase):
    def set_args(self):
        self.x = np.random.randn(3, 4, 4, 5).astype('float32')
        self.axis = 1
        self.shape = [2, 2]

    def setUp(self):
        self.set_args()
        self.places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not paddle.device.is_compiled_with_cuda()
        ):
            self.places.append(paddle.CPUPlace())
        if paddle.device.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def test_layer(self):
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not paddle.device.is_compiled_with_cuda()
        ):
            places.append(paddle.CPUPlace())
        if paddle.device.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))
        for place in places:
            paddle.disable_static()
            x = paddle.to_tensor(self.x, dtype='float32', place=place)
            unflatten = paddle.nn.Unflatten(self.axis, self.shape)
            dy_ret_value = unflatten(x)

            paddle.enable_static()

            def test_static_or_pir_mode():
                with paddle.static.program_guard(
                    paddle.static.Program(), paddle.static.Program()
                ):
                    x = paddle.static.data(
                        name="x", dtype=self.x.dtype, shape=self.x.shape
                    )
                    exe = paddle.static.Executor(place)
                    unflatten = paddle.nn.Unflatten(self.axis, self.shape)
                    out = unflatten(x)
                    static_ret = exe.run(
                        paddle.static.default_main_program(),
                        feed={
                            "x": self.x,
                            "axis": self.axis,
                            "shape": self.shape,
                        },
                        fetch_list=[out],
                    )[0]

                np.testing.assert_array_equal(static_ret, dy_ret_value)

            test_static_or_pir_mode()


class TestLayerName(unittest.TestCase):

    def test_name(self):
        self.x = np.random.randn(3, 4, 4, 5).astype('float32')
        self.axis = 1
        self.shape = [2, 2]
        self.name = 'unflatten'
        unflatten = paddle.nn.Unflatten(self.axis, self.shape, self.name)
        _name = unflatten.extra_repr()


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
