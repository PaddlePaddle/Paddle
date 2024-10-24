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

import unittest

import numpy
import numpy as np
import op_test

import paddle
from paddle import base
from paddle.base import core
from paddle.framework import in_pir_mode


def create_test_class(op_type, typename, callback, check_pir=False):
    class Cls(op_test.OpTest):
        def setUp(self):
            a = numpy.random.random(size=(10, 7)).astype(typename)
            b = numpy.random.random(size=(10, 7)).astype(typename)
            c = callback(a, b)
            self.python_api = eval("paddle." + op_type)
            self.inputs = {'X': a, 'Y': b}
            self.outputs = {'Out': c}
            self.op_type = op_type

        def test_output(self):
            self.check_output(check_cinn=True, check_pir=check_pir)

        def test_int16_support(self):
            paddle.enable_static()
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                a = paddle.static.data(name='a', shape=[-1, 2], dtype='int16')
                b = paddle.static.data(name='b', shape=[-1, 2], dtype='int16')
                op = eval(f"paddle.{self.op_type}")

                try:
                    result = op(x=a, y=b)
                except TypeError:
                    self.fail("TypeError should not be raised for int16 inputs")

    cls_name = f"{op_type}_{typename}"
    Cls.__name__ = cls_name
    globals()[cls_name] = Cls


for _type_name in {
    'float32',
    'float64',
    'uint8',
    'int8',
    'int16',
    'int32',
    'int64',
    'float16',
}:
    if _type_name == 'float64' and core.is_compiled_with_rocm():
        _type_name = 'float32'
    if _type_name == 'float16' and (not core.is_compiled_with_cuda()):
        continue

    create_test_class('less_than', _type_name, lambda _a, _b: _a < _b, True)
    create_test_class('less_equal', _type_name, lambda _a, _b: _a <= _b, True)
    create_test_class('greater_than', _type_name, lambda _a, _b: _a > _b, True)
    create_test_class(
        'greater_equal', _type_name, lambda _a, _b: _a >= _b, True
    )
    create_test_class('equal', _type_name, lambda _a, _b: _a == _b, True)
    create_test_class('not_equal', _type_name, lambda _a, _b: _a != _b, True)


def create_paddle_case(op_type, callback):
    class PaddleCls(unittest.TestCase):
        def setUp(self):
            self.op_type = op_type
            self.input_x = np.array([1, 2, 3, 4]).astype(np.int64)
            self.input_y = np.array([1, 3, 2, 4]).astype(np.int64)
            self.real_result = callback(self.input_x, self.input_y)
            self.place = base.CPUPlace()
            if core.is_compiled_with_cuda():
                self.place = paddle.CUDAPlace(0)

        def test_api(self):
            paddle.enable_static()
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x = paddle.static.data(name='x', shape=[4], dtype='int64')
                y = paddle.static.data(name='y', shape=[4], dtype='int64')
                op = eval(f"paddle.{self.op_type}")
                out = op(x, y)
                exe = base.Executor(self.place)
                (res,) = exe.run(
                    feed={"x": self.input_x, "y": self.input_y},
                    fetch_list=[out],
                )
            self.assertEqual((res == self.real_result).all(), True)

        def test_api_float(self):
            if self.op_type == "equal":
                paddle.enable_static()
                with paddle.static.program_guard(
                    paddle.static.Program(), paddle.static.Program()
                ):
                    x = paddle.static.data(name='x', shape=[4], dtype='int64')
                    y = paddle.static.data(name='y', shape=[], dtype='int64')
                    op = eval(f"paddle.{self.op_type}")
                    out = op(x, y)
                    exe = base.Executor(self.place)
                    (res,) = exe.run(
                        feed={"x": self.input_x, "y": 1.0}, fetch_list=[out]
                    )
                self.real_result = np.array([1, 0, 0, 0]).astype(np.int64)
                self.assertEqual((res == self.real_result).all(), True)

        def test_dynamic_api(self):
            paddle.disable_static()
            x = paddle.to_tensor(self.input_x)
            y = paddle.to_tensor(self.input_y)
            op = eval(f"paddle.{self.op_type}")
            out = op(x, y)
            self.assertEqual((out.numpy() == self.real_result).all(), True)
            paddle.enable_static()

        def test_dynamic_api_int(self):
            if self.op_type == "equal":
                paddle.disable_static()
                x = paddle.to_tensor(self.input_x)
                op = eval(f"paddle.{self.op_type}")
                out = op(x, 1)
                self.real_result = np.array([1, 0, 0, 0]).astype(np.int64)
                self.assertEqual((out.numpy() == self.real_result).all(), True)
                paddle.enable_static()

        def test_dynamic_api_float(self):
            if self.op_type == "equal":
                paddle.disable_static()
                x = paddle.to_tensor(self.input_x)
                op = eval(f"paddle.{self.op_type}")
                out = op(x, 1.0)
                self.real_result = np.array([1, 0, 0, 0]).astype(np.int64)
                self.assertEqual((out.numpy() == self.real_result).all(), True)
                paddle.enable_static()

        def test_dynamic_api_float16(self):
            paddle.disable_static()
            x = paddle.to_tensor(self.input_x, dtype="float16")
            y = paddle.to_tensor(self.input_y, dtype="float16")
            op = eval(f"paddle.{self.op_type}")
            out = op(x, y)
            self.assertEqual((out.numpy() == self.real_result).all(), True)
            paddle.enable_static()

        def test_dynamic_api_inf_1(self):
            if self.op_type == "equal":
                paddle.disable_static()
                x1 = np.array([1, float('inf'), float('inf')]).astype(np.int64)
                x = paddle.to_tensor(x1)
                y1 = np.array([1, float('-inf'), float('inf')]).astype(np.int64)
                y = paddle.to_tensor(y1)
                op = eval(f"paddle.{self.op_type}")
                out = op(x, y)
                self.real_result = (x1 == y1).astype(np.int64)
                self.assertEqual(
                    (out.numpy().astype(np.int64) == self.real_result).all(),
                    True,
                )
                paddle.enable_static()

        def test_dynamic_api_inf_2(self):
            if self.op_type == "equal":
                paddle.disable_static()
                x1 = np.array([1, float('inf'), float('inf')]).astype(
                    np.float32
                )
                x = paddle.to_tensor(x1)
                y1 = np.array([1, float('-inf'), float('inf')]).astype(
                    np.float32
                )
                y = paddle.to_tensor(y1)
                op = eval(f"paddle.{self.op_type}")
                out = op(x, y)
                self.real_result = (x1 == y1).astype(np.int64)
                self.assertEqual(
                    (out.numpy().astype(np.int64) == self.real_result).all(),
                    True,
                )
                paddle.enable_static()

        def test_dynamic_api_inf_3(self):
            if self.op_type == "equal":
                paddle.disable_static()
                x1 = np.array([1, float('inf'), float('-inf')]).astype(
                    np.float32
                )
                x = paddle.to_tensor(x1)
                y1 = np.array([1, 2, 3]).astype(np.float32)
                y = paddle.to_tensor(y1)
                op = eval(f"paddle.{self.op_type}")
                out = op(x, y)
                self.real_result = (x1 == y1).astype(np.int64)
                self.assertEqual(
                    (out.numpy().astype(np.int64) == self.real_result).all(),
                    True,
                )
                paddle.enable_static()

        def test_dynamic_api_nan_1(self):
            if self.op_type == "equal":
                paddle.disable_static()
                x1 = np.array([1, float('nan'), float('nan')]).astype(np.int64)
                x = paddle.to_tensor(x1)
                y1 = np.array([1, float('-nan'), float('nan')]).astype(np.int64)
                y = paddle.to_tensor(y1)
                op = eval(f"paddle.{self.op_type}")
                out = op(x, y)
                self.real_result = (x1 == y1).astype(np.int64)
                self.assertEqual(
                    (out.numpy().astype(np.int64) == self.real_result).all(),
                    True,
                )
                paddle.enable_static()

        def test_dynamic_api_nan_2(self):
            if self.op_type == "equal":
                paddle.disable_static()
                x1 = np.array([1, float('nan'), float('nan')]).astype(
                    np.float32
                )
                x = paddle.to_tensor(x1)
                y1 = np.array([1, float('-nan'), float('nan')]).astype(
                    np.float32
                )
                y = paddle.to_tensor(y1)
                op = eval(f"paddle.{self.op_type}")
                out = op(x, y)
                self.real_result = (x1 == y1).astype(np.int64)
                self.assertEqual(
                    (out.numpy().astype(np.int64) == self.real_result).all(),
                    True,
                )
                paddle.enable_static()

        def test_dynamic_api_nan_3(self):
            if self.op_type == "equal":
                paddle.disable_static()
                x1 = np.array([1, float('-nan'), float('nan')]).astype(
                    np.float32
                )
                x = paddle.to_tensor(x1)
                y1 = np.array([1, 2, 1]).astype(np.float32)
                y = paddle.to_tensor(y1)
                op = eval(f"paddle.{self.op_type}")
                out = op(x, y)
                self.real_result = (x1 == y1).astype(np.int64)
                self.assertEqual(
                    (out.numpy().astype(np.int64) == self.real_result).all(),
                    True,
                )
                paddle.enable_static()

        def test_not_equal(self):
            if self.op_type == "not_equal":
                paddle.disable_static()
                x = paddle.to_tensor(
                    np.array([1.2e-8, 2, 2, 1]), dtype="float32"
                )
                y = paddle.to_tensor(
                    np.array([1.1e-8, 2, 2, 1]), dtype="float32"
                )
                op = eval(f"paddle.{self.op_type}")
                out = op(x, y)
                self.real_result = np.array([0, 0, 0, 0]).astype(np.int64)
                self.assertEqual((out.numpy() == self.real_result).all(), True)
                paddle.enable_static()

        def test_assert(self):
            def test_dynamic_api_string(self):
                if self.op_type == "equal":
                    paddle.disable_static()
                    x = paddle.to_tensor(self.input_x)
                    op = eval(f"paddle.{self.op_type}")
                    out = op(x, "1.0")
                    paddle.enable_static()

            self.assertRaises(TypeError, test_dynamic_api_string)

        def test_dynamic_api_bool(self):
            if self.op_type == "equal":
                paddle.disable_static()
                x = paddle.to_tensor(self.input_x)
                op = eval(f"paddle.{self.op_type}")
                out = op(x, True)
                self.real_result = np.array([1, 0, 0, 0]).astype(np.int64)
                self.assertEqual((out.numpy() == self.real_result).all(), True)
                paddle.enable_static()

        def test_broadcast_api_1(self):
            paddle.enable_static()
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x = paddle.static.data(
                    name='x', shape=[1, 2, 1, 3], dtype='int32'
                )
                y = paddle.static.data(name='y', shape=[1, 2, 3], dtype='int32')
                op = eval(f"paddle.{self.op_type}")
                out = op(x, y)
                exe = paddle.static.Executor(self.place)
                input_x = np.arange(1, 7).reshape((1, 2, 1, 3)).astype(np.int32)
                input_y = np.arange(0, 6).reshape((1, 2, 3)).astype(np.int32)
                real_result = callback(input_x, input_y)
                (res,) = exe.run(
                    feed={"x": input_x, "y": input_y}, fetch_list=[out]
                )
            self.assertEqual((res == real_result).all(), True)

        def test_broadcast_api_2(self):
            paddle.enable_static()
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x = paddle.static.data(name='x', shape=[1, 2, 3], dtype='int32')
                y = paddle.static.data(
                    name='y', shape=[1, 2, 1, 3], dtype='int32'
                )
                op = eval(f"paddle.{self.op_type}")
                out = op(x, y)
                exe = paddle.static.Executor(self.place)
                input_x = np.arange(0, 6).reshape((1, 2, 3)).astype(np.int32)
                input_y = np.arange(1, 7).reshape((1, 2, 1, 3)).astype(np.int32)
                real_result = callback(input_x, input_y)
                (res,) = exe.run(
                    feed={"x": input_x, "y": input_y}, fetch_list=[out]
                )
            self.assertEqual((res == real_result).all(), True)

        def test_broadcast_api_3(self):
            paddle.enable_static()
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x = paddle.static.data(name='x', shape=[5], dtype='int32')
                y = paddle.static.data(name='y', shape=[3, 1], dtype='int32')
                op = eval(f"paddle.{self.op_type}")
                out = op(x, y)
                exe = paddle.static.Executor(self.place)
                input_x = np.arange(0, 5).reshape(5).astype(np.int32)
                input_y = np.array([5, 3, 2]).reshape((3, 1)).astype(np.int32)
                real_result = callback(input_x, input_y)
                (res,) = exe.run(
                    feed={"x": input_x, "y": input_y}, fetch_list=[out]
                )
            self.assertEqual((res == real_result).all(), True)

        def test_zero_dim_api_1(self):
            paddle.enable_static()
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x = paddle.randint(-3, 3, shape=[], dtype='int32')
                y = paddle.randint(-3, 3, shape=[], dtype='int32')
                op = eval(f"paddle.{self.op_type}")
                out = op(x, y)
                exe = paddle.static.Executor(self.place)
                (
                    x_np,
                    y_np,
                    res,
                ) = exe.run(fetch_list=[x, y, out])
                real_result = callback(x_np, y_np)
            self.assertEqual((res == real_result).all(), True)

        def test_zero_dim_api_2(self):
            paddle.enable_static()
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x = paddle.randint(-3, 3, shape=[2, 3, 4], dtype='int32')
                y = paddle.randint(-3, 3, shape=[], dtype='int32')
                op = eval(f"paddle.{self.op_type}")
                out = op(x, y)
                exe = paddle.static.Executor(self.place)
                (
                    x_np,
                    y_np,
                    res,
                ) = exe.run(fetch_list=[x, y, out])
                real_result = callback(x_np, y_np)
            self.assertEqual((res == real_result).all(), True)

        def test_zero_dim_api_3(self):
            paddle.enable_static()
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x = paddle.randint(-3, 3, shape=[], dtype='int32')
                y = paddle.randint(-3, 3, shape=[2, 3, 4], dtype='int32')
                op = eval(f"paddle.{self.op_type}")
                out = op(x, y)
                exe = paddle.static.Executor(self.place)
                (
                    x_np,
                    y_np,
                    res,
                ) = exe.run(fetch_list=[x, y, out])
                real_result = callback(x_np, y_np)
            self.assertEqual((res == real_result).all(), True)

        def test_bool_api_4(self):
            paddle.enable_static()
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x = paddle.static.data(name='x', shape=[3, 1], dtype='bool')
                y = paddle.static.data(name='y', shape=[3, 1], dtype='bool')
                op = eval(f"paddle.{self.op_type}")
                out = op(x, y)
                exe = paddle.static.Executor(self.place)
                input_x = np.array([True, False, True]).astype(np.bool_)
                input_y = np.array([True, True, False]).astype(np.bool_)
                real_result = callback(input_x, input_y)
                (res,) = exe.run(
                    feed={"x": input_x, "y": input_y}, fetch_list=[out]
                )
            self.assertEqual((res == real_result).all(), True)

        def test_bool_broadcast_api_4(self):
            paddle.enable_static()
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x = paddle.static.data(name='x', shape=[3, 1], dtype='bool')
                y = paddle.static.data(name='y', shape=[1], dtype='bool')
                op = eval(f"paddle.{self.op_type}")
                out = op(x, y)
                exe = paddle.static.Executor(self.place)
                input_x = np.array([True, False, True]).astype(np.bool_)
                input_y = np.array([True]).astype(np.bool_)
                real_result = callback(input_x, input_y)
                (res,) = exe.run(
                    feed={"x": input_x, "y": input_y}, fetch_list=[out]
                )
            self.assertEqual((res == real_result).all(), True)

        def test_attr_name(self):
            paddle.enable_static()
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x = paddle.static.data(name='x', shape=[-1, 4], dtype='int32')
                y = paddle.static.data(name='y', shape=[-1, 4], dtype='int32')
                op = eval(f"paddle.{self.op_type}")
                out = op(x=x, y=y, name=f"name_{self.op_type}")
            if not in_pir_mode():
                self.assertEqual(f"name_{self.op_type}" in out.name, True)

    cls_name = f"TestCase_{op_type}"
    PaddleCls.__name__ = cls_name
    globals()[cls_name] = PaddleCls


create_paddle_case('less_than', lambda _a, _b: _a < _b)
create_paddle_case('less_equal', lambda _a, _b: _a <= _b)
create_paddle_case('greater_than', lambda _a, _b: _a > _b)
create_paddle_case('greater_equal', lambda _a, _b: _a >= _b)
create_paddle_case('equal', lambda _a, _b: _a == _b)
create_paddle_case('not_equal', lambda _a, _b: _a != _b)


# add bf16 tests
def create_bf16_case(op_type, callback, check_pir=False):
    class TestCompareOpBF16Op(op_test.OpTest):
        def setUp(self):
            self.op_type = op_type
            self.dtype = np.uint16
            self.python_api = eval("paddle." + op_type)

            x = np.random.uniform(0, 1, [5, 5]).astype(np.float32)
            y = np.random.uniform(0, 1, [5, 5]).astype(np.float32)
            real_result = callback(x, y)
            self.inputs = {
                'X': op_test.convert_float_to_uint16(x),
                'Y': op_test.convert_float_to_uint16(y),
            }
            self.outputs = {'Out': real_result}

        def test_check_output(self):
            self.check_output(check_cinn=True, check_pir=check_pir)

    cls_name = f"BF16TestCase_{op_type}"
    TestCompareOpBF16Op.__name__ = cls_name
    globals()[cls_name] = TestCompareOpBF16Op


create_bf16_case('less_than', lambda _a, _b: _a < _b, True)
create_bf16_case('less_equal', lambda _a, _b: _a <= _b, True)
create_bf16_case('greater_than', lambda _a, _b: _a > _b, True)
create_bf16_case('greater_equal', lambda _a, _b: _a >= _b, True)
create_bf16_case('equal', lambda _a, _b: _a == _b, True)
create_bf16_case('not_equal', lambda _a, _b: _a != _b, True)


class TestCompareOpError(unittest.TestCase):

    def test_int16_support(self):
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            # The input x and y of compare_op must be Variable.
            x = paddle.static.data(name='x', shape=[-1, 1], dtype="float32")
            y = base.create_lod_tensor(
                numpy.array([[-1]]), [[1]], base.CPUPlace()
            )
            self.assertRaises(TypeError, paddle.greater_equal, x, y)


class API_TestElementwise_Equal(unittest.TestCase):

    def test_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            label = paddle.assign(np.array([3, 3], dtype="int32"))
            limit = paddle.assign(np.array([3, 2], dtype="int32"))
            out = paddle.equal(x=label, y=limit)
            place = base.CPUPlace()
            exe = base.Executor(place)
            (res,) = exe.run(fetch_list=[out])
        self.assertEqual((res == np.array([True, False])).all(), True)

        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            label = paddle.assign(np.array([3, 3], dtype="int32"))
            limit = paddle.assign(np.array([3, 3], dtype="int32"))
            out = paddle.equal(x=label, y=limit)
            place = base.CPUPlace()
            exe = base.Executor(place)
            (res,) = exe.run(fetch_list=[out])
        self.assertEqual((res == np.array([True, True])).all(), True)

    def test_api_fp16(self):
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            label = paddle.to_tensor([3, 3], dtype="float16")
            limit = paddle.to_tensor([3, 2], dtype="float16")
            out = paddle.equal(x=label, y=limit)
            if core.is_compiled_with_cuda():
                place = paddle.CUDAPlace(0)
                exe = base.Executor(place)
                (res,) = exe.run(fetch_list=[out])
                self.assertEqual((res == np.array([True, False])).all(), True)


class API_TestElementwise_Greater_Than(unittest.TestCase):

    def test_api_fp16(self):
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            label = paddle.to_tensor([3, 3], dtype="float16")
            limit = paddle.to_tensor([3, 2], dtype="float16")
            out = paddle.greater_than(x=label, y=limit)
            if core.is_compiled_with_cuda():
                place = paddle.CUDAPlace(0)
                exe = paddle.static.Executor(place)
                (res,) = exe.run(fetch_list=[out])
                self.assertEqual((res == np.array([False, True])).all(), True)


class TestCompareOpPlace(unittest.TestCase):

    def test_place_1(self):
        paddle.enable_static()
        place = paddle.CPUPlace()
        if core.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            label = paddle.assign(np.array([3, 3], dtype="int32"))
            limit = paddle.assign(np.array([3, 2], dtype="int32"))
            out = paddle.less_than(label, limit)
            exe = base.Executor(place)
            (res,) = exe.run(fetch_list=[out])
            self.assertEqual((res == np.array([False, False])).all(), True)

    def test_place_2(self):
        place = paddle.CPUPlace()
        data_place = place
        if core.is_compiled_with_cuda():
            place = paddle.CUDAPlace(0)
            data_place = paddle.CUDAPinnedPlace()
        paddle.disable_static(place)
        data = np.array([9], dtype="int64")
        data_tensor = paddle.to_tensor(data, place=data_place)
        result = data_tensor == 0
        self.assertEqual((result.numpy() == np.array([False])).all(), True)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
