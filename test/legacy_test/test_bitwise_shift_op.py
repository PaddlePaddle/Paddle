# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

_SIGNED_TO_UNSIGNED_TABLE = {
    "int8": "uint8",
    "int16": "uint16",
    "int32": "uint32",
    "int64": "uint64",
}

_UNSIGNED_TO_SIGNED_TABLE = {
    "uint8": "int8",
    "uint16": "int16",
    "uint32": "int32",
    "uint64": "int64",
}

_UNSIGNED_LIST = ['uint8', 'uint16', 'uint32', 'uint64']


def ref_left_shift_arithmetic(x, y):
    out = np.left_shift(x, y)
    return out


def ref_left_shift_logical(x, y):
    out = np.left_shift(x, y)
    return out


def ref_right_shift_arithmetic(x, y):
    return np.right_shift(x, y)


def ref_right_shift_logical(x, y):
    if str(x.dtype) in _UNSIGNED_LIST:
        return np.right_shift(x, y)
    else:
        orig_dtype = x.dtype
        unsigned_dtype = _SIGNED_TO_UNSIGNED_TABLE[str(orig_dtype)]
        x = x.astype(unsigned_dtype)
        y = y.astype(unsigned_dtype)
        res = np.right_shift(x, y)
        return res.astype(orig_dtype)


class TestBitwiseLeftShiftAPI(unittest.TestCase):
    def setUp(self):
        self.init_input()
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def init_input(self):
        self.x = np.random.randint(0, 256, [200, 300]).astype('uint8')
        self.y = np.random.randint(0, 256, [200, 300]).astype('uint8')

    def test_static_api_arithmetic(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('x', self.x.shape, dtype=self.x.dtype)
            y = paddle.static.data('y', self.y.shape, dtype=self.y.dtype)
            out = paddle.bitwise_left_shift(
                x,
                y,
            )
            out_ = x << y
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'x': self.x, 'y': self.y}, fetch_list=[out])
            res_ = exe.run(feed={'x': self.x, 'y': self.y}, fetch_list=[out_])
            out_ref = ref_left_shift_arithmetic(self.x, self.y)
            np.testing.assert_allclose(out_ref, res[0])
            np.testing.assert_allclose(out_ref, res_[0])

    def test_dygraph_api_arithmetic(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x)
        y = paddle.to_tensor(self.y)
        out = paddle.bitwise_left_shift(
            x,
            y,
        )
        out_ = x << y
        out_ref = ref_left_shift_arithmetic(self.x, self.y)
        np.testing.assert_allclose(out_ref, out.numpy())
        np.testing.assert_allclose(out_ref, out_.numpy())
        paddle.enable_static()

    def test_static_api_logical(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('x', self.x.shape, dtype=self.x.dtype)
            y = paddle.static.data('y', self.y.shape, dtype=self.y.dtype)
            out = paddle.bitwise_left_shift(x, y, False)
            out_ = x.__lshift__(y, False)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'x': self.x, 'y': self.y}, fetch_list=[out])
            res_ = exe.run(feed={'x': self.x, 'y': self.y}, fetch_list=[out_])
            out_ref = ref_left_shift_logical(self.x, self.y)
            np.testing.assert_allclose(out_ref, res[0])
            np.testing.assert_allclose(out_ref, res_[0])

    def test_dygraph_api_logical(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x)
        y = paddle.to_tensor(self.y)
        out = paddle.bitwise_left_shift(x, y, False)
        out_ = x.__lshift__(y, False)
        out_ref = ref_left_shift_logical(self.x, self.y)
        np.testing.assert_allclose(out_ref, out.numpy())
        np.testing.assert_allclose(out_ref, out_.numpy())
        paddle.enable_static()


class TestBitwiseLeftShiftAPI_UINT8(TestBitwiseLeftShiftAPI):
    def init_input(self):
        self.x = np.random.randint(0, 256, [200, 300]).astype('uint8')
        self.y = np.random.randint(0, 256, [200, 300]).astype('uint8')


class TestBitwiseLeftShiftAPI_UINT8_broadcast1(TestBitwiseLeftShiftAPI):
    def init_input(self):
        self.x = np.random.randint(0, 256, [200, 300]).astype('uint8')
        self.y = np.random.randint(0, 256, [300]).astype('uint8')


class TestBitwiseLeftShiftAPI_UINT8_broadcast2(TestBitwiseLeftShiftAPI):
    def init_input(self):
        self.x = np.random.randint(0, 256, [300]).astype('uint8')
        self.y = np.random.randint(0, 256, [200, 300]).astype('uint8')


class TestBitwiseLeftShiftAPI_INT8(TestBitwiseLeftShiftAPI):
    def init_input(self):
        self.x = np.random.randint(-(2**7), 2**7, [200, 300]).astype('int8')
        self.y = np.random.randint(-(2**7), 2**7, [200, 300]).astype('int8')


class TestBitwiseLeftShiftAPI_INT8_broadcast1(TestBitwiseLeftShiftAPI):
    def init_input(self):
        self.x = np.random.randint(-(2**7), 2**7, [200, 300]).astype('int8')
        self.y = np.random.randint(-(2**7), 2**7, [300]).astype('int8')


class TestBitwiseLeftShiftAPI_INT8_broadcast2(TestBitwiseLeftShiftAPI):
    def init_input(self):
        self.x = np.random.randint(-(2**7), 2**7, [300]).astype('int8')
        self.y = np.random.randint(-(2**7), 2**7, [200, 300]).astype('int8')


class TestBitwiseLeftShiftAPI_INT16(TestBitwiseLeftShiftAPI):
    def init_input(self):
        self.x = np.random.randint(-(2**15), 2**15, [200, 300]).astype('int16')
        self.y = np.random.randint(-(2**15), 2**15, [200, 300]).astype('int16')


class TestBitwiseLeftShiftAPI_INT16_broadcast1(TestBitwiseLeftShiftAPI):
    def init_input(self):
        self.x = np.random.randint(-(2**15), 2**15, [200, 300]).astype('int16')
        self.y = np.random.randint(-(2**15), 2**15, [300]).astype('int16')


class TestBitwiseLeftShiftAPI_INT16_broadcast2(TestBitwiseLeftShiftAPI):
    def init_input(self):
        self.x = np.random.randint(-(2**15), 2**15, [300]).astype('int16')
        self.y = np.random.randint(-(2**15), 2**15, [200, 300]).astype('int16')


class TestBitwiseLeftShiftAPI_INT32(TestBitwiseLeftShiftAPI):
    def init_input(self):
        self.x = np.random.randint(-(2**31), 2**31, [200, 300]).astype('int32')
        self.y = np.random.randint(-(2**31), 2**31, [200, 300]).astype('int32')


class TestBitwiseLeftShiftAPI_INT32_broadcast1(TestBitwiseLeftShiftAPI):
    def init_input(self):
        self.x = np.random.randint(-(2**31), 2**31, [200, 300]).astype('int32')
        self.y = np.random.randint(-(2**31), 2**31, [300]).astype('int32')


class TestBitwiseLeftShiftAPI_INT32_broadcast2(TestBitwiseLeftShiftAPI):
    def init_input(self):
        self.x = np.random.randint(-(2**31), 2**31, [300]).astype('int32')
        self.y = np.random.randint(-(2**31), 2**31, [200, 300]).astype('int32')


class TestBitwiseLeftShiftAPI_INT64(TestBitwiseLeftShiftAPI):
    def init_input(self):
        self.x = np.random.randint(-(2**63), 2**63, [200, 300], dtype=np.int64)
        self.y = np.random.randint(-(2**63), 2**63, [200, 300], dtype=np.int64)


class TestBitwiseLeftShiftAPI_INT64_broadcast1(TestBitwiseLeftShiftAPI):
    def init_input(self):
        self.x = np.random.randint(-(2**63), 2**63, [200, 300], dtype=np.int64)
        self.y = np.random.randint(-(2**63), 2**63, [300], dtype=np.int64)


class TestBitwiseLeftShiftAPI_INT64_broadcast2(TestBitwiseLeftShiftAPI):
    def init_input(self):
        self.x = np.random.randint(-(2**63), 2**63, [300], dtype=np.int64)
        self.y = np.random.randint(-(2**63), 2**63, [200, 300], dtype=np.int64)


class TestBitwiseLeftShiftAPI_special_case1(TestBitwiseLeftShiftAPI):
    def init_input(self):
        self.x = np.array([0b11111111], dtype='int16')
        self.y = np.array([1], dtype='int16')


class TestBitwiseLeftShiftAPI_special_case2(TestBitwiseLeftShiftAPI):
    def init_input(self):
        self.x = np.array([0b11111111], dtype='int16')
        self.y = np.array([10], dtype='int16')


class TestBitwiseLeftShiftAPI_special_case3(TestBitwiseLeftShiftAPI):
    def init_input(self):
        self.x = np.array([0b11111111], dtype='uint8')
        self.y = np.array([1], dtype='uint8')


class TestBitwiseLeftShiftAPI_special_case4(TestBitwiseLeftShiftAPI):
    def init_input(self):
        self.x = np.array([0b11111111], dtype='uint8')
        self.y = np.array([10], dtype='uint8')


class TestTensorRlshiftAPI(unittest.TestCase):
    def setUp(self):
        self.init_input()
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def init_input(self):
        self.x = np.random.randint(-255, 256)
        self.y = np.random.randint(0, 256, [200, 300]).astype('int32')

    def test_dygraph_tensor_rlshift(self):
        paddle.disable_static()
        x = self.x
        y = paddle.to_tensor(self.y, dtype=self.y.dtype)
        out = x << y
        expected_out = x << y.numpy()
        np.testing.assert_allclose(out.numpy(), expected_out)
        paddle.enable_static()

    def test_static_rlshift(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = self.x
            y = paddle.static.data('y', self.y.shape, dtype=self.y.dtype)
            out = x << y
            exe = paddle.static.Executor(self.place)
            res = exe.run(
                feed={'x': self.x, 'y': self.y},
                fetch_list=[out],
            )
            out_ref = ref_left_shift_arithmetic(self.x, self.y)
            np.testing.assert_allclose(out_ref, res[0])


class TestTensorRlshiftAPI_UINT8(TestTensorRlshiftAPI):
    def init_input(self):
        self.x = np.random.randint(0, 64)
        self.y = np.random.randint(0, 64, [200, 300]).astype('uint8')


class TestTensorRlshiftAPI_INT8(TestTensorRlshiftAPI):
    def init_input(self):
        self.x = np.random.randint(-64, 64)
        self.y = np.random.randint(0, 64, [200, 300]).astype('int8')


class TestTensorRlshiftAPI_INT16(TestTensorRlshiftAPI):
    def init_input(self):
        self.x = np.random.randint(-256, 256)
        self.y = np.random.randint(0, 256, [200, 300]).astype('int16')


class TestTensorRlshiftAPI_INT64(TestTensorRlshiftAPI):
    def init_input(self):
        self.x = np.random.randint(-255, 256)
        self.y = np.random.randint(0, 256, [200, 300]).astype('int64')


class TestBitwiseRightShiftAPI(unittest.TestCase):
    def setUp(self):
        self.init_input()
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def init_input(self):
        self.x = np.random.randint(0, 256, [200, 300]).astype('uint8')
        self.y = np.random.randint(0, 256, [200, 300]).astype('uint8')

    def test_static_api_arithmetic(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('x', self.x.shape, dtype=self.x.dtype)
            y = paddle.static.data('y', self.y.shape, dtype=self.y.dtype)
            out = paddle.bitwise_right_shift(
                x,
                y,
            )
            out_ = x >> y
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'x': self.x, 'y': self.y}, fetch_list=[out])
            res_ = exe.run(feed={'x': self.x, 'y': self.y}, fetch_list=[out_])
            out_ref = ref_right_shift_arithmetic(self.x, self.y)
            np.testing.assert_allclose(out_ref, res[0])
            np.testing.assert_allclose(out_ref, res_[0])

    def test_dygraph_api_arithmetic(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x)
        y = paddle.to_tensor(self.y)
        out = paddle.bitwise_right_shift(
            x,
            y,
        )
        out_ = x >> y
        out_ref = ref_right_shift_arithmetic(self.x, self.y)
        np.testing.assert_allclose(out_ref, out.numpy())
        np.testing.assert_allclose(out_ref, out_.numpy())
        paddle.enable_static()

    def test_static_api_logical(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('x', self.x.shape, dtype=self.x.dtype)
            y = paddle.static.data('y', self.y.shape, dtype=self.y.dtype)
            out = paddle.bitwise_right_shift(x, y, False)
            out_ = x.__rshift__(y, False)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'x': self.x, 'y': self.y}, fetch_list=[out])
            res_ = exe.run(feed={'x': self.x, 'y': self.y}, fetch_list=[out_])
            out_ref = ref_right_shift_logical(self.x, self.y)
            np.testing.assert_allclose(out_ref, res[0])
            np.testing.assert_allclose(out_ref, res_[0])

    def test_dygraph_api_logical(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x)
        y = paddle.to_tensor(self.y)
        out = paddle.bitwise_right_shift(x, y, False)
        out_ = x.__rshift__(y, False)
        out_ref = ref_right_shift_logical(self.x, self.y)
        np.testing.assert_allclose(out_ref, out.numpy())
        np.testing.assert_allclose(out_ref, out_.numpy())
        paddle.enable_static()


class TestBitwiseRightShiftAPI_UINT8(TestBitwiseRightShiftAPI):
    def init_input(self):
        self.x = np.random.randint(0, 256, [200, 300]).astype('uint8')
        self.y = np.random.randint(0, 256, [200, 300]).astype('uint8')


class TestBitwiseRightShiftAPI_UINT8_broadcast1(TestBitwiseRightShiftAPI):
    def init_input(self):
        self.x = np.random.randint(0, 256, [200, 300]).astype('uint8')
        self.y = np.random.randint(0, 256, [300]).astype('uint8')


class TestBitwiseRightShiftAPI_UINT8_broadcast2(TestBitwiseRightShiftAPI):
    def init_input(self):
        self.x = np.random.randint(0, 256, [300]).astype('uint8')
        self.y = np.random.randint(0, 256, [200, 300]).astype('uint8')


class TestBitwiseRightShiftAPI_INT8(TestBitwiseRightShiftAPI):
    def init_input(self):
        self.x = np.random.randint(-(2**7), 2**7, [200, 300]).astype('int8')
        self.y = np.random.randint(-(2**7), 2**7, [200, 300]).astype('int8')


class TestBitwiseRightShiftAPI_INT8_broadcast1(TestBitwiseRightShiftAPI):
    def init_input(self):
        self.x = np.random.randint(-(2**7), 2**7, [200, 300]).astype('int8')
        self.y = np.random.randint(-(2**7), 2**7, [300]).astype('int8')


class TestBitwiseRightShiftAPI_INT8_broadcast2(TestBitwiseRightShiftAPI):
    def init_input(self):
        self.x = np.random.randint(-(2**7), 2**7, [300]).astype('int8')
        self.y = np.random.randint(-(2**7), 2**7, [200, 300]).astype('int8')


class TestBitwiseRightShiftAPI_INT16(TestBitwiseRightShiftAPI):
    def init_input(self):
        self.x = np.random.randint(-(2**15), 2**15, [200, 300]).astype('int16')
        self.y = np.random.randint(-(2**15), 2**15, [200, 300]).astype('int16')


class TestBitwiseRightShiftAPI_INT16_broadcast1(TestBitwiseRightShiftAPI):
    def init_input(self):
        self.x = np.random.randint(-(2**15), 2**15, [200, 300]).astype('int16')
        self.y = np.random.randint(-(2**15), 2**15, [300]).astype('int16')


class TestBitwiseRightShiftAPI_INT16_broadcast2(TestBitwiseRightShiftAPI):
    def init_input(self):
        self.x = np.random.randint(-(2**15), 2**15, [300]).astype('int16')
        self.y = np.random.randint(-(2**15), 2**15, [200, 300]).astype('int16')


class TestBitwiseRightShiftAPI_INT32(TestBitwiseRightShiftAPI):
    def init_input(self):
        self.x = np.random.randint(-(2**31), 2**31, [200, 300]).astype('int32')
        self.y = np.random.randint(-(2**31), 2**31, [200, 300]).astype('int32')


class TestBitwiseRightShiftAPI_INT32_broadcast1(TestBitwiseRightShiftAPI):
    def init_input(self):
        self.x = np.random.randint(-(2**31), 2**31, [200, 300]).astype('int32')
        self.y = np.random.randint(-(2**31), 2**31, [300]).astype('int32')


class TestBitwiseRightShiftAPI_INT32_broadcast2(TestBitwiseRightShiftAPI):
    def init_input(self):
        self.x = np.random.randint(-(2**31), 2**31, [300]).astype('int32')
        self.y = np.random.randint(-(2**31), 2**31, [200, 300]).astype('int32')


class TestBitwiseRightShiftAPI_INT64(TestBitwiseRightShiftAPI):
    def init_input(self):
        self.x = np.random.randint(-(2**63), 2**63, [200, 300], dtype=np.int64)
        self.y = np.random.randint(-(2**63), 2**63, [200, 300], dtype=np.int64)


class TestBitwiseRightShiftAPI_INT64_broadcast1(TestBitwiseRightShiftAPI):
    def init_input(self):
        self.x = np.random.randint(-(2**63), 2**63, [200, 300], dtype=np.int64)
        self.y = np.random.randint(-(2**63), 2**63, [300], dtype=np.int64)


class TestBitwiseRightShiftAPI_INT64_broadcast2(TestBitwiseRightShiftAPI):
    def init_input(self):
        self.x = np.random.randint(-(2**63), 2**63, [300], dtype=np.int64)
        self.y = np.random.randint(-(2**63), 2**63, [200, 300], dtype=np.int64)


class TestBitwiseRightShiftAPI_special_case1(TestBitwiseRightShiftAPI):
    def init_input(self):
        self.x = np.array([0b11111111]).astype('int8')
        self.y = np.array([1]).astype('int8')


class TestBitwiseRightShiftAPI_special_case2(TestBitwiseRightShiftAPI):
    def init_input(self):
        self.x = np.array([0b11111111]).astype('int8')
        self.y = np.array([10]).astype('int8')


class TestBitwiseRightShiftAPI_special_case3(TestBitwiseRightShiftAPI):
    def init_input(self):
        self.x = np.array([0b11111111], dtype='uint8')
        self.y = np.array([1], dtype='uint8')


class TestBitwiseRightShiftAPI_special_case4(TestBitwiseRightShiftAPI):
    def init_input(self):
        self.x = np.array([0b11111111], dtype='uint8')
        self.y = np.array([10], dtype='uint8')


class TestTensorRrshiftAPI(unittest.TestCase):
    def setUp(self):
        self.init_input()
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def init_input(self):
        self.x = np.random.randint(-255, 256)
        self.y = np.random.randint(0, 256, [200, 300]).astype('int32')

    def test_dygraph_tensor_rrshift(self):
        paddle.disable_static()
        x = self.x
        y = paddle.to_tensor(self.y, dtype=self.y.dtype)
        out = x >> y
        expected_out = x >> y.numpy()
        np.testing.assert_allclose(out.numpy(), expected_out)
        paddle.enable_static()

    def test_static_rrshift(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = self.x
            y = paddle.static.data('y', self.y.shape, dtype=self.y.dtype)
            out = x >> y
            exe = paddle.static.Executor(self.place)
            res = exe.run(
                feed={'x': self.x, 'y': self.y},
                fetch_list=[out],
            )
            out_ref = ref_right_shift_arithmetic(self.x, self.y)
            np.testing.assert_allclose(out_ref, res[0])


class TestTensorRrshiftAPI_UINT8(TestTensorRrshiftAPI):
    def init_input(self):
        self.x = np.random.randint(0, 64)
        self.y = np.random.randint(0, 64, [200, 300]).astype('uint8')


class TestTensorRrshiftAPI_INT8(TestTensorRrshiftAPI):
    def init_input(self):
        self.x = np.random.randint(-64, 64)
        self.y = np.random.randint(0, 64, [200, 300]).astype('int8')


class TestTensorRrshiftAPI_INT16(TestTensorRrshiftAPI):
    def init_input(self):
        self.x = np.random.randint(-256, 256)
        self.y = np.random.randint(0, 256, [200, 300]).astype('int16')


class TestTensorRrshiftAPI_INT64(TestTensorRrshiftAPI):
    def init_input(self):
        self.x = np.random.randint(-255, 256)
        self.y = np.random.randint(0, 256, [200, 300]).astype('int64')


class TestTensorShiftAPI_FLOAT(unittest.TestCase):
    def setup(self):
        paddle.disable_static()
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_lshift_float(self):
        x = paddle.to_tensor(np.random.randint(-255, 256, [200, 300]))
        y = np.random.uniform(0, 256)
        with self.assertRaises(TypeError):
            x.__lshift__(y)

    def test_rshift_float(self):
        x = paddle.to_tensor(np.random.randint(-255, 256, [200, 300]))
        y = np.random.uniform(0, 256)
        with self.assertRaises(TypeError):
            x.__rshift__(y)

    def test_rlshift_float(self):
        x = np.random.uniform(0, 256)
        y = paddle.to_tensor(np.random.randint(-255, 256, [200, 300]))
        with self.assertRaises(TypeError):
            y.__rlshift__(x)

    def test_rrshift_float(self):
        x = np.random.uniform(0, 256)
        y = paddle.to_tensor(np.random.randint(-255, 256, [200, 300]))
        with self.assertRaises(TypeError):
            y.__rrshift__(x)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
