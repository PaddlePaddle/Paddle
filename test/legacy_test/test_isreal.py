# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import base, static

TEST_REAL_DATA = [
    np.array(1.0),
    np.random.randint(-10, 10, (2, 3)),
    np.random.randn(64, 32),
]
REAL_TYPE = [
    'float16',
    'float32',
    'float64',
    'bool',
    'int16',
    'int32',
    'int64',
    'uint16',
]
TEST_COMPLEX_DATA = [
    np.array(1.0 + 2j),
    np.array(1.0 + 0j),
    np.array([[0.2 + 3j, 3 + 0j, -0.7 - 6j], [-0.4 + 0j, 3.5 - 10j, 2.5 + 0j]]),
]
COMPLEX_TYPE = ['complex64', 'complex128']


def run_dygraph(data, type, use_gpu=False):
    place = paddle.CPUPlace()
    if use_gpu and base.core.is_compiled_with_cuda():
        place = paddle.CUDAPlace(0)
    paddle.disable_static(place)
    data = data.astype(type)
    x = paddle.to_tensor(data)
    return paddle.isreal(x)


def run_static(data, type, use_gpu=False):
    paddle.enable_static()
    startup_program = paddle.static.Program()
    main_program = paddle.static.Program()
    place = paddle.CPUPlace()
    if use_gpu and base.core.is_compiled_with_cuda():
        place = paddle.CUDAPlace(0)
    exe = base.Executor(place)
    with static.program_guard(main_program, startup_program):
        data = data.astype(type)
        x = paddle.static.data(name='x', shape=data.shape, dtype=type)
        res = paddle.isreal(x)
        static_result = exe.run(feed={'x': data}, fetch_list=[res])
        return static_result


def test(data_cases, type_cases, use_gpu=False):
    for data in data_cases:
        for type in type_cases:
            dygraph_result = run_dygraph(data, type, use_gpu).numpy()
            np_result = np.isreal(data.astype(type))
            np.testing.assert_equal(dygraph_result, np_result)

            def test_static_or_pir_mode():
                (static_result,) = run_static(data, type, use_gpu)
                np.testing.assert_equal(static_result, np_result)

            test_static_or_pir_mode()


class TestIsRealError(unittest.TestCase):
    def test_for_exception(self):
        with self.assertRaises(TypeError):
            paddle.isreal(np.array([1, 2]))


class TestIsReal(unittest.TestCase):
    def test_for_real_tensor_without_gpu(self):
        test(TEST_REAL_DATA, REAL_TYPE)

    def test_for_real_tensor_with_gpu(self):
        test(TEST_REAL_DATA, REAL_TYPE, True)

    def test_for_complex_tensor_without_gpu(self):
        test(TEST_COMPLEX_DATA, COMPLEX_TYPE)

    def test_for_complex_tensor_with_gpu(self):
        test(TEST_COMPLEX_DATA, COMPLEX_TYPE, True)


if __name__ == '__main__':
    unittest.main()
