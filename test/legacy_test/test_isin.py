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
from op_test import convert_float_to_uint16

import paddle
from paddle import base
from paddle.base import core

DATA_CASES = [
    {'x_data': np.array(1.0), 'test_x_data': np.array(-1.0)},
    {
        'x_data': np.random.randint(-10, 10, (4, 8)),
        'test_x_data': np.random.randint(0, 20, (2, 3)),
    },
    {
        'x_data': np.random.randint(-50, 50, (8, 64)),
        'test_x_data': np.random.randint(-20, 0, (4, 256)),
    },
]

DATA_CASES_UNIQUE = [
    {
        'x_data': np.arange(0, 1000).reshape([2, 5, 100]),
        'test_x_data': np.arange(200, 700),
    },
    {
        'x_data': np.arange(-100, 100).reshape([2, 2, 5, 10]),
        'test_x_data': np.arange(50, 150).reshape([4, 5, 5]),
    },
]

DATA_CASES_BF16 = [
    {'x_data': np.array(1.0), 'test_x_data': np.array(0.0)},
    {
        'x_data': np.random.randint(0, 10, (4, 8)),
        'test_x_data': np.random.randint(5, 15, (2, 3)),
    },
    {
        'x_data': np.random.randint(0, 50, (8, 64)),
        'test_x_data': np.random.randint(0, 20, (4, 256)),
    },
]


DATA_CASES_UNIQUE_BF16 = [
    {
        'x_data': np.arange(0, 100).reshape([2, 5, 10]),
        'test_x_data': np.arange(50, 150),
    },
]


DATA_TYPE = ['float32', 'float64', 'int32', 'int64']


def run_dygraph(
    x_data,
    test_x_data,
    type,
    assume_unique=False,
    invert=False,
    use_gpu=False,
):
    place = paddle.CPUPlace()
    if use_gpu and base.core.is_compiled_with_cuda():
        place = paddle.CUDAPlace(0)
    paddle.disable_static(place)
    x_data = x_data.astype(type)
    test_x_data = test_x_data.astype(type)
    x_e = paddle.to_tensor(x_data)
    x_t = paddle.to_tensor(test_x_data)
    return paddle.isin(x_e, x_t, assume_unique, invert)


def run_static(
    x_data,
    test_x_data,
    type,
    assume_unique=False,
    invert=False,
    use_gpu=False,
):
    paddle.enable_static()
    startup_program = paddle.static.Program()
    main_program = paddle.static.Program()
    place = paddle.CPUPlace()
    if use_gpu and base.core.is_compiled_with_cuda():
        place = paddle.CUDAPlace(0)
    exe = base.Executor(place)
    with paddle.static.program_guard(main_program, startup_program):
        x_data = x_data.astype(type)
        test_x_data = test_x_data.astype(type)
        x_e = paddle.static.data(name='x_e', shape=x_data.shape, dtype=type)
        x_t = paddle.static.data(
            name='x_t', shape=test_x_data.shape, dtype=type
        )
        res = paddle.isin(x_e, x_t, assume_unique, invert)
        static_result = exe.run(
            feed={'x_e': x_data, 'x_t': test_x_data},
            fetch_list=[res],
        )
        return static_result


def test(
    data_cases, type_cases, assume_unique=False, invert=False, use_gpu=False
):
    for type in type_cases:
        for case in data_cases:
            x_data = case['x_data']
            test_x_data = case['test_x_data']
            dygraph_result = run_dygraph(
                x_data,
                test_x_data,
                type,
                assume_unique,
                invert,
                use_gpu,
            ).numpy()
            np_result = np.isin(
                x_data.astype(type),
                test_x_data.astype(type),
                assume_unique=assume_unique,
                invert=invert,
            )
            np.testing.assert_equal(dygraph_result, np_result)

            def test_static():
                (static_result,) = run_static(
                    x_data,
                    test_x_data,
                    type,
                    assume_unique,
                    invert,
                    use_gpu,
                )
                np.testing.assert_equal(static_result, np_result)

            test_static()


def run_dygraph_bf16(
    x_data,
    test_x_data,
    assume_unique=False,
    invert=False,
    use_gpu=False,
):
    place = paddle.CPUPlace()
    if use_gpu and base.core.is_compiled_with_cuda():
        place = paddle.CUDAPlace(0)
    paddle.disable_static(place)
    x_e = paddle.to_tensor(convert_float_to_uint16(x_data))
    x_t = paddle.to_tensor(convert_float_to_uint16(test_x_data))
    return paddle.isin(x_e, x_t, assume_unique, invert)


def run_static_bf16(
    x_data,
    test_x_data,
    assume_unique=False,
    invert=False,
    use_gpu=False,
):
    paddle.enable_static()
    startup_program = paddle.static.Program()
    main_program = paddle.static.Program()
    place = paddle.CPUPlace()
    if use_gpu and base.core.is_compiled_with_cuda():
        place = paddle.CUDAPlace(0)
    exe = base.Executor(place)
    with paddle.static.program_guard(main_program, startup_program):
        x_data = convert_float_to_uint16(x_data)
        test_x_data = convert_float_to_uint16(test_x_data)
        x_e = paddle.static.data(
            name='x_e', shape=x_data.shape, dtype=np.uint16
        )
        x_t = paddle.static.data(
            name='x_t', shape=test_x_data.shape, dtype=np.uint16
        )
        res = paddle.isin(x_e, x_t, assume_unique, invert)
        static_result = exe.run(
            feed={'x_e': x_data, 'x_t': test_x_data},
            fetch_list=[res],
        )
        return static_result


def test_bf16(data_cases, assume_unique=False, invert=False, use_gpu=False):
    for case in data_cases:
        x_data = case['x_data'].astype("float32")
        test_x_data = case['test_x_data'].astype("float32")
        dygraph_result = run_dygraph_bf16(
            x_data,
            test_x_data,
            assume_unique,
            invert,
            use_gpu,
        ).numpy()
        np_result = np.isin(
            x_data,
            test_x_data,
            assume_unique=assume_unique,
            invert=invert,
        )
        np.testing.assert_equal(dygraph_result, np_result)

        def test_static():
            (static_result,) = run_static_bf16(
                x_data,
                test_x_data,
                assume_unique,
                invert,
                use_gpu,
            )
            np.testing.assert_equal(static_result, np_result)

        test_static()


class TestIsInError(unittest.TestCase):
    def test_for_exception(self):
        with self.assertRaises(TypeError):
            paddle.isin(np.array([1, 2]), np.array([1, 2]))


class TestIsIn(unittest.TestCase):
    def test_without_gpu(self):
        test(DATA_CASES, DATA_TYPE)

    def test_with_gpu(self):
        test(DATA_CASES, DATA_TYPE, use_gpu=True)

    def test_invert_without_gpu(self):
        test(DATA_CASES, DATA_TYPE, invert=True)

    def test_invert_with_gpu(self):
        test(DATA_CASES, DATA_TYPE, invert=True, use_gpu=True)

    def test_unique_without_gpu(self):
        test(DATA_CASES_UNIQUE, DATA_TYPE, assume_unique=True)

    def test_unique_with_gpu(self):
        test(DATA_CASES_UNIQUE, DATA_TYPE, assume_unique=True, use_gpu=True)

    def test_unique_invert_without_gpu(self):
        test(DATA_CASES_UNIQUE, DATA_TYPE, assume_unique=True, invert=True)

    def test_unique_invert_with_gpu(self):
        test(
            DATA_CASES_UNIQUE,
            DATA_TYPE,
            assume_unique=True,
            invert=True,
            use_gpu=True,
        )


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_float16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the float16",
)
class TestIsInFP16(unittest.TestCase):
    def test_default(self):
        test(DATA_CASES, ['float16'], use_gpu=True)

    def test_invert(self):
        test(DATA_CASES, ['float16'], invert=True, use_gpu=True)

    def test_unique(self):
        test(DATA_CASES_UNIQUE, ['float16'], assume_unique=True, use_gpu=True)

    def test_unique_invert(self):
        test(
            DATA_CASES_UNIQUE,
            ['float16'],
            assume_unique=True,
            invert=True,
            use_gpu=True,
        )


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_float16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the float16",
)
class TestIsInBF16(unittest.TestCase):
    def test_default(self):
        test_bf16(DATA_CASES_BF16, use_gpu=True)

    def test_invert(self):
        test_bf16(DATA_CASES_BF16, invert=True, use_gpu=True)

    def test_unique(self):
        test_bf16(DATA_CASES_UNIQUE_BF16, assume_unique=True, use_gpu=True)

    def test_unique_invert(self):
        test_bf16(
            DATA_CASES_UNIQUE_BF16,
            assume_unique=True,
            invert=True,
            use_gpu=True,
        )


if __name__ == '__main__':
    unittest.main()
