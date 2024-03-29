# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

DATA_CASES = [
    {'elements_data': np.array(1.0), 'test_elements_data': np.array(-1.0)},
    {
        'elements_data': np.random.randint(-10, 10, (4, 8)),
        'test_elements_data': np.random.randint(0, 20, (2, 3)),
    },
    {
        'elements_data': np.random.randint(-50, 50, (8, 64)),
        'test_elements_data': np.random.randint(-20, 0, (4, 256)),
    },
]
DATA_TYPE = ['float32', 'float64', 'int32', 'int64']


def run_dygraph(
    elements_data, test_elements_data, type, invert=False, use_gpu=False
):
    place = paddle.CPUPlace()
    if use_gpu and base.core.is_compiled_with_cuda():
        place = paddle.CUDAPlace(0)
    paddle.disable_static(place)
    elements_data = elements_data.astype(type)
    test_elements_data = test_elements_data.astype(type)
    x_e = paddle.to_tensor(elements_data)
    x_t = paddle.to_tensor(test_elements_data)
    return paddle.isin(x_e, x_t, invert)


def run_static(
    elements_data, test_elements_data, type, invert=False, use_gpu=False
):
    paddle.enable_static()
    startup_program = paddle.static.Program()
    main_program = paddle.static.Program()
    place = paddle.CPUPlace()
    if use_gpu and base.core.is_compiled_with_cuda():
        place = paddle.CUDAPlace(0)
    exe = base.Executor(place)
    with static.program_guard(main_program, startup_program):
        elements_data = elements_data.astype(type)
        test_elements_data = test_elements_data.astype(type)
        x_e = paddle.static.data(
            name='x_e', shape=elements_data.shape, dtype=type
        )
        x_t = paddle.static.data(
            name='x_t', shape=test_elements_data.shape, dtype=type
        )
        res = paddle.isin(x_e, x_t, invert)
        static_result = exe.run(
            feed={'x_e': elements_data, 'x_t': test_elements_data},
            fetch_list=[res],
        )
        return static_result


def test(data_cases, type_cases, invert=False, use_gpu=False):
    for type in type_cases:
        for case in data_cases:
            elements_data = case['elements_data']
            test_elements_data = case['test_elements_data']
            dygraph_result = run_dygraph(
                elements_data, test_elements_data, type, invert, use_gpu
            ).numpy()
            np_result = np.isin(
                elements_data.astype(type),
                test_elements_data.astype(type),
                invert=invert,
            )
            np.testing.assert_equal(dygraph_result, np_result)

            def test_static():
                (static_result,) = run_static(
                    elements_data, test_elements_data, type, invert, use_gpu
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


if __name__ == '__main__':
    unittest.main()
