# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved
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

import os
import unittest

import numpy as np

import paddle
from paddle import base

paddle.enable_static()


class TestPythonOperatorOverride(unittest.TestCase):
    def check_result(self, fn, place, dtype):
        shape = [9, 10]

        x_data = np.random.random(size=shape).astype(dtype)
        y_data = np.random.random(size=shape).astype(dtype)
        python_out = fn(x_data, y_data)

        x_var = paddle.static.data(name='x', shape=shape, dtype=dtype)
        y_var = paddle.static.data(name='y', shape=shape, dtype=dtype)
        out = fn(x_var, y_var)

        exe = paddle.static.Executor(place)

        exe.run(paddle.static.default_startup_program())
        base_out = exe.run(
            paddle.static.default_main_program(),
            feed={'x': x_data, 'y': y_data},
            fetch_list=[out],
        )

        np.testing.assert_array_equal(python_out, base_out[0])

    def test_override(self):
        # compare func to check
        compare_fns = [
            lambda _a, _b: _a == _b,
            lambda _a, _b: _a != _b,
            lambda _a, _b: _a < _b,
            lambda _a, _b: _a <= _b,
            lambda _a, _b: _a > _b,
            lambda _a, _b: _a >= _b,
        ]

        # places to check
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not base.core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if base.core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))

        # dtypes to check
        dtypes = ['int32', 'float32']

        for place in places:
            for dtype in dtypes:
                for compare_fn in compare_fns:
                    with paddle.static.program_guard(
                        paddle.static.Program(), paddle.static.Program()
                    ):
                        self.check_result(compare_fn, place, dtype)


if __name__ == '__main__':
    unittest.main()
