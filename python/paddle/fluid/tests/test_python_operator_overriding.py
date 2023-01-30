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

<<<<<<< HEAD
=======
from __future__ import print_function

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
import unittest

import numpy as np

<<<<<<< HEAD
import paddle
import paddle.fluid as fluid
import paddle.fluid.framework as framework
=======
import paddle.fluid.layers as layers
import paddle.fluid.framework as framework
import paddle.fluid as fluid
import paddle
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

paddle.enable_static()


class TestPythonOperatorOverride(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def check_result(self, fn, place, dtype):
        shape = [9, 10]

        x_data = np.random.random(size=shape).astype(dtype)
        y_data = np.random.random(size=shape).astype(dtype)
        python_out = fn(x_data, y_data)

<<<<<<< HEAD
        x_var = paddle.static.create_global_var(
            name='x', shape=shape, value=0.0, dtype=dtype, persistable=True
        )
        y_var = paddle.static.create_global_var(
            name='y', shape=shape, value=0.0, dtype=dtype, persistable=True
        )
=======
        x_var = layers.create_global_var(name='x',
                                         shape=shape,
                                         value=0.0,
                                         dtype=dtype,
                                         persistable=True)
        y_var = layers.create_global_var(name='y',
                                         shape=shape,
                                         value=0.0,
                                         dtype=dtype,
                                         persistable=True)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        out = fn(x_var, y_var)

        exe = fluid.Executor(place)

        exe.run(fluid.default_startup_program())
<<<<<<< HEAD
        fluid_out = exe.run(
            fluid.default_main_program(),
            feed={'x': x_data, 'y': y_data},
            fetch_list=[out],
        )
=======
        fluid_out = exe.run(fluid.default_main_program(),
                            feed={
                                'x': x_data,
                                'y': y_data
                            },
                            fetch_list=[out])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        np.testing.assert_array_equal(python_out, fluid_out[0])

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
        places = [fluid.CPUPlace()]
        if fluid.core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))

        # dtypes to check
        dtypes = ['int32', 'float32']

        for place in places:
            for dtype in dtypes:
                for compare_fn in compare_fns:
<<<<<<< HEAD
                    with framework.program_guard(
                        framework.Program(), framework.Program()
                    ):
=======
                    with framework.program_guard(framework.Program(),
                                                 framework.Program()):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                        self.check_result(compare_fn, place, dtype)


if __name__ == '__main__':
    unittest.main()
