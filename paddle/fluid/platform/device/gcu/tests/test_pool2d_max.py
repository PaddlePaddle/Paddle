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

import logging

import numpy as np
import pytest

import paddle

np.random.seed(33)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def compare(cpu_res, gcu_res):
    assert len(cpu_res) == len(gcu_res)
    for i in range(len(cpu_res)):
        out = gcu_res[i]
        exp = cpu_res[i]
        assert out.shape == exp.shape
        assert out.dtype == exp.dtype
        if exp.dtype == np.float32:
            diff = np.abs(out - exp)
            err = np.ones(shape=exp.shape) * 1e-5
            assert np.all(diff < err)
        elif exp.dtype in [np.bool_, np.int64]:
            assert np.all(out == exp)


paddle.enable_static()
main_program = paddle.static.Program()
startup_program = paddle.static.Program()
main_program.random_seed = 33
startup_program.random_seed = 33


@pytest.mark.pool2d
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_pool2d_max():
    with paddle.utils.unique_name.guard():
        with paddle.static.program_guard(
            main_program=main_program, startup_program=startup_program
        ):
            data = paddle.static.data(
                name='data', shape=[1, 64, 320, 320], dtype='float32'
            )
            data.stop_gradient = False
            pool = paddle.nn.MaxPool2D(
                kernel_size=3, stride=2, padding=1, ceil_mode=False
            )
            out = pool(data)
            loss = paddle.mean(out)
            g = paddle.static.gradients(loss, data)

            input = np.random.uniform(-1, 1, (1, 64, 320, 320)).astype(
                np.float32
            )

            # get cpu result
            cpu_place = paddle.CPUPlace()
            cpu_exe = paddle.static.Executor(cpu_place)
            cpu_exe.run(startup_program)
            cpu_res = cpu_exe.run(
                main_program,
                feed={'data': input},
                fetch_list=[out.name, g[0].name],
                return_numpy=True,
            )

            # get gcu result
            gcu_exe = paddle.static.Executor('gcu')
            gcu_res = gcu_exe.run(
                main_program,
                feed={'data': input},
                fetch_list=[out.name, g[0].name],
                return_numpy=True,
            )

            compare(cpu_res, gcu_res)
