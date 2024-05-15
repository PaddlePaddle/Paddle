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
        elif exp.dtype in [bool, np.int64]:
            assert np.all(out == exp)


@pytest.mark.argsort
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_argsort():
    paddle.enable_static()
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    main_program.random_seed = 33
    startup_program.random_seed = 33
    with paddle.utils.unique_name.guard():
        with paddle.static.program_guard(
            main_program=main_program, startup_program=startup_program
        ):
            data = paddle.static.data(
                name='data', shape=[2, 3, 4], dtype='float32'
            )
            data.stop_gradient = False
            out = paddle.argsort(x=data, axis=-1)  # same as axis=2
            # out2 = paddle.argsort(x=data, axis=0)
            # out3 = paddle.argsort(x=data, axis=1)
            out[0].stop_gradient = False
            out[1].stop_gradient = False
            # loss = paddle.mean(out[0])
            g = paddle.static.gradients(out[0], data)  # argsort without grad

            feed = {
                "data": np.array(
                    [
                        [[5, 8, 9, 5], [0, 0, 1, 7], [6, 9, 2, 4]],
                        [[5, 2, 4, 2], [4, 7, 7, 9], [1, 7, 0, 6]],
                    ]
                ).astype("float32")
            }

            cpu_place = paddle.CPUPlace()
            cpu_exe = paddle.static.Executor(cpu_place)
            cpu_exe.run(startup_program)
            cpu_res = cpu_exe.run(
                main_program,
                feed=feed,
                fetch_list=[out[0].name],
                return_numpy=True,
            )

            gcu_exe = paddle.static.Executor('gcu')
            gcu_res = gcu_exe.run(
                main_program,
                feed=feed,
                fetch_list=[out[0].name],
                return_numpy=True,
            )

            print('[cpu result]')
            print(cpu_res)
            print('[gcu result]')
            print(gcu_res)

            compare(cpu_res, gcu_res)
