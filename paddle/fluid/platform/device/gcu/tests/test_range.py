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


paddle.enable_static()
seed = 33
main_program = paddle.static.default_main_program()
startup_program = paddle.static.default_startup_program()
main_program.random_seed = seed
startup_program.random_seed = seed
paddle.seed(seed)
np.random.seed(seed)
startup_program.random_seed = seed


@pytest.mark.range
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_range():
    with paddle.utils.unique_name.guard():
        with paddle.static.program_guard(
            main_program=main_program, startup_program=startup_program
        ):
            data = paddle.static.data(
                name='data', shape=[6, 128], dtype='int64'
            )
            data.stop_gradient = False
            out = paddle.arange(1, 10, 1, 'int32')
            fetch_list = [out.name]
            print(fetch_list)
            print("start to debug run")
            exec = paddle.static.Executor(paddle.CPUPlace())
            exec.run(startup_program)
            exe = paddle.static.Executor('gcu')
            output_gcu = exe.run(main_program, fetch_list=fetch_list)

            output = exec.run(main_program, fetch_list=fetch_list)
            print("output num:", len(output))
            for i in range(len(output)):
                print("------------")
                print(
                    np.allclose(output[i], output_gcu[i], atol=1e-5, rtol=1e-5)
                )
                print("---- cpu result ----")
                print(fetch_list[i], output[i])
                print("---- gcu result ----")
                print(fetch_list[i], output_gcu[i])


test_range()
