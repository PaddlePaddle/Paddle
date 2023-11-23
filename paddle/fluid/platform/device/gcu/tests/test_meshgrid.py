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
np.random.seed(33)
main_program = paddle.static.Program()
startup_program = paddle.static.Program()
main_program.random_seed = 33
startup_program.random_seed = 33


@pytest.mark.meshgrid
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_meshgrid():
    with paddle.utils.unique_name.guard():
        with paddle.static.program_guard(
            main_program=main_program, startup_program=startup_program
        ):
            data1 = paddle.static.data(
                name='data1', shape=[100], dtype='float32'
            )
            data1.stop_gradient = True
            data2 = paddle.static.data(
                name='data2', shape=[200], dtype='float32'
            )
            data2.stop_gradient = True
            data3 = paddle.static.data(
                name='data3', shape=[300], dtype='float32'
            )
            data3.stop_gradient = True
            out = paddle.meshgrid(data1, data2, data3)
            fetch_list = [out[0].name]
            fetch_list.append(out[1].name)
            fetch_list.append(out[2].name)
            print(fetch_list)
            print(main_program)
            print("start to debug run")
            exe = paddle.static.Executor('gcu')
            x = np.random.randn(100).astype('float32')
            y = np.random.randn(200).astype('float32')
            z = np.random.randn(300).astype('float32')
            output_dtu = exe.run(
                main_program,
                feed={"data1": x, "data2": y, "data3": z},
                fetch_list=fetch_list,
                return_numpy=True,
            )
            exec = paddle.static.Executor(paddle.CPUPlace())
            exec.run(startup_program)
            output_cpu = exec.run(
                main_program,
                feed={"data1": x, "data2": y, "data3": z},
                fetch_list=fetch_list,
                return_numpy=True,
            )
            print("output num:", len(output_dtu))
            for i in range(len(output_dtu)):
                print("------------")
                print(
                    np.allclose(
                        output_dtu[i], output_cpu[i], atol=1e-5, rtol=1e-5
                    )
                )
                print(fetch_list[i], output_dtu[i].reshape(-1))
                print(fetch_list[i], output_cpu[i].reshape(-1))
