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

import numpy as np
import pytest

import paddle


@pytest.mark.cross_entropy2
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_cross_entropy1():
    paddle.enable_static()
    np.random.seed(33)
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    main_program.random_seed = 33
    startup_program.random_seed = 33
    with paddle.utils.unique_name.guard():
        with paddle.static.program_guard(
            main_program=main_program, startup_program=startup_program
        ):
            data = paddle.static.data(
                name='data', shape=[4, 1000], dtype='float32'
            )
            label = paddle.static.data(
                name='label', shape=[4, 1], dtype='int64'
            )
            data.stop_gradient = False
            label.stop_gradient = True
            loss = paddle.nn.functional.cross_entropy(data, label)
            feed_list = ["data", "label"]
            fetch_list = [loss.name]
            g = paddle.static.gradients(loss, [data, label])
            print(g)
            fetch_list.append(g[0].name)
            print(fetch_list)
            print(main_program)
            print("start to debug run")
            exe = paddle.static.Executor('gcu')
            x = np.random.uniform(0.1, 1.0, (4, 1000)).astype('float32')
            x2 = np.random.randint(0, 1000, (4, 1)).astype('int64')
            output = exe.run(
                main_program,
                feed={"data": x, "label": x2},
                fetch_list=fetch_list,
            )
            exec = paddle.static.Executor(paddle.CPUPlace())
            exec.run(startup_program)
            output_cpu = exec.run(
                main_program,
                feed={"data": x, "label": x2},
                fetch_list=fetch_list,
            )
            print("output num:", len(output_cpu))
            for i in range(len(output_cpu)):
                print("------------")
                print(
                    np.allclose(output[i], output_cpu[i], atol=1e-5, rtol=1e-5)
                )
                print(fetch_list[i], output[i].reshape(-1))
                print(fetch_list[i], output_cpu[i].reshape(-1))


@pytest.mark.cross_entropy2
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_cross_entropy2():
    paddle.enable_static()
    np.random.seed(33)
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    main_program.random_seed = 33
    startup_program.random_seed = 33
    with paddle.utils.unique_name.guard():
        with paddle.static.program_guard(
            main_program=main_program, startup_program=startup_program
        ):
            data = paddle.static.data(
                name='data', shape=[4, 1000], dtype='float32'
            )
            label = paddle.static.data(name='label', shape=[4], dtype='int64')
            data.stop_gradient = False
            label.stop_gradient = True
            loss = paddle.nn.functional.cross_entropy(data, label)
            feed_list = ["data", "label"]
            fetch_list = [loss.name]
            g = paddle.static.gradients(loss, [data, label])
            print(g)
            fetch_list.append(g[0].name)
            print(fetch_list)
            print(main_program)
            print("start to debug run")
            exe = paddle.static.Executor('gcu')
            x = np.random.uniform(0.1, 1.0, (4, 1000)).astype('float32')
            x2 = np.random.randint(0, 1000, (4)).astype('int64')
            output = exe.run(
                main_program,
                feed={"data": x, "label": x2},
                fetch_list=fetch_list,
            )
            exec = paddle.static.Executor(paddle.CPUPlace())
            exec.run(startup_program)
            output_cpu = exec.run(
                main_program,
                feed={"data": x, "label": x2},
                fetch_list=fetch_list,
            )
            print("output num:", len(output_cpu))
            for i in range(len(output_cpu)):
                print("------------")
                print(
                    np.allclose(output[i], output_cpu[i], atol=1e-5, rtol=1e-5)
                )
                print(fetch_list[i], output[i].reshape(-1))
                print(fetch_list[i], output_cpu[i].reshape(-1))
