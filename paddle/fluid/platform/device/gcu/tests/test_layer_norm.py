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

import paddle

paddle.enable_static()
paddle.seed(33)
np.random.seed(33)


def cosSim(x, y):
    '''
    余弦相似度
    '''
    tmp = np.sum(x * y)
    non = np.linalg.norm(x) * np.linalg.norm(y)
    return np.round(tmp / (float(non) + 1e-30), decimals=9)


def test_layer_norm_0():
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    main_program.random_seed = 33
    startup_program.random_seed = 33
    with paddle.utils.unique_name.guard():
        with paddle.static.program_guard(
            main_program=main_program, startup_program=startup_program
        ):
            data = paddle.static.data(
                name='data', shape=[8, 256, 2048], dtype='float32'
            )

            weight3 = paddle.static.data(
                name='weight3', shape=[2048], dtype='float32'
            )
            bias3 = paddle.static.data(
                name='bias3', shape=[2048], dtype='float32'
            )
            data.stop_gradient = False
            weight3.stop_gradient = False
            bias3.stop_gradient = False
            out = paddle.nn.functional.layer_norm(
                data, 2048, weight=weight3, bias=bias3
            )
            loss = paddle.mean(out)
            fetch_list = ["data"]
            fetch_list = [loss.name, out.name]
            g = paddle.static.gradients(loss, [data, weight3, bias3])
            for grad in g:
                fetch_list.append(grad.name)
            print(fetch_list)
            print(main_program)
            print("start to debug run")
            exe = paddle.static.Executor('gcu')
            x = np.random.random(size=(8, 256, 2048)).astype("float32")
            x6 = np.random.random(size=(2048)).astype("float32")
            x7 = np.random.random(size=(2048)).astype("float32")
            exec = paddle.static.Executor(paddle.CPUPlace())
            exec.run(startup_program)
            output = exe.run(
                main_program,
                feed={"data": x, "weight3": x6, "bias3": x7},
                fetch_list=fetch_list,
            )
            output_cpu = exec.run(
                main_program,
                feed={"data": x, "weight3": x6, "bias3": x7},
                fetch_list=fetch_list,
            )
            print("output num:", len(output))
            for i in range(len(output)):
                print("-----------------------------")
                print(f"cossim is: {cosSim(output[i], output_cpu[i])}")
                print(
                    np.allclose(output[i], output_cpu[i], atol=1e-5, rtol=1e-5)
                )
                print(fetch_list[i], output[i].reshape(-1))
                print(fetch_list[i], output_cpu[i].reshape(-1))
                print("-------------------------------")


def test_layer_norm_1():
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    main_program.random_seed = 33
    startup_program.random_seed = 33
    with paddle.utils.unique_name.guard():
        with paddle.static.program_guard(
            main_program=main_program, startup_program=startup_program
        ):
            data = paddle.static.data(
                name='data', shape=[2, 2, 2, 3], dtype='float32'
            )
            weight3 = paddle.static.data(
                name='weight3', shape=[3], dtype='float32'
            )
            bias3 = paddle.static.data(name='bias3', shape=[3], dtype='float32')
            data.stop_gradient = False
            weight3.stop_gradient = False
            bias3.stop_gradient = False
            out = paddle.nn.functional.layer_norm(
                data, 3, weight=weight3, bias=bias3
            )
            loss = paddle.mean(out)
            fetch_list = ["data"]
            fetch_list = [loss.name, out.name]
            g = paddle.static.gradients(loss, [data, weight3, bias3])
            for grad in g:
                fetch_list.append(grad.name)
            print(fetch_list)
            print(main_program)
            print("start to debug run")
            exe = paddle.static.Executor('gcu')
            x = np.random.random(size=(2, 2, 2, 3)).astype("float32")
            x6 = np.random.random(size=(3)).astype("float32")
            x7 = np.random.random(size=(3)).astype("float32")
            exec = paddle.static.Executor(paddle.CPUPlace())
            exec.run(startup_program)
            output = exe.run(
                main_program,
                feed={"data": x, "weight3": x6, "bias3": x7},
                fetch_list=fetch_list,
            )
            output_cpu = exec.run(
                main_program,
                feed={"data": x, "weight3": x6, "bias3": x7},
                fetch_list=fetch_list,
            )
            print("output num:", len(output))
            for i in range(len(output)):
                print("-----------------------------")
                print(f"cossim is: {cosSim(output[i], output_cpu[i])}")
                print(
                    np.allclose(output[i], output_cpu[i], atol=1e-5, rtol=1e-5)
                )
                print(fetch_list[i], output[i].reshape(-1))
                print(fetch_list[i], output_cpu[i].reshape(-1))
                print("-------------------------------")


def test_layer_norm_2():
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    main_program.random_seed = 33
    startup_program.random_seed = 33
    with paddle.utils.unique_name.guard():
        with paddle.static.program_guard(
            main_program=main_program, startup_program=startup_program
        ):
            data = paddle.static.data(
                name='data', shape=[2, 2, 2, 3], dtype='float32'
            )
            data.stop_gradient = False
            out = paddle.nn.functional.layer_norm(data, 3)
            loss = paddle.mean(out)
            fetch_list = ["data"]
            fetch_list = [loss.name, out.name]
            g = paddle.static.gradients(loss, [data])
            for grad in g:
                fetch_list.append(grad.name)
            print(fetch_list)
            print(main_program)
            print("start to debug run")
            exe = paddle.static.Executor('gcu')
            x = np.random.random(size=(2, 2, 2, 3)).astype("float32")
            exec = paddle.static.Executor(paddle.CPUPlace())
            exec.run(startup_program)
            output = exe.run(
                main_program, feed={"data": x}, fetch_list=fetch_list
            )
            output_cpu = exec.run(
                main_program, feed={"data": x}, fetch_list=fetch_list
            )
            print("output num:", len(output))
            for i in range(len(output)):
                print("-----------------------------")
                print(f"cossim is: {cosSim(output[i], output_cpu[i])}")
                print(
                    np.allclose(output[i], output_cpu[i], atol=1e-5, rtol=1e-5)
                )
                print(fetch_list[i], output[i].reshape(-1))
                print(fetch_list[i], output_cpu[i].reshape(-1))
                print("-------------------------------")
