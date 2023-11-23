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
from api_base import ApiBase

import paddle

paddle.enable_static()


@pytest.mark.index_select
@pytest.mark.filterwarnings('ignore::UserWarning')
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_index_select_with_grad_rank3_axis_1():
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    with paddle.utils.unique_name.guard():
        with paddle.static.program_guard(
            main_program=main_program, startup_program=startup_program
        ):
            data = paddle.static.data(
                name='data', shape=[2, 3, 4], dtype='float64'
            )
            index = paddle.static.data(name='index', shape=[2], dtype='int32')
            data.stop_gradient = False
            index.stop_gradient = True
            loss = paddle.index_select(x=data, index=index, axis=1)
            fetch_list = [loss.name]
            g = paddle.static.gradients(loss, [data, index])
            print(g)
            fetch_list.append(g[0].name)
            print(fetch_list)
            print(main_program)
            feed_data = np.arange(0, 24).astype('float64')
            feed_data = np.reshape(feed_data, [2, 3, 4])
            feed_index = np.array([1, 1], dtype='int32')
            cpu_place = paddle.CPUPlace()
            exe_cpu = paddle.static.Executor(cpu_place)
            exe_cpu.run(startup_program)
            output_cpu = exe_cpu.run(
                main_program,
                feed={"data": feed_data, "index": feed_index},
                fetch_list=fetch_list,
            )

            exe_gcu = paddle.static.Executor('gcu')
            output_gcu = exe_gcu.run(
                main_program,
                feed={"data": feed_data, "index": feed_index},
                fetch_list=fetch_list,
            )

            print("cpu output num:", len(output_cpu))
            for i in range(len(output_cpu)):
                print("------------")
                print(fetch_list[i], output_gcu[i].reshape(-1)[:100])
                print(fetch_list[i], output_cpu[i].reshape(-1)[:100])
                assert np.allclose(
                    output_gcu[i], output_cpu[i], atol=1e-5, rtol=1e-5
                )


def test_index_select_with_grad_rank4_axis0():
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    with paddle.utils.unique_name.guard():
        with paddle.static.program_guard(
            main_program=main_program, startup_program=startup_program
        ):
            data = paddle.static.data(
                name='data', shape=[2, 3, 2, 2], dtype='int64'
            )
            index = paddle.static.data(name='index', shape=[2], dtype='int32')
            data.stop_gradient = False
            index.stop_gradient = True
            loss = paddle.index_select(x=data, index=index, axis=0)
            fetch_list = [loss.name]
            g = paddle.static.gradients(loss, [data, index])
            print(g)
            fetch_list.append(g[0].name)
            print(fetch_list)
            print(main_program)
            feed_data = np.arange(0, 24).astype('int64')
            feed_data = np.reshape(feed_data, [2, 3, 2, 2])
            feed_index = np.array([1, 1], dtype='int32')
            cpu_place = paddle.CPUPlace()
            exe_cpu = paddle.static.Executor(cpu_place)
            exe_cpu.run(startup_program)
            output_cpu = exe_cpu.run(
                main_program,
                feed={"data": feed_data, "index": feed_index},
                fetch_list=fetch_list,
            )

            exe_gcu = paddle.static.Executor('gcu')
            output_gcu = exe_gcu.run(
                main_program,
                feed={"data": feed_data, "index": feed_index},
                fetch_list=fetch_list,
            )

            print("cpu output num:", len(output_cpu))
            for i in range(len(output_cpu)):
                print("------------")
                print(fetch_list[i], output_gcu[i].reshape(-1)[:100])
                print(fetch_list[i], output_cpu[i].reshape(-1)[:100])
                assert np.allclose(
                    output_gcu[i], output_cpu[i], atol=1e-5, rtol=1e-5
                )


def test_index_select_with_grad_rank4_axis2():
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    with paddle.utils.unique_name.guard():
        with paddle.static.program_guard(
            main_program=main_program, startup_program=startup_program
        ):
            data = paddle.static.data(
                name='data', shape=[2, 3, 2, 2], dtype='float32'
            )
            index = paddle.static.data(name='index', shape=[2], dtype='int32')
            data.stop_gradient = False
            index.stop_gradient = True
            loss = paddle.index_select(x=data, index=index, axis=2)
            fetch_list = [loss.name]
            g = paddle.static.gradients(loss, [data, index])
            print(g)
            fetch_list.append(g[0].name)
            print(fetch_list)
            print(main_program)
            feed_data = np.random.uniform(0.1, 1.0, (2, 3, 2, 2)).astype(
                'float32'
            )
            feed_index = np.array([1, 1], dtype='int32')
            cpu_place = paddle.CPUPlace()
            exe_cpu = paddle.static.Executor(cpu_place)
            exe_cpu.run(startup_program)
            output_cpu = exe_cpu.run(
                main_program,
                feed={"data": feed_data, "index": feed_index},
                fetch_list=fetch_list,
            )

            exe_gcu = paddle.static.Executor('gcu')
            output_gcu = exe_gcu.run(
                main_program,
                feed={"data": feed_data, "index": feed_index},
                fetch_list=fetch_list,
            )

            print("cpu output num:", len(output_cpu))
            for i in range(len(output_cpu)):
                print("------------")
                print(fetch_list[i], output_gcu[i].reshape(-1)[:100])
                print(fetch_list[i], output_cpu[i].reshape(-1)[:100])
                assert np.allclose(
                    output_gcu[i], output_cpu[i], atol=1e-5, rtol=1e-5
                )


def test_index_select_with_grad_rank4_axis2_float64():
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    with paddle.utils.unique_name.guard():
        with paddle.static.program_guard(
            main_program=main_program, startup_program=startup_program
        ):
            data = paddle.static.data(
                name='data', shape=[2, 3, 2, 2], dtype='float64'
            )
            index = paddle.static.data(name='index', shape=[2], dtype='int32')
            data.stop_gradient = False
            index.stop_gradient = True
            loss = paddle.index_select(x=data, index=index, axis=2)
            fetch_list = [loss.name]
            g = paddle.static.gradients(loss, [data, index])
            print(g)
            fetch_list.append(g[0].name)
            print(fetch_list)
            print(main_program)
            feed_data = np.random.uniform(0.1, 1.0, (2, 3, 2, 2)).astype(
                'float64'
            )
            feed_index = np.array([1, 1], dtype='int32')
            cpu_place = paddle.CPUPlace()
            exe_cpu = paddle.static.Executor(cpu_place)
            exe_cpu.run(startup_program)
            output_cpu = exe_cpu.run(
                main_program,
                feed={"data": feed_data, "index": feed_index},
                fetch_list=fetch_list,
            )

            exe_gcu = paddle.static.Executor('gcu')
            output_gcu = exe_gcu.run(
                main_program,
                feed={"data": feed_data, "index": feed_index},
                fetch_list=fetch_list,
            )

            print("cpu output num:", len(output_cpu))
            for i in range(len(output_cpu)):
                print("------------")
                print(fetch_list[i], output_gcu[i].reshape(-1)[:100])
                print(fetch_list[i], output_cpu[i].reshape(-1)[:100])
                assert np.allclose(
                    output_gcu[i], output_cpu[i], atol=1e-5, rtol=1e-5
                )


def test_index_select_without_grad_axis0():
    test = ApiBase(
        func=paddle.index_select,
        feed_names=['data', 'index'],
        feed_shapes=[[3, 4], [3]],
        feed_dtypes=['int64', 'int32'],
        is_train=False,
    )
    data = np.arange(0, 12).astype('int64')
    data = np.reshape(data, [3, 4])
    index = np.array([0, 1, 1], dtype='int32')
    test.run(feed=[data, index], axis=0)


def test_index_select_without_grad_axis1():
    test = ApiBase(
        func=paddle.index_select,
        feed_names=['data', 'index'],
        feed_shapes=[[3, 4], [3]],
        feed_dtypes=['float64', 'int32'],
        is_train=False,
    )
    data = np.array(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]
    ).astype('float64')
    index = np.array([0, 1, 1], dtype='int32')
    test.run(feed=[data, index], axis=1)
