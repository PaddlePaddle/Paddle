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

paddle.enable_static()
np.random.seed(33)
main_program = paddle.static.Program()
startup_program = paddle.static.Program()
main_program.random_seed = 33
startup_program.random_seed = 33


def compare(cpu_res, gcu_res):
    assert len(cpu_res) == len(gcu_res)
    for i in range(len(cpu_res)):
        out = gcu_res[i]
        exp = cpu_res[i]
        assert out.shape == exp.shape
        assert out.dtype == exp.dtype
        if exp.dtype == np.float32:
            diff = np.abs(out - exp)
            err = np.maximum(
                (np.ones(shape=exp.shape).astype(np.float32) * 1.0e-5),
                (1.0e-6 * np.maximum(np.abs(out), np.abs(exp))),
            )
            assert np.all(diff <= err)
        elif exp.dtype in [np.bool_, np.int64]:
            assert np.all(out == exp)


@pytest.mark.gather
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_gather():
    with paddle.utils.unique_name.guard():
        with paddle.static.program_guard(
            main_program=main_program, startup_program=startup_program
        ):

            data = paddle.static.data(
                name='data', shape=[2, 3, 4], dtype='float32'
            )
            index = paddle.static.data(name='index', shape=[2], dtype='int64')
            data.stop_gradient = False
            index.stop_gradient = True

            out = paddle.gather(data, index, axis=1)
            loss = paddle.mean(out)

            feed_list = ["data", "index"]
            fetch_list = [loss.name]
            g = paddle.static.gradients(loss, data)
            for grad in g:
                fetch_list.append(grad.name)

            print("start to run compare")
            x = np.random.uniform(0.1, 1.0, (2, 3, 4)).astype('float32')
            index = np.array([0, 1]).astype('int64')

            cpu_exe = paddle.static.Executor(paddle.CPUPlace())
            cpu_res = cpu_exe.run(
                main_program,
                feed={"data": x, "index": index},
                fetch_list=fetch_list,
                return_numpy=True,
            )

            gcu_exe = paddle.static.Executor('gcu')
            gcu_res = gcu_exe.run(
                main_program,
                feed={"data": x, "index": index},
                fetch_list=fetch_list,
                return_numpy=True,
            )

            logging.info(
                'result number: '
                + str(len(fetch_list))
                + ", result names: "
                + str(fetch_list)
            )
            logging.info('[cpu result]')
            logging.info(cpu_res)
            logging.info('[gcu result]')
            logging.info(gcu_res)

            compare(cpu_res, gcu_res)
