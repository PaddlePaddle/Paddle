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
from paddle import nn

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


def cosSim(x, y):
    '''
    余弦相似度
    '''
    tmp = np.sum(x * y)
    non = np.linalg.norm(x) * np.linalg.norm(y)
    return np.round(tmp / (float(non) + 1e-30), decimals=9)


@pytest.mark.embedding
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_embedding():
    with paddle.utils.unique_name.guard():
        with paddle.static.program_guard(
            main_program=main_program, startup_program=startup_program
        ):
            data = paddle.static.data(
                name='data', shape=[6, 128], dtype='int64'
            )
            embed = nn.Embedding(128, 16)
            out = embed(data)
            loss = paddle.mean(out)
            fetch_list = [loss.name]
            g = paddle.static.gradients(loss, [data, embed.weight])
            fetch_list.append(g[1].name)
            fetch_list.append('embedding_1.tmp_0')
            fetch_list.append('mean_0.tmp_0@GRAD')
            fetch_list.append('embedding_1.tmp_0@GRAD')
            fetch_list.append('embedding_0.w_0@GRAD')
            print(fetch_list)
            print("start to debug run")
            x = np.random.randint(0, 128, (6, 128)).astype('int64')
            exec = paddle.static.Executor(paddle.CPUPlace())
            exec.run(startup_program)
            output = exec.run(
                main_program, feed={"data": x}, fetch_list=fetch_list
            )
            exe = paddle.static.Executor('gcu')
            output_gcu = exe.run(
                main_program, feed={"data": x}, fetch_list=fetch_list
            )
            output = exec.run(
                main_program, feed={"data": x}, fetch_list=fetch_list
            )
            print("output num:", len(output))
            for i in range(len(output)):
                print("------------------------------------------")
                print(cosSim(output[i], output_gcu[i]))
                print(
                    np.allclose(output[i], output_gcu[i], atol=1e-5, rtol=1e-5)
                )
