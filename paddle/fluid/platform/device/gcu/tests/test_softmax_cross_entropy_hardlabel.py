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


@pytest.mark.softmax_with_cross_entropy
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_softmax_with_cross_entropy():
    with paddle.utils.unique_name.guard():
        # soft_label=False  use_softmax=true
        with paddle.static.program_guard(
            main_program=main_program, startup_program=startup_program
        ):
            data = paddle.static.data(
                name='data', shape=[48, 10], dtype='float32'
            )
            label = paddle.static.data(name='label', shape=[48], dtype='int32')
            data.stop_gradient = False
            label.stop_gradient = True
            loss = paddle.nn.loss.CrossEntropyLoss()
            out = loss(data, label)

            fetch_list = [out.name]
            g = paddle.static.gradients(out, [data, label])
            fetch_list.append(g[0].name)
            print(fetch_list)
            print(main_program)
            print("start to debug run")
            exe = paddle.static.Executor('gcu')
            x = np.random.randn(48, 10).astype('float32')
            y = np.random.randint(0, 10, (48)).astype('int32')
            output_dtu = exe.run(
                main_program,
                feed={"data": x, "label": y},
                fetch_list=fetch_list,
                return_numpy=True,
            )
            exec = paddle.static.Executor(paddle.CPUPlace())
            exec.run(startup_program)
            output_cpu = exec.run(
                main_program,
                feed={"data": x, "label": y},
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
