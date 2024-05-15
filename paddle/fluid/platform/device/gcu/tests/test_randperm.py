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

# paddle.seed(2021)

# test = ApiBase(func=paddle.randperm,
#                feed_names=[],
#                feed_shapes=[])

# @pytest.mark.randperm
# @pytest.mark.filterwarnings('ignore::UserWarning')
# def test_randperm():
#     test.run(feed=[], n = 300)


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


paddle.enable_static()
seed = 2021
main_program = paddle.static.default_main_program()
startup_program = paddle.static.default_startup_program()
main_program.random_seed = seed
startup_program.random_seed = seed
paddle.seed(seed)
np.random.seed(seed)
startup_program.random_seed = seed


@pytest.mark.randperm
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_truncated_gaussian_random():
    with paddle.utils.unique_name.guard():
        with paddle.static.program_guard(
            main_program=main_program, startup_program=startup_program
        ):
            block = main_program.global_block()
            vout = block.create_var(name="Out")
            op = block.append_op(
                type="randperm",
                outputs={"Out": vout},
                attrs={"dtype": 3, "n": 30000},
            )
            op.desc.infer_var_type(block.desc)
            op.desc.infer_shape(block.desc)

            fetch_list = []
            for var_name in ["Out"]:
                fetch_list.append(block.var(var_name))

            print(fetch_list)
            print("start to debug run")
            exe_cpu = paddle.static.Executor(paddle.CPUPlace())
            exe_cpu.run(startup_program)
            output_cpu = exe_cpu.run(main_program, fetch_list=fetch_list)

            exe_gcu = paddle.static.Executor('gcu')
            output_gcu = exe_gcu.run(main_program, fetch_list=fetch_list)

            # print(f'------cpu: {len(output_cpu)}')
            # print(f'------gcu: {len(output_gcu)}')
            print("output num:", len(output_cpu))
            for i in range(len(output_cpu)):
                print("------------")
                print(
                    np.allclose(
                        output_cpu[i][0:10],
                        output_gcu[i][0:10],
                        atol=1e-5,
                        rtol=1e-5,
                    )
                )
                print("---- cpu result ----")
                print(fetch_list[i], output_gcu[i][0:10])
                print("---- gcu result ----")
                print(fetch_list[i], output_gcu[i][0:10])
