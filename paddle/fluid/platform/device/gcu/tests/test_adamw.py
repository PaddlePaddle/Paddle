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
        # assert out.shape == exp.shape
        assert out.dtype == exp.dtype
        if exp.dtype == np.float32:
            diff = np.abs(out - exp)
            err = np.ones(shape=exp.shape) * 1e-5
            assert np.all(diff < err)
        elif exp.dtype in [np.bool_, np.int64]:
            assert np.all(out == exp)


@pytest.mark.adamw
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_adamw():
    paddle.enable_static()
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    main_program.random_seed = 33
    startup_program.random_seed = 33

    gcu_options = paddle.static.GcuOptions()
    gcu_options.set_global_option('gcu_weights_sync_mode', 'manually')
    with paddle.utils.unique_name.guard():
        with paddle.static.program_guard(
            main_program=main_program, startup_program=startup_program
        ):
            data = paddle.static.data(
                name='data', shape=[4, 2, 4, 4], dtype='float32'
            )
            data.stop_gradient = False
            conv = paddle.nn.Conv2D(
                2,
                2,
                (3, 3),
                bias_attr=False,
                weight_attr=paddle.nn.initializer.Constant(value=0.5),
            )
            moment_optimizer = paddle.optimizer.AdamW(
                learning_rate=0.001,
                parameters=[conv.weight],
                beta1=0.9,
                beta2=0.999,
                weight_decay=0.01,
            )
            out = conv(data)
            loss = paddle.mean(out)
            moment_optimizer.minimize(loss)

            input = (
                np.array(
                    [
                        -0.01433557,
                        0.5931416,
                        -0.43119228,
                        0.38800803,
                        -0.4111048,
                        0.5461155,
                        -0.2005271,
                        -0.09387056,
                        -0.6605675,
                        0.00123398,
                        0.41237578,
                        -0.78077316,
                        0.5132639,
                        0.35805455,
                        0.4673452,
                        -0.07142179,
                        0.14276928,
                        0.5966507,
                        -0.71268463,
                        0.7278599,
                        0.62913686,
                        -0.7392282,
                        0.11245467,
                        -0.34481817,
                        -0.8540824,
                        -0.14133406,
                        -0.37151954,
                        -0.03198902,
                        0.20855112,
                        0.17116332,
                        -0.15859579,
                        -0.33735827,
                    ]
                )
                .reshape(1, 2, 4, 4)
                .astype(np.float32)
            )
            input = np.repeat(input, 4, axis=0)  # [4,2,4,4]

            # get cpu result
            cpu_place = paddle.CPUPlace()
            cpu_exe = paddle.static.Executor(cpu_place)
            cpu_exe.run(startup_program)
            eval_program = main_program.clone(for_test=True)
            cpu_results = []
            gcu_results = []
            # 1. firstly run 5 times on cpu
            for i in range(5):
                cpu_res = cpu_exe.run(
                    main_program,
                    feed={'data': input},
                    fetch_list=[loss],
                    return_numpy=True,
                )
                cpu_results.append(cpu_res[0])
            cpu_eval_res = cpu_exe.run(
                eval_program,
                feed={'data': input},
                fetch_list=[loss],
                return_numpy=True,
            )

            # get gcu result
            # cpu_exe.run(startup_program)
            gcu_exe = paddle.static.Executor('gcu')
            gcu_exe.run(startup_program)
            cpu_exe.run(startup_program)
            # 2. secondly run 5 times on gcu
            for i in range(5):
                gcu_res = gcu_exe.run(
                    main_program,
                    feed={'data': input},
                    fetch_list=[loss],
                    return_numpy=True,
                )
                gcu_results.append(gcu_res[0])

            gcu_exe.synchronize()
            gcu_eval_res = gcu_exe.run(
                eval_program,
                feed={'data': input},
                fetch_list=[loss],
                return_numpy=True,
            )
            gcu_exe.synchronize()

            print("eval loss")
            print(gcu_eval_res)
            print("-------")
            print(cpu_eval_res)

            print("\ntrain loss")
            print(gcu_results)
            print("-------")
            print(cpu_results)

            compare(cpu_results, gcu_results)
            compare(cpu_eval_res, gcu_eval_res)

    gcu_options.clear_all_options()
