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
        assert out.shape == exp.shape
        assert out.dtype == exp.dtype
        if exp.dtype == np.float32:
            diff = np.abs(out - exp)
            err = np.ones(shape=exp.shape) * 1e-5
            assert np.all(diff < err)
        elif exp.dtype in [np.bool_, np.int64]:
            assert np.all(out == exp)


paddle.enable_static()


@pytest.mark.merged_momentum
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_merged_momentum1():
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    main_program.random_seed = 33
    startup_program.random_seed = 33
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
            moment_optimizer = paddle.optimizer.Momentum(
                learning_rate=0.001,
                momentum=0.9,
                weight_decay=0.001,
                grad_clip=None,
                multi_precision=False,
                use_multi_tensor=True,
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

            # # get cpu result
            cpu_place = paddle.CPUPlace()
            cpu_exe = paddle.static.Executor(cpu_place)
            cpu_exe.run(startup_program)
            # # 1. firstly run 5 times on cpu
            for i in range(5):
                cpu_res = cpu_exe.run(
                    main_program,
                    feed={'data': input},
                    fetch_list=[out],
                    return_numpy=True,
                )
            # get gcu result
            cpu_exe.run(startup_program)
            gcu_exe = paddle.static.Executor('gcu')
            # 2. secondly run 5 times on gcu
            for i in range(5):
                gcu_res = gcu_exe.run(
                    main_program,
                    feed={'data': input},
                    fetch_list=[out],
                    return_numpy=True,
                )

            print(cpu_res)
            print(gcu_res)
            compare(cpu_res, gcu_res)


@pytest.mark.merged_momentum
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_merged_momentum2():
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    main_program.random_seed = 33
    startup_program.random_seed = 33
    with paddle.utils.unique_name.guard():
        with paddle.static.program_guard(
            main_program=main_program, startup_program=startup_program
        ):
            data = paddle.static.data(
                name='data', shape=[4, 10], dtype='float32'
            )
            data.stop_gradient = False
            linear1 = paddle.nn.Linear(10, 4)
            linear2 = paddle.nn.Linear(4, 10)
            moment_optimizer = paddle.optimizer.Momentum(
                learning_rate=0.001,
                momentum=0.9,
                weight_decay=0.001,
                grad_clip=None,
                multi_precision=False,
                use_multi_tensor=True,
            )
            out1 = linear1(data)
            out2 = linear2(out1)
            loss = paddle.mean(out2)
            moment_optimizer.minimize(loss)
            input = (
                np.array(
                    [
                        -1.1045314,
                        -0.24136452,
                        1.8703375,
                        -1.1941437,
                        -1.2117761,
                        0.08814529,
                        0.24922983,
                        -0.46522725,
                        -0.9125267,
                        1.1357822,
                        -0.4180504,
                        1.6631924,
                        -2.3349247,
                        0.12183782,
                        -0.20258728,
                        0.96573776,
                        1.0522023,
                        -1.1099466,
                        2.1103911,
                        -0.04271242,
                        -0.8485446,
                        -2.9438388,
                        -0.46665922,
                        -1.393594,
                        1.440952,
                        -0.59755176,
                        1.4488506,
                        0.10697991,
                        -2.7158506,
                        -0.6388879,
                        1.4546522,
                        -0.00309283,
                        2.5021417,
                        0.77642894,
                        -0.36866572,
                        -0.08921445,
                        1.7057414,
                        -0.8443171,
                        0.8232965,
                        -0.34043625,
                    ]
                )
                .reshape(4, 10)
                .astype('float32')
            )

            # # get cpu result
            cpu_place = paddle.CPUPlace()
            cpu_exe = paddle.static.Executor(cpu_place)
            cpu_exe.run(startup_program)
            # # 1. firstly run 5 times on cpu
            for i in range(5):
                cpu_res = cpu_exe.run(
                    main_program,
                    feed={'data': input},
                    fetch_list=[out2, linear1.weight, linear2.weight],
                    return_numpy=True,
                )
            # get gcu result
            gcu_exe = paddle.static.Executor('gcu')
            # 2. secondly run 5 times on gcu
            for i in range(5):
                gcu_res = gcu_exe.run(
                    main_program,
                    feed={'data': input},
                    fetch_list=[out2, linear1.weight, linear2.weight],
                    return_numpy=True,
                )

            cpu_out = np.array(
                [
                    1.0265418,
                    0.5102334,
                    0.12734565,
                    0.5299988,
                    -0.1838639,
                    -1.1992538,
                    -0.7738419,
                    0.3769602,
                    -0.74879736,
                    1.0831891,
                    -1.114617,
                    0.23838653,
                    0.72727954,
                    -1.7606783,
                    -0.3703414,
                    1.955406,
                    2.3179526,
                    -0.5923667,
                    0.7557231,
                    -1.6739687,
                    0.33690774,
                    -0.521091,
                    0.05446241,
                    -1.8037113,
                    -1.1241956,
                    -0.5060293,
                    0.68719923,
                    0.7576096,
                    -0.1773548,
                    0.09980698,
                    -1.0154629,
                    -1.2353334,
                    -0.3856181,
                    -1.0614029,
                    -0.02710527,
                    0.612765,
                    0.66890776,
                    0.14932385,
                    0.83822167,
                    -1.1086351,
                ]
            ).astype(np.float32)
            cpu_weight_out1 = np.array(
                [
                    -0.29009148,
                    -0.33639044,
                    0.20157081,
                    -0.37743866,
                    0.2099683,
                    -0.23151858,
                    0.42595392,
                    0.27582368,
                    0.35091934,
                    -0.6041192,
                    -0.05378386,
                    0.16939516,
                    0.3111113,
                    -0.4470445,
                    -0.12960136,
                    0.15088047,
                    0.5485424,
                    0.01457348,
                    0.06065221,
                    -0.27163178,
                    -0.4076804,
                    0.21849234,
                    0.05038582,
                    0.5045542,
                    0.45335606,
                    0.6368949,
                    0.07995939,
                    -0.47953638,
                    0.2474267,
                    -0.37185845,
                    0.33528,
                    0.4238264,
                    0.30056754,
                    0.52559346,
                    0.29137352,
                    -0.5504425,
                    -0.22246248,
                    0.08179406,
                    -0.20216773,
                    0.19925694,
                ]
            ).astype(np.float32)
            cpu_weight_out2 = np.array(
                [
                    -0.2905963,
                    -0.3375837,
                    0.19980706,
                    -0.37942642,
                    0.20970437,
                    -0.23292822,
                    0.4235949,
                    0.27309185,
                    0.34943032,
                    -0.6044281,
                    -0.05627537,
                    0.16728765,
                    0.3077519,
                    -0.45167306,
                    -0.13528164,
                    0.14478722,
                    0.54464984,
                    0.01042394,
                    0.05628973,
                    -0.2760779,
                    -0.40645662,
                    0.2199919,
                    0.05211389,
                    0.5063719,
                    0.45296183,
                    0.63984793,
                    0.08568661,
                    -0.47272,
                    0.24971081,
                    -0.37131155,
                    0.3355717,
                    0.42355287,
                    0.30339584,
                    0.5279,
                    0.29324752,
                    -0.5487383,
                    -0.21995406,
                    0.08438794,
                    -0.19950306,
                    0.20194945,
                ]
            ).astype(np.float32)
            cpu_res = [cpu_out, cpu_weight_out1, cpu_weight_out2]
            gcu_res = [
                gcu_res[0].reshape(-1),
                gcu_res[1].reshape(-1),
                gcu_res[2].reshape(-1),
            ]
            print(cpu_res)
            print(gcu_res)
            compare(cpu_res, gcu_res)
