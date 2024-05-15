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


@pytest.mark.rmsprop
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_rmsprop_1():
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
            rmsprop_optimizer = paddle.optimizer.RMSProp(
                learning_rate=0.001,
                rho=0.8999999761581421,
                epsilon=0.0010000000474974513,
                momentum=0.8999999761581421,
                centered=False,
            )
            out = conv(data)
            loss = paddle.mean(out)
            rmsprop_optimizer.minimize(loss)

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
            # 1. firstly run 5 times on cpu
            for _ in range(5):
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
            for _ in range(5):
                gcu_res = gcu_exe.run(
                    main_program,
                    feed={'data': input},
                    fetch_list=[out],
                    return_numpy=True,
                )

            print(cpu_res)
            print(gcu_res)
            # the gcu is not update based on cpu's data
            compare(cpu_res, gcu_res)


@pytest.mark.rmsprop
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_rmsprop_2():
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
            rmsprop_optimizer = paddle.optimizer.RMSProp(
                learning_rate=0.001,
                rho=0.8999999761581421,
                epsilon=0.0010000000474974513,
                momentum=0.8999999761581421,
                centered=True,
            )
            out = conv(data)
            loss = paddle.mean(out)
            rmsprop_optimizer.minimize(loss)

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
            # 1. firstly run 5 times on cpu
            for _ in range(5):
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
            for _ in range(5):
                gcu_res = gcu_exe.run(
                    main_program,
                    feed={'data': input},
                    fetch_list=[out],
                    return_numpy=True,
                )

            print(cpu_res)
            print(gcu_res)
            # the gcu is not update based on cpu's data
            compare(cpu_res, gcu_res)


@pytest.mark.rmsprop
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_rmsprop_3():
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
            rmsprop_optimizer = paddle.optimizer.RMSProp(
                learning_rate=0.001,
                rho=0.8999999761581421,
                epsilon=0.0010000000474974513,
                momentum=0.8999999761581421,
                centered=False,
            )
            out1 = linear1(data)
            out2 = linear2(out1)
            loss = paddle.mean(out2)
            rmsprop_optimizer.minimize(loss)
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
            for _ in range(5):
                cpu_res = cpu_exe.run(
                    main_program,
                    feed={'data': input},
                    fetch_list=[out2, linear1.weight, linear2.weight],
                    return_numpy=True,
                )
            # get gcu result
            gcu_exe = paddle.static.Executor('gcu')
            # 2. secondly run 5 times on gcu
            for _ in range(5):
                gcu_res = gcu_exe.run(
                    main_program,
                    feed={'data': input},
                    fetch_list=[out2, linear1.weight, linear2.weight],
                    return_numpy=True,
                )

            # output for run 10 times on cpu
            cpu_out = np.array(
                [
                    0.9741428,
                    0.24238437,
                    0.00539757,
                    0.19598693,
                    -0.3735733,
                    -1.2674633,
                    -0.73275465,
                    0.4925601,
                    -0.7306787,
                    1.0342182,
                    -1.3816588,
                    -0.03476756,
                    0.46511894,
                    -2.0231047,
                    -0.62538236,
                    1.67203,
                    2.050822,
                    -0.8482914,
                    0.49533927,
                    -1.9472803,
                    0.06528003,
                    -1.0510296,
                    -0.29761246,
                    -2.4074123,
                    -1.5535182,
                    -0.8035267,
                    0.5263775,
                    0.6901291,
                    -0.3624345,
                    -0.17044547,
                    -1.1830238,
                    -1.5699614,
                    -0.6004701,
                    -1.4347831,
                    -0.28563872,
                    0.41594443,
                    0.5703205,
                    0.11799531,
                    0.72914493,
                    -1.2799954,
                ]
            ).astype(np.float32)
            cpu_weight_out1 = np.array(
                [
                    -0.30266374,
                    -0.3308738,
                    0.22902276,
                    -0.34411913,
                    0.18967265,
                    -0.22242738,
                    0.46657136,
                    0.32308167,
                    0.37179255,
                    -0.6135084,
                    -0.09520376,
                    0.12134466,
                    0.2888499,
                    -0.43702084,
                    -0.08617214,
                    0.20091636,
                    0.54380596,
                    0.01664989,
                    0.07167733,
                    -0.2577539,
                    -0.40259165,
                    0.21627395,
                    0.03858194,
                    0.4897159,
                    0.50020504,
                    0.61323315,
                    0.01446525,
                    -0.54804134,
                    0.21822475,
                    -0.35837474,
                    0.3872435,
                    0.48174936,
                    0.290989,
                    0.52981895,
                    0.31291682,
                    -0.5238916,
                    -0.22088566,
                    0.08110666,
                    -0.20588507,
                    0.19455859,
                ]
            ).astype(np.float32)
            cpu_weight_out2 = np.array(
                [
                    -0.31845367,
                    -0.36544302,
                    0.1719699,
                    -0.4072875,
                    0.18186761,
                    -0.26078326,
                    0.39576703,
                    0.2452578,
                    0.32159942,
                    -0.6322986,
                    -0.1171589,
                    0.10641333,
                    0.2468834,
                    -0.512573,
                    -0.19616845,
                    0.08391199,
                    0.48379123,
                    -0.05045684,
                    -0.00458916,
                    -0.33697057,
                    -0.36644012,
                    0.2600344,
                    0.09214943,
                    0.5464263,
                    0.49301395,
                    0.6799079,
                    0.12572354,
                    -0.4327062,
                    0.28975448,
                    -0.33129358,
                    0.39075494,
                    0.47873968,
                    0.35857773,
                    0.583091,
                    0.34842896,
                    -0.49359176,
                    -0.1647938,
                    0.13956076,
                    -0.14434199,
                    0.2571271,
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


@pytest.mark.rmsprop
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_rmsprop_4():
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
            rmsprop_optimizer = paddle.optimizer.RMSProp(
                learning_rate=0.001,
                rho=0.9,
                epsilon=0.001,
                momentum=0.9,
                centered=True,
            )
            out1 = linear1(data)
            out2 = linear2(out1)
            loss = paddle.mean(out2)
            rmsprop_optimizer.minimize(loss)
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

            # get cpu result
            cpu_place = paddle.CPUPlace()
            cpu_exe = paddle.static.Executor(cpu_place)
            cpu_exe.run(startup_program)
            # 1. firstly run 5 times on cpu
            for _ in range(5):
                cpu_res = cpu_exe.run(
                    main_program,
                    feed={'data': input},
                    fetch_list=[out2, linear1.weight, linear2.weight],
                    return_numpy=True,
                )
            # get gcu result
            gcu_exe = paddle.static.Executor('gcu')
            # 2. secondly run 5 times on gcu
            for _ in range(5):
                gcu_res = gcu_exe.run(
                    main_program,
                    feed={'data': input},
                    fetch_list=[out2, linear1.weight, linear2.weight],
                    return_numpy=True,
                )

            # output for run 10 times on cpu
            cpu_out = np.array(
                [
                    0.96528304,
                    0.2164373,
                    -0.00936548,
                    0.1579294,
                    -0.39861393,
                    -1.2712287,
                    -0.7294314,
                    0.49668187,
                    -0.7336873,
                    1.026491,
                    -1.4250073,
                    -0.08824064,
                    0.4187718,
                    -2.0819418,
                    -0.676933,
                    1.6259234,
                    2.012049,
                    -0.8840463,
                    0.45373103,
                    -1.9912609,
                    0.02877443,
                    -1.1120571,
                    -0.34223902,
                    -2.4850347,
                    -1.6124524,
                    -0.83478343,
                    0.50658906,
                    0.6724137,
                    -0.3909019,
                    -0.20588824,
                    -1.205172,
                    -1.615506,
                    -0.6303657,
                    -1.4951072,
                    -0.32864755,
                    0.39586484,
                    0.5624044,
                    0.11348046,
                    0.7135352,
                    -1.3016179,
                ]
            ).astype(np.float32)
            cpu_weight_out1 = np.array(
                [
                    -0.30279356,
                    -0.33128834,
                    0.23010014,
                    -0.3421233,
                    0.1892077,
                    -0.22308162,
                    0.4698337,
                    0.32825595,
                    0.3722935,
                    -0.6128378,
                    -0.09865596,
                    0.11592362,
                    0.28824872,
                    -0.43773213,
                    -0.08222188,
                    0.20696783,
                    0.54378885,
                    0.01649109,
                    0.07177828,
                    -0.2575234,
                    -0.40257254,
                    0.21644428,
                    0.03846361,
                    0.4894492,
                    0.5050924,
                    0.61425954,
                    0.0018592,
                    -0.56260985,
                    0.21693541,
                    -0.35926607,
                    0.393788,
                    0.49076477,
                    0.29092142,
                    0.5295006,
                    0.31346723,
                    -0.52280056,
                    -0.2208816,
                    0.08115966,
                    -0.2058997,
                    0.19451942,
                ]
            ).astype(np.float32)
            cpu_weight_out2 = np.array(
                [
                    -0.31963634,
                    -0.3666257,
                    0.17078729,
                    -0.40847018,
                    0.180685,
                    -0.26196587,
                    0.39458436,
                    0.2440751,
                    0.32041675,
                    -0.63348126,
                    -0.12749134,
                    0.0960809,
                    0.23655102,
                    -0.5229054,
                    -0.20650089,
                    0.07357956,
                    0.47345874,
                    -0.06078928,
                    -0.0149216,
                    -0.34730294,
                    -0.3629782,
                    0.26349628,
                    0.09561133,
                    0.54988813,
                    0.49647588,
                    0.68336976,
                    0.12918544,
                    -0.42924428,
                    0.2932164,
                    -0.32783166,
                    0.3987024,
                    0.48668715,
                    0.3665252,
                    0.5910386,
                    0.35637644,
                    -0.48564422,
                    -0.15684633,
                    0.14750825,
                    -0.13639452,
                    0.2650746,
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
