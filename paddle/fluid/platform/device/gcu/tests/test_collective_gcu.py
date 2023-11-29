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

import os
import time

import numpy as np

import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.distributed_utils import get_rank, get_world_size


class ConvNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = paddle.nn.Conv2D(
            in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2
        )
        self.max_pool1 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        self.conv2 = paddle.nn.Conv2D(
            in_channels=6, out_channels=16, kernel_size=5, stride=1
        )
        self.max_pool2 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)

    def forward(self, x):
        x = paddle.abs(x)
        x = self.conv1(x)
        x = paddle.nn.functional.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = paddle.nn.functional.relu(x)
        x = self.max_pool2(x)
        return x


def gen_data(batch_size):
    return {
        "x": np.random.random(size=(batch_size, 1, 16, 16)).astype('float32'),
        "y": np.random.random(size=(batch_size, 16, 2, 2)).astype('float32'),
    }


def batch_gen_data(batch_size, nums):
    input_datas = []
    np.random.seed(2022)
    for i in range(nums):
        input_datas.append(gen_data(batch_size))
    return input_datas


def get_dist_data(data, rank, dist_mini_batch):
    x = data['x'][(rank * dist_mini_batch) : ((rank + 1) * dist_mini_batch)]
    y = data['y'][(rank * dist_mini_batch) : ((rank + 1) * dist_mini_batch)]
    return {'x': x, 'y': y}


def create_model(batch_size, is_dist=False):
    # create input/label
    input_x = paddle.static.data(
        name="x", shape=[batch_size, 1, 16, 16], dtype='float32'
    )
    label_y = paddle.static.data(
        name="y", shape=[batch_size, 16, 2, 2], dtype='float32'
    )

    # create network
    model = ConvNet()
    prediction = model(input_x)
    loss = paddle.nn.functional.l1_loss(input=prediction, label=label_y)

    # distributed init
    if is_dist:
        strategy = fleet.DistributedStrategy()
        fleet.init(is_collective=True, strategy=strategy)

    optimizer = paddle.fluid.optimizer.Adam(learning_rate=0.01)
    # distributed optimizer
    if is_dist:
        optimizer = fleet.distributed_optimizer(optimizer)
    optimizer.minimize(loss)


def build_program(batch_size, seed, is_dist=False):
    paddle.enable_static()
    startup_program = paddle.static.Program()
    main_program = paddle.static.Program()
    startup_program.random_seed = seed
    main_program.random_seed = seed
    with paddle.fluid.unique_name.guard():
        with paddle.static.program_guard(
            main_program=main_program, startup_program=startup_program
        ):
            create_model(batch_size, is_dist)
    return startup_program, main_program


def check_state_dict(state_dict, dist_state_dict):
    atol = 1e-01
    for key in sorted(state_dict):
        tensor = paddle.fluid.executor.as_numpy(state_dict[key])
        dist_tensor = paddle.fluid.executor.as_numpy(dist_state_dict[key])
        assert np.allclose(
            tensor, dist_tensor, atol=atol
        ), 'state_dict check failed, var name:{}, tensor:{}, vs dist tensor:{}'.format(
            key, tensor, dist_tensor
        )


def distributed_test(batch_size=8, total_steps=10, compare_with_cpu=False):
    grads = [
        'conv2d_0.b_0@GRAD',
        'conv2d_0.w_0@GRAD',
        'conv2d_1.b_0@GRAD',
        'conv2d_1.w_0@GRAD',
    ]
    dist_grads = [
        'conv2d_0.b_0@GRAD_gcu_all_reduce',
        'conv2d_0.w_0@GRAD_gcu_all_reduce',
        'conv2d_1.b_0@GRAD_gcu_all_reduce',
        'conv2d_1.w_0@GRAD_gcu_all_reduce',
    ]
    output_list = ["mean_0.tmp_0"]  # + grads
    dist_output_list = ["mean_0.tmp_0"]  # + dist_grads

    input_datas = batch_gen_data(batch_size, total_steps)
    rank = get_rank()
    world_size = get_world_size()
    dist_mini_batch = batch_size // world_size
    print(
        'test init rank:{}, world_size:{}, batch_size:{}, dist_mini_batch:{}'.format(
            rank, world_size, batch_size, dist_mini_batch
        )
    )

    # single card
    seed = 2036
    single_run_scope = paddle.static.Scope()
    with paddle.static.scope_guard(single_run_scope):
        startup_prog, main_prog = build_program(batch_size, seed, False)
        cpu_place = paddle.CPUPlace()
        cpu_exe = paddle.static.Executor(cpu_place)
        cpu_exe.run(startup_prog)

        gcu_exe = paddle.static.Executor('gcu')
        if not compare_with_cpu:
            res = gcu_exe.run(
                main_prog, feed=input_datas[0], fetch_list=output_list
            )
        else:
            res = cpu_exe.run(
                main_prog, feed=input_datas[0], fetch_list=output_list
            )
        single_run_state = main_prog.state_dict()

    # distributed
    dist_run_scope = paddle.static.Scope()
    with paddle.static.scope_guard(dist_run_scope):
        dist_startup_prog, dist_main_prog = build_program(
            dist_mini_batch, seed, True
        )
        cpu_exe.run(dist_startup_prog)
        dist_res = gcu_exe.run(
            dist_main_prog,
            feed=get_dist_data(input_datas[0], rank, dist_mini_batch),
            fetch_list=dist_output_list,
        )
        dist_run_state = dist_main_prog.state_dict()

    check_state_dict(single_run_state, dist_run_state)
    if rank == 0:
        print(
            'step: {}, loss: {}, dist_loss: {}, compare with {}, check state_dict pass'.format(
                0, res[0], dist_res[0], 'CPU' if compare_with_cpu else 'GCU'
            )
        )

    # test every steps
    for i in range(1, total_steps):
        with paddle.static.scope_guard(single_run_scope):
            if not compare_with_cpu:
                res = gcu_exe.run(
                    main_prog, feed=input_datas[i], fetch_list=output_list
                )
            else:
                res = cpu_exe.run(
                    main_prog, feed=input_datas[i], fetch_list=output_list
                )
            single_run_state = main_prog.state_dict()

        with paddle.static.scope_guard(dist_run_scope):
            dist_res = gcu_exe.run(
                dist_main_prog,
                feed=get_dist_data(input_datas[i], rank, dist_mini_batch),
                fetch_list=dist_output_list,
            )
            dist_run_state = dist_main_prog.state_dict()

        check_state_dict(single_run_state, dist_run_state)
        if rank == 0:
            print(
                'step: {}, loss: {}, dist_loss: {}, compare with {}, check state_dict pass'.format(
                    i, res[0], dist_res[0], 'CPU' if compare_with_cpu else 'GCU'
                )
            )


def distributed_test_with_gcu():
    rank = get_rank()
    t1 = time.time()
    distributed_test(batch_size=16, total_steps=10, compare_with_cpu=False)
    t2 = time.time()
    print(
        'Run distributed_test_with_gcu successfully, rank:{}, time: {}s'.format(
            rank, (t2 - t1)
        )
    )


def distributed_test_with_cpu():
    rank = get_rank()
    t1 = time.time()
    distributed_test(batch_size=16, total_steps=10, compare_with_cpu=True)
    t2 = time.time()
    print(
        'Run distributed_test_with_cpu successfully, rank:{}, time: {}s'.format(
            rank, (t2 - t1)
        )
    )


if __name__ == '__main__':
    os.environ[
        "ENFLAME_COMPILE_OPTIONS_HLIR"
    ] = "hlir-pipeline{tensor-split=false op-key=pavo}"
    os.environ["PADDLE_GCU_RUNNING_MODE"] = "force_serial"
    os.environ["ECCL_RUNTIME_3_0_ENABLE"] = "true"
    distributed_test_with_gcu()
    distributed_test_with_cpu()

# python -m paddle.distributed.launch --gcus=0,1,2,3,4,5,6,7  test_collective_gcu.py 2>&1 | tee grads_test.log
