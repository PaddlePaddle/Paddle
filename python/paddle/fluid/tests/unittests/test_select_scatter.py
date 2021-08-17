# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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


def select_scatter_dygraph():
    import numpy as np
    import paddle
    from paddle.distributed import init_parallel_env

    init_parallel_env()
    n_expert = 2
    world_size = 2
    d_model = 2
    in_feat = d_model
    local_input_buf = np.array(
        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=np.float32)
    if paddle.distributed.ParallelEnv().local_rank == 0:
        local_expert_count = np.array(
            [2, 1, 1, 1])  # (world_size * num_experts)
        global_expert_count = np.array(
            [2, 1, 1, 1])  # (world_size * num_experts)
    else:
        local_expert_count = np.array(
            [1, 1, 2, 1])  # (world_size * num_experts)
        global_expert_count = np.array(
            [1, 1, 2, 1])  # (world_size * num_experts)
    input_buf = np.empty(
        shape=(np.sum(global_expert_count), in_feat), dtype=np.
        float32)  # (batch_size, d_model) batch_size是global_expert_count的所有数量

    local_input_buf = paddle.to_tensor(
        local_input_buf, dtype="float32", stop_gradient=False)
    local_expert_count = paddle.to_tensor(local_expert_count, dtype="int32")
    global_expert_count = paddle.to_tensor(global_expert_count, dtype="int32")
    input_buf = paddle.to_tensor(
        input_buf, dtype="float32", stop_gradient=False)
    in_feat = paddle.to_tensor(in_feat, dtype="int32")
    n_expert = paddle.to_tensor(n_expert, dtype="int32")
    world_size = paddle.to_tensor(world_size, dtype="int32")

    a = paddle.distributed.selectscatter(local_input_buf, \
    local_expert_count, global_expert_count, \
    in_feat, n_expert, world_size)
    a.stop_gradient = False
    print(a)
    # out for rank 0: [[1, 2], [3, 4], [1, 2], [5, 6], [3, 4]]
    # out for rank 1: [[7, 8], [5, 6], [7, 8], [9, 10], [9, 10]]
    # a.backward()
    # print("a.grad: ", a.grad)
    # print("local_input_buf.grad: ", local_input_buf.grad)
    # backward test
    b = paddle.ones(shape=a.shape)
    b.stop_gradient = False
    c = a * b * 3
    c.backward()
    print("c.grad: ", c.grad)
    print("b.grad: ", b.grad)
    print("a.grad: ", a.grad)
    print("local_input_buf.grad: ", local_input_buf.grad)


def select_scatter_static():
    import numpy as np
    import paddle
    from paddle.distributed import init_parallel_env
    import paddle.fluid as fluid
    from paddle.distributed import fleet
    import os

    paddle.enable_static()
    init_parallel_env()
    main_program = paddle.static.Program()
    main_startup_program = paddle.static.Program()
    rank = int(os.getenv("FLAGS_selected_gpus", "0"))
    with fluid.program_guard(main_program, main_startup_program):
        with fluid.unique_name.guard():
            local_input_buf = paddle.static.data(
                name="local_input_buf", shape=[5, 2], dtype="float32")
            local_expert_count = paddle.static.data(
                name="local_expert_count", shape=[4], dtype="int32")
            global_expert_count = paddle.static.data(
                name="global_expert_count", shape=[4], dtype="int32")
            input_x = paddle.distributed.selectscatter(
                local_input_buf,
                local_expert_count,
                global_expert_count,
                in_feat=2,
                n_expert=2,
                world_size=2)
            fluid.layers.Print(input_x, message="input_x")

    def gen_data():
        n_expert = 2
        world_size = 2
        d_model = 2
        in_feat = d_model
        local_input_buf = np.array(
            [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=np.float32)
        rank = int(os.getenv("FLAGS_selected_gpus", "0"))
        if rank == 0:
            local_expert_count = np.array(
                [2, 1, 1, 1], dtype="int32")  # (world_size * num_experts)
            global_expert_count = np.array(
                [2, 1, 1, 1], dtype="int32")  # (world_size * num_experts)
        else:
            local_expert_count = np.array(
                [1, 1, 2, 1], dtype="int32")  # (world_size * num_experts)
            global_expert_count = np.array(
                [1, 1, 2, 1], dtype="int32")  # (world_size * num_experts)
        return {
            "local_input_buf": local_input_buf,
            "local_expert_count": local_expert_count,
            "global_expert_count": global_expert_count
        }

    trainer_id = int(os.getenv("FLAGS_selected_gpus", "0"))
    exe = paddle.static.Executor(paddle.CUDAPlace(trainer_id))
    exe.run(main_startup_program)
    exe.run(program=main_program, feed=gen_data())


def moe_expert_exchange_dygraph():
    # def selectscatter(local_input_buf, local_expert_count, 
    #                 global_expert_count, input_buf, \
    #                 in_feat, n_expert, world_size, \
    #                 out_tensor_list, group=None, use_calc_stream=True):
    import numpy as np
    import paddle
    from paddle.distributed import init_parallel_env
    # paddle.enable_static()
    init_parallel_env()
    n_expert = 2
    world_size = 2
    d_model = 2
    in_feat = d_model
    local_input_buf = np.array(
        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=np.float32)
    if paddle.distributed.ParallelEnv().local_rank == 0:
        local_expert_count = np.array(
            [2, 1, 1, 1])  # (world_size * num_experts)
        # global_expert_count = np.array([2, 1, 1, 1]) # (world_size * num_experts)
        # global_expert_count = np.empty_like(local_expert_count)
    else:
        local_expert_count = np.array(
            [1, 1, 2, 1])  # (world_size * num_experts)
        # global_expert_count = np.array([1, 1, 2, 1]) # (world_size * num_experts)
        # global_expert_count = np.empty_like(local_expert_count)
    # input_buf = np.empty(shape=(np.sum(global_expert_count), in_feat), dtype=np.float32) # (batch_size, d_model) batch_size是global_expert_count的所有数量

    local_input_buf = paddle.to_tensor(
        local_input_buf, dtype="float32", stop_gradient=False)
    local_expert_count = paddle.to_tensor(local_expert_count, dtype="int32")
    # global_expert_count = paddle.to_tensor(global_expert_count, dtype="int32")
    # input_buf = paddle.to_tensor(input_buf, dtype="float32", stop_gradient=False)
    in_feat = paddle.to_tensor(in_feat, dtype="int32")
    n_expert = paddle.to_tensor(n_expert, dtype="int32")
    world_size = paddle.to_tensor(world_size, dtype="int32")
    # print(input_buf.dtype)
    # print(local_input_buf.dtype)
    # print(local_expert_count.dtype)
    # print(global_expert_count.dtype)
    # local_input_buf.stop_gradient = False
    # input_buf.stop_gradient = False
    global_expert_count = paddle.distributed.moe_expert_exchange(local_expert_count, \
                                                                 n_expert, world_size)
    print(global_expert_count)
    # out for rank 0: [2, 1, 1, 1]
    # out for rank 1: [1, 1, 2, 1]


def moe_expert_exchange_static():
    import numpy as np
    import paddle
    from paddle.distributed import init_parallel_env
    import paddle.fluid as fluid
    from paddle.distributed import fleet
    import os

    paddle.enable_static()
    init_parallel_env()
    main_program = paddle.static.Program()
    main_startup_program = paddle.static.Program()
    rank = int(os.getenv("FLAGS_selected_gpus", "0"))
    with fluid.program_guard(main_program, main_startup_program):
        with fluid.unique_name.guard():
            local_expert_count = paddle.static.data(
                name="local_expert_count", shape=[4], dtype="int32")
            input_x = paddle.distributed.moe_expert_exchange(
                local_expert_count, n_expert=2, world_size=2)
            fluid.layers.Print(input_x, message="input_x")
            # output for rank 0: [2, 1, 1, 1]
            # output for rank 1: [1, 1, 2, 1]

    def gen_data():
        rank = int(os.getenv("FLAGS_selected_gpus", "0"))
        if rank == 0:
            local_expert_count = np.array(
                [2, 1, 1, 1], dtype="int32")  # (world_size * num_experts)
        else:
            local_expert_count = np.array(
                [1, 1, 2, 1], dtype="int32")  # (world_size * num_experts)
        return {"local_expert_count": local_expert_count}

    trainer_id = int(os.getenv("FLAGS_selected_gpus", "0"))
    exe = paddle.static.Executor(paddle.CUDAPlace(trainer_id))
    exe.run(main_startup_program)
    exe.run(program=main_program, feed=gen_data())


def select_gather_dygraph():
    import numpy as np
    import paddle
    from paddle.distributed import init_parallel_env

    init_parallel_env()
    n_expert = 2
    world_size = 2
    d_model = 2
    in_feat = d_model
    local_input_buf = np.array(
        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=np.float32)
    if paddle.distributed.ParallelEnv().local_rank == 0:
        local_expert_count = np.array(
            [2, 1, 1, 1])  # (world_size * num_experts)
        global_expert_count = np.array(
            [2, 1, 1, 1])  # (world_size * num_experts)
    else:
        local_expert_count = np.array(
            [1, 1, 2, 1])  # (world_size * num_experts)
        global_expert_count = np.array(
            [1, 1, 2, 1])  # (world_size * num_experts)
    input_buf = np.empty(
        shape=(np.sum(global_expert_count), in_feat), dtype=np.
        float32)  # (batch_size, d_model) batch_size是global_expert_count的所有数量

    local_input_buf = paddle.to_tensor(
        local_input_buf, dtype="float32", stop_gradient=False)
    local_expert_count = paddle.to_tensor(local_expert_count, dtype="int32")
    global_expert_count = paddle.to_tensor(global_expert_count, dtype="int32")
    input_buf = paddle.to_tensor(
        input_buf, dtype="float32", stop_gradient=False)
    in_feat = paddle.to_tensor(in_feat, dtype="int32")
    n_expert = paddle.to_tensor(n_expert, dtype="int32")
    world_size = paddle.to_tensor(world_size, dtype="int32")

    a = paddle.distributed.selectgather(local_input_buf, \
    global_expert_count, local_expert_count, \
    in_feat, n_expert, world_size)
    a.stop_gradient = False
    print(a)
    # out for rank 0: [[1, 2], [3, 4], [1, 2], [5, 6], [3, 4]]
    # out for rank 1: [[7, 8], [5, 6], [7, 8], [9, 10], [9, 10]]
    # a.backward()
    # print("a.grad: ", a.grad)
    # print("local_input_buf.grad: ", local_input_buf.grad)
    # backward test
    b = paddle.ones(shape=a.shape)
    b.stop_gradient = False
    c = a * b * 3
    c.backward()
    print("c.grad: ", c.grad)
    print("b.grad: ", b.grad)
    print("a.grad: ", a.grad)
    print("local_input_buf.grad: ", local_input_buf.grad)


def select_gather_static():
    import numpy as np
    import paddle
    from paddle.distributed import init_parallel_env
    import paddle.fluid as fluid
    from paddle.distributed import fleet
    import os

    paddle.enable_static()
    init_parallel_env()
    main_program = paddle.static.Program()
    main_startup_program = paddle.static.Program()
    rank = int(os.getenv("FLAGS_selected_gpus", "0"))
    with fluid.program_guard(main_program, main_startup_program):
        with fluid.unique_name.guard():
            local_input_buf = paddle.static.data(
                name="local_input_buf", shape=[5, 2], dtype="float32")
            local_expert_count = paddle.static.data(
                name="local_expert_count", shape=[4], dtype="int32")
            global_expert_count = paddle.static.data(
                name="global_expert_count", shape=[4], dtype="int32")
            input_x = paddle.distributed.selectscatter(
                local_input_buf,
                local_expert_count,
                global_expert_count,
                in_feat=2,
                n_expert=2,
                world_size=2)
            fluid.layers.Print(input_x, message="input_x")

    def gen_data():
        n_expert = 2
        world_size = 2
        d_model = 2
        in_feat = d_model
        local_input_buf = np.array(
            [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=np.float32)
        rank = int(os.getenv("FLAGS_selected_gpus", "0"))
        if rank == 0:
            local_expert_count = np.array(
                [2, 1, 1, 1], dtype="int32")  # (world_size * num_experts)
            global_expert_count = np.array(
                [2, 1, 1, 1], dtype="int32")  # (world_size * num_experts)
        else:
            local_expert_count = np.array(
                [1, 1, 2, 1], dtype="int32")  # (world_size * num_experts)
            global_expert_count = np.array(
                [1, 1, 2, 1], dtype="int32")  # (world_size * num_experts)
        return {
            "local_input_buf": local_input_buf,
            "local_expert_count": global_expert_count,
            "global_expert_count": local_expert_count
        }

    trainer_id = int(os.getenv("FLAGS_selected_gpus", "0"))
    exe = paddle.static.Executor(paddle.CUDAPlace(trainer_id))
    exe.run(main_startup_program)
    exe.run(program=main_program, feed=gen_data())


# moe_expert_exchange_static()
# moe_expert_exchange()
# select_scatter_static()
# select_scatter_static()
# select_gather_static()
select_gather_dygraph()
