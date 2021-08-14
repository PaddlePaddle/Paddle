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

import os
import sys
import time
import multiprocessing
from contextlib import closing
import socket

import paddle
import paddle.fluid as fluid

port_set = set()
paddle.enable_static()


def find_free_port():
    def _free_port():
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(('', 0))
            return s.getsockname()[1]

    while True:
        port = _free_port()
        if port not in port_set:
            port_set.add(port)
            return port


def test_barrier_func(id, rank_num, server_endpoint, out_dict, sleep_time):
    paddle.distributed.init_gloo_parallel_env(id, rank_num, server_endpoint)
    # 1st barrier
    # Run barrier to synchronize processes after starting
    paddle.distributed.barrier_func()

    # 2nd barrier
    # Let rank 0 sleep for one second and check that all processes
    # saw that artificial delay through the barrier
    start = time.time()
    if (id == 0):
        time.sleep(sleep_time)
    paddle.distributed.barrier_func()
    end = time.time()

    out_dict[id] = end - start
    # Release
    paddle.distributed.release_gloo(id)


def test_barrier_op(id, rank_num, server_endpoint, out_dict, sleep_time):
    main_prog = fluid.Program()
    startup_prog = fluid.Program()
    paddle.distributed.init_gloo_parallel_env(id, rank_num, server_endpoint)
    place = fluid.CPUPlace()
    with fluid.program_guard(main_prog, startup_prog):
        paddle.distributed.barrier()
    exe = fluid.Executor(place)
    # Run barrier to synchronize processes after starting
    exe.run(main_prog)

    # Let rank 0 sleep for one second and check that all processes
    # saw that artificial delay through the barrier
    start = time.time()
    if (id == 0):
        time.sleep(sleep_time)
    exe.run(main_prog)
    end = time.time()

    out_dict[id] = end - start
    # Release
    paddle.distributed.release_gloo(id)


def test_barrier_with_multiprocess(test_barrier, num_of_ranks, sleep_time):
    if num_of_ranks <= 0 or sleep_time < 0:
        return
    # create endpoints
    ep_str = "127.0.0.1:%s" % (find_free_port())
    #print(ep_str)

    # call barrier op inside each process
    all_procs_start = time.time()
    manager = multiprocessing.Manager()
    procs_out_dict = manager.dict()
    jobs = []
    for id in range(num_of_ranks):
        p = multiprocessing.Process(
            target=test_barrier,
            args=(id, num_of_ranks, ep_str, procs_out_dict, sleep_time))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()
    all_procs_end = time.time()

    # check results
    print("***************************************************")
    print("Barrier op exection time recorded in each process: ")
    print("***************************************************")
    print(procs_out_dict)

    sum_of_barrier_time = 0
    for _, v in procs_out_dict.items():
        if v <= sleep_time:
            print("Failed")
            return
        sum_of_barrier_time += v

    print("**********************************")
    print("Average barrier op exection time: ")
    print("**********************************")
    print((sum_of_barrier_time - sleep_time * num_of_ranks) / num_of_ranks)

    print("")
    print("Passed   %.2fs" % (all_procs_end - all_procs_start))
    return


if __name__ == '__main__':
    # Arg 0: test_barrier_func or test_barrier_op
    # Arg 1: number of ranks (processes)
    # Arg 2: time sleeping in second in #1 process
    test_barrier_with_multiprocess(test_barrier_func, 16, 1)
