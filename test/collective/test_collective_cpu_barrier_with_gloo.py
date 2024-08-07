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

import multiprocessing
import socket
import time
import unittest
from contextlib import closing

import paddle
from paddle import base

port_set = set()
paddle.enable_static()


class CollectiveCPUBarrierWithGlooTest(unittest.TestCase):
    def find_free_port(self):
        def _free_port():
            with closing(
                socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            ) as s:
                s.bind(('', 0))
                return s.getsockname()[1]

        while True:
            port = _free_port()
            if port not in port_set:
                port_set.add(port)
                return port

    def barrier_func(self, id, rank_num, server_endpoint, out_dict, sleep_time):
        try:
            paddle.distributed.gloo_init_parallel_env(
                id, rank_num, server_endpoint
            )
            # 1st barrier
            # Run barrier to synchronize processes after starting
            paddle.distributed.gloo_barrier()
            # 2nd barrier
            # Let rank 0 sleep for one second and check that all processes
            # saw that artificial delay through the barrier
            start = time.time()
            if id == 0:
                time.sleep(sleep_time)
            paddle.distributed.gloo_barrier()
            end = time.time()
            out_dict[id] = end - start
            # Release
            paddle.distributed.gloo_release()
        except:
            out_dict[id] = 0

    def barrier_op(self, id, rank_num, server_endpoint, out_dict, sleep_time):
        try:
            main_prog = base.Program()
            startup_prog = base.Program()
            paddle.distributed.gloo_init_parallel_env(
                id, rank_num, server_endpoint
            )
            place = base.CPUPlace()
            with base.program_guard(main_prog, startup_prog):
                paddle.distributed.barrier()
            exe = base.Executor(place)
            # Run barrier to synchronize processes after starting
            exe.run(main_prog)
            # Let rank 0 sleep for one second and check that all processes
            # saw that artificial delay through the barrier
            start = time.time()
            if id == 0:
                time.sleep(sleep_time)
            exe.run(main_prog)
            end = time.time()
            out_dict[id] = end - start
            # Release
            paddle.distributed.gloo_release()
        except:
            out_dict[id] = 0

    def test_barrier_func_with_multiprocess(self):
        num_of_ranks = 4
        sleep_time = 1
        # create endpoints
        ep_str = f"127.0.0.1:{self.find_free_port()}"
        # call barrier op inside each process
        manager = multiprocessing.Manager()
        procs_out_dict = manager.dict()
        jobs = []
        for id in range(num_of_ranks):
            p = multiprocessing.Process(
                target=self.barrier_func,
                args=(id, num_of_ranks, ep_str, procs_out_dict, sleep_time),
            )
            jobs.append(p)
            p.start()
        for proc in jobs:
            proc.join()
        for _, v in procs_out_dict.items():
            self.assertTrue(v > sleep_time)

    def test_barrier_op_with_multiprocess(self):
        num_of_ranks = 4
        sleep_time = 1
        # create endpoints
        ep_str = f"127.0.0.1:{self.find_free_port()}"
        # call barrier op inside each process
        manager = multiprocessing.Manager()
        procs_out_dict = manager.dict()
        jobs = []
        for id in range(num_of_ranks):
            p = multiprocessing.Process(
                target=self.barrier_op,
                args=(id, num_of_ranks, ep_str, procs_out_dict, sleep_time),
            )
            jobs.append(p)
            p.start()
        for proc in jobs:
            proc.join()
        for _, v in procs_out_dict.items():
            self.assertTrue(v > sleep_time)


if __name__ == '__main__':
    unittest.main()
