# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import unittest
from multiprocessing import Process, Queue
import subprocess
import socket
from contextlib import closing

import paddle.distributed as dist
import numpy as np


def worker_name(rank):
    return "worker{}".format(rank)


def run_rpc_sync(
    rank,
    world_size,
    master_endpoint,
    queue,
    fn,
    args=None,
    kwargs=None,
):
    dist.rpc.init_rpc(
        worker_name(rank),
        rank,
        world_size,
        master_endpoint,
    )
    res = dist.rpc.rpc_sync(worker_name(0), fn, args=args, kwargs=kwargs)
    queue.put(res)
    dist.rpc.shutdown()


def run_rpc_sync_master_working(
    rank,
    world_size,
    master_endpoint,
    queue,
    fn,
    args=None,
    kwargs=None,
):
    dist.rpc.init_rpc(
        worker_name(rank),
        rank,
        world_size,
        master_endpoint,
    )
    if dist.get_rank() == 0:
        for i in range(1, dist.get_rank()):
            res = dist.rpc.rpc_sync(worker_name(i),
                                    fn,
                                    args=args,
                                    kwargs=kwargs)
            queue.put(res)
    dist.rpc.shutdown()


def run_rpc_async(
    rank,
    world_size,
    master_endpoint,
    queue,
    fn,
    args=None,
    kwargs=None,
):
    dist.rpc.init_rpc(
        worker_name(rank),
        rank,
        world_size,
        master_endpoint,
    )
    res = dist.rpc.rpc_async(worker_name(0), fn, args=args, kwargs=kwargs)
    queue.put(res.wait())
    dist.rpc.shutdown()


def run_rpc_async_master_working(
    rank,
    world_size,
    master_endpoint,
    queue,
    fn,
    args=None,
    kwargs=None,
):
    dist.rpc.init_rpc(
        worker_name(rank),
        rank,
        world_size,
        master_endpoint,
    )
    if dist.get_rank() == 0:
        for i in range(1, dist.get_rank()):
            res = dist.rpc.rpc_async(worker_name(i),
                                     fn,
                                     args=args,
                                     kwargs=kwargs)
            queue.put(res.wait())
    dist.rpc.shutdown()


class RpcTestBase(unittest.TestCase):

    def setUp(self):
        self._port_set = set()
        print("RPC setUp...")

    def tearDown(self):
        if len(self.processes) != 0:
            [p.join() for p in self.processes]
        print("RPC tearDown...")

    def _find_free_port(self):

        def __free_port():
            with closing(socket.socket(socket.AF_INET,
                                       socket.SOCK_STREAM)) as s:
                s.bind(("", 0))
                return s.getsockname()[1]

        while True:
            port = __free_port()
            if port not in self._port_set:
                self._port_set.add(port)
                return port

    def run_rpc(self, sync, world_size, fn, fn_args=None, fn_kwargs=None):
        self.processes = []
        queues = []
        master_endpoint = "127.0.0.1:{}".format(self._find_free_port())
        for rank in range(world_size):
            q = Queue()
            queues.append(q)
            if sync:
                self.processes.append(
                    Process(
                        target=run_rpc_sync,
                        args=(
                            rank,
                            world_size,
                            master_endpoint,
                            q,
                            fn,
                            fn_args,
                            fn_kwargs,
                        ),
                    ))
            else:
                self.processes.append(
                    Process(
                        target=run_rpc_async,
                        args=(
                            rank,
                            world_size,
                            master_endpoint,
                            q,
                            fn,
                            fn_args,
                            fn_kwargs,
                        ),
                    ))
        [p.start() for p in self.processes]
        return queues


class RpcLaunchTestBase(unittest.TestCase):

    def setUp(self):
        self._port_set = set()
        print("Launch RPC setUp...")

    def tearDown(self):
        self.remove_data()
        print("Launch RPC tearDown...")

    def _find_free_port(self):

        def __free_port():
            with closing(socket.socket(socket.AF_INET,
                                       socket.SOCK_STREAM)) as s:
                s.bind(("", 0))
                return s.getsockname()[1]

        while True:
            port = __free_port()
            if port not in self._port_set:
                self._port_set.add(port)
                return port

    def create_data(self, nnodes, nproc_per_node):
        mmap_data1 = np.memmap(
            "rpc_launch_data1.npy",
            dtype=np.float32,
            mode="w+",
            shape=(10 * nnodes * nproc_per_node, 100),
        )
        mmap_data2 = np.memmap(
            "rpc_launch_data2.npy",
            dtype=np.float32,
            mode="w+",
            shape=(10 * nnodes * nproc_per_node, 100),
        )
        for i in range(nnodes * nproc_per_node):
            a = np.random.random((10, 100)).astype(np.float32)
            b = np.random.random((10, 100)).astype(np.float32)
            mmap_data1[i * 10:(i + 1) * 10, :] = a
            mmap_data2[i * 10:(i + 1) * 10, :] = b
        return mmap_data1, mmap_data2

    def remove_data(self):
        os.remove("rpc_launch_data1.npy")
        os.remove("rpc_launch_data2.npy")

    def launch_rpc(self, nnodes, nproc_per_node, model_file):
        master_endpoint = "127.0.0.1:{}".format(self._find_free_port())
        log_dir = "log"
        tr_cmd = "python -m paddle.distributed.launch --master {} --rank {} --nnodes {} --nproc_per_node {} --run_mode rpc {} --log_dir {}"
        cmds = [
            tr_cmd.format(master_endpoint, rank, nnodes, nproc_per_node,
                          model_file, log_dir) for rank in range(nnodes)
        ]
        processes = [subprocess.Popen(cmd.strip().split()) for cmd in cmds]
        [proc.communicate() for proc in processes]
        out = np.memmap(
            "rpc_launch_result.npy",
            dtype=np.float32,
            mode="r",
            shape=(10 * nnodes * nproc_per_node, 100),
        )
        os.remove("rpc_launch_result.npy")
        import shutil

        shutil.rmtree(log_dir)
        return out
