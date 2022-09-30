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
import socket
from contextlib import closing
from multiprocessing import Process, Queue

import paddle
import paddle.distributed as dist
import numpy as np

MASTER_ENDPOINT = "127.0.0.1:8765"
paddle.device.set_device("cpu")


def find_free_port():
    port_set = set()

    def _free_port():
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    while True:
        port = _free_port()
        if port not in port_set:
            port_set.add(port)
            return port


def worker_name(rank):
    return "worker{}".format(rank)


def server_endpoint():
    return "127.0.0.1:{}".format(find_free_port())


def paddle_add(a, b):
    a = paddle.to_tensor(a)
    b = paddle.to_tensor(b)
    res = paddle.add(a, b).numpy()
    return res


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
        print("RPC setUp...")

    def tearDown(self):
        if len(self.processes) != 0:
            [p.join() for p in self.processes]
        print("RPC tearDown...")

    def run_rpc(self, sync, world_size, fn, fn_args=None, fn_kwargs=None):
        self.processes = []
        queues = []
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
                            MASTER_ENDPOINT,
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
                            MASTER_ENDPOINT,
                            q,
                            fn,
                            fn_args,
                            fn_kwargs,
                        ),
                    ))
        [p.start() for p in self.processes]
        return queues


class TestRpc(RpcTestBase):

    def test_one_server_sync_paddle_add(self):
        a = np.random.random((10, 100))
        b = np.random.random((10, 100))
        res = np.add(a, b)
        args = (a, b)
        queues = self.run_rpc(True, 1, paddle_add, args)
        out = queues[0].get()
        np.testing.assert_allclose(out, res, rtol=1e-05)

    def test_one_server_async_paddle_add(self):
        a = np.random.random((10, 100))
        b = np.random.random((10, 100))
        res = np.add(a, b)
        args = (a, b)
        queues = self.run_rpc(False, 1, paddle_add, args)
        out = queues[0].get()
        np.testing.assert_allclose(out, res, rtol=1e-05)

    def test_two_server_sync_paddle_add(self):
        a = np.random.random((10, 100))
        b = np.random.random((10, 100))
        res = np.add(a, b)
        args = (a, b)
        queues = self.run_rpc(True, 2, paddle_add, args)
        out1 = queues[0].get()
        out2 = queues[1].get()
        np.testing.assert_allclose(out1, res, rtol=1e-05)
        np.testing.assert_allclose(out2, res, rtol=1e-05)

    def test_two_server_async_paddle_add(self):
        a = np.random.random((10, 100))
        b = np.random.random((10, 100))
        res = np.add(a, b)
        args = (a, b)
        queues = self.run_rpc(False, 2, paddle_add, args)
        out1 = queues[0].get()
        out2 = queues[1].get()
        np.testing.assert_allclose(out1, res, rtol=1e-05)
        np.testing.assert_allclose(out2, res, rtol=1e-05)


class TestSingleProcessRpc(unittest.TestCase):

    def setUp(self):
        self.server_endpoint = server_endpoint()
        os.environ["PADDLE_SERVER_ENDPOINT"] = self.server_endpoint
        dist.rpc.init_rpc(
            worker_name(0),
            0,
            1,
            server_endpoint(),
        )
        print("Single Process RPC setUp...")

    def tearDown(self):
        dist.rpc.shutdown()
        print("Single Process RPC tearDown...")

    def test_sync_rpc_paddle_add(self):
        a = np.random.random((10, 100))
        b = np.random.random((10, 100))
        res = np.add(a, b)
        args = (a, b)
        out = dist.rpc.rpc_sync(worker_name(0), paddle_add, args=args)
        np.testing.assert_allclose(out, res, rtol=1e-05)

    def test_async_rpc_paddle_add(self):
        a = np.random.random((10, 100))
        b = np.random.random((10, 100))
        res = np.add(a, b)
        args = (a, b)
        out = dist.rpc.rpc_async(worker_name(0), paddle_add, args=args).wait()
        np.testing.assert_allclose(out, res, rtol=1e-05)

    def test_get_worker_info(self):
        info = dist.rpc.get_worker_info(worker_name(0))
        self.assertEqual(info.name, worker_name(0))
        self.assertEqual(info.rank, 0)
        ip, port = self.server_endpoint.split(":")
        port = int(port)
        self.assertEqual(info.ip, ip)
        self.assertEqual(info.port, port)

    def test_get_all_worker_infos(self):
        infos = dist.rpc.get_all_worker_infos()
        info = infos[0]
        self.assertEqual(info.name, worker_name(0))
        self.assertEqual(info.rank, 0)
        ip, port = self.server_endpoint.split(":")
        port = int(port)
        self.assertEqual(info.ip, ip)
        self.assertEqual(info.port, port)

    def test_get_current_worker_info(self):
        info = dist.rpc.get_current_worker_info()
        self.assertEqual(info.name, worker_name(0))
        self.assertEqual(info.rank, 0)
        ip, port = self.server_endpoint.split(":")
        port = int(port)
        self.assertEqual(info.ip, ip)
        self.assertEqual(info.port, port)


if __name__ == "__main__":
    unittest.main()
