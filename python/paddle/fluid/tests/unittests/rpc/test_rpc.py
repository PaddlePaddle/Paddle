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

import paddle
import paddle.distributed as dist
import numpy as np
from test_rpc_base import RpcTestBase, RpcLaunchTestBase

paddle.device.set_device("cpu")


def worker_name(rank):
    return "worker{}".format(rank)


def paddle_add(a, b):
    a = paddle.to_tensor(a)
    b = paddle.to_tensor(b)
    res = paddle.add(a, b).numpy()
    return res


class TestMultiProcessRpc(RpcTestBase):

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


class TestSingleProcessRpc(RpcTestBase):

    def setUp(self):
        self._port_set = set()
        master_endpoint = "127.0.0.1:{}".format(self._find_free_port())
        dist.rpc.init_rpc(worker_name(0), 0, 1, master_endpoint)
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

    def test_get_all_worker_infos(self):
        infos = dist.rpc.get_all_worker_infos()
        info = infos[0]
        self.assertEqual(info.name, worker_name(0))
        self.assertEqual(info.rank, 0)

    def test_get_current_worker_info(self):
        info = dist.rpc.get_current_worker_info()
        self.assertEqual(info.name, worker_name(0))
        self.assertEqual(info.rank, 0)


class RpcLaunchTest(RpcLaunchTestBase):

    def test_sync_rpc_paddle_add1(self):
        nnodes = 2
        nproc_per_node = 1
        pwd, _ = os.path.split(os.path.realpath(__file__))
        model_file = os.path.join(pwd, "rpc_launch_sync_add.py")
        a, b = self.create_data(nnodes, nproc_per_node)
        res = np.add(a, b)
        out = self.launch_rpc(nnodes, nproc_per_node, model_file)
        np.testing.assert_allclose(out, res, rtol=1e-05)

    def test_sync_rpc_paddle_add2(self):
        nnodes = 2
        nproc_per_node = 2
        pwd, _ = os.path.split(os.path.realpath(__file__))
        model_file = os.path.join(pwd, "rpc_launch_sync_add.py")
        a, b = self.create_data(nnodes, nproc_per_node)
        res = np.add(a, b)
        out = self.launch_rpc(nnodes, nproc_per_node, model_file)
        np.testing.assert_allclose(out, res, rtol=1e-05)


if __name__ == "__main__":
    unittest.main()
