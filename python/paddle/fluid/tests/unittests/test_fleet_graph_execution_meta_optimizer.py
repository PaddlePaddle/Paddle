# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import paddle
import os
from launch_function_helper import launch_func, wait, _find_free_port

paddle.enable_static()


class TestFleetGraphExecutionMetaOptimizer(unittest.TestCase):
    def setUp(self):
        try:
            self._dist_ut_port_0 = int(os.environ["PADDLE_DIST_UT_PORT"])
            self._dist_ut_port_1 = self._dist_ut_port_0 + 1
        except Exception as e:
            self._dist_ut_port_0 = _find_free_port(set())
            self._dist_ut_port_1 = _find_free_port(set())

    def test_graph_execution_optimizer_not_apply(self):
        port_a = self._dist_ut_port_0
        port_b = self._dist_ut_port_1
        node_a = {
            "PADDLE_TRAINER_ID": "0",
            "PADDLE_CURRENT_ENDPOINT": "127.0.0.1:{}".format(port_a),
            "PADDLE_TRAINERS_NUM": "2",
            "PADDLE_TRAINER_ENDPOINTS":
            "127.0.0.1:{},127.0.0.1:{}".format(port_a, port_b),
            "http_proxy": "",
            "https_proxy": ""
        }

        node_b = {
            "PADDLE_TRAINER_ID": "1",
            "PADDLE_CURRENT_ENDPOINT": "127.0.0.1:{}".format(port_b),
            "PADDLE_TRAINERS_NUM": "2",
            "PADDLE_TRAINER_ENDPOINTS":
            "127.0.0.1:{},127.0.0.1:{}".format(port_a, port_b),
            "http_proxy": "",
            "https_proxy": ""
        }

        def node_func():
            import paddle.distributed.fleet as fleet
            fleet.init(is_collective=True)
            input_x = paddle.fluid.layers.data(
                name="x", shape=[32], dtype='float32')
            input_y = paddle.fluid.layers.data(
                name="y", shape=[1], dtype='int64')

            fc_1 = paddle.fluid.layers.fc(input=input_x, size=64, act='tanh')
            fc_2 = paddle.fluid.layers.fc(input=fc_1, size=64, act='tanh')
            prediction = paddle.fluid.layers.fc(input=[fc_2],
                                                size=2,
                                                act='softmax')
            cost = paddle.fluid.layers.cross_entropy(
                input=prediction, label=input_y)
            avg_cost = paddle.fluid.layers.mean(x=cost)

            strategy = paddle.distributed.fleet.DistributedStrategy()
            optimizer = paddle.fluid.optimizer.SGD(learning_rate=0.01)
            optimizer = fleet.distributed_optimizer(
                optimizer, strategy=strategy)
            optimizer.minimize(avg_cost)

            exe = paddle.fluid.Executor(place=paddle.fluid.CPUPlace())
            exe.run(paddle.fluid.default_startup_program())

        proc_a = launch_func(node_func, node_a)
        proc_a.start()
        proc_b = launch_func(node_func, node_b)
        proc_b.start()
        wait([proc_a, proc_b])

    def test_graph_execution_optimizer(self):
        port_a = self._dist_ut_port_0 + 2
        port_b = self._dist_ut_port_1 + 2

        node_a = {
            "PADDLE_TRAINER_ID": "0",
            "PADDLE_CURRENT_ENDPOINT": "127.0.0.1:{}".format(port_a),
            "PADDLE_TRAINERS_NUM": "2",
            "PADDLE_TRAINER_ENDPOINTS":
            "127.0.0.1:{},127.0.0.1:{}".format(port_a, port_b),
            "http_proxy": "",
            "https_proxy": ""
        }

        node_b = {
            "PADDLE_TRAINER_ID": "1",
            "PADDLE_CURRENT_ENDPOINT": "127.0.0.1:{}".format(port_b),
            "PADDLE_TRAINERS_NUM": "2",
            "PADDLE_TRAINER_ENDPOINTS":
            "127.0.0.1:{},127.0.0.1:{}".format(port_a, port_b),
            "http_proxy": "",
            "https_proxy": ""
        }

        def node_func():
            import paddle.distributed.fleet as fleet
            fleet.init(is_collective=True)
            input_x = paddle.fluid.layers.data(
                name="x", shape=[32], dtype='float32')
            input_y = paddle.fluid.layers.data(
                name="y", shape=[1], dtype='int64')

            fc_1 = paddle.fluid.layers.fc(input=input_x, size=64, act='tanh')
            fc_2 = paddle.fluid.layers.fc(input=fc_1, size=64, act='tanh')
            prediction = paddle.fluid.layers.fc(input=[fc_2],
                                                size=2,
                                                act='softmax')
            cost = paddle.fluid.layers.cross_entropy(
                input=prediction, label=input_y)
            avg_cost = paddle.fluid.layers.mean(x=cost)

            strategy = paddle.distributed.fleet.DistributedStrategy()
            strategy.nccl_comm_num = 2
            strategy.sync_nccl_allreduce = True
            optimizer = paddle.fluid.optimizer.SGD(learning_rate=0.01)
            optimizer = fleet.distributed_optimizer(
                optimizer, strategy=strategy)
            optimizer.minimize(avg_cost)
            exe = paddle.fluid.Executor(place=paddle.fluid.CPUPlace())
            exe.run(paddle.fluid.default_startup_program())

            import numpy as np

            def gen_data():
                return {
                    "x": np.random.random(size=(128, 32)).astype('float32'),
                    "y": np.random.randint(
                        2, size=(128, 1)).astype('int64')
                }

            for i in range(10):
                cost_val = exe.run(feed=gen_data(), fetch_list=[avg_cost.name])
                print("cost of step[{}] = {}".format(i, cost_val))

        proc_a = launch_func(node_func, node_a)
        proc_a.start()
        proc_b = launch_func(node_func, node_b)
        proc_b.start()
        wait([proc_a, proc_b])

    def test_graph_execution_optimizer_not_apply_v2(self):
        port_a = self._dist_ut_port_0 + 4
        port_b = self._dist_ut_port_1 + 4
        node_a = {
            "PADDLE_TRAINER_ID": "0",
            "PADDLE_CURRENT_ENDPOINT": "127.0.0.1:{}".format(port_a),
            "PADDLE_TRAINERS_NUM": "2",
            "PADDLE_TRAINER_ENDPOINTS":
            "127.0.0.1:{},127.0.0.1:{}".format(port_a, port_b),
            "http_proxy": "",
            "https_proxy": ""
        }

        node_b = {
            "PADDLE_TRAINER_ID": "1",
            "PADDLE_CURRENT_ENDPOINT": "127.0.0.1:{}".format(port_b),
            "PADDLE_TRAINERS_NUM": "2",
            "PADDLE_TRAINER_ENDPOINTS":
            "127.0.0.1:{},127.0.0.1:{}".format(port_a, port_b),
            "http_proxy": "",
            "https_proxy": ""
        }

        def node_func():
            import paddle.distributed.fleet as fleet
            fleet.init(is_collective=True)
            input_x = paddle.fluid.layers.data(
                name="x", shape=[32], dtype='float32')
            input_y = paddle.fluid.layers.data(
                name="y", shape=[1], dtype='int64')

            fc_1 = paddle.fluid.layers.fc(input=input_x, size=64, act='tanh')
            fc_2 = paddle.fluid.layers.fc(input=fc_1, size=64, act='tanh')
            prediction = paddle.fluid.layers.fc(input=[fc_2],
                                                size=2,
                                                act='softmax')
            cost = paddle.fluid.layers.cross_entropy(
                input=prediction, label=input_y)
            avg_cost = paddle.fluid.layers.mean(x=cost)

            strategy = paddle.distributed.fleet.DistributedStrategy()
            optimizer = paddle.fluid.optimizer.SGD(learning_rate=0.01)
            optimizer = fleet.distributed_optimizer(
                optimizer, strategy=strategy)
            optimizer.minimize(avg_cost)

            exe = paddle.fluid.Executor(place=paddle.fluid.CPUPlace())
            exe.run(paddle.fluid.default_startup_program())

        proc_a = launch_func(node_func, node_a)
        proc_a.start()
        proc_b = launch_func(node_func, node_b)
        proc_b.start()
        wait([proc_a, proc_b])

    def test_graph_execution_optimizer_v2(self):
        port_a = self._dist_ut_port_0 + 6
        port_b = self._dist_ut_port_1 + 6
        node_a = {
            "PADDLE_TRAINER_ID": "0",
            "PADDLE_CURRENT_ENDPOINT": "127.0.0.1:{}".format(port_a),
            "PADDLE_TRAINERS_NUM": "2",
            "PADDLE_TRAINER_ENDPOINTS":
            "127.0.0.1:{},127.0.0.1:{}".format(port_a, port_b),
            "http_proxy": "",
            "https_proxy": ""
        }

        node_b = {
            "PADDLE_TRAINER_ID": "1",
            "PADDLE_CURRENT_ENDPOINT": "127.0.0.1:{}".format(port_b),
            "PADDLE_TRAINERS_NUM": "2",
            "PADDLE_TRAINER_ENDPOINTS":
            "127.0.0.1:{},127.0.0.1:{}".format(port_a, port_b),
            "http_proxy": "",
            "https_proxy": ""
        }

        def node_func():
            import paddle.distributed.fleet as fleet
            fleet.init(is_collective=True)
            input_x = paddle.fluid.layers.data(
                name="x", shape=[32], dtype='float32')
            input_y = paddle.fluid.layers.data(
                name="y", shape=[1], dtype='int64')

            fc_1 = paddle.fluid.layers.fc(input=input_x, size=64, act='tanh')
            fc_2 = paddle.fluid.layers.fc(input=fc_1, size=64, act='tanh')
            prediction = paddle.fluid.layers.fc(input=[fc_2],
                                                size=2,
                                                act='softmax')
            cost = paddle.fluid.layers.cross_entropy(
                input=prediction, label=input_y)
            avg_cost = paddle.fluid.layers.mean(x=cost)

            strategy = paddle.distributed.fleet.DistributedStrategy()
            strategy.nccl_comm_num = 2
            strategy.sync_nccl_allreduce = True
            optimizer = paddle.fluid.optimizer.SGD(learning_rate=0.01)
            optimizer = fleet.distributed_optimizer(
                optimizer, strategy=strategy)
            optimizer.minimize(avg_cost)
            exe = paddle.fluid.Executor(place=paddle.fluid.CPUPlace())
            exe.run(paddle.fluid.default_startup_program())

            import numpy as np

            def gen_data():
                return {
                    "x": np.random.random(size=(128, 32)).astype('float32'),
                    "y": np.random.randint(
                        2, size=(128, 1)).astype('int64')
                }

            for i in range(10):
                cost_val = exe.run(feed=gen_data(), fetch_list=[avg_cost.name])
                print("cost of step[{}] = {}".format(i, cost_val))

        proc_a = launch_func(node_func, node_a)
        proc_a.start()
        proc_b = launch_func(node_func, node_b)
        proc_b.start()
        wait([proc_a, proc_b])


if __name__ == "__main__":
    unittest.main()
