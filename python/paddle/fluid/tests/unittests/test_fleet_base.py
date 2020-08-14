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
import paddle.fluid as fluid


class TestFleetBase(unittest.TestCase):
    def setUp(self):
        os.environ["POD_IP"] = "127.0.0.1"
        os.environ["PADDLE_TRAINER_ENDPOINTS"] = "127.0.0.1:36001"
        os.environ["PADDLE_TRAINERS_NUM"] = "2"
        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = \
                       "127.0.0.1:36001,127.0.0.2:36001"

    def test_init(self):
        import paddle.distributed.fleet as fleet
        import paddle.fluid.incubate.fleet.base.role_maker as role_maker
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)

    def test_is_first_worker(self):
        import paddle.distributed.fleet as fleet
        import paddle.fluid.incubate.fleet.base.role_maker as role_maker
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        if fleet.is_first_worker():
            print("test fleet first worker done.")

    def test_worker_index(self):
        import paddle.distributed.fleet as fleet
        import paddle.fluid.incubate.fleet.base.role_maker as role_maker
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        print(fleet.worker_index())

    def test_worker_num(self):
        import paddle.distributed.fleet as fleet
        import paddle.fluid.incubate.fleet.base.role_maker as role_maker
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        print(fleet.worker_num())

    def test_is_worker(self):
        import paddle.distributed.fleet as fleet
        import paddle.fluid.incubate.fleet.base.role_maker as role_maker
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        if fleet.is_worker():
            print("test fleet is worker")

    def test_worker_endpoints(self):
        import paddle.distributed.fleet as fleet
        import paddle.fluid.incubate.fleet.base.role_maker as role_maker
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        print(fleet.worker_endpoints(to_string=True))

    def test_server_num(self):
        import paddle.distributed.fleet as fleet
        import paddle.fluid.incubate.fleet.base.role_maker as role_maker
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        if fleet.is_server():
            print("fleet server num: {}".format(fleet.server_num()))

    def test_server_index(self):
        import paddle.distributed.fleet as fleet
        import paddle.fluid.incubate.fleet.base.role_maker as role_maker
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        if fleet.is_server():
            print("fleet server index: {}".format(fleet.server_index()))

    def test_server_endpoints(self):
        import paddle.distributed.fleet as fleet
        import paddle.fluid.incubate.fleet.base.role_maker as role_maker
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        if fleet.is_server():
            print("fleet server index: {}".format(
                fleet.server_endpoints(to_string=True)))

    def test_is_server(self):
        import paddle.distributed.fleet as fleet
        import paddle.fluid.incubate.fleet.base.role_maker as role_maker
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        if fleet.is_server():
            print("test fleet is server")

    def test_util(self):
        import paddle.distributed.fleet as fleet
        import paddle.fluid.incubate.fleet.base.role_maker as role_maker
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        self.assertEqual(fleet.util, None)

    def test_barrier_worker(self):
        import paddle.distributed.fleet as fleet
        import paddle.fluid.incubate.fleet.base.role_maker as role_maker
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        if fleet.is_worker():
            fleet.barrier_worker()

    def test_init_worker(self):
        import paddle.distributed.fleet as fleet
        import paddle.fluid.incubate.fleet.base.role_maker as role_maker
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        if fleet.is_worker():
            fleet.init_worker()

    def test_run_server(self):
        import paddle.distributed.fleet as fleet
        import paddle.fluid.incubate.fleet.base.role_maker as role_maker
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        if fleet.is_worker():
            fleet.run_worker()

    def test_stop_worker(self):
        import paddle.distributed.fleet as fleet
        import paddle.fluid.incubate.fleet.base.role_maker as role_maker
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        if fleet.is_worker():
            fleet.stop_worker()

    def test_distributed_optimizer(self):
        import paddle.distributed.fleet as fleet
        import paddle.fluid.incubate.fleet.base.role_maker as role_maker
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        strategy = fleet.DistributedStrategy()
        optimizer = paddle.optimizer.SGD(learning_rate=0.001)
        optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)

    def test_exception(self):
        import paddle.distributed.fleet as fleet
        self.assertRaises(Exception, fleet.init_worker)

    def _net(self):
        import paddle

        input_x = paddle.fluid.layers.data(
            name="x", shape=[32], dtype='float32')
        input_y = paddle.fluid.layers.data(name="y", shape=[1], dtype='int64')

        fc_1 = paddle.fluid.layers.fc(input=input_x, size=64, act='tanh')
        fc_2 = paddle.fluid.layers.fc(input=fc_1, size=64, act='tanh')
        prediction = paddle.fluid.layers.fc(input=[fc_2], size=2, act='softmax')
        cost = paddle.fluid.layers.cross_entropy(
            input=prediction, label=input_y)
        avg_cost = paddle.fluid.layers.mean(x=cost)
        return avg_cost

    def test_collective_minimize(self):
        import paddle
        import paddle.distributed.fleet as fleet
        import paddle.fluid.incubate.fleet.base.role_maker as role_maker

        avg_cost = self._net()
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        strategy = fleet.DistributedStrategy()
        optimizer = paddle.optimizer.SGD(learning_rate=0.001)
        optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
        optimizer.minimize(avg_cost)

    def test_ps_minimize(self):
        import paddle
        from paddle.distributed.fleet import fleet
        import paddle.fluid.incubate.fleet.base.role_maker as role_maker

        os.environ["TRAINING_ROLE"] = "PSERVER"
        os.environ["POD_IP"] = "127.0.0.1"
        os.environ["PADDLE_PORT"] = "36001"

        avg_cost = self._net()
        role = role_maker.PaddleCloudRoleMaker(is_collective=False)
        fleet.init(role)
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.a_sync = False
        optimizer = paddle.optimizer.SGD(learning_rate=0.001)
        optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
        optimizer.minimize(avg_cost)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        pe = fluid.ParallelExecutor(use_cuda=False, loss_name=avg_cost.name)
        compiled_prog = fluid.compiler.CompiledProgram(
            fluid.default_main_program())
        self.assertRaises(
            Exception,
            fleet.save_inference_model,
            dirname='/tmp/',
            feeded_var_names=['x', 'y'],
            target_vars=[avg_cost],
            executor=pe)

        self.assertRaises(
            Exception,
            fleet.save_inference_model,
            dirname='/tmp/',
            feeded_var_names=['x', 'y'],
            target_vars=[avg_cost],
            executor="exe")

        self.assertRaises(
            Exception,
            fleet.save_inference_model,
            dirname='/tmp/',
            feeded_var_names=['x', 'y'],
            target_vars=[avg_cost],
            executor=exe,
            main_program=compiled_prog)

        self.assertRaises(
            Exception,
            fleet.save_persistables,
            dirname='/tmp/',
            feeded_var_names=['x', 'y'],
            target_vars=[avg_cost],
            executor=pe)

        self.assertRaises(
            Exception,
            fleet.save_persistables,
            dirname='/tmp/',
            feeded_var_names=['x', 'y'],
            target_vars=[avg_cost],
            executor="exe")

        self.assertRaises(
            Exception,
            fleet.save_persistables,
            dirname='/tmp/',
            feeded_var_names=['x', 'y'],
            target_vars=[avg_cost],
            executor=exe,
            main_program=compiled_prog)


if __name__ == "__main__":
    unittest.main()
