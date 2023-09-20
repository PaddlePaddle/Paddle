#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import subprocess
import sys
import tempfile
import unittest

import numpy

import paddle
from paddle import base
from paddle.distributed import fleet
from paddle.distributed.fleet.base import role_maker
from paddle.distributed.utils.launch_utils import find_free_ports

paddle.enable_static()


class TestCommunicatorGeoEnd2End(unittest.TestCase):
    def net(self):
        x = paddle.static.data(name='x', shape=[-1, 13], dtype='float32')
        x1 = paddle.static.data(
            name='x1', shape=[-1, 1], dtype='int64', lod_level=1
        )

        emb = paddle.static.nn.embedding(
            input=x1,
            size=[10000, 10],
            param_attr=base.ParamAttr(
                name="embedding",
                initializer=paddle.nn.initializer.Constant(value=0.01),
            ),
            is_sparse=True,
        )

        pool = paddle.static.nn.sequence_lod.sequence_pool(
            input=emb.squeeze(-2), pool_type="sum"
        )
        z = paddle.concat([x, pool], axis=1)

        y_predict = paddle.static.nn.fc(x=z, size=1)
        y = paddle.static.data(name='y', shape=[-1, 1], dtype='float32')
        cost = paddle.nn.functional.square_error_cost(input=y_predict, label=y)
        avg_cost = paddle.mean(cost)
        return avg_cost, x, x1, y

    def fake_reader(self):
        def reader():
            for i in range(10000):
                x = numpy.random.random((1, 13)).astype('float32')
                z = numpy.random.randint(0, 9999, (1, 1)).astype('int64')
                y = numpy.random.randint(0, 2, (1, 1)).astype('int64')
                yield x, z, y

        return reader

    def run_pserver(self, role, strategy):
        fleet.init(role)
        avg_cost, x, z, y = self.net()
        optimizer = paddle.optimizer.SGD(0.01)
        optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(avg_cost)

        fleet.init_server()
        fleet.run_server()

    def run_trainer(self, role, strategy):
        place = base.core.CPUPlace()
        exe = base.Executor(place)

        fleet.init(role)
        avg_cost, x, z, y = self.net()
        optimizer = paddle.optimizer.SGD(0.01)
        optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(avg_cost)

        exe.run(base.default_startup_program())
        fleet.init_worker()

        train_reader = paddle.batch(self.fake_reader(), batch_size=24)
        feeder = base.DataFeeder(place=place, feed_list=[x, z, y])

        for batch_id, data in enumerate(train_reader()):
            exe.run(
                base.default_main_program(),
                feed=feeder.feed(data),
                fetch_list=[],
            )

        fleet.stop_worker()

    def run_ut(self):
        training_role = os.getenv("TRAINING_ROLE", "TRAINER")

        os.environ["PADDLE_PSERVER_NUMS"] = "1"
        os.environ["PADDLE_TRAINERS_NUM"] = "1"
        os.environ["PADDLE_TRAINER_ID"] = "0"
        os.environ["PADDLE_TRAINERS_NUM"] = "1"
        os.environ["POD_IP"] = "127.0.0.1"

        role = role_maker.PaddleCloudRoleMaker()

        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.a_sync = True
        strategy.a_sync_configs = {"k_steps": 100}
        strategy.a_sync_configs = {"launch_barrier": False}

        if training_role == "TRAINER":
            self.run_trainer(role, strategy)
        else:
            self.run_pserver(role, strategy)

    def test_communicator(self):
        temp_dir = tempfile.TemporaryDirectory()
        pipe_name = os.path.join(temp_dir.name, 'mypipe')
        try:
            os.mkfifo(pipe_name)
        except OSError as oe:
            print(f"Failed to create pipe: {oe}")

        port = find_free_ports(1).pop()

        os.environ["TRAINING_ROLE"] = "PSERVER"
        os.environ["PADDLE_PORT"] = str(port)
        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = f"127.0.0.1:{port}"
        os.environ["PIPE_FILE"] = pipe_name

        _python = sys.executable
        server_file = "run_server_for_communicator_geo.py"
        ps_cmd = f"{_python} {server_file}"

        ps_proc = subprocess.Popen(
            ps_cmd.strip().split(" "),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        with open(pipe_name, 'r') as pipe:
            start_command = pipe.read()

        os.environ["TRAINING_ROLE"] = "TRAINER"

        self.run_ut()
        ps_proc.kill()
        ps_proc.wait()
        outs, errs = ps_proc.communicate()


if __name__ == '__main__':
    unittest.main()
