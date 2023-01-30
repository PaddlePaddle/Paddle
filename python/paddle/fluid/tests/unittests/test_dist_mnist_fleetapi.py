# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

<<<<<<< HEAD
import unittest

from test_dist_base import TestDistBase

=======
from __future__ import print_function
import unittest
from test_dist_base import TestDistBase
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
import paddle

paddle.enable_static()


class TestDistMnistNCCL2FleetApi(TestDistBase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def _setup_config(self):
        self._sync_mode = True
        self._use_reduce = False
        self._use_reader_alloc = False
        self._nccl2_mode = True
        self._use_fleet_api = True
        self._sync_batch_norm = True

    def test_dist_train(self):
        import paddle.fluid as fluid
<<<<<<< HEAD

=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        if fluid.core.is_compiled_with_cuda():
            self.check_with_place(
                "dist_mnist.py",
                delta=1e-5,
                check_error_log=True,
<<<<<<< HEAD
                need_envs={'FLAGS_allreduce_record_one_event': '1'},
            )


class FleetCollectiveTest(unittest.TestCase):
    def test_open_sync_batch_norm(self):
        import paddle.fluid as fluid
        import paddle.fluid.incubate.fleet.base.role_maker as role_maker
        from paddle.fluid.incubate.fleet.collective import (
            DistributedStrategy,
            fleet,
        )
=======
                need_envs={'FLAGS_allreduce_record_one_event': '1'})


class FleetCollectiveTest(unittest.TestCase):

    def test_open_sync_batch_norm(self):
        import paddle.fluid as fluid
        import paddle.fluid.incubate.fleet.base.role_maker as role_maker
        from paddle.fluid.incubate.fleet.collective import fleet, DistributedStrategy
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        if not fluid.core.is_compiled_with_cuda():
            # Operator "gen_nccl_id" has not been registered
            return

<<<<<<< HEAD
        data = paddle.static.data(name='X', shape=[-1, 1], dtype='float32')
        hidden = paddle.static.nn.fc(x=data, size=10)
=======
        data = fluid.layers.data(name='X', shape=[1], dtype='float32')
        hidden = fluid.layers.fc(input=data, size=10)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        loss = paddle.mean(hidden)

        optimizer = fluid.optimizer.AdamOptimizer()

        role = role_maker.UserDefinedCollectiveRoleMaker(0, ['127.0.0.1:6170'])
        fleet.init(role)

        dist_strategy = DistributedStrategy()
        dist_strategy.sync_batch_norm = True

<<<<<<< HEAD
        dist_optimizer = fleet.distributed_optimizer(
            optimizer, strategy=dist_strategy
        )
=======
        dist_optimizer = fleet.distributed_optimizer(optimizer,
                                                     strategy=dist_strategy)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        dist_optimizer.minimize(loss)

        self.assertEqual(dist_strategy.exec_strategy.num_threads, 1)


if __name__ == "__main__":
    unittest.main()
