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

import paddle.fluid as fluid
import paddle.compat as cpt
import paddle.fluid.core as core
import numpy as np
import os
import shutil
import unittest
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspilerConfig
from paddle.fluid.incubate.fleet.base.role_maker import PaddleCloudRoleMaker


class TestCloudRoleMaker(unittest.TestCase):
    def setUp(self):
        """Set up, set envs."""
        os.environ["PADDLE_TRAINERS_NUM"] = "2"
        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:36001"
        """Test tr rolenamer."""
        os.environ["TRAINING_ROLE"] = "TRAINER"
        os.environ["PADDLE_TRAINER_ID"] = "0"
        """Test ps rolemaker."""
        os.environ["TRAINING_ROLE"] = "PSERVER"
        os.environ["POD_IP"] = "127.0.0.1"
        os.environ["PADDLE_PORT"] = "36001"

        # fs_name="afs://yinglong.afs.baidu.com:9902"
        # fs_ugi="paddle,paddle"

    def test_tr_rolemaker(self):
        ro = PaddleCloudRoleMaker(is_collective=False, path="./tmp_tr")
        ro.generate_role()
        self.assertTrue(ro.is_worker())
        self.assertFalse(ro.is_server())
        self.assertEqual(ro.worker_num(), 2)

    def test_ps_rolemaker(self):
        ro = PaddleCloudRoleMaker(is_collective=False, path="./tmp_ps")
        ro.generate_role()
        self.assertFalse(ro.is_worker())
        self.assertTrue(ro.is_server())
        self.assertEqual(ro.worker_num(), 2)

    def test_traing_role(self):
        """Test training role."""
        os.environ["TRAINING_ROLE"] = "TEST"
        ro = PaddleCloudRoleMaker(is_collective=False, path="./tmp_traing")
        self.assertRaises(ValueError, ro.generate_role)

    def test_paddlecloud_gloo(self):
        """Test cases for paddlecloud rolemaker."""
        import paddle.fluid as fluid
        from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
        from paddle.fluid.incubate.fleet.base.role_maker import PaddleCloudRoleMaker
        from paddle.fluid.incubate.fleet.base.role_maker import RoleMakerBase
        try:
            import netifaces
        except:
            print("warning: no netifaces, skip test_paddlecloud_rolemaker_1")
            return
        os.environ["POD_IP"] = "127.0.0.1"
        os.environ["PADDLE_PORT"] = "36001"
        os.environ["TRAINING_ROLE"] = "TRAINER"
        os.environ["PADDLE_TRAINER_ENDPOINTS"] = "127.0.0.1:36001"
        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:36002"
        os.environ["PADDLE_TRAINER_ID"] = "0"
        role_maker = PaddleCloudRoleMaker(
            is_collective=False, path="./tmp_paddlecloud_gloo")
        role_maker.generate_role()
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        fleet.init(role_maker)
        train_program = fluid.Program()
        startup_program = fluid.Program()
        scope = fluid.Scope()
        with fluid.program_guard(train_program, startup_program):
            show = fluid.layers.data(name="show", shape=[-1, 1], \
                dtype="float32", lod_level=1, append_batch_size=False)
            fc = fluid.layers.fc(input=show, size=1, act=None)
            label = fluid.layers.data(name="click", shape=[-1, 1], \
                dtype="int64", lod_level=1, append_batch_size=False)
            label_cast = fluid.layers.cast(label, dtype='float32')
            cost = fluid.layers.log_loss(fc, label_cast)
        try:
            adam = fluid.optimizer.Adam(learning_rate=0.000005)
            adam = fleet.distributed_optimizer(adam)
            adam.minimize([cost], [scope])
            fleet.run_server()
        except:
            print("do not support pslib test, skip")
            return


if __name__ == '__main__':
    #test_fleet1()
    unittest.main()
