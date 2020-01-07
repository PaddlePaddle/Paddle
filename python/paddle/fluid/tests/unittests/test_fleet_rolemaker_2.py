#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
"""Test cases for role makers."""

from __future__ import print_function
import os
import unittest

import paddle.fluid.incubate.fleet.base.role_maker as role_maker


class TestCloudRoleMaker2(unittest.TestCase):
    """
    Test cases for paddle cloud role makers.
    """

    def setUp(self):
        """Set up, set envs."""
        pass

    def test_pslib_2(self):
        """Test cases for pslib."""
        import paddle.fluid as fluid
        from paddle.fluid.incubate.fleet.parameter_server.pslib import fleet
        from paddle.fluid.incubate.fleet.parameter_server.pslib import PSLib
        from paddle.fluid.incubate.fleet.base.role_maker import \
            PaddleCloudGlooRoleMaker
        os.environ["POD_IP"] = "127.0.0.1"
        os.environ["PADDLE_PORT"] = "36001"
        os.environ["TRAINING_ROLE"] = "TRAINER"
        os.environ["PADDLE_TRAINER_ENDPOINTS"] = "127.0.0.1:36001"
        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:36002"
        os.environ["PADDLE_TRAINER_ID"] = "0"
        os.environ["PADDLE_TRAINERS_NUM"] = "1"

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        try:
            fleet.init(exe)
        except:
            print("no mpi4py, skip test_pslib_2")
            return

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

        os.environ["TRAINING_ROLE"] = "wrong"
        try:
            role1 = PaddleCloudGlooRoleMaker("", "", "./test_gloo_1", "lo")
            role1.generate_role()
        except:
            pass
        os.environ["TRAINING_ROLE"] = "PSERVER"
        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:36001"
        role2 = PaddleCloudGlooRoleMaker("", "", "./test_gloo_2", "lo")
        role2._finalize()
        role2._all_gather(1)
        role2._all_gather(1)
        role2._barrier_server()

        role3 = PaddleCloudGlooRoleMaker("", "", "./test_gloo_3", "lo")
        role3._worker_gather(1)
        role3._worker_gather(1)

        os.environ["TRAINING_ROLE"] = "TRAINER"
        os.environ["PADDLE_PSERVERS_IP_PORT_LIST"] = "127.0.0.1:36002"
        role4 = PaddleCloudGlooRoleMaker("", "", "./test_gloo_4", "lo")
        role4._worker_gather(1)
        role4._get_rank()
        role4._get_size()
        role4._all_comm.init(0, 0, "", "", "", "", "")

        role5 = PaddleCloudGlooRoleMaker("", "", "./test_gloo_5", "lo")
        role5.get_local_endpoint()
        role5.get_local_endpoint()

        role6 = PaddleCloudGlooRoleMaker("", "", "./test_gloo_6", "lo")
        role6.get_trainer_endpoints()
        role6.get_trainer_endpoints()

        role7 = PaddleCloudGlooRoleMaker("", "", "./test_gloo_7", "lo")
        role7.get_pserver_endpoints()
        role7.get_pserver_endpoints()

        role8 = PaddleCloudGlooRoleMaker("", "", "./test_gloo_8", "lo")
        role8.is_worker()
        role8.is_worker()

        role9 = PaddleCloudGlooRoleMaker("", "", "./test_gloo_9", "lo")
        role9.is_server()
        role9.is_server()

        role10 = PaddleCloudGlooRoleMaker("", "", "./test_gloo_10", "lo")
        role10.is_first_worker()
        role10.is_first_worker()

        role11 = PaddleCloudGlooRoleMaker("", "", "./test_gloo_11", "lo")
        role11.worker_index()
        role11.worker_index()

        role12 = PaddleCloudGlooRoleMaker("", "", "./test_gloo_12", "lo")
        role12.server_index()
        role12.server_index()

        role13 = PaddleCloudGlooRoleMaker("", "", "./test_gloo_13", "lo")
        role13.worker_num()
        role13.worker_num()

        role14 = PaddleCloudGlooRoleMaker("", "", "./test_gloo_14", "lo")
        role14.server_num()
        role14.server_num()

        role15 = PaddleCloudGlooRoleMaker("", "", "./test_gloo_15", "lo")
        role15._barrier_worker()
        role15._barrier_worker()

        role16 = PaddleCloudGlooRoleMaker("", "", "./test_gloo_16", "lo")
        role16._barrier_all()
        role16._barrier_all()

        role17 = PaddleCloudGlooRoleMaker("", "", "./test_gloo_17", "lo")
        role17._barrier_server()
        role17._barrier_server()

        role18 = PaddleCloudGlooRoleMaker("", "", "./test_gloo_18", "lo")
        role18._worker_num()
        role18._worker_num()

        role19 = PaddleCloudGlooRoleMaker("", "", "./test_gloo_19", "lo")
        role19._server_num()
        role19._server_num()

        role20 = PaddleCloudGlooRoleMaker("", "", "./test_gloo_20", "lo")
        a = [1]
        b = [0]
        role20.Allreduce(a, b)

        with open("test_fleet_gloo_role_maker_1.txt", "w") as f:
            data = "1 1 1 1\n"
            f.write(data)

        dataset = fluid.DatasetFactory().create_dataset("InMemoryDataset")
        dataset.set_filelist(["test_fleet_gloo_role_maker_1.txt"])
        dataset.set_use_var([show, label])
        dataset.load_into_memory()
        dataset.get_memory_data_size(fleet)
        dataset.get_shuffle_data_size(fleet)
        os.remove("./test_fleet_gloo_role_maker_1.txt")

if __name__ == "__main__":
    unittest.main()
