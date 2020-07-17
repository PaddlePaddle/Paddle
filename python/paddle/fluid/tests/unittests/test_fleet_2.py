#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""Test fleet."""

from __future__ import print_function
import os
import paddle.fluid as fluid
import unittest
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.incubate.fleet.parameter_server.pslib import fleet
from paddle.fluid.incubate.fleet.parameter_server.pslib import \
    fleet_embedding, _prepare_params, _fleet_embedding, \
    _fleet_embedding_v2, FLEET_GLOBAL_DICT
from paddle.fluid.incubate.fleet.base.role_maker import GeneralRoleMaker


class TestFleet2(unittest.TestCase):
    """Test cases for fleet ops."""

    def test_in_memory_dataset_run_fleet(self):
        """
        Testcase for InMemoryDataset from create to run.
        """
        with open("test_in_memory_dataset_run_fleet_a.txt", "w") as f:
            data = "1 1 1 2 2 3 3 4 5 5 5 5 1 1\n"
            data += "1 0 1 3 2 3 4 4 6 6 6 6 1 2\n"
            data += "1 1 1 4 2 3 5 4 7 7 7 7 1 3\n"
            f.write(data)
        with open("test_in_memory_dataset_run_fleet_b.txt", "w") as f:
            data = "1 0 1 5 2 3 3 4 5 5 5 5 1 4\n"
            data += "1 1 1 6 2 3 4 4 6 6 6 6 1 5\n"
            data += "1 0 1 7 2 3 5 4 7 7 7 7 1 6\n"
            data += "1 1 1 8 2 3 6 4 8 8 8 8 1 7\n"
            f.write(data)

        slots = ["click", "slot1", "slot2", "slot3", "slot4"]
        slots_vars = []
        for slot in slots:
            var = fluid.layers.data(
                name=slot, shape=[1], dtype="int64", lod_level=1)
            slots_vars.append(var)
        click = slots_vars[0]
        embs = []
        for slot in slots_vars[1:3]:
            with fleet_embedding(click_name=click.name):
                emb = fluid.layers.embedding(input=slot, size=[-1, 11], \
                    is_sparse=True, is_distributed=True, \
                    param_attr=fluid.ParamAttr(name="embedding"))
                embs.append(emb)
        for slot in slots_vars[3:5]:
            with fleet_embedding(click_name=click.name):
                emb = fluid.embedding(input=slot, size=[-1, 11], \
                    is_sparse=True, is_distributed=True, \
                    param_attr=fluid.ParamAttr(name="embedding"))
                emb = fluid.layers.reshape(emb, [-1, 11])
                embs.append(emb)
        concat = fluid.layers.concat([embs[0], embs[3]], axis=1)
        fc = fluid.layers.fc(input=concat, size=1, act=None)
        label_cast = fluid.layers.cast(slots_vars[1], dtype='float32')
        cost = fluid.layers.log_loss(fc, label_cast)
        cost = fluid.layers.mean(cost)

        try:
            fleet.init()
            adam = fluid.optimizer.Adam(learning_rate=0.000005)
            adam = fleet.distributed_optimizer(adam)
            scope = fluid.Scope()
            adam.minimize([cost], [scope])
        except:
            print("do not support pslib test, skip")
            return

        dataset = fluid.DatasetFactory().create_dataset("InMemoryDataset")
        dataset.set_batch_size(1)
        dataset.set_thread(2)
        dataset.set_filelist([
            "test_in_memory_dataset_run_fleet_a.txt",
            "test_in_memory_dataset_run_fleet_b.txt"
        ])
        dataset.set_pipe_command("cat")
        dataset.set_use_var(slots_vars)
        dataset.load_into_memory()

        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(fluid.default_startup_program())
        exe.train_from_dataset(fluid.default_main_program(), dataset)
        fleet._opt_info["stat_var_names"] = ["233"]
        exe.infer_from_dataset(fluid.default_main_program(), dataset)
        fleet._opt_info = None
        fleet._fleet_ptr = None
        os.remove("./test_in_memory_dataset_run_fleet_a.txt")
        os.remove("./test_in_memory_dataset_run_fleet_b.txt")


if __name__ == "__main__":
    unittest.main()
