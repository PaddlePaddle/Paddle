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
"""
TestCases for Monitor
"""

from __future__ import print_function
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import numpy as np
import os
import unittest


class TestDatasetWithStat(unittest.TestCase):
    """  TestCases for Dataset. """

    def setUp(self):
        self.use_data_loader = False
        self.epoch_num = 10
        self.drop_last = False

    def test_dataset_run_with_stat(self):
        with open("test_in_memory_dataset_run_a.txt", "w") as f:
            data = "1 1 2 3 3 4 5 5 5 5 1 1\n"
            data += "1 2 2 3 4 4 6 6 6 6 1 2\n"
            data += "1 3 2 3 5 4 7 7 7 7 1 3\n"
            f.write(data)
        with open("test_in_memory_dataset_run_b.txt", "w") as f:
            data = "1 4 2 3 3 4 5 5 5 5 1 4\n"
            data += "1 5 2 3 4 4 6 6 6 6 1 5\n"
            data += "1 6 2 3 5 4 7 7 7 7 1 6\n"
            data += "1 7 2 3 6 4 8 8 8 8 1 7\n"
            f.write(data)

        slots = ["slot1", "slot2", "slot3", "slot4"]
        slots_vars = []
        for slot in slots:
            var = fluid.layers.data(
                name=slot, shape=[1], dtype="int64", lod_level=1)
            slots_vars.append(var)

        dataset = paddle.distributed.fleet.DatasetFactory().create_dataset(
            "InMemoryDataset")
        dataset.set_batch_size(32)
        dataset.set_thread(3)
        dataset.set_filelist([
            "test_in_memory_dataset_run_a.txt",
            "test_in_memory_dataset_run_b.txt"
        ])
        dataset.set_pipe_command("cat")
        dataset.set_use_var(slots_vars)
        dataset.load_into_memory()
        dataset.set_fea_eval(1, True)
        dataset.slots_shuffle(["slot1"])

        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(fluid.default_startup_program())
        if self.use_data_loader:
            data_loader = fluid.io.DataLoader.from_dataset(dataset,
                                                           fluid.cpu_places(),
                                                           self.drop_last)
            for i in range(self.epoch_num):
                for data in data_loader():
                    exe.run(fluid.default_main_program(), feed=data)
        else:
            for i in range(self.epoch_num):
                try:
                    exe.train_from_dataset(fluid.default_main_program(),
                                           dataset)
                except Exception as e:
                    self.assertTrue(False)

        int_stat = core.get_int_stats()
        # total 56 keys
        print(int_stat["STAT_total_feasign_num_in_mem"])

        os.remove("./test_in_memory_dataset_run_a.txt")
        os.remove("./test_in_memory_dataset_run_b.txt")


if __name__ == '__main__':
    unittest.main()
