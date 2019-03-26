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

from __future__ import print_function
import paddle.fluid as fluid
import numpy as np
import os
import shutil
import unittest


class TestDataset(unittest.TestCase):
    """  TestCases for Dataset. """
    def test_dataset_create(self):
        """ Testcase for dataset create """
        try:
            dataset = fluid.DatasetFactory().create_dataset("InMemoryDataset")
        except:
            self.assertTrue(False)

        try:
            dataset = fluid.DatasetFactory().create_dataset("QueueDataset")
        except:
            self.assertTrue(False)

        try:
            dataset = fluid.DatasetFactory().create_dataset("MyOwnDataset")
            self.assertTrue(False)
        except:
            self.assertTrue(True)

    def test_dataset_config(self):
        """ Testcase for dataset configuration """
        dataset = fluid.core.Dataset("MultiSlotDataset")
        dataset.set_thread_num(12)
        dataset.set_filelist(["a.txt", "b.txt", "c.txt"])
        dataset.set_trainer_num(4)
        dataset.set_hdfs_config("my_fs_name", "my_fs_ugi")

        thread_num = dataset.get_thread_num()
        self.assertEqual(thread_num, 12)

        filelist = dataset.get_filelist()
        self.assertEqual(len(filelist), 3)
        self.assertEqual(filelist[0], "a.txt")
        self.assertEqual(filelist[1], "b.txt")
        self.assertEqual(filelist[2], "c.txt")

        trainer_num = dataset.get_trainer_num()
        self.assertEqual(trainer_num, 4)

        name, ugi = dataset.get_hdfs_config()
        self.assertEqual(name, "my_fs_name")
        self.assertEqual(ugi, "my_fs_ugi")

    def test_in_memory_dataset_run(self):
        """
        Testcase for InMemoryDataset from create to run
        """
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

        slots = ["slot1","slot2","slot3","slot4"]
        slots_vars = []
        for slot in slots:
            var = fluid.layers.data(name=slot, shape=[1],
                                    dtype="int64", lod_level=1)
            slots_vars.append(var)

        dataset = fluid.DatasetFactory().create_dataset("InMemoryDataset")
        dataset.set_batch_size(32)
        dataset.set_thread(3)
        dataset.set_filelist(["test_in_memory_dataset_run_a.txt",
                              "test_in_memory_dataset_run_b.txt"])
        dataset.set_pipe_command("cat")
        dataset.set_use_var(slots_vars)
        dataset.load_into_memory()
        dataset.local_shuffle()

        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(fluid.default_startup_program())
        for i in range(2):
            try:
                exe.train_from_dataset(fluid.default_main_program(), dataset)
            except:
                self.assertTrue(False)

        os.remove("./test_in_memory_dataset_run_a.txt")
        os.remove("./test_in_memory_dataset_run_b.txt")

    def test_queue_dataset_run(self):
        """
        Testcase for QueueDataset from create to run
        """
        with open("test_queue_dataset_run_a.txt", "w") as f:
            data = "1 1 2 3 3 4 5 5 5 5 1 1\n"
            data += "1 2 2 3 4 4 6 6 6 6 1 2\n"
            data += "1 3 2 3 5 4 7 7 7 7 1 3\n"
            f.write(data)
        with open("test_queue_dataset_run_b.txt", "w") as f:
            data = "1 4 2 3 3 4 5 5 5 5 1 4\n"
            data += "1 5 2 3 4 4 6 6 6 6 1 5\n"
            data += "1 6 2 3 5 4 7 7 7 7 1 6\n"
            data += "1 7 2 3 6 4 8 8 8 8 1 7\n"
            f.write(data)

        slots = ["slot1","slot2","slot3","slot4"]
        slots_vars = []
        for slot in slots:
            var = fluid.layers.data(name=slot, shape=[1],
                                    dtype="int64", lod_level=1)
            slots_vars.append(var)

        dataset = fluid.DatasetFactory().create_dataset("QueueDataset")
        dataset.set_batch_size(32)
        dataset.set_thread(3)
        dataset.set_filelist(["test_queue_dataset_run_a.txt",
                              "test_queue_dataset_run_b.txt"])
        dataset.set_pipe_command("cat")
        dataset.set_use_var(slots_vars)

        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(fluid.default_startup_program())
        for i in range(2):
            try:
                exe.train_from_dataset(fluid.default_main_program(), dataset)
            except:
                self.assertTrue(False)

        os.remove("./test_queue_dataset_run_a.txt")
        os.remove("./test_queue_dataset_run_b.txt")


if __name__ == '__main__':
    unittest.main()
