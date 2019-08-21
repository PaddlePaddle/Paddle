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

from __future__ import print_function
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import numpy as np
import os
import shutil
import paddle.fluid.core as core
import unittest


class TestBoxPS(unittest.TestCase):
    """  TestCases for BoxPS """

    def test_boxps_cpu(self):
        self.run_boxps(True)

    def test_boxps_gpu(self):
        self.run_boxps(False)

    def run_boxps(self, is_cpu=True):
        x = fluid.layers.data(name='x', shape=[1], dtype='int64', lod_level=0)
        y = fluid.layers.data(name='y', shape=[1], dtype='int64', lod_level=0)
        emb_x, emb_y = layers.pull_box_sparse([x, y], size=2)
        concat = layers.concat([emb_x, emb_y], axis=1)
        fc = layers.fc(input=concat,
                       name="fc",
                       size=1,
                       num_flatten_dims=1,
                       bias_attr=False)
        loss = layers.reduce_mean(fc)

        place = fluid.CPUPlace() if is_cpu or not core.is_compiled_with_cuda(
        ) else fluid.CUDAPlace(0)
        exe = fluid.Executor(place)

        optimizer = fluid.optimizer.SGD(learning_rate=0.5)

        batch_size = 2

        def binary_print(slot, fout):
            fout.write(str(len(slot)) + " ")
            for e in slot:
                fout.write(str(e) + " ")

        batch1 = np.ones(
            (batch_size, 2, 1)).astype("int64").reshape(batch_size, 2, 1)
        batch2 = np.ones(
            (batch_size, 2, 1)).astype("int64").reshape(batch_size, 2, 1)
        data = [batch1, batch2]
        filelist = []
        for i in range(2):
            filelist.append("test_hdfs_" + str(i))
        for f in filelist:
            with open(f, "w") as fout:
                for batch_data in data:
                    for ins in batch_data:
                        for slot in ins:
                            binary_print(slot, fout)
                    fout.write("\n")
        dataset = fluid.DatasetFactory().create_dataset("InMemoryDataset")
        dataset.set_use_var([x, y])
        dataset.set_batch_size(2)
        dataset.set_thread(1)
        dataset.set_filelist(filelist)
        dataset.set_boxps_flag()
        dataset.load_into_memory()
        optimizer.minimize(loss)
        exe.run(fluid.default_startup_program())
        dataset.begin_pass()
        for epoch in range(1):
            exe.train_from_dataset(
                fluid.default_main_program(),
                dataset,
                thread=1,
                debug=False,
                fetch_list=[],
                fetch_info=[],
                print_period=1)
        dataset.end_pass()
        for f in filelist:
            os.remove(f)


class TestBoxPSPreload(unittest.TestCase):
    """  TestCases for BoxPS Preload """

    def test_boxps_cpu(self):
        self.run_boxps_preload(True)

    def test_boxps_gpu(self):
        self.run_boxps_preload(False)

    def run_boxps_preload(self, is_cpu=True):
        x = fluid.layers.data(name='x', shape=[1], dtype='int64', lod_level=0)
        y = fluid.layers.data(name='y', shape=[1], dtype='int64', lod_level=0)
        emb_x, emb_y = layers.pull_box_sparse([x, y], size=2)

        concat = layers.concat([emb_x, emb_y], axis=1)
        fc = layers.fc(input=concat,
                       name="fc",
                       size=1,
                       num_flatten_dims=1,
                       bias_attr=False)
        loss = layers.reduce_mean(fc)
        layers.Print(loss)

        place = fluid.CPUPlace() if is_cpu or not core.is_compiled_with_cuda(
        ) else fluid.CUDAPlace(0)
        exe = fluid.Executor(place)

        optimizer = fluid.optimizer.SGD(learning_rate=0.5)

        batch_size = 2

        def binary_print(slot, fout):
            fout.write(str(len(slot)) + " ")
            for e in slot:
                fout.write(str(e) + " ")

        batch1 = np.ones(
            (batch_size, 2, 1)).astype("int64").reshape(batch_size, 2, 1)
        batch2 = np.ones(
            (batch_size, 2, 1)).astype("int64").reshape(batch_size, 2, 1)
        data = [batch1, batch2]
        filelist = []
        for i in range(2):
            filelist.append("test_hdfs_" + str(i))
        for f in filelist:
            with open(f, "w") as fout:
                for batch_data in data:
                    for ins in batch_data:
                        for slot in ins:
                            binary_print(slot, fout)
                    fout.write("\n")

        def create_dataset():
            dataset = fluid.DatasetFactory().create_dataset("InMemoryDataset")
            dataset.set_use_var([x, y])
            dataset.set_batch_size(2)
            dataset.set_thread(1)
            dataset.set_boxps_flag()
            dataset.set_filelist(filelist)
            return dataset

        datasets = []
        datasets.append(create_dataset())
        datasets.append(create_dataset())
        datasets.append(create_dataset())

        optimizer.minimize(loss)
        exe.run(fluid.default_startup_program())

        datasets[0].load_into_memory()
        datasets[0].begin_pass()
        datasets[1].preload_into_memory()
        exe.train_from_dataset(
            program=fluid.default_main_program(),
            dataset=datasets[0],
            print_period=1)

        datasets[0].end_pass()
        datasets[1].wait_preload_done()
        datasets[2].preload_into_memory()
        exe.train_from_dataset(
            program=fluid.default_main_program(),
            dataset=datasets[1],
            print_period=1)
        datasets[1].end_pass()
        datasets[2].wait_preload_done()
        exe.train_from_dataset(
            program=fluid.default_main_program(),
            dataset=datasets[2],
            print_period=1)
        datasets[2].end_pass()

        for f in filelist:
            os.remove(f)


if __name__ == '__main__':
    unittest.main()
