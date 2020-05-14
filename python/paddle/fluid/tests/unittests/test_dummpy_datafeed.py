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
from __future__ import print_function
import paddle.fluid as fluid
import paddle.fluid.core as core
import os
import unittest
import paddle.fluid.layers as layers
from paddle.fluid.layers.nn import _pull_box_sparse


class TestDataFeed(unittest.TestCase):
    """  TestBaseCase(Merge PV)   """

    def setUp(self):
        self.batch_size = 10

    def set_data_config(self):
        self.dataset = fluid.DatasetFactory().create_dataset("DummpyDataset")
        self.dataset.set_thread(1)
        self.dataset.set_batch_size(self.batch_size)

    def test_dummpydatafeed(self):
        self.run_dataset(False)

    def run_dataset(self, is_cpu):
        x = fluid.layers.data(name='x', shape=[1], dtype='int64', lod_level=0)
        y = fluid.layers.data(name='y', shape=[1], dtype='int64', lod_level=0)

        fc = layers.fc(input=x,
                       name="fc",
                       size=1,
                       num_flatten_dims=1,
                       bias_attr=False)
        loss = layers.reduce_mean(fc)
        place = fluid.CPUPlace() if is_cpu or not core.is_compiled_with_cuda(
        ) else fluid.CUDAPlace(0)
        exe = fluid.Executor(place)

        with open("test_run_with_dump_b.txt", "w") as f:
            data = "1 1702f830fff22201ad7429505f715c1d 1 1 1 1\n"
            data += "1 1702f830fff22202ad7429505f715c1d 1 2 1 2\n"
            data += "1 1702f830fff22203ad7429505f715c1d 1 3 1 3\n"
            data += "1 1702f830fff22101ad7429505f714ccd 1 4 1 4\n"
            data += "1 1702f830fff22102ad7429505f714ccd 1 5 1 5\n"
            data += "1 1702f830fff22103ad7429505f714ccd 1 6 1 6\n"
            data += "1 1702f830fff22104ad7429505f714ccd 1 6 1 7\n"
            f.write(data)

        self.set_data_config()
        self.dataset.set_use_var([x, y])
        self.dataset.set_filelist(["test_run_with_dump_b.txt"])

        optimizer = fluid.optimizer.SGD(learning_rate=0.5)
        optimizer = fluid.optimizer.PipelineOptimizer(
            optimizer,
            cut_list=[],
            place_list=[place],
            concurrency_list=[1],
            queue_size=1,
            sync_steps=-1)
        optimizer.minimize(loss)
        exe.run(fluid.default_startup_program())

        exe.train_from_dataset(
            program=fluid.default_main_program(),
            dataset=self.dataset,
            print_period=1)
        exe.train_from_dataset(
            program=fluid.default_main_program(),
            dataset=self.dataset,
            print_period=1)
        os.remove("test_run_with_dump_b.txt")


if __name__ == '__main__':
    unittest.main()
