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
import paddle.fluid.layers as layers
import numpy as np
import os
import shutil
import unittest


class TestPipelineConfig(unittest.TestCase):
    """  TestCases for Config in Pipeline Training. """

    def config(self, filelist_length, pipeline_num, reader_concurrency):
        filelist = []
        for i in range(filelist_length):
            filelist.append("file" + str(i))
        self.dataset.set_filelist(filelist)
        self.pipeline_opt["concurrency_list"][0] = reader_concurrency
        self.pipeline_num = pipeline_num

    def helper(self, in_filelist_length, in_pipeline_num, in_reader_concurrency,
               out_pipeline_num, out_reader_concurrency, out_dataset_thread):
        self.config(in_filelist_length, in_pipeline_num, in_reader_concurrency)
        res = self.exe._adjust_pipeline_resource(
            self.pipeline_opt, self.dataset, self.pipeline_num)
        self.assertEqual(self.pipeline_opt["concurrency_list"][0],
                         out_reader_concurrency)
        self.assertEqual(res, out_pipeline_num)
        self.assertEqual(self.dataset.thread_num, out_dataset_thread)

    def test_adjust_pipeline_resource(self):
        self.exe = fluid.Executor(fluid.CPUPlace())
        self.dataset = fluid.DatasetFactory().create_dataset(
            "FileInstantDataset")
        self.pipeline_opt = {"concurrency_list": [0, 1, 2]}
        self.pipeline_num = 0

        self.helper(7, 2, 2, 2, 2, 4)
        self.helper(7, 2, 3, 2, 3, 6)
        self.helper(7, 2, 4, 2, 3, 6)

        self.helper(8, 2, 3, 2, 3, 6)
        self.helper(8, 2, 4, 2, 4, 8)
        self.helper(8, 2, 5, 2, 4, 8)

        self.helper(3, 4, 1, 3, 1, 3)
        self.helper(3, 4, 2, 3, 1, 3)


class TestPipeline(unittest.TestCase):
    """  TestCases for Pipeline Training. """

    def test_pipeline(self):
        x = fluid.layers.data(name='x', shape=[1], dtype='int64', lod_level=0)
        y = fluid.layers.data(name='y', shape=[1], dtype='int64', lod_level=0)
        emb_x = layers.embedding(
            input=x,
            param_attr=fluid.ParamAttr(name="embx"),
            size=[10, 2],
            is_sparse=False)
        emb_y = layers.embedding(
            input=y,
            param_attr=fluid.ParamAttr(
                name="emby", learning_rate=0.9),
            size=[10, 2],
            is_sparse=False)

        concat = layers.concat([emb_x, emb_y], axis=1)

        fc = layers.fc(input=concat,
                       name="fc",
                       size=1,
                       num_flatten_dims=1,
                       bias_attr=False)
        loss = layers.reduce_mean(fc)

        optimizer = fluid.optimizer.SGD(learning_rate=0.5)
        optimizer = fluid.optimizer.PipelineOptimizer(
            optimizer,
            cut_list=[[emb_x, emb_y], [loss]],
            place_list=[
                fluid.CPUPlace(), fluid.CUDAPlace(0), fluid.CPUPlace()
            ],
            concurrency_list=[1, 1, 1],
            queue_size=1,
            sync_steps=10000000, )
        optimizer.minimize(loss)
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        #prepare data
        batch_size = 100

        def binary_print(slot, fout):
            num = np.int16(len(slot) + 1)
            num.tofile(fout)
            a = np.int64(batch_size)
            a.tofile(fout)
            slot.tofile(fout)

        #batch1 = np.array([[0,1], [1,2], [2,3]]).astype("int64").reshape(batch_size,2,1)
        #batch2 = np.array([[1,2], [2,3], [3,4]]).astype("int64").reshape(batch_size,2,1)
        batch1 = np.ones(
            (batch_size, 2, 1)).astype("int64").reshape(batch_size, 2, 1)
        batch2 = np.ones(
            (batch_size, 2, 1)).astype("int64").reshape(batch_size, 2, 1)
        data = [batch1, batch2]
        filelist = []
        for i in range(2):
            filelist.append("test_pipeline_input_" + str(i))
        for f in filelist:
            with open(f, "wb") as fout:
                for batch_data in data:
                    for ins in batch_data:
                        for slot in ins:
                            binary_print(slot, fout)

        dataset = fluid.DatasetFactory().create_dataset("FileInstantDataset")
        dataset.set_use_var([x, y])
        dataset.set_batch_size(batch_size)
        dataset.set_filelist(filelist)

        for epoch in range(1):
            exe.train_from_dataset(
                fluid.default_main_program(),
                dataset,
                thread=1,
                debug=False,
                fetch_list=[],
                fetch_info=[],
                print_period=1)

        for f in filelist:
            os.remove(f)

    def test_pipeline_single_section(self):
        program = fluid.Program()
        with fluid.program_guard(program):
            x = fluid.layers.data(
                name='x', shape=[1], dtype='int64', lod_level=0)
            y = fluid.layers.data(
                name='y', shape=[1], dtype='int64', lod_level=0)
            emb_x = layers.embedding(
                input=x,
                param_attr=fluid.ParamAttr(name="embx"),
                size=[10, 2],
                is_sparse=False)
            emb_y = layers.embedding(
                input=y,
                param_attr=fluid.ParamAttr(
                    name="emby", learning_rate=0.9),
                size=[10, 2],
                is_sparse=False)

            concat = layers.concat([emb_x, emb_y], axis=1)

            fc = layers.fc(input=concat,
                           name="fc",
                           size=1,
                           num_flatten_dims=1,
                           bias_attr=False)
            loss = layers.reduce_mean(fc)

            optimizer = fluid.optimizer.SGD(learning_rate=0.5)
            optimizer = fluid.optimizer.PipelineOptimizer(
                optimizer,
                cut_list=[],
                place_list=[fluid.CUDAPlace(0)],
                concurrency_list=[1],
                queue_size=1,
                sync_steps=-1)
            optimizer.minimize(loss)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            #prepare data
            batch_size = 100

            def binary_print(slot, fout):
                num = np.int16(len(slot) + 1)
                num.tofile(fout)
                a = np.int64(batch_size)
                a.tofile(fout)
                slot.tofile(fout)

            #batch1 = np.array([[0,1], [1,2], [2,3]]).astype("int64").reshape(batch_size,2,1)
            #batch2 = np.array([[1,2], [2,3], [3,4]]).astype("int64").reshape(batch_size,2,1)
            batch1 = np.ones(
                (batch_size, 2, 1)).astype("int64").reshape(batch_size, 2, 1)
            batch2 = np.ones(
                (batch_size, 2, 1)).astype("int64").reshape(batch_size, 2, 1)
            data = [batch1, batch2]
            filelist = []
            for i in range(2):
                filelist.append("test_pipeline_input_" + str(i))
            for f in filelist:
                with open(f, "wb") as fout:
                    for batch_data in data:
                        for ins in batch_data:
                            for slot in ins:
                                binary_print(slot, fout)

            dataset = fluid.DatasetFactory().create_dataset(
                "FileInstantDataset")
            dataset.set_use_var([x, y])
            dataset.set_batch_size(batch_size)
            dataset.set_filelist(filelist)

            for epoch in range(1):
                exe.train_from_dataset(
                    fluid.default_main_program(),
                    dataset,
                    thread=1,
                    debug=False,
                    fetch_list=[],
                    fetch_info=[],
                    print_period=1)

            for f in filelist:
                os.remove(f)


if __name__ == '__main__':
    unittest.main()
