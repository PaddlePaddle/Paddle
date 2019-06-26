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


if __name__ == '__main__':
    unittest.main()
