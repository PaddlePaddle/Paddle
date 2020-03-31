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
import paddle.compat as cpt
import paddle.fluid.core as core
import numpy as np
import os
import shutil
import unittest


class TestDataFeed(unittest.TestCase):
    """  TestBaseCase(Merge PV)   """

    def setUp(self):
        self.batch_size = 2
        self.pv_batch_size = 2
        self.dataset = fluid.DatasetFactory().create_dataset("BoxPSDataset")
        self.dataset.set_feed_type("PaddleBoxDataFeed")
        self.dataset.set_pv_batch_size(self.pv_batch_size)
        self.dataset.set_parse_logkey(True)
        self.dataset.set_merge_by_sid(True)
        self.dataset.set_rank_offset("rank_offset")
        self.dataset.set_enable_pv_predict(True)
        self.dataset.set_thread(3)

    def test_config(self):
        self.assertTrue(self.dataset.parse_logkey)
        self.assertTrue(self.dataset.merge_by_sid)

    def test_run_dataset(self):
        x = fluid.layers.data(name='x', shape=[1], dtype='int64', lod_level=0)
        y = fluid.layers.data(name='y', shape=[1], dtype='int64', lod_level=0)
        emb_x, emb_y = _pull_box_sparse([x, y], size=2)
        emb_xp = _pull_box_sparse(x, size=2)
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

        with open("test_run_with_dump_a.txt", "w") as f:
            data = "1 1702f830eee19501ad7429505f714c1d 1 1 1 9\n"
            data += "1 1702f830eee19502ad7429505f714c1d 1 2 1 8\n"
            data += "1 1702f830eee19503ad7429505f714c1d 1 3 1 7\n"
            data += "1 1702f830eee22201ad7429505f714c2d 1 4 1 6\n"
            data += "1 1702f830eee19101ad7429505f714c3d 1 5 1 5\n"
            data += "1 1702f830eee19102ad7429505f714c3d 1 6 1 4\n"
            f.write(data)
        with open("test_run_with_dump_b.txt", "w") as f:
            data = "1 1702f830fff22201ad7429505f715c1d 1 1 1 1\n"
            data += "1 1702f830fff22202ad7429505f715c1d 1 2 1 2\n"
            data += "1 1702f830fff22203ad7429505f715c1d 1 3 1 3\n"
            data += "1 1702f830fff22101ad7429505f714ccd 1 4 1 4\n"
            data += "1 1702f830fff22102ad7429505f714ccd 1 5 1 5\n"
            data += "1 1702f830fff22103ad7429505f714ccd 1 6 1 6\n"
            data += "1 1702f830fff22104ad7429505f714ccd 1 6 1 7\n"
            f.write(data)

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
        self.datasets.load_into_memory()

        self.datasets.begin_pass()
        exe.train_from_dataset(
            program=fluid.default_main_program(),
            dataset=datasets[0],
            print_period=1)
        self.datasets.end_pass(True)

        os.remove("test_run_with_dump_a.txt")
        os.remove("test_run_with_dump_b.txt")


if __name__ == '__main__':
    unittest.main()
