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

import shutil
import tempfile
import time

import paddle.fluid as fluid
import os

import ctr_dataset_reader
from test_dist_fleet_base import runtime_main, FleetDistRunnerBase

# Fix seed for test
fluid.default_startup_program().random_seed = 1
fluid.default_main_program().random_seed = 1


class TestDistCTR2x2(FleetDistRunnerBase):
    def net(self, batch_size=4, lr=0.01):
        dnn_input_dim, lr_input_dim, train_file_path = ctr_dataset_reader.prepare_data(
        )
        """ network definition """
        dnn_data = fluid.layers.data(
            name="dnn_data",
            shape=[-1, 1],
            dtype="int64",
            lod_level=1,
            append_batch_size=False)
        lr_data = fluid.layers.data(
            name="lr_data",
            shape=[-1, 1],
            dtype="int64",
            lod_level=1,
            append_batch_size=False)
        label = fluid.layers.data(
            name="click",
            shape=[-1, 1],
            dtype="int64",
            lod_level=0,
            append_batch_size=False)

        datas = [dnn_data, lr_data, label]

        # build dnn model
        dnn_layer_dims = [128, 64, 32, 1]
        dnn_embedding = fluid.layers.embedding(
            is_distributed=False,
            input=dnn_data,
            size=[dnn_input_dim, dnn_layer_dims[0]],
            param_attr=fluid.ParamAttr(
                name="deep_embedding",
                initializer=fluid.initializer.Constant(value=0.01)),
            is_sparse=True)
        dnn_pool = fluid.layers.sequence_pool(
            input=dnn_embedding, pool_type="sum")
        dnn_out = dnn_pool
        for i, dim in enumerate(dnn_layer_dims[1:]):
            fc = fluid.layers.fc(
                input=dnn_out,
                size=dim,
                act="relu",
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Constant(value=0.01)),
                name='dnn-fc-%d' % i)
            dnn_out = fc

        # build lr model
        lr_embbding = fluid.layers.embedding(
            is_distributed=False,
            input=lr_data,
            size=[lr_input_dim, 1],
            param_attr=fluid.ParamAttr(
                name="wide_embedding",
                initializer=fluid.initializer.Constant(value=0.01)),
            is_sparse=True)
        lr_pool = fluid.layers.sequence_pool(input=lr_embbding, pool_type="sum")

        merge_layer = fluid.layers.concat(input=[dnn_out, lr_pool], axis=1)

        predict = fluid.layers.fc(input=merge_layer, size=2, act='softmax')
        acc = fluid.layers.accuracy(input=predict, label=label)
        auc_var, batch_auc_var, auc_states = fluid.layers.auc(input=predict,
                                                              label=label)
        cost = fluid.layers.cross_entropy(input=predict, label=label)
        avg_cost = fluid.layers.mean(x=cost)

        self.feeds = datas
        self.train_file_path = train_file_path
        self.avg_cost = avg_cost
        self.predict = predict

        return avg_cost

    def check_model_right(self, dirname):
        model_filename = os.path.join(dirname, "__model__")

        with open(model_filename, "rb") as f:
            program_desc_str = f.read()

        program = fluid.Program.parse_from_string(program_desc_str)
        with open(os.path.join(dirname, "__model__.proto"), "w") as wn:
            wn.write(str(program))

    def do_training(self, fleet):
        dnn_input_dim, lr_input_dim, train_file_path = ctr_dataset_reader.prepare_data(
        )

        exe = fluid.Executor(fluid.CPUPlace())

        fleet.init_worker()
        exe.run(fleet.startup_program)

        thread_num = 2
        filelist = []
        for _ in range(thread_num):
            filelist.append(train_file_path)

        # config dataset
        dataset = fluid.DatasetFactory().create_dataset()
        dataset.set_batch_size(128)
        dataset.set_use_var(self.feeds)
        pipe_command = 'python ctr_dataset_reader.py'
        dataset.set_pipe_command(pipe_command)

        dataset.set_filelist(filelist)
        dataset.set_thread(thread_num)

        for epoch_id in range(2):
            pass_start = time.time()
            dataset.set_filelist(filelist)
            exe.train_from_dataset(
                program=fleet.main_program,
                dataset=dataset,
                fetch_list=[self.avg_cost],
                fetch_info=["cost"],
                print_period=100,
                debug=False)
            pass_time = time.time() - pass_start

        model_dir = tempfile.mkdtemp()
        fleet.save_inference_model(
            exe, model_dir, [feed.name for feed in self.feeds], self.avg_cost)
        self.check_model_right(model_dir)
        shutil.rmtree(model_dir)
        fleet.stop_worker()


if __name__ == "__main__":
    runtime_main(TestDistCTR2x2)
