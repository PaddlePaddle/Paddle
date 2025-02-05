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
"""
Distribute CTR model for test fleet api
"""

import os
import shutil
import tempfile
import time

import ctr_dataset_reader
import numpy as np
from test_dist_fleet_base import FleetDistRunnerBase, runtime_main

import paddle
from paddle import base

paddle.enable_static()

# Fix seed for test
paddle.seed(1)


def fake_ctr_reader():
    def reader():
        for _ in range(1000):
            deep = np.random.random_integers(0, 1e5 - 1, size=16).tolist()
            wide = np.random.random_integers(0, 1e5 - 1, size=8).tolist()
            label = np.random.random_integers(0, 1, size=1).tolist()
            yield [deep, wide, label]

    return reader


class TestDistCTR2x2(FleetDistRunnerBase):
    """
    For test CTR model, using Fleet api
    """

    def net(self, args, is_train=True, batch_size=4, lr=0.01):
        """
        network definition

        Args:
            batch_size(int): the size of mini-batch for training
            lr(float): learning rate of training
        Returns:
            avg_cost: DenseTensor of cost.
        """
        dnn_input_dim, lr_input_dim = int(1e5), int(1e5)

        dnn_data = paddle.static.data(
            name="dnn_data",
            shape=[-1, 1],
            dtype="int64",
        )
        lr_data = paddle.static.data(
            name="lr_data",
            shape=[-1, 1],
            dtype="int64",
        )
        label = paddle.static.data(
            name="click",
            shape=[-1, 1],
            dtype="int64",
        )

        datas = [dnn_data, lr_data, label]

        if args.reader == "pyreader":
            if is_train:
                self.reader = base.io.PyReader(
                    feed_list=datas,
                    capacity=64,
                    iterable=False,
                    use_double_buffer=False,
                )
            else:
                self.test_reader = base.io.PyReader(
                    feed_list=datas,
                    capacity=64,
                    iterable=False,
                    use_double_buffer=False,
                )

        # build dnn model
        dnn_layer_dims = [128, 128, 64, 32, 1]
        dnn_embedding = paddle.static.nn.embedding(
            is_distributed=False,
            input=dnn_data,
            size=[dnn_input_dim, dnn_layer_dims[0]],
            param_attr=base.ParamAttr(
                name="deep_embedding",
                initializer=paddle.nn.initializer.Constant(value=0.01),
            ),
            is_sparse=True,
            padding_idx=0,
        )
        dnn_pool = paddle.static.nn.sequence_lod.sequence_pool(
            input=dnn_embedding.squeeze(-2), pool_type="sum"
        )
        dnn_out = dnn_pool
        for i, dim in enumerate(dnn_layer_dims[1:]):
            fc = paddle.static.nn.fc(
                x=dnn_out,
                size=dim,
                activation="relu",
                weight_attr=base.ParamAttr(
                    initializer=paddle.nn.initializer.Constant(value=0.01)
                ),
                name=f'dnn-fc-{i}',
            )
            dnn_out = fc

        # build lr model
        lr_embedding = paddle.static.nn.embedding(
            is_distributed=False,
            input=lr_data,
            size=[lr_input_dim, 1],
            param_attr=base.ParamAttr(
                name="wide_embedding",
                initializer=paddle.nn.initializer.Constant(value=0.01),
            ),
            is_sparse=True,
            padding_idx=0,
        )
        lr_pool = paddle.static.nn.sequence_lod.sequence_pool(
            input=lr_embedding.squeeze(-2), pool_type="sum"
        )

        merge_layer = paddle.concat([dnn_out, lr_pool], axis=1)

        predict = paddle.static.nn.fc(
            x=merge_layer, size=2, activation='softmax'
        )
        acc = paddle.static.accuracy(input=predict, label=label)

        auc_var, batch_auc_var, auc_states = paddle.static.auc(
            input=predict, label=label
        )

        cost = paddle.nn.functional.cross_entropy(
            input=predict, label=label, reduction='none', use_softmax=False
        )
        avg_cost = paddle.mean(x=cost)

        self.feeds = datas
        self.train_file_path = ["fake1", "fake2"]
        self.avg_cost = avg_cost
        self.predict = predict

        return avg_cost

    def check_model_right(self, dirname):
        dirname = dirname + '/dnn_plugin/'
        model_filename = os.path.join(dirname, "__model__")

        with open(model_filename, "rb") as f:
            program_desc_str = f.read()

        program = base.Program.parse_from_string(program_desc_str)
        with open(os.path.join(dirname, "__model__.proto"), "w") as wn:
            wn.write(str(program))

    def do_distributed_testing(self, fleet):
        """
        do distributed
        """
        exe = self.get_executor()

        batch_size = 4
        test_reader = paddle.batch(fake_ctr_reader(), batch_size=batch_size)
        self.test_reader.decorate_sample_list_generator(test_reader)

        pass_start = time.time()
        batch_idx = 0

        self.test_reader.start()
        try:
            while True:
                batch_idx += 1
                loss_val = exe.run(
                    program=paddle.static.default_main_program(),
                    fetch_list=[self.avg_cost.name],
                )
                loss_val = np.mean(loss_val)
                message = f"TEST ---> batch_idx: {batch_idx} loss: {loss_val}\n"
                fleet.util.print_on_rank(message, 0)
        except base.core.EOFException:
            self.test_reader.reset()

        pass_time = time.time() - pass_start
        message = f"Distributed Test Succeed, Using Time {pass_time}\n"
        fleet.util.print_on_rank(message, 0)

    def do_pyreader_training(self, fleet):
        """
        do training using dataset, using fetch handler to catch variable
        Args:
            fleet(Fleet api): the fleet object of Parameter Server, define distribute training role
        """
        exe = self.get_executor()
        exe.run(base.default_startup_program())
        fleet.init_worker()

        batch_size = 4
        train_reader = paddle.batch(fake_ctr_reader(), batch_size=batch_size)
        self.reader.decorate_sample_list_generator(train_reader)

        for epoch_id in range(1):
            self.reader.start()
            try:
                pass_start = time.time()
                while True:
                    loss_val = exe.run(
                        program=base.default_main_program(),
                        fetch_list=[self.avg_cost.name],
                    )
                    loss_val = np.mean(loss_val)
                    # TODO(randomly fail)
                    #   reduce_output = fleet.util.all_reduce(
                    #       np.array(loss_val), mode="sum")
                    #   loss_all_trainer = fleet.util.all_gather(float(loss_val))
                    #   loss_val = float(reduce_output) / len(loss_all_trainer)
                    message = f"TRAIN ---> pass: {epoch_id} loss: {loss_val}\n"
                    fleet.util.print_on_rank(message, 0)

                pass_time = time.time() - pass_start
            except base.core.EOFException:
                self.reader.reset()

        dirname = os.getenv("SAVE_DIRNAME", None)
        if dirname:
            fleet.save_persistables(exe, dirname=dirname)

        model_dir = tempfile.mkdtemp()
        fleet.save_inference_model(
            exe, model_dir, [feed.name for feed in self.feeds], self.avg_cost
        )
        if fleet.is_first_worker():
            self.check_model_right(model_dir)
        shutil.rmtree(model_dir)

    def do_dataset_training_queuedataset(self, fleet):
        train_file_list = ctr_dataset_reader.prepare_fake_data()

        exe = self.get_executor()
        exe.run(base.default_startup_program())
        fleet.init_worker()

        thread_num = 2
        batch_size = 128
        filelist = train_file_list

        # config dataset
        dataset = paddle.distributed.QueueDataset()
        pipe_command = 'python ctr_dataset_reader.py'

        dataset.init(
            batch_size=batch_size,
            use_var=self.feeds,
            pipe_command=pipe_command,
            thread_num=thread_num,
        )

        dataset.set_filelist(filelist)

        for epoch_id in range(1):
            pass_start = time.time()
            dataset.set_filelist(filelist)
            exe.train_from_dataset(
                program=base.default_main_program(),
                dataset=dataset,
                fetch_list=[self.avg_cost],
                fetch_info=["cost"],
                print_period=2,
                debug=int(os.getenv("Debug", "0")),
            )
            pass_time = time.time() - pass_start

        if os.getenv("SAVE_MODEL") == "1":
            model_dir = tempfile.mkdtemp()
            fleet.save_inference_model(
                exe,
                model_dir,
                [feed.name for feed in self.feeds],
                self.avg_cost,
            )
            if fleet.is_first_worker():
                self.check_model_right(model_dir)
            shutil.rmtree(model_dir)

        dirname = os.getenv("SAVE_DIRNAME", None)
        if dirname:
            fleet.save_persistables(exe, dirname=dirname)

    def do_dataset_training(self, fleet):
        train_file_list = ctr_dataset_reader.prepare_fake_data()

        exe = self.get_executor()
        exe.run(base.default_startup_program())
        fleet.init_worker()

        thread_num = 2
        batch_size = 128
        filelist = train_file_list

        # config dataset
        dataset = base.DatasetFactory().create_dataset("InMemoryDataset")
        dataset.set_use_var(self.feeds)
        dataset.set_batch_size(128)
        dataset.set_thread(2)
        dataset.set_filelist(filelist)
        dataset.set_pipe_command('python ctr_dataset_reader.py')
        dataset.load_into_memory()

        dataset.global_shuffle(fleet, 12)  # TODO: thread configure
        shuffle_data_size = dataset.get_shuffle_data_size(fleet)
        local_data_size = dataset.get_shuffle_data_size()
        data_size_list = fleet.util.all_gather(local_data_size)
        print('after global_shuffle data_size_list: ', data_size_list)
        print('after global_shuffle data_size: ', shuffle_data_size)

        for epoch_id in range(1):
            pass_start = time.time()
            exe.train_from_dataset(
                program=base.default_main_program(),
                dataset=dataset,
                fetch_list=[self.avg_cost],
                fetch_info=["cost"],
                print_period=2,
                debug=int(os.getenv("Debug", "0")),
            )
            pass_time = time.time() - pass_start
        dataset.release_memory()

        if os.getenv("SAVE_MODEL") == "1":
            model_dir = tempfile.mkdtemp()
            fleet.save_inference_model(
                exe,
                model_dir,
                [feed.name for feed in self.feeds],
                self.avg_cost,
            )
            fleet.load_inference_model(model_dir, mode=0)
            if fleet.is_first_worker():
                self.check_model_right(model_dir)
            shutil.rmtree(model_dir)

        dirname = os.getenv("SAVE_DIRNAME", None)
        if dirname:
            fleet.save_persistables(exe, dirname=dirname)
            fleet.load_model(dirname, mode=0)

        cache_dirname = os.getenv("SAVE_CACHE_DIRNAME", None)
        if cache_dirname:
            fleet.save_cache_model(cache_dirname)

        dense_param_dirname = os.getenv("SAVE_DENSE_PARAM_DIRNAME", None)
        if dense_param_dirname:
            fleet.save_dense_params(
                exe,
                dense_param_dirname,
                base.global_scope(),
                base.default_main_program(),
            )

        save_one_table_dirname = os.getenv("SAVE_ONE_TABLE_DIRNAME", None)
        if save_one_table_dirname:
            fleet.save_one_table(0, save_one_table_dirname, 0)
            fleet.load_one_table(0, save_one_table_dirname, 0)

        patch_dirname = os.getenv("SAVE_PATCH_DIRNAME", None)
        if patch_dirname:
            fleet.save_persistables(exe, patch_dirname, None, 5)
            fleet.check_save_pre_patch_done()

        # add for gpu graph
        fleet.save_cache_table(0, 0)
        fleet.shrink()


if __name__ == "__main__":
    runtime_main(TestDistCTR2x2)
