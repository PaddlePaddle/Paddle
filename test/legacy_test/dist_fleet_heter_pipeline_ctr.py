#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import time

import ctr_dataset_reader
from test_dist_fleet_heter_base import FleetDistHeterRunnerBase, runtime_main

import paddle
from paddle import base

paddle.enable_static()

# Fix seed for test
paddle.seed(1)


class TestHeterPipelinePsCTR2x2(FleetDistHeterRunnerBase):
    """
    For test CTR model, using Fleet api
    """

    def net(self, args, batch_size=4, lr=0.01):
        """
        network definition

        Args:
            batch_size(int): the size of mini-batch for training
            lr(float): learning rate of training
        Returns:
            avg_cost: DenseTensor of cost.
        """
        dnn_input_dim, lr_input_dim = int(1e5), int(1e5)

        with base.device_guard("cpu"):
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
                dtype="float32",
            )

            datas = [dnn_data, lr_data, label]

            # build dnn model
            dnn_layer_dims = [128, 64, 32, 1]
            dnn_embedding = paddle.static.nn.embedding(
                is_distributed=False,
                input=dnn_data,
                size=[dnn_input_dim, dnn_layer_dims[0]],
                param_attr=base.ParamAttr(
                    name="deep_embedding",
                    initializer=paddle.nn.initializer.Constant(value=0.01),
                ),
                is_sparse=True,
            )
            dnn_pool = paddle.static.nn.sequence_lod.sequence_pool(
                input=dnn_embedding, pool_type="sum"
            )
            dnn_out = dnn_pool

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
            )
            lr_pool = paddle.static.nn.sequence_lod.sequence_pool(
                input=lr_embedding, pool_type="sum"
            )

        with base.device_guard("gpu"):
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

        with base.device_guard("cpu"):
            merge_layer = paddle.concat([dnn_out, lr_pool], axis=1)
            label = paddle.cast(label, dtype="int64")
            predict = paddle.static.nn.fc(
                x=merge_layer, size=2, activation='softmax'
            )

            cost = paddle.nn.functional.cross_entropy(
                input=predict, label=label, reduction='none', use_softmax=False
            )
            avg_cost = paddle.mean(x=cost)
            paddle.static.Print(avg_cost, message="avg_cost")

        self.feeds = datas
        self.train_file_path = ["fake1", "fake2"]
        self.avg_cost = avg_cost
        self.predict = predict

        return avg_cost

    def check_model_right(self, dirname):
        model_filename = os.path.join(dirname, "__model__")

        with open(model_filename, "rb") as f:
            program_desc_str = f.read()

        program = base.Program.parse_from_string(program_desc_str)
        with open(os.path.join(dirname, "__model__.proto"), "w") as wn:
            wn.write(str(program))

    def do_dataset_training(self, fleet):
        train_file_list = ctr_dataset_reader.prepare_fake_data()

        exe = base.Executor(base.CPUPlace())
        real_program = base.default_main_program()._heter_pipeline_opt[
            "section_program"
        ]
        print(real_program)

        exe.run(base.default_startup_program())
        fleet.init_worker()

        thread_num = int(os.getenv("CPU_NUM", 2))
        batch_size = 128

        filelist = fleet.util.get_file_shard(train_file_list)
        print(f"filelist: {filelist}")

        # config dataset
        dataset = base.DatasetFactory().create_dataset()
        dataset.set_batch_size(batch_size)
        dataset.set_use_var(self.feeds)
        pipe_command = 'python3 ctr_dataset_reader.py'
        dataset.set_pipe_command(pipe_command)

        dataset.set_filelist(filelist)
        dataset.set_thread(thread_num)

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
            print(f"do_dataset_training done. using time {pass_time}")
        exe.close()

    def do_dataset_heter_training(self, fleet):
        exe = base.Executor()
        exe.run(base.default_startup_program())
        fleet.init_worker()
        real_program = base.default_main_program()._heter_pipeline_opt[
            "section_program"
        ]
        print(real_program)

        thread_num = int(os.getenv("CPU_NUM", 2))
        batch_size = 128

        pass_start = time.time()
        exe.train_from_dataset(
            program=base.default_main_program(),
            fetch_list=[self.avg_cost],
            fetch_info=["cost"],
            print_period=2,
            debug=int(os.getenv("Debug", "0")),
        )
        exe.close()
        pass_time = time.time() - pass_start
        print(f"do_dataset_heter_training done. using time {pass_time}")

        # for epoch_id in range(1):
        #    pass_start = time.time()
        #    dataset.set_filelist(filelist)
        #    exe.train_from_dataset(
        #        program=base.default_main_program(),
        #        dataset=dataset,
        #        fetch_list=[self.avg_cost],
        #        fetch_info=["cost"],
        #        print_period=2,
        #        debug=int(os.getenv("Debug", "0")))
        #    pass_time = time.time() - pass_start
        #    print("do_dataset_heter_training done. using time {}".format(pass_time))


if __name__ == "__main__":
    runtime_main(TestHeterPipelinePsCTR2x2)
