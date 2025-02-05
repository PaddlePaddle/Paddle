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

import numpy as np
from test_dist_fleet_base import FleetDistRunnerBase, runtime_main

import paddle
from paddle import base


def fake_ctr_reader():
    def reader():
        for _ in range(1000):
            deep = np.random.random_integers(0, 1e10, size=16).tolist()
            wide = np.random.random_integers(0, 1e10, size=8).tolist()
            label = np.random.random_integers(0, 1, size=1).tolist()
            yield [deep, wide, label]

    return reader


class TestDistCTR2x2(FleetDistRunnerBase):
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
        dnn_input_dim, lr_input_dim = 10, 10

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
            self.reader = base.io.PyReader(
                feed_list=datas,
                capacity=64,
                iterable=False,
                use_double_buffer=False,
            )

        # build dnn model
        initializer = int(os.getenv("INITIALIZER", "0"))
        inference = bool(int(os.getenv("INFERENCE", "0")))

        if initializer == 0:
            init = paddle.nn.initializer.Constant(value=0.01)
        elif initializer == 1:
            init = paddle.nn.initializer.Uniform()
        elif initializer == 2:
            init = paddle.nn.initializer.Normal()
        else:
            raise ValueError(f"error initializer code: {initializer}")

        entry = paddle.distributed.ShowClickEntry("show", "click")
        dnn_layer_dims = [128, 64, 32]
        dnn_embedding = paddle.static.nn.sparse_embedding(
            input=dnn_data,
            size=[dnn_input_dim, dnn_layer_dims[0]],
            is_test=inference,
            entry=entry,
            param_attr=base.ParamAttr(name="deep_embedding", initializer=init),
        )
        dnn_pool = paddle.static.nn.sequence_lod.sequence_pool(
            input=dnn_embedding, pool_type="sum"
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
        lr_embedding = paddle.static.nn.sparse_embedding(
            input=lr_data,
            size=[lr_input_dim, 1],
            is_test=inference,
            entry=entry,
            param_attr=base.ParamAttr(
                name="wide_embedding",
                initializer=paddle.nn.initializer.Constant(value=0.01),
            ),
        )

        lr_pool = paddle.static.nn.sequence_lod.sequence_pool(
            input=lr_embedding, pool_type="sum"
        )
        merge_layer = paddle.concat([dnn_out, lr_pool], axis=1)

        predict = paddle.static.nn.fc(
            x=merge_layer, size=2, activation='softmax'
        )

        acc = paddle.static.accuracy(input=predict, label=label)
        auc_var, _, _ = paddle.static.auc(input=predict, label=label)
        cost = paddle.nn.functional.cross_entropy(
            input=predict, label=label, reduction='none', use_softmax=False
        )
        avg_cost = paddle.mean(x=cost)

        self.feeds = datas
        self.train_file_path = ["fake1", "fake2"]
        self.avg_cost = avg_cost
        self.predict = predict

        return avg_cost

    def do_pyreader_training(self, fleet):
        """
        do training using dataset, using fetch handler to catch variable
        Args:
            fleet(Fleet api): the fleet object of Parameter Server, define distribute training role
        """

        exe = base.Executor(base.CPUPlace())

        exe.run(base.default_startup_program())
        fleet.init_worker()

        batch_size = 4

        train_reader = paddle.batch(fake_ctr_reader(), batch_size=batch_size)
        self.reader.decorate_sample_list_generator(train_reader)

        for epoch_id in range(1):
            self.reader.start()
            try:
                while True:
                    loss_val = exe.run(
                        program=base.default_main_program(),
                        fetch_list=[self.avg_cost.name],
                    )
                    loss_val = np.mean(loss_val)
                    print(f"TRAIN ---> pass: {epoch_id} loss: {loss_val}\n")
            except base.core.EOFException:
                self.reader.reset()

        model_dir = os.getenv("MODEL_DIR", None)
        if model_dir:
            fleet.save_inference_model(
                exe,
                model_dir,
                [feed.name for feed in self.feeds],
                self.avg_cost,
            )
            fleet.load_model(model_dir, mode=1)


if __name__ == "__main__":
    runtime_main(TestDistCTR2x2)
