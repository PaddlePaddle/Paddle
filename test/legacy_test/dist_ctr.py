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

import os

import dist_ctr_reader
from test_dist_base import TestDistRunnerBase, runtime_main

import paddle
from paddle import base

IS_SPARSE = True
os.environ['PADDLE_ENABLE_REMOTE_PREFETCH'] = "1"

# Fix seed for test
paddle.seed(1)


class TestDistCTR2x2(TestDistRunnerBase):
    def get_model(self, batch_size=2):
        dnn_input_dim, lr_input_dim = dist_ctr_reader.load_data_meta()
        """ network definition """
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
            is_sparse=IS_SPARSE,
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
        lr_embedding = paddle.static.nn.embedding(
            is_distributed=False,
            input=lr_data,
            size=[lr_input_dim, 1],
            param_attr=base.ParamAttr(
                name="wide_embedding",
                initializer=paddle.nn.initializer.Constant(value=0.01),
            ),
            is_sparse=IS_SPARSE,
        )
        lr_pool = paddle.static.nn.sequence_lod.sequence_pool(
            input=lr_embedding, pool_type="sum"
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

        inference_program = paddle.base.default_main_program().clone()

        regularization = None
        use_l2_decay = bool(os.getenv('USE_L2_DECAY', 0))
        if use_l2_decay:
            regularization = paddle.regularizer.L2Decay(coeff=1e-1)
        use_lr_decay = bool(os.getenv('LR_DECAY', 0))
        lr = 0.0001
        if use_lr_decay:
            lr = paddle.optimizer.lr.ExponentialDecay(
                learning_rate=0.0001,
                gamma=0.999,
            )

        sgd_optimizer = paddle.optimizer.SGD(
            learning_rate=lr, weight_decay=regularization
        )
        sgd_optimizer.minimize(avg_cost)

        dataset = dist_ctr_reader.Dataset()
        train_reader = paddle.batch(dataset.train(), batch_size=batch_size)
        test_reader = paddle.batch(dataset.test(), batch_size=batch_size)

        return (
            inference_program,
            avg_cost,
            train_reader,
            test_reader,
            None,
            predict,
        )


if __name__ == "__main__":
    runtime_main(TestDistCTR2x2)
