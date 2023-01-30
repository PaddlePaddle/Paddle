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

<<<<<<< HEAD
=======
from __future__ import print_function

import paddle
import paddle.fluid as fluid
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
import os

import dist_ctr_reader
from test_dist_base import TestDistRunnerBase, runtime_main

<<<<<<< HEAD
import paddle
import paddle.fluid as fluid

=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
IS_SPARSE = True
os.environ['PADDLE_ENABLE_REMOTE_PREFETCH'] = "1"

# Fix seed for test
fluid.default_startup_program().random_seed = 1
fluid.default_main_program().random_seed = 1


class TestDistCTR2x2(TestDistRunnerBase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def get_model(self, batch_size=2):

        dnn_input_dim, lr_input_dim = dist_ctr_reader.load_data_meta()
        """ network definition """
<<<<<<< HEAD
        dnn_data = paddle.static.data(
            name="dnn_data",
            shape=[-1, 1],
            dtype="int64",
            lod_level=1,
        )
        lr_data = paddle.static.data(
            name="lr_data",
            shape=[-1, 1],
            dtype="int64",
            lod_level=1,
        )
        label = paddle.static.data(
            name="click",
            shape=[-1, 1],
            dtype="int64",
            lod_level=0,
        )
=======
        dnn_data = fluid.layers.data(name="dnn_data",
                                     shape=[-1, 1],
                                     dtype="int64",
                                     lod_level=1,
                                     append_batch_size=False)
        lr_data = fluid.layers.data(name="lr_data",
                                    shape=[-1, 1],
                                    dtype="int64",
                                    lod_level=1,
                                    append_batch_size=False)
        label = fluid.layers.data(name="click",
                                  shape=[-1, 1],
                                  dtype="int64",
                                  lod_level=0,
                                  append_batch_size=False)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        # build dnn model
        dnn_layer_dims = [128, 64, 32, 1]
        dnn_embedding = fluid.layers.embedding(
            is_distributed=False,
            input=dnn_data,
            size=[dnn_input_dim, dnn_layer_dims[0]],
            param_attr=fluid.ParamAttr(
                name="deep_embedding",
<<<<<<< HEAD
                initializer=fluid.initializer.Constant(value=0.01),
            ),
            is_sparse=IS_SPARSE,
        )
        dnn_pool = fluid.layers.sequence_pool(
            input=dnn_embedding, pool_type="sum"
        )
        dnn_out = dnn_pool
        for i, dim in enumerate(dnn_layer_dims[1:]):
            fc = paddle.static.nn.fc(
                x=dnn_out,
                size=dim,
                activation="relu",
                weight_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Constant(value=0.01)
                ),
                name='dnn-fc-%d' % i,
            )
=======
                initializer=fluid.initializer.Constant(value=0.01)),
            is_sparse=IS_SPARSE)
        dnn_pool = fluid.layers.sequence_pool(input=dnn_embedding,
                                              pool_type="sum")
        dnn_out = dnn_pool
        for i, dim in enumerate(dnn_layer_dims[1:]):
            fc = fluid.layers.fc(
                input=dnn_out,
                size=dim,
                act="relu",
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Constant(value=0.01)),
                name='dnn-fc-%d' % i)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            dnn_out = fc

        # build lr model
        lr_embbding = fluid.layers.embedding(
            is_distributed=False,
            input=lr_data,
            size=[lr_input_dim, 1],
            param_attr=fluid.ParamAttr(
                name="wide_embedding",
<<<<<<< HEAD
                initializer=fluid.initializer.Constant(value=0.01),
            ),
            is_sparse=IS_SPARSE,
        )
=======
                initializer=fluid.initializer.Constant(value=0.01)),
            is_sparse=IS_SPARSE)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        lr_pool = fluid.layers.sequence_pool(input=lr_embbding, pool_type="sum")

        merge_layer = fluid.layers.concat(input=[dnn_out, lr_pool], axis=1)

<<<<<<< HEAD
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
=======
        predict = fluid.layers.fc(input=merge_layer, size=2, act='softmax')
        acc = fluid.layers.accuracy(input=predict, label=label)
        auc_var, batch_auc_var, auc_states = fluid.layers.auc(input=predict,
                                                              label=label)
        cost = fluid.layers.cross_entropy(input=predict, label=label)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        avg_cost = paddle.mean(x=cost)

        inference_program = paddle.fluid.default_main_program().clone()

        regularization = None
        use_l2_decay = bool(os.getenv('USE_L2_DECAY', 0))
        if use_l2_decay:
            regularization = fluid.regularizer.L2DecayRegularizer(
<<<<<<< HEAD
                regularization_coeff=1e-1
            )
        use_lr_decay = bool(os.getenv('LR_DECAY', 0))
        lr = 0.0001
        if use_lr_decay:
            lr = fluid.layers.exponential_decay(
                learning_rate=0.0001,
                decay_steps=10000,
                decay_rate=0.999,
                staircase=True,
            )

        sgd_optimizer = fluid.optimizer.SGD(
            learning_rate=lr, regularization=regularization
        )
=======
                regularization_coeff=1e-1)
        use_lr_decay = bool(os.getenv('LR_DECAY', 0))
        lr = 0.0001
        if use_lr_decay:
            lr = fluid.layers.exponential_decay(learning_rate=0.0001,
                                                decay_steps=10000,
                                                decay_rate=0.999,
                                                staircase=True)

        sgd_optimizer = fluid.optimizer.SGD(learning_rate=lr,
                                            regularization=regularization)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        sgd_optimizer.minimize(avg_cost)

        dataset = dist_ctr_reader.Dataset()
        train_reader = paddle.batch(dataset.train(), batch_size=batch_size)
        test_reader = paddle.batch(dataset.test(), batch_size=batch_size)

<<<<<<< HEAD
        return (
            inference_program,
            avg_cost,
            train_reader,
            test_reader,
            None,
            predict,
        )
=======
        return inference_program, avg_cost, train_reader, test_reader, None, predict
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


if __name__ == "__main__":
    runtime_main(TestDistCTR2x2)
