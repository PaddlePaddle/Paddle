# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle import base
from paddle.distributed import fleet

base.disable_dygraph()


def get_dataset(inputs):
    dataset = base.DatasetFactory().create_dataset()
    dataset.set_use_var(inputs)
    dataset.set_batch_size(1)
    dataset.set_filelist([])
    dataset.set_thread(1)
    return dataset


def net(batch_size=4, lr=0.01):
    """
    network definition

    Args:
        batch_size(int): the size of mini-batch for training
        lr(float): learning rate of training
    Returns:
        avg_cost: DenseTensor of cost.
    """
    dnn_input_dim, lr_input_dim = 2, 2

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
        dnn_layer_dims = [2, 1]
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

        merge_layer = paddle.concat([dnn_out, lr_pool], axis=1)
        label = paddle.cast(label, dtype="int64")
        predict = paddle.static.nn.fc(
            x=merge_layer, size=2, activation='softmax'
        )

        cost = paddle.nn.functional.cross_entropy(
            input=predict, label=label, reduction='none', use_softmax=False
        )
        avg_cost = paddle.mean(x=cost)
    return datas, avg_cost


'''
optimizer = paddle.optimizer.Adam(learning_rate=0.01)

role = role_maker.PaddleCloudRoleMaker()
fleet.init(role)

strategy = paddle.distributed.fleet.DistributedStrategy()
strategy.a_sync = True
strategy.a_sync_configs = {"heter_worker_device_guard": 'gpu'}

strategy.pipeline = True
strategy.pipeline_configs = {"accumulate_steps": 1, "micro_batch_size": 2048}
feeds, avg_cost = net()
optimizer = fleet.distributed_optimizer(optimizer, strategy)
optimizer.minimize(avg_cost)
dataset = get_dataset(feeds)
'''

if fleet.is_server():
    pass
    # fleet.init_server()
    # fleet.run_server()
elif fleet.is_heter_worker():
    pass
    # fleet.init_heter_worker()
    # fleet.run_heter_worker(dataset=dataset)
    fleet.stop_worker()
elif fleet.is_worker():
    pass
    # place = base.CPUPlace()
    # exe = base.Executor(place)
    # exe.run(base.default_startup_program())
    # fleet.init_worker()
    # step = 1
    # for i in range(step):
    #    exe.train_from_dataset(
    #        program=base.default_main_program(), dataset=dataset, debug=False)
    # exe.close()
    # fleet.stop_worker()
