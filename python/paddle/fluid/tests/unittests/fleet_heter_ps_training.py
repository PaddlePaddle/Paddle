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
import paddle.distributed.fleet as fleet
import paddle.fluid as fluid

fluid.disable_dygraph()


def get_dataset(inputs):
    dataset = fluid.DatasetFactory().create_dataset()
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
        avg_cost: LoDTensor of cost.
    """
    dnn_input_dim, lr_input_dim = int(2), int(2)

    with fluid.device_guard("cpu"):
        dnn_data = fluid.layers.data(
            name="dnn_data",
            shape=[-1, 1],
            dtype="int64",
            lod_level=1,
            append_batch_size=False,
        )
        lr_data = fluid.layers.data(
            name="lr_data",
            shape=[-1, 1],
            dtype="int64",
            lod_level=1,
            append_batch_size=False,
        )
        label = fluid.layers.data(
            name="click",
            shape=[-1, 1],
            dtype="float32",
            lod_level=0,
            append_batch_size=False,
        )

        datas = [dnn_data, lr_data, label]

        # build dnn model
        dnn_layer_dims = [2, 1]
        dnn_embedding = fluid.layers.embedding(
            is_distributed=False,
            input=dnn_data,
            size=[dnn_input_dim, dnn_layer_dims[0]],
            param_attr=fluid.ParamAttr(
                name="deep_embedding",
                initializer=fluid.initializer.Constant(value=0.01),
            ),
            is_sparse=True,
        )
        dnn_pool = fluid.layers.sequence_pool(
            input=dnn_embedding, pool_type="sum"
        )
        dnn_out = dnn_pool

        # build lr model
        lr_embbding = fluid.layers.embedding(
            is_distributed=False,
            input=lr_data,
            size=[lr_input_dim, 1],
            param_attr=fluid.ParamAttr(
                name="wide_embedding",
                initializer=fluid.initializer.Constant(value=0.01),
            ),
            is_sparse=True,
        )
        lr_pool = fluid.layers.sequence_pool(input=lr_embbding, pool_type="sum")

    with fluid.device_guard("gpu"):
        for i, dim in enumerate(dnn_layer_dims[1:]):
            fc = fluid.layers.fc(
                input=dnn_out,
                size=dim,
                act="relu",
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Constant(value=0.01)
                ),
                name='dnn-fc-%d' % i,
            )
            dnn_out = fc

        merge_layer = fluid.layers.concat(input=[dnn_out, lr_pool], axis=1)
        label = fluid.layers.cast(label, dtype="int64")
        predict = fluid.layers.fc(input=merge_layer, size=2, act='softmax')

        cost = paddle.nn.functional.cross_entropy(
            input=predict, label=label, reduction='none', use_softmax=False
        )
        avg_cost = paddle.mean(x=cost)
    return datas, avg_cost


'''
optimizer = fluid.optimizer.Adam(learning_rate=0.01)

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
    # place = fluid.CPUPlace()
    # exe = fluid.Executor(place)
    # exe.run(fluid.default_startup_program())
    # fleet.init_worker()
    # step = 1
    # for i in range(step):
    #    exe.train_from_dataset(
    #        program=fluid.default_main_program(), dataset=dataset, debug=False)
    # exe.close()
    # fleet.stop_worker()
