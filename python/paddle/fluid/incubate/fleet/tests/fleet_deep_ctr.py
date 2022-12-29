# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import logging
import time

import paddle
import paddle.fluid as fluid
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import (
    fleet,
)
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler.distributed_strategy import (
    StrategyFactory,
)

from paddle.fluid.log_helper import get_logger

import ctr_dataset_reader

logger = get_logger(
    "fluid", logging.INFO, fmt='%(asctime)s - %(levelname)s - %(message)s'
)


def parse_args():
    parser = argparse.ArgumentParser(description="PaddlePaddle Fleet ctr")

    # the following arguments is used for distributed train, if is_local == false, then you should set them
    parser.add_argument(
        '--role',
        type=str,
        default='pserver',  # trainer or pserver
        help='The path for model to store (default: models)',
    )
    parser.add_argument(
        '--endpoints',
        type=str,
        default='127.0.0.1:6000',
        help='The pserver endpoints, like: 127.0.0.1:6000,127.0.0.1:6001',
    )
    parser.add_argument(
        '--current_endpoint',
        type=str,
        default='127.0.0.1:6000',
        help='The path for model to store (default: 127.0.0.1:6000)',
    )
    parser.add_argument(
        '--trainer_id',
        type=int,
        default=0,
        help='The path for model to store (default: models)',
    )
    parser.add_argument(
        '--trainers',
        type=int,
        default=1,
        help='The num of trainers, (default: 1)',
    )

    return parser.parse_args()


def model():
    (
        dnn_input_dim,
        lr_input_dim,
        train_file_path,
    ) = ctr_dataset_reader.prepare_data()
    """ network definition """
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
        dtype="int64",
        lod_level=0,
        append_batch_size=False,
    )

    datas = [dnn_data, lr_data, label]

    # build dnn model
    dnn_layer_dims = [128, 64, 32, 1]
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
    dnn_pool = fluid.layers.sequence_pool(input=dnn_embedding, pool_type="sum")
    dnn_out = dnn_pool
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

    merge_layer = fluid.layers.concat(input=[dnn_out, lr_pool], axis=1)

    predict = fluid.layers.fc(input=merge_layer, size=2, act='softmax')
    acc = paddle.static.accuracy(input=predict, label=label)
    auc_var, batch_auc_var, auc_states = paddle.static.auc(
        input=predict, label=label
    )
    cost = paddle.nn.functional.cross_entropy(
        input=predict, label=label, reduction='none', use_softmax=False
    )
    avg_cost = paddle.mean(x=cost)

    return datas, avg_cost, predict, train_file_path


def train(args):
    datas, avg_cost, predict, train_file_path = model()

    endpoints = args.endpoints.split(",")
    if args.role.upper() == "PSERVER":
        current_id = endpoints.index(args.current_endpoint)
    else:
        current_id = 0
    role = role_maker.UserDefinedRoleMaker(
        current_id=current_id,
        role=role_maker.Role.WORKER
        if args.role.upper() == "TRAINER"
        else role_maker.Role.SERVER,
        worker_num=args.trainers,
        server_endpoints=endpoints,
    )

    exe = fluid.Executor(fluid.CPUPlace())
    fleet.init(role)

    strategy = StrategyFactory.create_half_async_strategy()

    optimizer = fluid.optimizer.SGD(learning_rate=0.0001)
    optimizer = fleet.distributed_optimizer(optimizer, strategy)
    optimizer.minimize(avg_cost)

    if fleet.is_server():
        logger.info("run pserver")

        fleet.init_server()
        fleet.run_server()
    elif fleet.is_worker():
        logger.info("run trainer")

        exe.run(fleet.startup_program)
        fleet.init_worker()

        thread_num = 2
        filelist = []
        for _ in range(thread_num):
            filelist.append(train_file_path)

        # config dataset
        dataset = fluid.DatasetFactory().create_dataset()
        dataset.set_batch_size(128)
        dataset.set_use_var(datas)
        pipe_command = 'python ctr_dataset_reader.py'
        dataset.set_pipe_command(pipe_command)

        dataset.set_filelist(filelist)
        dataset.set_thread(thread_num)

        for epoch_id in range(10):
            logger.info("epoch {} start".format(epoch_id))
            pass_start = time.time()
            dataset.set_filelist(filelist)
            exe.train_from_dataset(
                program=fleet.main_program,
                dataset=dataset,
                fetch_list=[avg_cost],
                fetch_info=["cost"],
                print_period=100,
                debug=False,
            )
            pass_time = time.time() - pass_start
            logger.info(
                "epoch {} finished, pass_time {}".format(epoch_id, pass_time)
            )
        fleet.stop_worker()


if __name__ == "__main__":
    args = parse_args()
    train(args)
