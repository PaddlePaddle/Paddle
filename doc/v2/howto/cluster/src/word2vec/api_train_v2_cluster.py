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

import math
import os
import paddle.v2 as paddle
import pickle

embsize = 32
hiddensize = 256
N = 5
cluster_train_file = "./train_data_dir/train/train.txt"
cluster_test_file = "./test_data_dir/test/test.txt"
node_id = os.getenv("OMPI_COMM_WORLD_RANK")
if not node_id:
    raise EnvironmentError("must provied OMPI_COMM_WORLD_RANK")


def wordemb(inlayer):
    wordemb = paddle.layer.embedding(
        input=inlayer,
        size=embsize,
        param_attr=paddle.attr.Param(
            name="_proj",
            initial_std=0.001,
            learning_rate=1,
            l2_rate=0,
            sparse_update=True))
    return wordemb


def cluster_reader_cluster(filename, node_id):
    def cluster_reader():
        with open("-".join([filename, "%05d" % int(node_id)]), "r") as f:
            for l in f:
                csv_data = [int(cell) for cell in l.split(",")]
                yield tuple(csv_data)

    return cluster_reader


def main():
    # get arguments from env

    # for local training
    TRUTH = ["true", "True", "TRUE", "1", "yes", "Yes", "YES"]
    cluster_train = os.getenv('PADDLE_CLUSTER_TRAIN', "False") in TRUTH
    use_gpu = os.getenv('PADDLE_INIT_USE_GPU', "False")

    if not cluster_train:
        paddle.init(
            use_gpu=use_gpu,
            trainer_count=int(os.getenv("PADDLE_INIT_TRAINER_COUNT", "1")))
    else:
        paddle.init(
            use_gpu=use_gpu,
            trainer_count=int(os.getenv("PADDLE_INIT_TRAINER_COUNT", "1")),
            port=int(os.getenv("PADDLE_INIT_PORT", "7164")),
            ports_num=int(os.getenv("PADDLE_INIT_PORTS_NUM", "1")),
            ports_num_for_sparse=int(
                os.getenv("PADDLE_INIT_PORTS_NUM_FOR_SPARSE", "1")),
            num_gradient_servers=int(
                os.getenv("PADDLE_INIT_NUM_GRADIENT_SERVERS", "1")),
            trainer_id=int(os.getenv("PADDLE_INIT_TRAINER_ID", "0")),
            pservers=os.getenv("PADDLE_INIT_PSERVERS", "127.0.0.1"))
    fn = open("thirdparty/wuyi_train_thdpty/word_dict.pickle", "r")
    word_dict = pickle.load(fn)
    fn.close()
    dict_size = len(word_dict)
    firstword = paddle.layer.data(
        name="firstw", type=paddle.data_type.integer_value(dict_size))
    secondword = paddle.layer.data(
        name="secondw", type=paddle.data_type.integer_value(dict_size))
    thirdword = paddle.layer.data(
        name="thirdw", type=paddle.data_type.integer_value(dict_size))
    fourthword = paddle.layer.data(
        name="fourthw", type=paddle.data_type.integer_value(dict_size))
    nextword = paddle.layer.data(
        name="fifthw", type=paddle.data_type.integer_value(dict_size))

    Efirst = wordemb(firstword)
    Esecond = wordemb(secondword)
    Ethird = wordemb(thirdword)
    Efourth = wordemb(fourthword)

    contextemb = paddle.layer.concat(input=[Efirst, Esecond, Ethird, Efourth])
    hidden1 = paddle.layer.fc(input=contextemb,
                              size=hiddensize,
                              act=paddle.activation.Sigmoid(),
                              layer_attr=paddle.attr.Extra(drop_rate=0.5),
                              bias_attr=paddle.attr.Param(learning_rate=2),
                              param_attr=paddle.attr.Param(
                                  initial_std=1. / math.sqrt(embsize * 8),
                                  learning_rate=1))
    predictword = paddle.layer.fc(input=hidden1,
                                  size=dict_size,
                                  bias_attr=paddle.attr.Param(learning_rate=2),
                                  act=paddle.activation.Softmax())

    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0:
                result = trainer.test(
                    paddle.batch(
                        cluster_reader_cluster(cluster_test_file, node_id), 32))
                print "Pass %d, Batch %d, Cost %f, %s, Testing metrics %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics,
                    result.metrics)

    cost = paddle.layer.classification_cost(input=predictword, label=nextword)
    parameters = paddle.parameters.create(cost)
    adagrad = paddle.optimizer.AdaGrad(
        learning_rate=3e-3,
        regularization=paddle.optimizer.L2Regularization(8e-4))
    trainer = paddle.trainer.SGD(cost,
                                 parameters,
                                 adagrad,
                                 is_local=not cluster_train)
    trainer.train(
        paddle.batch(cluster_reader_cluster(cluster_train_file, node_id), 32),
        num_passes=30,
        event_handler=event_handler)


if __name__ == '__main__':
    main()
