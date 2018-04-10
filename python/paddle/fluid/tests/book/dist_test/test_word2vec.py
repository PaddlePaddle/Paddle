#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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
import paddle.fluid as fluid
import paddle.v2
import unittest
import os
import numpy as np
import math
import sys


def train(use_cuda=False, is_sparse=True, is_local=False):
    PASS_NUM = 1000
    EMBED_SIZE = 32
    HIDDEN_SIZE = 256
    N = 5
    BATCH_SIZE = 32
    IS_SPARSE = is_sparse

    def __network__(words):
        id_distributed = True
        embed_first = fluid.layers.embedding(
            input=words[0],
            size=[dict_size, EMBED_SIZE],
            dtype='float32',
            is_sparse=IS_SPARSE,
            param_attr='shared_w',
            is_distributed=id_distributed)
        embed_second = fluid.layers.embedding(
            input=words[1],
            size=[dict_size, EMBED_SIZE],
            dtype='float32',
            is_sparse=IS_SPARSE,
            param_attr='shared_w',
            is_distributed=id_distributed)
        embed_third = fluid.layers.embedding(
            input=words[2],
            size=[dict_size, EMBED_SIZE],
            dtype='float32',
            is_sparse=IS_SPARSE,
            param_attr='shared_w',
            is_distributed=id_distributed)
        embed_forth = fluid.layers.embedding(
            input=words[3],
            size=[dict_size, EMBED_SIZE],
            dtype='float32',
            is_sparse=IS_SPARSE,
            param_attr='shared_w',
            is_distributed=id_distributed)

        concat_embed = fluid.layers.concat(
            input=[embed_first, embed_second, embed_third, embed_forth], axis=1)
        hidden1 = fluid.layers.fc(input=concat_embed,
                                  size=HIDDEN_SIZE,
                                  act='sigmoid')
        predict_word = fluid.layers.fc(input=hidden1,
                                       size=dict_size,
                                       act='softmax')
        cost = fluid.layers.cross_entropy(input=predict_word, label=words[4])
        avg_cost = fluid.layers.mean(cost)
        return avg_cost, predict_word

    word_dict = paddle.v2.dataset.imikolov.build_dict()
    dict_size = len(word_dict)

    first_word = fluid.layers.data(name='firstw', shape=[1], dtype='int64')
    second_word = fluid.layers.data(name='secondw', shape=[1], dtype='int64')
    third_word = fluid.layers.data(name='thirdw', shape=[1], dtype='int64')
    forth_word = fluid.layers.data(name='forthw', shape=[1], dtype='int64')
    next_word = fluid.layers.data(name='nextw', shape=[1], dtype='int64')

    avg_cost, predict_word = __network__(
        [first_word, second_word, third_word, forth_word, next_word])

    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
    optimize_ops, params_grads = sgd_optimizer.minimize(avg_cost)

    train_reader = paddle.v2.batch(
        paddle.v2.dataset.imikolov.train(word_dict, N), BATCH_SIZE)

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    feeder = fluid.DataFeeder(
        feed_list=[first_word, second_word, third_word, forth_word, next_word],
        place=place)

    def train_loop(main_program):
        exe.run(fluid.default_startup_program())

        TRAINING_BATCHES = 5
        batch_num = 0
        for pass_id in range(PASS_NUM):
            for data in train_reader():
                avg_cost_np = exe.run(main_program,
                                      feed=feeder.feed(data),
                                      fetch_list=[avg_cost])
                #if batch_num == TRAINING_BATCHES:
                #    return
                if batch_num % 10 == 0:
                    print("pass_id=" + str(pass_id) + ", batch_id=" + str(
                        batch_num) + ", cost=" + str(avg_cost_np[0]))
                if avg_cost_np[0] < 5.0:
                    return
                if math.isnan(float(avg_cost_np[0])):
                    sys.exit("got NaN loss, training failed.")
                batch_num += 1

        raise AssertionError("Cost is too large {0:2.2}".format(avg_cost_np[0]))

    if is_local:
        train_loop(fluid.default_main_program())
    else:
        port = os.getenv("PADDLE_INIT_PORT", "6174")
        pserver_ips = os.getenv("PADDLE_INIT_PSERVERS")  # ip,ip...
        eplist = []
        for ip in pserver_ips.split(","):
            eplist.append(':'.join([ip, port]))
        pserver_endpoints = ",".join(eplist)  # ip:port,ip:port...
        trainers = int(os.getenv("TRAINERS"))
        current_endpoint = os.getenv("POD_IP") + ":" + port
        trainer_id = int(os.getenv("PADDLE_INIT_TRAINER_ID"))
        training_role = os.getenv("TRAINING_ROLE", "TRAINER")
        t = fluid.DistributeTranspiler()
        t.transpile(
            optimize_ops,
            params_grads,
            trainer_id,
            pservers=pserver_endpoints,
            trainer_num=trainers)
        with open("program.proto", "w") as f:
            f.write(str(fluid.default_main_program()))
        if training_role == "PSERVER":
            pserver_prog = t.get_pserver_program(current_endpoint)
            with open("pserver.proto", "w") as f:
                f.write(str(pserver_prog))
            pserver_startup = t.get_startup_program(current_endpoint,
                                                    pserver_prog)
            with open("startup.proto", "w") as f:
                f.write(str(pserver_startup))
            exe.run(pserver_startup)
            exe.run(pserver_prog)
        elif training_role == "TRAINER":
            trainer_program = t.get_trainer_program()
            with open("trainer.proto", "w") as f:
                f.write(str(trainer_program))
            train_loop(trainer_program)


class W2VTest(unittest.TestCase):
    def test_main(self):
        train()


if __name__ == '__main__':
    unittest.main()
