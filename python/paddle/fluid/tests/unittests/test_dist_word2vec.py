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

import numpy as np
import argparse
import time
import math
import paddle
import paddle.fluid as fluid
import paddle.fluid.profiler as profiler
from paddle.fluid import core
import unittest
from multiprocessing import Process
import os
import signal

IS_SPARSE = True
EMBED_SIZE = 32
HIDDEN_SIZE = 256
N = 5
BATCH_SIZE = 32
ExecutionStrategy = core.ParallelExecutor.ExecutionStrategy


def get_model():
    def __network__(words):
        embed_first = fluid.layers.embedding(
            input=words[0],
            size=[dict_size, EMBED_SIZE],
            dtype='float32',
            is_sparse=IS_SPARSE,
            param_attr='shared_w')
        embed_second = fluid.layers.embedding(
            input=words[1],
            size=[dict_size, EMBED_SIZE],
            dtype='float32',
            is_sparse=IS_SPARSE,
            param_attr='shared_w')
        embed_third = fluid.layers.embedding(
            input=words[2],
            size=[dict_size, EMBED_SIZE],
            dtype='float32',
            is_sparse=IS_SPARSE,
            param_attr='shared_w')
        embed_forth = fluid.layers.embedding(
            input=words[3],
            size=[dict_size, EMBED_SIZE],
            dtype='float32',
            is_sparse=IS_SPARSE,
            param_attr='shared_w')

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

    word_dict = paddle.dataset.imikolov.build_dict()
    dict_size = len(word_dict)

    first_word = fluid.layers.data(name='firstw', shape=[1], dtype='int64')
    second_word = fluid.layers.data(name='secondw', shape=[1], dtype='int64')
    third_word = fluid.layers.data(name='thirdw', shape=[1], dtype='int64')
    forth_word = fluid.layers.data(name='forthw', shape=[1], dtype='int64')
    next_word = fluid.layers.data(name='nextw', shape=[1], dtype='int64')
    avg_cost, predict_word = __network__(
        [first_word, second_word, third_word, forth_word, next_word])

    inference_program = paddle.fluid.default_main_program().clone()

    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
    sgd_optimizer.minimize(avg_cost)

    train_reader = paddle.batch(
        paddle.dataset.imikolov.train(word_dict, N), BATCH_SIZE)
    test_reader = paddle.batch(
        paddle.dataset.imikolov.test(word_dict, N), BATCH_SIZE)

    return inference_program, avg_cost, train_reader, test_reader, predict_word


def get_transpiler(trainer_id, main_program, pserver_endpoints, trainers):
    t = fluid.DistributeTranspiler()
    t.transpile(
        trainer_id=trainer_id,
        program=main_program,
        pservers=pserver_endpoints,
        trainers=trainers)
    return t


def run_pserver(pserver_endpoints, trainers, current_endpoint):
    get_model()
    t = get_transpiler(0,
                       fluid.default_main_program(), pserver_endpoints,
                       trainers)
    pserver_prog = t.get_pserver_program(current_endpoint)
    startup_prog = t.get_startup_program(current_endpoint, pserver_prog)

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    exe.run(pserver_prog)


class TestDistMnist(unittest.TestCase):
    def setUp(self):
        self._trainers = 1
        self._pservers = 1
        self._ps_endpoints = "127.0.0.1:9123"

    def start_pserver(self, endpoint):
        p = Process(
            target=run_pserver,
            args=(self._ps_endpoints, self._trainers, endpoint))
        p.start()
        return p.pid

    def _wait_ps_ready(self, pid):
        retry_times = 5
        while True:
            assert retry_times >= 0, "wait ps ready failed"
            time.sleep(1)
            try:
                # the listen_and_serv_op would touch a file which contains the listen port
                # on the /tmp directory until it was ready to process all the RPC call.
                os.stat("/tmp/paddle.%d.port" % pid)
                return
            except os.error:
                retry_times -= 1

    def stop_pserver(self, pid):
        os.kill(pid, signal.SIGKILL)

    def test_with_place(self):
        p = fluid.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()

        pserver_pid = self.start_pserver(self._ps_endpoints)
        self._wait_ps_ready(pserver_pid)

        self.run_trainer(p, 0)

        self.stop_pserver(pserver_pid)

    def run_trainer(self, place, trainer_id):
        test_program, avg_cost, train_reader, test_reader, predict = get_model()
        t = get_transpiler(trainer_id,
                           fluid.default_main_program(), self._ps_endpoints,
                           self._trainers)

        trainer_prog = t.get_trainer_program()

        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())

        use_gpu = True if core.is_compiled_with_cuda() else False

        exec_strategy = ExecutionStrategy()
        exec_strategy.use_cuda = use_gpu
        train_exe = fluid.ParallelExecutor(
            use_cuda=use_gpu,
            main_program=trainer_prog,
            loss_name=avg_cost.name,
            exec_strategy=exec_strategy)

        feed_var_list = [
            var for var in trainer_prog.global_block().vars.itervalues()
            if var.is_data
        ]

        feeder = fluid.DataFeeder(feed_var_list, place)
        for pass_id in xrange(10):
            for batch_id, data in enumerate(train_reader()):
                avg_loss_np = train_exe.run(feed=feeder.feed(data),
                                            fetch_list=[avg_cost.name])
                loss = np.array(avg_loss_np).mean()
                if float(loss) < 5.0:
                    return
                if math.isnan(loss):
                    assert ("Got Nan loss, training failed")


if __name__ == "__main__":
    unittest.main()
