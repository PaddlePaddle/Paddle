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

import contextlib

import unittest
from functools import partial
import numpy as np
import paddle
import paddle.fluid.core as core

import paddle.fluid as fluid
from paddle.fluid import compiler


def get_places():
    places = []
    if core.is_compiled_with_cuda():
        places.append(core.CUDAPlace(0))
    return places


@contextlib.contextmanager
def prog_scope_guard(main_prog, startup_prog):
    scope = fluid.core.Scope()
    with fluid.unique_name.guard():
        with fluid.scope_guard(scope):
            with fluid.program_guard(main_prog, startup_prog):
                yield


def bow_net(data,
            label,
            dict_dim,
            is_sparse=False,
            emb_dim=128,
            hid_dim=128,
            hid_dim2=96,
            class_dim=2):
    """
    BOW net
    This model is from https://github.com/PaddlePaddle/models:
    fluid/PaddleNLP/text_classification/nets.py
    """
    emb = fluid.layers.embedding(input=data,
                                 is_sparse=is_sparse,
                                 size=[dict_dim, emb_dim])
    bow = fluid.layers.sequence_pool(input=emb, pool_type='sum')
    bow_tanh = fluid.layers.tanh(bow)
    fc_1 = fluid.layers.fc(input=bow_tanh, size=hid_dim, act="tanh")
    fc_2 = fluid.layers.fc(input=fc_1, size=hid_dim2, act="tanh")
    prediction = fluid.layers.fc(input=[fc_2], size=class_dim, act="softmax")
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = paddle.mean(x=cost)

    return avg_cost


class TestWeightDecay(unittest.TestCase):

    def setUp(self):
        self.word_dict = paddle.dataset.imdb.word_dict()
        reader = paddle.batch(paddle.dataset.imdb.train(self.word_dict),
                              batch_size=4)()
        self.train_data = [next(reader) for _ in range(5)]
        self.learning_rate = .5

    def run_executor(self, place, feed_list, loss):
        exe = fluid.Executor(place)
        feeder = fluid.DataFeeder(feed_list=feed_list, place=place)
        exe.run(fluid.default_startup_program())
        main_prog = fluid.default_main_program()
        loss_set = []
        for data in self.train_data:
            out = exe.run(main_prog,
                          feed=feeder.feed(data),
                          fetch_list=[loss.name])

            print("loss              %s" % (np.average(out)))
            loss_set.append(np.average(out))

        return loss_set

    def run_parallel_exe(self,
                         place,
                         feed_list,
                         loss,
                         use_reduce=False,
                         use_fast_executor=False,
                         use_ir_memory_optimize=False):
        exe = fluid.Executor(place)
        feeder = fluid.DataFeeder(feed_list=feed_list, place=place)
        exe.run(fluid.default_startup_program())

        exec_strategy = fluid.ExecutionStrategy()
        if use_fast_executor:
            exec_strategy.use_experimental_executor = True

        build_strategy = fluid.BuildStrategy()
        build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce \
                if use_reduce else fluid.BuildStrategy.ReduceStrategy.AllReduce
        build_strategy.memory_optimize = use_ir_memory_optimize

        train_cp = compiler.CompiledProgram(
            fluid.default_main_program()).with_data_parallel(
                loss_name=loss.name,
                exec_strategy=exec_strategy,
                build_strategy=build_strategy)

        loss_set = []
        for data in self.train_data:
            out = exe.run(train_cp,
                          feed=feeder.feed(data),
                          fetch_list=[loss.name])
            loss_set.append(np.average(out))

        return loss_set

    def check_weight_decay(self,
                           place,
                           model,
                           use_parallel_exe=False,
                           use_reduce=False):
        main_prog = fluid.framework.Program()
        startup_prog = fluid.framework.Program()
        startup_prog.random_seed = 1
        with prog_scope_guard(main_prog=main_prog, startup_prog=startup_prog):
            data = fluid.layers.data(name="words",
                                     shape=[1],
                                     dtype="int64",
                                     lod_level=1)
            label = fluid.layers.data(name="label", shape=[1], dtype="int64")
            avg_cost = model(data, label, len(self.word_dict))

            param_list = [(var, var * self.learning_rate)
                          for var in main_prog.block(0).all_parameters()]

            optimizer = fluid.optimizer.Adagrad(
                learning_rate=self.learning_rate)
            optimizer.minimize(avg_cost)

            for params in param_list:
                updated_p = fluid.layers.elementwise_sub(x=params[0],
                                                         y=params[1])
                fluid.layers.assign(input=updated_p, output=params[0])

            if use_parallel_exe:
                loss = self.run_parallel_exe(place, [data, label],
                                             loss=avg_cost,
                                             use_reduce=use_reduce)
            else:
                loss = self.run_executor(place, [data, label], loss=avg_cost)

        return loss

    def test_weight_decay(self):
        model = partial(bow_net, is_sparse=False)
        for place in get_places():
            loss = self.check_weight_decay(place, model, use_parallel_exe=False)

            # TODO(zcd): should test use_reduce=True
            loss2 = self.check_weight_decay(place,
                                            model,
                                            use_parallel_exe=True,
                                            use_reduce=False)

            for i in range(len(loss)):
                self.assertTrue(
                    np.isclose(a=loss[i], b=loss2[i], rtol=5e-5),
                    "Expect " + str(loss[i]) + "\n" + "But Got" +
                    str(loss2[i]) + " in class " + self.__class__.__name__)


if __name__ == '__main__':
    unittest.main()
