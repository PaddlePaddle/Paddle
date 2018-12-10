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

from __future__ import print_function

import unittest
from functools import partial
import contextlib
import numpy as np
import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid

# import paddle.fluid.framework as framework
# import paddle.fluid.optimizer as optimizer
# import paddle.fluid.regularizer as regularizer
# from paddle.fluid.backward import append_backward


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
    emb = fluid.layers.embedding(
        input=data, is_sparse=is_sparse, size=[dict_dim, emb_dim])
    bow = fluid.layers.sequence_pool(input=emb, pool_type='sum')
    bow_tanh = fluid.layers.tanh(bow)
    fc_1 = fluid.layers.fc(input=bow_tanh, size=hid_dim, act="tanh")
    fc_2 = fluid.layers.fc(input=fc_1, size=hid_dim2, act="tanh")
    prediction = fluid.layers.fc(input=[fc_2], size=class_dim, act="softmax")
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)

    return avg_cost


class TestWeightDecay(unittest.TestCase):
    def setUp(self):
        self.word_dict = paddle.dataset.imdb.word_dict()
        reader = paddle.batch(
            paddle.dataset.imdb.train(self.word_dict), batch_size=2)()
        self.train_data = [next(reader) for _ in range(5)]

    def get_places(self):
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        return places

    @contextlib.contextmanager
    def scope_prog_guard(self, main_prog, startup_prog):
        scope = fluid.core.Scope()
        with fluid.unique_name.guard():
            with fluid.scope_guard(scope):
                with fluid.program_guard(main_prog, startup_prog):
                    yield

    def run_program(self, place, feed_list):
        exe = fluid.Executor(place)
        feeder = fluid.DataFeeder(feed_list=feed_list, place=place)
        exe.run(fluid.default_startup_program())

        main_prog = fluid.default_main_program()
        param_list = [var.name for var in main_prog.block(0).all_parameters()]

        param_sum = []
        for data in self.train_data:
            out = exe.run(main_prog,
                          feed=feeder.feed(data),
                          fetch_list=param_list)
            p_sum = 0
            for v in out:
                p_sum += np.sum(np.abs(v))
            param_sum.append(p_sum)
            print(p_sum)
        return param_sum

    def check_weight_decay(self, place, model):
        main_prog = fluid.framework.Program()
        startup_prog = fluid.framework.Program()
        startup_prog.random_seed = 1
        with self.scope_prog_guard(
                main_prog=main_prog, startup_prog=startup_prog):

            data = fluid.layers.data(
                name="words", shape=[1], dtype="int64", lod_level=1)
            label = fluid.layers.data(name="label", shape=[1], dtype="int64")

            avg_cost = model(data, label, len(self.word_dict))

            weight_decay = fluid.weight_decay.WeightDecay(
                fluid.default_main_program(), coeff=0.1)

            optimizer = fluid.optimizer.Adagrad(learning_rate=0.1)

            weight_decay.decay()

            optimizer.minimize(avg_cost)
            param_sum = self.run_program(place, [data, label])

        return param_sum

    def test_weight_decay(self):
        for place in self.get_places():
            model = partial(bow_net, is_sparse=False)
            self.check_weight_decay(place, model)


if __name__ == '__main__':
    unittest.main()
