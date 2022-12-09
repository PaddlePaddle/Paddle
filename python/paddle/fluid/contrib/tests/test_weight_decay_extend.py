#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
from functools import partial
import numpy as np
import paddle
import paddle.fluid as fluid
import contextlib

paddle.enable_static()

SEED = 2020


def fake_imdb_reader(
    word_dict_size,
    sample_num,
    lower_seq_len=100,
    upper_seq_len=200,
    class_dim=2,
):
    def __reader__():
        for _ in range(sample_num):
            length = np.random.random_integers(
                low=lower_seq_len, high=upper_seq_len, size=[1]
            )[0]
            ids = np.random.random_integers(
                low=0, high=word_dict_size - 1, size=[length]
            ).astype('int64')
            label = np.random.random_integers(
                low=0, high=class_dim - 1, size=[1]
            ).astype('int64')[0]
            yield ids, label

    return __reader__


def get_places():
    places = [fluid.CPUPlace()]
    if fluid.core.is_compiled_with_cuda():
        places.append(fluid.CUDAPlace(0))
    return places


@contextlib.contextmanager
def prog_scope_guard(main_prog, startup_prog):
    scope = fluid.core.Scope()
    with fluid.unique_name.guard():
        with fluid.scope_guard(scope):
            with fluid.program_guard(main_prog, startup_prog):
                yield


def bow_net(
    data,
    label,
    dict_dim,
    is_sparse=False,
    emb_dim=128,
    hid_dim=128,
    hid_dim2=96,
    class_dim=2,
):
    """
    BOW net
    This model is from https://github.com/PaddlePaddle/models:
    fluid/PaddleNLP/text_classification/nets.py
    """
    emb = fluid.layers.embedding(
        input=data, is_sparse=is_sparse, size=[dict_dim, emb_dim]
    )
    bow = fluid.layers.sequence_pool(input=emb, pool_type='sum')
    bow_tanh = paddle.tanh(bow)
    fc_1 = fluid.layers.fc(input=bow_tanh, size=hid_dim, act="tanh")
    fc_2 = fluid.layers.fc(input=fc_1, size=hid_dim2, act="tanh")
    prediction = fluid.layers.fc(input=[fc_2], size=class_dim, act="softmax")
    cost = paddle.nn.functional.cross_entropy(
        input=prediction, label=label, reduction='none', use_softmax=False
    )
    avg_cost = paddle.mean(x=cost)

    return avg_cost


class TestWeightDecay(unittest.TestCase):
    def setUp(self):
        # set seed
        np.random.seed(SEED)
        paddle.seed(SEED)
        paddle.framework.random._manual_program_seed(SEED)
        # configs
        self.word_dict_len = 5147
        batch_size = 2
        reader = fake_imdb_reader(self.word_dict_len, batch_size * 100)
        reader = paddle.batch(reader, batch_size=batch_size)()
        self.train_data = [next(reader) for _ in range(3)]
        self.learning_rate = 0.5

    def run_program(self, place, feed_list):
        exe = fluid.Executor(place)
        feeder = fluid.DataFeeder(feed_list=feed_list, place=place)
        exe.run(fluid.default_startup_program())

        main_prog = fluid.default_main_program()
        param_list = [var.name for var in main_prog.block(0).all_parameters()]

        param_sum = []
        for data in self.train_data:
            out = exe.run(
                main_prog, feed=feeder.feed(data), fetch_list=param_list
            )
            p_sum = 0
            for v in out:
                p_sum += np.sum(np.abs(v))
            param_sum.append(p_sum)
        return param_sum

    def check_weight_decay(self, place, model):
        main_prog = fluid.framework.Program()
        startup_prog = fluid.framework.Program()

        with prog_scope_guard(main_prog=main_prog, startup_prog=startup_prog):
            data = fluid.layers.data(
                name="words", shape=[1], dtype="int64", lod_level=1
            )
            label = fluid.layers.data(name="label", shape=[1], dtype="int64")
            avg_cost = model(data, label, self.word_dict_len)
            AdamW = fluid.contrib.extend_with_decoupled_weight_decay(
                fluid.optimizer.Adam
            )

            optimizer = AdamW(
                learning_rate=self.learning_rate,
                weight_decay=self.learning_rate,
            )

            optimizer.minimize(avg_cost)
            param_sum = self.run_program(place, [data, label])

        return param_sum

    def check_weight_decay2(self, place, model):
        main_prog = fluid.framework.Program()
        startup_prog = fluid.framework.Program()

        with prog_scope_guard(main_prog=main_prog, startup_prog=startup_prog):
            data = fluid.layers.data(
                name="words", shape=[1], dtype="int64", lod_level=1
            )
            label = fluid.layers.data(name="label", shape=[1], dtype="int64")

            avg_cost = model(data, label, self.word_dict_len)

            optimizer = fluid.optimizer.Adam(learning_rate=self.learning_rate)

            params_grads = optimizer.backward(avg_cost)

            param_list = [
                (var, var * self.learning_rate)
                for var in main_prog.block(0).all_parameters()
            ]

            for params in param_list:
                updated_p = paddle.subtract(x=params[0], y=params[1])
                fluid.layers.assign(input=updated_p, output=params[0])

            optimizer.apply_optimize(avg_cost, startup_prog, params_grads)

            param_sum = self.run_program(place, [data, label])
        return param_sum

    def test_weight_decay(self):
        for place in get_places():
            model = partial(bow_net, is_sparse=False)
            param_sum1 = self.check_weight_decay(place, model)
            param_sum2 = self.check_weight_decay2(place, model)

            for i in range(len(param_sum1)):
                np.testing.assert_allclose(
                    param_sum1[i],
                    param_sum2[i],
                    rtol=1e-05,
                    err_msg='Current place: {}, i: {}, sum1: {}, sum2: {}'.format(
                        place,
                        i,
                        param_sum1[i][
                            ~np.isclose(param_sum1[i], param_sum2[i])
                        ],
                        param_sum2[i][
                            ~np.isclose(param_sum1[i], param_sum2[i])
                        ],
                    ),
                )


if __name__ == '__main__':
    unittest.main()
