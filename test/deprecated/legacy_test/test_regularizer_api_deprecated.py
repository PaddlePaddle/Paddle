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
import os
import random
import unittest
from functools import partial

import numpy as np

import paddle
from paddle import base
from paddle.base import core


def bow_net(
    data,
    label,
    dict_dim,
    is_sparse=False,
    emb_dim=8,
    hid_dim=8,
    hid_dim2=6,
    class_dim=2,
):
    """
    BOW net
    This model is from https://github.com/PaddlePaddle/models:
    base/PaddleNLP/text_classification/nets.py
    """
    emb = paddle.static.nn.embedding(
        input=data, is_sparse=is_sparse, size=[dict_dim, emb_dim]
    )
    bow = paddle.static.nn.sequence_lod.sequence_pool(
        input=emb, pool_type='sum'
    )
    bow_tanh = paddle.tanh(bow)
    fc_1 = paddle.static.nn.fc(x=bow_tanh, size=hid_dim, activation="tanh")
    fc_2 = paddle.static.nn.fc(x=fc_1, size=hid_dim2, activation="tanh")
    prediction = paddle.static.nn.fc(
        x=[fc_2], size=class_dim, activation="softmax"
    )
    cost = paddle.nn.functional.cross_entropy(
        input=prediction, label=label, reduction='none', use_softmax=False
    )
    avg_cost = paddle.mean(x=cost)

    return avg_cost


class TestRegularizer(unittest.TestCase):
    def setUp(self):
        self.word_len = 1500
        self.train_data = [
            [(random.sample(range(1000), 10), [0])] for _ in range(2)
        ]

    def get_places(self):
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(core.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        return places

    @contextlib.contextmanager
    def scope_prog_guard(self, main_prog, startup_prog):
        scope = base.core.Scope()
        with base.unique_name.guard():
            with base.scope_guard(scope):
                with base.program_guard(main_prog, startup_prog):
                    yield

    def run_program(self, place, feed_list):
        exe = base.Executor(place)
        feeder = base.DataFeeder(feed_list=feed_list, place=place)
        exe.run(base.default_startup_program())

        main_prog = base.default_main_program()
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

    def check_l2decay_regularizer(self, place, model):
        paddle.seed(1)
        paddle.framework.random._manual_program_seed(1)
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        with self.scope_prog_guard(
            main_prog=main_prog, startup_prog=startup_prog
        ):
            data = paddle.static.data(
                name="words", shape=[-1, 1], dtype="int64", lod_level=1
            )
            label = paddle.static.data(
                name="label", shape=[-1, 1], dtype="int64"
            )

            avg_cost = model(data, label, self.word_len)

            optimizer = paddle.optimizer.Adagrad(
                learning_rate=0.1,
                weight_decay=paddle.regularizer.L2Decay(1.0),
            )
            optimizer.minimize(avg_cost)
            param_sum = self.run_program(place, [data, label])
        return param_sum

    def check_l2decay(self, place, model):
        paddle.seed(1)
        paddle.framework.random._manual_program_seed(1)
        main_prog = base.framework.Program()
        startup_prog = base.framework.Program()

        with self.scope_prog_guard(
            main_prog=main_prog, startup_prog=startup_prog
        ):
            data = paddle.static.data(
                name="words", shape=[-1, 1], dtype="int64", lod_level=1
            )
            label = paddle.static.data(
                name="label", shape=[-1, 1], dtype="int64"
            )

            avg_cost_l2 = model(data, label, self.word_len)

            param_list = base.default_main_program().block(0).all_parameters()
            para_sum = []
            for para in param_list:
                para_mul = paddle.square(x=para)
                para_sum.append(paddle.sum(para_mul))
            avg_cost_l2 += paddle.add_n(para_sum) * 0.5

            optimizer = paddle.optimizer.Adagrad(learning_rate=0.1)
            optimizer.minimize(avg_cost_l2)
            param_sum = self.run_program(place, [data, label])
        return param_sum

    def test_l2(self):
        paddle.enable_static()
        for place in self.get_places():
            dense_sparse_p_sum = []
            for sparse in [True, False]:
                model = partial(bow_net, is_sparse=sparse)
                framework_l2 = self.check_l2decay_regularizer(place, model)
                l2 = self.check_l2decay(place, model)
                assert len(l2) == len(framework_l2)
                for i in range(len(l2)):
                    assert np.isclose(a=framework_l2[i], b=l2[i], rtol=5e-5)
                dense_sparse_p_sum.append(framework_l2)

            assert len(dense_sparse_p_sum[0]) == len(dense_sparse_p_sum[1])
            for i in range(len(dense_sparse_p_sum[0])):
                assert np.isclose(
                    a=dense_sparse_p_sum[0][i],
                    b=dense_sparse_p_sum[1][i],
                    rtol=5e-5,
                )


if __name__ == '__main__':
    unittest.main()
