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

import os

from test_dist_base import TestDistRunnerBase, runtime_main

import paddle
from paddle import base

IS_SPARSE = True
EMBED_SIZE = 32
HIDDEN_SIZE = 256
N = 5

# Fix seed for test
paddle.seed(1)


class TestDistWord2vec2x2(TestDistRunnerBase):
    def get_model(self, batch_size=2):
        BATCH_SIZE = batch_size

        def __network__(words):
            embed_first = paddle.static.nn.embedding(
                input=words[0],
                size=[dict_size, EMBED_SIZE],
                dtype='float32',
                is_sparse=IS_SPARSE,
                param_attr=base.ParamAttr(
                    name='shared_w',
                    initializer=paddle.nn.initializer.Constant(value=0.1),
                ),
            )
            embed_second = paddle.static.nn.embedding(
                input=words[1],
                size=[dict_size, EMBED_SIZE],
                dtype='float32',
                is_sparse=IS_SPARSE,
                param_attr=base.ParamAttr(
                    name='shared_w',
                    initializer=paddle.nn.initializer.Constant(value=0.1),
                ),
            )
            embed_third = paddle.static.nn.embedding(
                input=words[2],
                size=[dict_size, EMBED_SIZE],
                dtype='float32',
                is_sparse=IS_SPARSE,
                param_attr=base.ParamAttr(
                    name='shared_w',
                    initializer=paddle.nn.initializer.Constant(value=0.1),
                ),
            )
            embed_forth = paddle.static.nn.embedding(
                input=words[3],
                size=[dict_size, EMBED_SIZE],
                dtype='float32',
                is_sparse=IS_SPARSE,
                param_attr=base.ParamAttr(
                    name='shared_w',
                    initializer=paddle.nn.initializer.Constant(value=0.1),
                ),
            )

            concat_embed = paddle.concat(
                [embed_first, embed_second, embed_third, embed_forth],
                axis=1,
            )
            hidden1 = paddle.static.nn.fc(
                x=concat_embed,
                size=HIDDEN_SIZE,
                activation='sigmoid',
                weight_attr=base.ParamAttr(
                    initializer=paddle.nn.initializer.Constant(value=0.1)
                ),
            )
            predict_word = paddle.static.nn.fc(
                x=hidden1,
                size=dict_size,
                activation='softmax',
                weight_attr=base.ParamAttr(
                    initializer=paddle.nn.initializer.Constant(value=0.1)
                ),
            )
            cost = paddle.nn.functional.cross_entropy(
                input=predict_word,
                label=words[4],
                reduction='none',
                use_softmax=False,
            )
            avg_cost = paddle.mean(cost)
            return avg_cost, predict_word

        word_dict = paddle.dataset.imikolov.build_dict()
        dict_size = len(word_dict)

        first_word = paddle.static.data(
            name='firstw', shape=[-1, 1], dtype='int64'
        )
        second_word = paddle.static.data(
            name='secondw', shape=[-1, 1], dtype='int64'
        )
        third_word = paddle.static.data(
            name='thirdw', shape=[-1, 1], dtype='int64'
        )
        forth_word = paddle.static.data(
            name='forthw', shape=[-1, 1], dtype='int64'
        )
        next_word = paddle.static.data(
            name='nextw', shape=[-1, 1], dtype='int64'
        )
        avg_cost, predict_word = __network__(
            [first_word, second_word, third_word, forth_word, next_word]
        )

        inference_program = paddle.base.default_main_program().clone()

        sgd_optimizer = paddle.optimizer.SGD(learning_rate=0.001)
        sgd_optimizer.minimize(avg_cost)

        train_reader = paddle.batch(
            paddle.dataset.imikolov.train(word_dict, N), BATCH_SIZE
        )
        test_reader = paddle.batch(
            paddle.dataset.imikolov.test(word_dict, N), BATCH_SIZE
        )

        return (
            inference_program,
            avg_cost,
            train_reader,
            test_reader,
            None,
            predict_word,
        )


if __name__ == "__main__":
    os.environ['CPU_NUM'] = '1'
    os.environ['USE_CUDA'] = "FALSE"
    runtime_main(TestDistWord2vec2x2)
