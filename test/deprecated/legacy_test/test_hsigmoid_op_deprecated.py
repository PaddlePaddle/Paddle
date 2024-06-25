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
import unittest

import numpy as np

import paddle
from paddle import base

paddle.enable_static()
np.random.seed(100)


class TestHSigmoidOpWithSparseGrad(unittest.TestCase):
    def hs_net_conf(self, is_sparse):
        input_word = paddle.static.data(name="x", shape=[-1, 1], dtype='int64')
        path_table = paddle.static.data(
            name='path_table', shape=[-1, 3], dtype='int64'
        )
        path_code = paddle.static.data(
            name='path_code', shape=[-1, 3], dtype='int64'
        )
        label = paddle.static.data(name='label', shape=[-1, 1], dtype='int64')

        data_list = [input_word, path_table, path_code, label]

        emb = paddle.static.nn.embedding(
            input=input_word,
            is_sparse=is_sparse,
            size=[3, 3],
            param_attr=base.ParamAttr(
                initializer=paddle.nn.initializer.Normal(std=1 / math.sqrt(3))
            ),
        )

        loss = paddle.nn.HSigmoidLoss(
            feature_size=emb.shape[1],
            num_classes=3,
            bias_attr=True,
            is_custom=True,
            is_sparse=is_sparse,
        )

        cost = loss(
            input=emb,
            label=label,
            path_table=path_table,
            path_code=path_code,
        )

        avg_cost = paddle.mean(cost)

        return avg_cost, data_list

    def training_test(self, is_sparse):
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            paddle.seed(1)
            start_up = paddle.static.default_startup_program()
            x = np.arange(6).reshape(6)
            path_table = np.array([(1, 2, -1), (1, 2, -1)]).astype('int64')
            path_code = np.array([(1, 0, -1), (0, 0, -1)]).astype('int64')
            label = np.array([1, 4]).astype('int64')

            loss, data_list = self.hs_net_conf(is_sparse)
            optimizer = paddle.optimizer.SGD(learning_rate=1e-3)
            optimizer.minimize(loss)

            main_program = paddle.static.default_main_program()
            place = base.CPUPlace()
            feeder = base.DataFeeder(feed_list=data_list, place=place)
            exe = paddle.static.Executor(place)

            exe.run(start_up)
            result = []
            for i in range(10):
                data = [
                    (
                        [[x[i % 2]]],
                        [list(path_table[i % 2])],
                        [list(path_code[i % 2])],
                        [label[i % 2]],
                    )
                ]

                loss_val = exe.run(
                    main_program, feed=feeder.feed(data), fetch_list=[loss]
                )
                result.append(loss_val)
        return result

    def test_hs_grad_with_sparse(self):
        dense_result = self.training_test(is_sparse=False)
        sparse_result = self.training_test(is_sparse=True)
        assert dense_result == sparse_result


if __name__ == '__main__':
    unittest.main()
