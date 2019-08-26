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

import math

import unittest
import paddle.fluid as fluid
import paddle.fluid.incubate.fleet.base.role_maker as role_maker
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspilerConfig


class TestTranspilerWithNCE(unittest.TestCase):
    def skip_gram_word2vec(self):
        def nce_layer(input, label, embedding_size, num_total_classes,
                      num_neg_samples, sampler, word_frequencys, sample_weight):
            w_param_name = "nce_w"
            b_param_name = "nce_b"

            w_param = fluid.default_main_program().global_block(
            ).create_parameter(
                shape=[num_total_classes, embedding_size],
                dtype='float32',
                type=fluid.core.VarDesc.VarType.LOD_TENSOR,
                name=w_param_name, )
            b_param = fluid.default_main_program().global_block(
            ).create_parameter(
                shape=[num_total_classes, 1],
                dtype='float32',
                name=b_param_name, )

            cost = fluid.layers.nce(
                input=input,
                label=label,
                num_total_classes=num_total_classes,
                sampler=sampler,
                custom_dist=word_frequencys,
                sample_weight=sample_weight,
                param_attr=fluid.ParamAttr(
                    name=w_param_name,
                    initializer=fluid.initializer.Normal(
                        scale=1 / math.sqrt(num_total_classes))),
                bias_attr=fluid.ParamAttr(
                    name=b_param_name, initializer=fluid.initializer.Normal()),
                num_neg_samples=num_neg_samples,
                is_sparse=is_sparse)

            return cost

        datas = []
        word_frequencys = []

        input_word = fluid.layers.data(
            name="input_word", shape=[1], dtype='int64')
        predict_word = fluid.layers.data(
            name='predict_word', shape=[1], dtype='int64')
        datas.append(input_word)
        datas.append(predict_word)

        py_reader = fluid.layers.create_py_reader_by_data(
            capacity=64,
            feed_list=datas,
            name='py_reader',
            use_double_buffer=True)

        words = fluid.layers.read_file(py_reader)

        dict_size = 10001
        embedding_size = 11
        is_sparse = True

        emb = fluid.layers.embedding(
            input=words[0],
            is_sparse=is_sparse,
            size=[dict_size + 10, embedding_size],
            param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                scale=1 / math.sqrt(dict_size))))

        fc0 = fluid.layers.fc(emb, size=11)

        cost = nce_layer(fc0, words[1], embedding_size, dict_size, 5, "uniform",
                         word_frequencys, [])

        avg_cost = fluid.layers.reduce_mean(cost)
        return avg_cost, py_reader

    def get_trainer_program(self):
        role = role_maker.UserDefinedRoleMaker(
            current_id=0,
            role=role_maker.Role.WORKER,
            worker_num=2,
            server_endpoints=["127.0.0.1:6001", "127.0.0.1:6002"])

        fleet.init(role)
        avg_cost, py_reader = self.skip_gram_word2vec()

        optimizer = fluid.optimizer.SGD(0.01)

        strategy = DistributeTranspilerConfig()
        strategy.sync_mode = True
        strategy.wait_port = False
        optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(avg_cost)

        return fleet.main_program

    def test_nce_at_transpiler(self):
        trainer_pro = self.get_trainer_program()

        nce_op = None
        for op in trainer_pro.global_block().ops:
            if op.type == "nce":
                nce_op = op
                break

        self.assertEqual(nce_op.type, "nce")
        self.assertEqual(nce_op.attr('is_sparse'), True)
        self.assertEqual(nce_op.attr('remote_prefetch'), True)


if __name__ == '__main__':
    unittest.main()
