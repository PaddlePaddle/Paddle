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

# Copyright PaddlePaddle contributors. All Rights Reservedd
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
import math
import paddle.v2 as paddle


def wordemb(inlayer):
    wordemb = paddle.layer.table_projection(
        input=inlayer,
        size=5,
        param_attr=paddle.attr.Param(
            name="_proj", initial_std=0.001, learning_rate=1, l2_rate=0))
    return wordemb


def train():
    word_dict = paddle.dataset.imikolov.build_dict()
    dict_size = len(word_dict)
    # Every layer takes integer value of range [0, dict_size)
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
    hidden1 = paddle.layer.fc(name="fc1",
                              input=contextemb,
                              size=128,
                              act=paddle.activation.Sigmoid(),
                              layer_attr=paddle.attr.Extra(drop_rate=0.5),
                              bias_attr=paddle.attr.Param(learning_rate=2),
                              param_attr=paddle.attr.Param(
                                  initial_std=1. / math.sqrt(5 * 8),
                                  learning_rate=1,
                                  l2_rate=6e-4))
    predictword = paddle.layer.fc(input=hidden1,
                                  size=dict_size,
                                  bias_attr=paddle.attr.Param(learning_rate=2),
                                  act=paddle.activation.Softmax())

    return paddle.layer.classification_cost(input=predictword, label=nextword)


class TestParamConfOrder(unittest.TestCase):
    def test_param_conf_order(self):
        paddle.init()
        cost = train()
        parameters = paddle.parameters.create(cost)
        adagrad = paddle.optimizer.AdaGrad(
            learning_rate=3e-3,
            regularization=paddle.optimizer.L2Regularization(rate=8e-4))

        trainer = paddle.trainer.SGD(cost, parameters, adagrad)
        for p in trainer.get_topology_proto().parameters:
            if p.name == "_fc1.w0":
                self.assertEqual(p.decay_rate, 6e-4)
            else:
                self.assertEqual(p.decay_rate, 8e-4)


if __name__ == '__main__':
    unittest.main()
