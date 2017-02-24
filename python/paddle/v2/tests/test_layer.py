# Copyright PaddlePaddle contributors. All Rights Reserved
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
import difflib
import unittest

import paddle.trainer_config_helpers as conf_helps
import paddle.v2.activation as activation
import paddle.v2.attr as attr
import paddle.v2.data_type as data_type
import paddle.v2.layer as layer
from paddle.trainer_config_helpers.config_parser_utils import \
    parse_network_config as parse_network

pixel = layer.data(name='pixel', type=data_type.dense_vector(784))
label = layer.data(name='label', type=data_type.integer_value(10))
weight = layer.data(name='weight', type=data_type.dense_vector(10))
score = layer.data(name='score', type=data_type.dense_vector(1))
hidden = layer.fc(input=pixel,
                  size=100,
                  act=activation.Sigmoid(),
                  param_attr=attr.Param(name='hidden'))
inference = layer.fc(input=hidden, size=10, act=activation.Softmax())


class CostLayerTest(unittest.TestCase):
    def test_cost_layer(self):
        cost1 = layer.classification_cost(input=inference, label=label)
        cost2 = layer.classification_cost(
            input=inference, label=label, weight=weight)
        cost3 = layer.cross_entropy_cost(input=inference, label=label)
        cost4 = layer.cross_entropy_with_selfnorm_cost(
            input=inference, label=label)
        cost5 = layer.regression_cost(input=inference, label=label)
        cost6 = layer.regression_cost(
            input=inference, label=label, weight=weight)
        cost7 = layer.multi_binary_label_cross_entropy_cost(
            input=inference, label=label)
        cost8 = layer.rank_cost(left=score, right=score, label=score)
        cost9 = layer.lambda_cost(input=inference, score=score)
        cost10 = layer.sum_cost(input=inference)
        cost11 = layer.huber_cost(input=score, label=label)

        print dir(layer)
        layer.parse_network(cost1, cost2)
        print dir(layer)
        #print layer.parse_network(cost3, cost4)
        #print layer.parse_network(cost5, cost6)
        #print layer.parse_network(cost7, cost8, cost9, cost10, cost11)


if __name__ == '__main__':
    unittest.main()
