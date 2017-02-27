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

    def test_projection(self):
        input = layer.data(name='data', type=data_type.dense_vector(784))
        word = layer.data(
            name='word', type=data_type.integer_value_sequence(10000))
        fc0 = layer.fc(input=input, size=100, act=activation.Sigmoid())
        fc1 = layer.fc(input=input, size=200, act=activation.Sigmoid())
        mixed0 = layer.mixed(
            size=256,
            input=[
                layer.full_matrix_projection(input=fc0),
                layer.full_matrix_projection(input=fc1)
            ])
        with layer.mixed(size=200) as mixed1:
            mixed1 += layer.full_matrix_projection(input=fc0)
            mixed1 += layer.identity_projection(input=fc1)

        table = layer.table_projection(input=word)
        emb0 = layer.mixed(size=512, input=table)
        with layer.mixed(size=512) as emb1:
            emb1 += table

        scale = layer.scaling_projection(input=fc0)
        scale0 = layer.mixed(size=100, input=scale)
        with layer.mixed(size=100) as scale1:
            scale1 += scale

        dotmul = layer.dotmul_projection(input=fc0)
        dotmul0 = layer.mixed(size=100, input=dotmul)
        with layer.mixed(size=100) as dotmul1:
            dotmul1 += dotmul

        context = layer.context_projection(input=fc0, context_len=5)
        context0 = layer.mixed(size=100, input=context)
        with layer.mixed(size=100) as context1:
            context1 += context

        conv = layer.conv_projection(
            input=input,
            filter_size=1,
            num_channels=1,
            num_filters=128,
            stride=1,
            padding=0)
        conv0 = layer.mixed(input=conv, bias_attr=True)
        with layer.mixed(bias_attr=True) as conv1:
            conv1 += conv

        print layer.parse_network(mixed0)
        print layer.parse_network(mixed1)
        print layer.parse_network(emb0)
        print layer.parse_network(emb1)
        print layer.parse_network(scale0)
        print layer.parse_network(scale1)
        print layer.parse_network(dotmul0)
        print layer.parse_network(dotmul1)
        print layer.parse_network(conv0)
        print layer.parse_network(conv1)

    def test_operator(self):
        ipt0 = layer.data(name='data', type=data_type.dense_vector(784))
        ipt1 = layer.data(name='word', type=data_type.dense_vector(128))
        fc0 = layer.fc(input=ipt0, size=100, act=activation.Sigmoid())
        fc1 = layer.fc(input=ipt0, size=100, act=activation.Sigmoid())

        dotmul_op = layer.dotmul_operator(a=fc0, b=fc1)
        dotmul0 = layer.mixed(input=dotmul_op)
        with layer.mixed() as dotmul1:
            dotmul1 += dotmul_op

        conv = layer.conv_operator(
            img=ipt0,
            filter=ipt1,
            filter_size=1,
            num_channels=1,
            num_filters=128,
            stride=1,
            padding=0)
        conv0 = layer.mixed(input=conv)
        with layer.mixed() as conv1:
            conv1 += conv

        print layer.parse_network(dotmul0)
        print layer.parse_network(dotmul1)
        print layer.parse_network(conv0)
        print layer.parse_network(conv1)


if __name__ == '__main__':
    unittest.main()
