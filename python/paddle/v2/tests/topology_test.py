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
import unittest
import paddle.v2.layer as layer
import paddle.v2.topology as topology
import paddle.v2.data_type as data_type
import paddle.trainer_config_helpers as conf_helps


class TestTopology(unittest.TestCase):
    def test_parse(self):
        pixel = layer.data(name='pixel', type=data_type.dense_vector(784))
        label = layer.data(name='label', type=data_type.integer_value(10))
        hidden = layer.fc(input=pixel,
                          size=100,
                          act=conf_helps.SigmoidActivation())
        inference = layer.fc(input=hidden,
                             size=10,
                             act=conf_helps.SoftmaxActivation())
        maxid = layer.max_id(input=inference)
        cost1 = layer.classification_cost(input=inference, label=label)
        cost2 = layer.cross_entropy_cost(input=inference, label=label)

        print topology.Topology(cost2).proto()
        print topology.Topology([cost1]).proto()
        print topology.Topology([cost1, cost2]).proto()
        print topology.Topology(cost2).proto()
        print topology.Topology([inference, maxid]).proto()

    def test_data_type(self):
        pixel = layer.data(name='pixel', type=data_type.dense_vector(784))
        label = layer.data(name='label', type=data_type.integer_value(10))
        hidden = layer.fc(input=pixel,
                          size=100,
                          act=conf_helps.SigmoidActivation())
        inference = layer.fc(input=hidden,
                             size=10,
                             act=conf_helps.SoftmaxActivation())
        cost = layer.classification_cost(input=inference, label=label)
        topo = topology.Topology(cost)
        type = topo.data_type()
        self.assertEqual(len(type), 2)
        self.assertEqual(type[0][0], "pixel")
        self.assertEqual(type[0][1].type, data_type.DataType.Dense)
        self.assertEqual(type[0][1].dim, 784)
        self.assertEqual(type[1][0], "label")
        self.assertEqual(type[1][1].type, data_type.DataType.Index)
        self.assertEqual(type[1][1].dim, 10)

    def test_get_layer(self):
        pixel = layer.data(name='pixel', type=data_type.dense_vector(784))
        label = layer.data(name='label', type=data_type.integer_value(10))
        hidden = layer.fc(input=pixel,
                          size=100,
                          act=conf_helps.SigmoidActivation())
        inference = layer.fc(input=hidden,
                             size=10,
                             act=conf_helps.SoftmaxActivation())
        cost = layer.classification_cost(input=inference, label=label)
        topo = topology.Topology(cost)
        pixel_layer = topo.get_layer("pixel")
        label_layer = topo.get_layer("label")
        self.assertEqual(pixel_layer, pixel)
        self.assertEqual(label_layer, label)


if __name__ == '__main__':
    unittest.main()
