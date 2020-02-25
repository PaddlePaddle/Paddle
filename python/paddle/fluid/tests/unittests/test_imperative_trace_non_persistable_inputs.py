# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
import numpy as np
import six
import os


class SimpleFCLayer(fluid.dygraph.Layer):
    def __init__(self, feature_size, batch_size, fc_size):
        super(SimpleFCLayer, self).__init__()
        self._linear = fluid.dygraph.Linear(feature_size, fc_size)
        self._offset = fluid.dygraph.to_variable(
            np.random.random((batch_size, fc_size)).astype('float32'))

    def forward(self, x):
        fc = self._linear(x)
        return fc + self._offset


class TestTracedLayerRecordNonPersistableInput(unittest.TestCase):
    def test_main(self):
        traced_layer = None
        with fluid.dygraph.guard():
            feature_size = 3
            batch_size = 4
            fc_size = 2
            layer = SimpleFCLayer(feature_size, batch_size, fc_size)
            optimizer = fluid.optimizer.SGD(learning_rate=1e-3,
                                            parameter_list=layer.parameters())

            expected_persistable_vars = set([
                layer._linear.weight.name, layer._linear.bias.name,
                layer._offset.name
            ])

            for _ in six.moves.range(10):
                in_x = fluid.dygraph.to_variable(
                    np.random.random((batch_size, feature_size)).astype(
                        'float32'))
                if traced_layer is None:
                    dygraph_out, traced_layer = fluid.dygraph.TracedLayer.trace(
                        layer, [in_x])
                else:
                    dygraph_out = layer(in_x)
                dygraph_out_numpy = dygraph_out.numpy()
                static_out = traced_layer([in_x])[0]
                self.assertTrue(np.array_equal(dygraph_out_numpy, static_out))

                loss = fluid.layers.reduce_mean(dygraph_out)
                loss.backward()

                optimizer.minimize(loss)

            del layer

        program = traced_layer.program
        actual_persistable_vars = set()
        for var in program.list_vars():
            if var.persistable:
                actual_persistable_vars.add(var.name)

        self.assertEqual(actual_persistable_vars, expected_persistable_vars)

        dirname = './traced_layer_test_non_persistable_vars'
        traced_layer.save_inference_model(dirname=dirname)
        filenames = set([f for f in os.listdir(dirname) if f != '__model__'])
        self.assertEqual(filenames, expected_persistable_vars)


if __name__ == '__main__':
    unittest.main()
