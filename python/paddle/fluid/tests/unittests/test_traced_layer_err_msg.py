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

import numpy as np
import paddle.fluid as fluid
import six
import unittest


class SimpleFCLayer(fluid.dygraph.Layer):
    def __init__(self, feature_size, batch_size, fc_size):
        super(SimpleFCLayer, self).__init__()
        self._linear = fluid.dygraph.Linear(feature_size, fc_size)
        self._offset = fluid.dygraph.to_variable(
            np.random.random((batch_size, fc_size)).astype('float32'))

    def forward(self, x):
        fc = self._linear(x)
        return fc + self._offset


class TestTracedLayerErrMsg(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.feature_size = 3
        self.fc_size = 2
        self.layer = self._train_simple_net()
        if six.PY2:
            self.type_str = 'type'
        else:
            self.type_str = 'class'

    def test_trace_err(self):
        with fluid.dygraph.guard():
            in_x = fluid.dygraph.to_variable(
                np.random.random((self.batch_size, self.feature_size)).astype(
                    'float32'))

            with self.assertRaises(AssertionError) as e:
                dygraph_out, traced_layer = fluid.dygraph.TracedLayer.trace(
                    None, [in_x])
            self.assertEqual(
                "The type of 'layer' in fluid.dygraph.jit.TracedLayer.trace must be fluid.dygraph.Layer, but received <{} 'NoneType'>.".
                format(self.type_str), str(e.exception))
            with self.assertRaises(TypeError) as e:
                dygraph_out, traced_layer = fluid.dygraph.TracedLayer.trace(
                    self.layer, 3)
            self.assertEqual(
                "The type of 'each element of inputs' in fluid.dygraph.jit.TracedLayer.trace must be fluid.Variable, but received <{} 'int'>.".
                format(self.type_str, self.type_str), str(e.exception))
            with self.assertRaises(TypeError) as e:
                dygraph_out, traced_layer = fluid.dygraph.TracedLayer.trace(
                    self.layer, [True, 1])
            self.assertEqual(
                "The type of 'each element of inputs' in fluid.dygraph.jit.TracedLayer.trace must be fluid.Variable, but received <{} 'bool'>.".
                format(self.type_str), str(e.exception))

            dygraph_out, traced_layer = fluid.dygraph.TracedLayer.trace(
                self.layer, [in_x])

    def test_set_strategy_err(self):
        with fluid.dygraph.guard():
            in_x = fluid.dygraph.to_variable(
                np.random.random((self.batch_size, self.feature_size)).astype(
                    'float32'))
            dygraph_out, traced_layer = fluid.dygraph.TracedLayer.trace(
                self.layer, [in_x])

            with self.assertRaises(AssertionError) as e:
                traced_layer.set_strategy(1, fluid.ExecutionStrategy())
            self.assertEqual(
                "The type of 'build_strategy' in fluid.dygraph.jit.TracedLayer.set_strategy must be fluid.BuildStrategy, but received <{} 'int'>.".
                format(self.type_str), str(e.exception))

            with self.assertRaises(AssertionError) as e:
                traced_layer.set_strategy(fluid.BuildStrategy(), False)
            self.assertEqual(
                "The type of 'exec_strategy' in fluid.dygraph.jit.TracedLayer.set_strategy must be fluid.ExecutionStrategy, but received <{} 'bool'>.".
                format(self.type_str), str(e.exception))

            traced_layer.set_strategy(build_strategy=fluid.BuildStrategy())
            traced_layer.set_strategy(exec_strategy=fluid.ExecutionStrategy())
            traced_layer.set_strategy(fluid.BuildStrategy(),
                                      fluid.ExecutionStrategy())

    def test_save_inference_model_err(self):
        with fluid.dygraph.guard():
            in_x = fluid.dygraph.to_variable(
                np.random.random((self.batch_size, self.feature_size)).astype(
                    'float32'))
            dygraph_out, traced_layer = fluid.dygraph.TracedLayer.trace(
                self.layer, [in_x])

            dirname = './traced_layer_err_msg'
            with self.assertRaises(TypeError) as e:
                traced_layer.save_inference_model([0])
            self.assertEqual(
                "The type of 'dirname' in fluid.dygraph.jit.TracedLayer.save_inference_model must be <{} 'str'>, but received <{} 'list'>. ".
                format(self.type_str, self.type_str), str(e.exception))
            with self.assertRaises(TypeError) as e:
                traced_layer.save_inference_model(dirname, [0], [None])
            self.assertEqual(
                "The type of 'each element of fetch' in fluid.dygraph.jit.TracedLayer.save_inference_model must be <{} 'int'>, but received <{} 'NoneType'>. ".
                format(self.type_str, self.type_str), str(e.exception))
            with self.assertRaises(TypeError) as e:
                traced_layer.save_inference_model(dirname, [0], False)
            self.assertEqual(
                "The type of 'fetch' in fluid.dygraph.jit.TracedLayer.save_inference_model must be (<{} 'NoneType'>, <{} 'list'>), but received <{} 'bool'>. ".
                format(self.type_str, self.type_str, self.type_str),
                str(e.exception))
            with self.assertRaises(TypeError) as e:
                traced_layer.save_inference_model(dirname, [None], [0])
            self.assertEqual(
                "The type of 'each element of feed' in fluid.dygraph.jit.TracedLayer.save_inference_model must be <{} 'int'>, but received <{} 'NoneType'>. ".
                format(self.type_str, self.type_str), str(e.exception))
            with self.assertRaises(TypeError) as e:
                traced_layer.save_inference_model(dirname, True, [0])
            self.assertEqual(
                "The type of 'feed' in fluid.dygraph.jit.TracedLayer.save_inference_model must be (<{} 'NoneType'>, <{} 'list'>), but received <{} 'bool'>. ".
                format(self.type_str, self.type_str, self.type_str),
                str(e.exception))

            traced_layer.save_inference_model(dirname)

    def _train_simple_net(self):
        layer = None
        with fluid.dygraph.guard():
            layer = SimpleFCLayer(self.feature_size, self.batch_size,
                                  self.fc_size)
            optimizer = fluid.optimizer.SGD(learning_rate=1e-3,
                                            parameter_list=layer.parameters())

            for i in range(5):
                in_x = fluid.dygraph.to_variable(
                    np.random.random((self.batch_size, self.feature_size))
                    .astype('float32'))
                dygraph_out = layer(in_x)
                loss = fluid.layers.reduce_mean(dygraph_out)
                loss.backward()
                optimizer.minimize(loss)
        return layer


if __name__ == '__main__':
    unittest.main()
