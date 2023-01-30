# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os
<<<<<<< HEAD
import tempfile
import unittest

import numpy as np

=======
import numpy as np
import tempfile
import unittest

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
import paddle
import paddle.fluid as fluid
import paddle.nn as nn


class SimpleFCLayer(nn.Layer):
<<<<<<< HEAD
    def __init__(self, feature_size, batch_size, fc_size):
        super().__init__()
        self._linear = nn.Linear(feature_size, fc_size)
        self._offset = paddle.to_tensor(
            np.random.random((batch_size, fc_size)).astype('float32')
        )
=======

    def __init__(self, feature_size, batch_size, fc_size):
        super(SimpleFCLayer, self).__init__()
        self._linear = nn.Linear(feature_size, fc_size)
        self._offset = paddle.to_tensor(
            np.random.random((batch_size, fc_size)).astype('float32'))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def forward(self, x):
        fc = self._linear(x)
        return fc + self._offset


class LinearNetWithNone(nn.Layer):
<<<<<<< HEAD
    def __init__(self, feature_size, fc_size):
        super().__init__()
=======

    def __init__(self, feature_size, fc_size):
        super(LinearNetWithNone, self).__init__()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self._linear = nn.Linear(feature_size, fc_size)

    def forward(self, x):
        fc = self._linear(x)

        return [fc, [None, 2]]


class TestTracedLayerErrMsg(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.batch_size = 4
        self.feature_size = 3
        self.fc_size = 2
        self.layer = self._train_simple_net()
        self.type_str = 'class'
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_trace_err(self):
        if fluid.framework.in_dygraph_mode():
            return
        with fluid.dygraph.guard():
            in_x = fluid.dygraph.to_variable(
<<<<<<< HEAD
                np.random.random((self.batch_size, self.feature_size)).astype(
                    'float32'
                )
            )

            with self.assertRaises(AssertionError) as e:
                dygraph_out, traced_layer = fluid.dygraph.TracedLayer.trace(
                    None, [in_x]
                )
            self.assertEqual(
                "The type of 'layer' in paddle.jit.TracedLayer.trace must be fluid.dygraph.Layer, but received <{} 'NoneType'>.".format(
                    self.type_str
                ),
                str(e.exception),
            )
            with self.assertRaises(TypeError) as e:
                dygraph_out, traced_layer = fluid.dygraph.TracedLayer.trace(
                    self.layer, 3
                )
            self.assertEqual(
                "The type of 'each element of inputs' in paddle.jit.TracedLayer.trace must be fluid.Variable, but received <{} 'int'>.".format(
                    self.type_str
                ),
                str(e.exception),
            )
            with self.assertRaises(TypeError) as e:
                dygraph_out, traced_layer = fluid.dygraph.TracedLayer.trace(
                    self.layer, [True, 1]
                )
            self.assertEqual(
                "The type of 'each element of inputs' in paddle.jit.TracedLayer.trace must be fluid.Variable, but received <{} 'bool'>.".format(
                    self.type_str
                ),
                str(e.exception),
            )

            dygraph_out, traced_layer = fluid.dygraph.TracedLayer.trace(
                self.layer, [in_x]
            )
=======
                np.random.random(
                    (self.batch_size, self.feature_size)).astype('float32'))

            with self.assertRaises(AssertionError) as e:
                dygraph_out, traced_layer = fluid.dygraph.TracedLayer.trace(
                    None, [in_x])
            self.assertEqual(
                "The type of 'layer' in fluid.dygraph.jit.TracedLayer.trace must be fluid.dygraph.Layer, but received <{} 'NoneType'>."
                .format(self.type_str), str(e.exception))
            with self.assertRaises(TypeError) as e:
                dygraph_out, traced_layer = fluid.dygraph.TracedLayer.trace(
                    self.layer, 3)
            self.assertEqual(
                "The type of 'each element of inputs' in fluid.dygraph.jit.TracedLayer.trace must be fluid.Variable, but received <{} 'int'>."
                .format(self.type_str), str(e.exception))
            with self.assertRaises(TypeError) as e:
                dygraph_out, traced_layer = fluid.dygraph.TracedLayer.trace(
                    self.layer, [True, 1])
            self.assertEqual(
                "The type of 'each element of inputs' in fluid.dygraph.jit.TracedLayer.trace must be fluid.Variable, but received <{} 'bool'>."
                .format(self.type_str), str(e.exception))

            dygraph_out, traced_layer = fluid.dygraph.TracedLayer.trace(
                self.layer, [in_x])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_set_strategy_err(self):
        if fluid.framework.in_dygraph_mode():
            return
        with fluid.dygraph.guard():
            in_x = fluid.dygraph.to_variable(
<<<<<<< HEAD
                np.random.random((self.batch_size, self.feature_size)).astype(
                    'float32'
                )
            )
            dygraph_out, traced_layer = fluid.dygraph.TracedLayer.trace(
                self.layer, [in_x]
            )
=======
                np.random.random(
                    (self.batch_size, self.feature_size)).astype('float32'))
            dygraph_out, traced_layer = fluid.dygraph.TracedLayer.trace(
                self.layer, [in_x])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            with self.assertRaises(AssertionError) as e:
                traced_layer.set_strategy(1, fluid.ExecutionStrategy())
            self.assertEqual(
<<<<<<< HEAD
                "The type of 'build_strategy' in paddle.jit.TracedLayer.set_strategy must be fluid.BuildStrategy, but received <{} 'int'>.".format(
                    self.type_str
                ),
                str(e.exception),
            )
=======
                "The type of 'build_strategy' in fluid.dygraph.jit.TracedLayer.set_strategy must be fluid.BuildStrategy, but received <{} 'int'>."
                .format(self.type_str), str(e.exception))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            with self.assertRaises(AssertionError) as e:
                traced_layer.set_strategy(fluid.BuildStrategy(), False)
            self.assertEqual(
<<<<<<< HEAD
                "The type of 'exec_strategy' in paddle.jit.TracedLayer.set_strategy must be fluid.ExecutionStrategy, but received <{} 'bool'>.".format(
                    self.type_str
                ),
                str(e.exception),
            )

            traced_layer.set_strategy(build_strategy=fluid.BuildStrategy())
            traced_layer.set_strategy(exec_strategy=fluid.ExecutionStrategy())
            traced_layer.set_strategy(
                fluid.BuildStrategy(), fluid.ExecutionStrategy()
            )
=======
                "The type of 'exec_strategy' in fluid.dygraph.jit.TracedLayer.set_strategy must be fluid.ExecutionStrategy, but received <{} 'bool'>."
                .format(self.type_str), str(e.exception))

            traced_layer.set_strategy(build_strategy=fluid.BuildStrategy())
            traced_layer.set_strategy(exec_strategy=fluid.ExecutionStrategy())
            traced_layer.set_strategy(fluid.BuildStrategy(),
                                      fluid.ExecutionStrategy())
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_save_inference_model_err(self):
        if fluid.framework.in_dygraph_mode():
            return
        with fluid.dygraph.guard():
            in_x = fluid.dygraph.to_variable(
<<<<<<< HEAD
                np.random.random((self.batch_size, self.feature_size)).astype(
                    'float32'
                )
            )
            dygraph_out, traced_layer = fluid.dygraph.TracedLayer.trace(
                self.layer, [in_x]
            )
=======
                np.random.random(
                    (self.batch_size, self.feature_size)).astype('float32'))
            dygraph_out, traced_layer = fluid.dygraph.TracedLayer.trace(
                self.layer, [in_x])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            path = os.path.join(self.temp_dir.name, './traced_layer_err_msg')
            with self.assertRaises(TypeError) as e:
                traced_layer.save_inference_model([0])
            self.assertEqual(
<<<<<<< HEAD
                "The type of 'path' in paddle.jit.TracedLayer.save_inference_model must be <{} 'str'>, but received <{} 'list'>. ".format(
                    self.type_str, self.type_str
                ),
                str(e.exception),
            )
            with self.assertRaises(TypeError) as e:
                traced_layer.save_inference_model(path, [0], [None])
            self.assertEqual(
                "The type of 'each element of fetch' in paddle.jit.TracedLayer.save_inference_model must be <{} 'int'>, but received <{} 'NoneType'>. ".format(
                    self.type_str, self.type_str
                ),
                str(e.exception),
            )
            with self.assertRaises(TypeError) as e:
                traced_layer.save_inference_model(path, [0], False)
            self.assertEqual(
                "The type of 'fetch' in paddle.jit.TracedLayer.save_inference_model must be (<{} 'NoneType'>, <{} 'list'>), but received <{} 'bool'>. ".format(
                    self.type_str, self.type_str, self.type_str
                ),
                str(e.exception),
            )
            with self.assertRaises(TypeError) as e:
                traced_layer.save_inference_model(path, [None], [0])
            self.assertEqual(
                "The type of 'each element of feed' in paddle.jit.TracedLayer.save_inference_model must be <{} 'int'>, but received <{} 'NoneType'>. ".format(
                    self.type_str, self.type_str
                ),
                str(e.exception),
            )
            with self.assertRaises(TypeError) as e:
                traced_layer.save_inference_model(path, True, [0])
            self.assertEqual(
                "The type of 'feed' in paddle.jit.TracedLayer.save_inference_model must be (<{} 'NoneType'>, <{} 'list'>), but received <{} 'bool'>. ".format(
                    self.type_str, self.type_str, self.type_str
                ),
                str(e.exception),
            )
=======
                "The type of 'path' in fluid.dygraph.jit.TracedLayer.save_inference_model must be <{} 'str'>, but received <{} 'list'>. "
                .format(self.type_str, self.type_str), str(e.exception))
            with self.assertRaises(TypeError) as e:
                traced_layer.save_inference_model(path, [0], [None])
            self.assertEqual(
                "The type of 'each element of fetch' in fluid.dygraph.jit.TracedLayer.save_inference_model must be <{} 'int'>, but received <{} 'NoneType'>. "
                .format(self.type_str, self.type_str), str(e.exception))
            with self.assertRaises(TypeError) as e:
                traced_layer.save_inference_model(path, [0], False)
            self.assertEqual(
                "The type of 'fetch' in fluid.dygraph.jit.TracedLayer.save_inference_model must be (<{} 'NoneType'>, <{} 'list'>), but received <{} 'bool'>. "
                .format(self.type_str, self.type_str, self.type_str),
                str(e.exception))
            with self.assertRaises(TypeError) as e:
                traced_layer.save_inference_model(path, [None], [0])
            self.assertEqual(
                "The type of 'each element of feed' in fluid.dygraph.jit.TracedLayer.save_inference_model must be <{} 'int'>, but received <{} 'NoneType'>. "
                .format(self.type_str, self.type_str), str(e.exception))
            with self.assertRaises(TypeError) as e:
                traced_layer.save_inference_model(path, True, [0])
            self.assertEqual(
                "The type of 'feed' in fluid.dygraph.jit.TracedLayer.save_inference_model must be (<{} 'NoneType'>, <{} 'list'>), but received <{} 'bool'>. "
                .format(self.type_str, self.type_str, self.type_str),
                str(e.exception))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            with self.assertRaises(ValueError) as e:
                traced_layer.save_inference_model("")
            self.assertEqual(
                "The input path MUST be format of dirname/file_prefix [dirname\\file_prefix in Windows system], "
<<<<<<< HEAD
                "but received file_prefix is empty string.",
                str(e.exception),
            )
=======
                "but received file_prefix is empty string.", str(e.exception))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            traced_layer.save_inference_model(path)

    def _train_simple_net(self):
        layer = None
        with fluid.dygraph.guard():
<<<<<<< HEAD
            layer = SimpleFCLayer(
                self.feature_size, self.batch_size, self.fc_size
            )
            optimizer = fluid.optimizer.SGD(
                learning_rate=1e-3, parameter_list=layer.parameters()
            )
=======
            layer = SimpleFCLayer(self.feature_size, self.batch_size,
                                  self.fc_size)
            optimizer = fluid.optimizer.SGD(learning_rate=1e-3,
                                            parameter_list=layer.parameters())
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            for i in range(5):
                in_x = fluid.dygraph.to_variable(
                    np.random.random(
<<<<<<< HEAD
                        (self.batch_size, self.feature_size)
                    ).astype('float32')
                )
                dygraph_out = layer(in_x)
                loss = paddle.mean(dygraph_out)
=======
                        (self.batch_size, self.feature_size)).astype('float32'))
                dygraph_out = layer(in_x)
                loss = fluid.layers.reduce_mean(dygraph_out)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                loss.backward()
                optimizer.minimize(loss)
        return layer


class TestOutVarWithNoneErrMsg(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_linear_net_with_none(self):
        if fluid.framework.in_dygraph_mode():
            return
        model = LinearNetWithNone(100, 16)
        in_x = paddle.to_tensor(np.random.random((4, 100)).astype('float32'))
        with self.assertRaises(TypeError):
            dygraph_out, traced_layer = fluid.dygraph.TracedLayer.trace(
<<<<<<< HEAD
                model, [in_x]
            )
=======
                model, [in_x])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


class TestTracedLayerSaveInferenceModel(unittest.TestCase):
    """test save_inference_model will automaticlly create non-exist dir"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.save_path = os.path.join(self.temp_dir.name, "./nonexist_dir/fc")
        import shutil
<<<<<<< HEAD

=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        if os.path.exists(os.path.dirname(self.save_path)):
            shutil.rmtree(os.path.dirname(self.save_path))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_mkdir_when_input_path_non_exist(self):
        if fluid.framework.in_dygraph_mode():
            return
        fc_layer = SimpleFCLayer(3, 4, 2)
        input_var = paddle.to_tensor(np.random.random([4, 3]).astype('float32'))
        with fluid.dygraph.guard():
            dygraph_out, traced_layer = fluid.dygraph.TracedLayer.trace(
<<<<<<< HEAD
                fc_layer, inputs=[input_var]
            )
=======
                fc_layer, inputs=[input_var])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            self.assertFalse(os.path.exists(os.path.dirname(self.save_path)))
            traced_layer.save_inference_model(self.save_path)
            self.assertTrue(os.path.exists(os.path.dirname(self.save_path)))


if __name__ == '__main__':
    unittest.main()
