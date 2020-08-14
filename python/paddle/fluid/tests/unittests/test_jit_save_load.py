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

from __future__ import print_function

import os
import unittest
import numpy as np

import paddle.fluid as fluid
from paddle.fluid.dygraph import Linear
from paddle.fluid.dygraph import declarative, ProgramTranslator, TensorSpec
from paddle.fluid.dygraph.io import EXTRA_VAR_INFO_FILENAME

BATCH_SIZE = 32
BATCH_NUM = 20
SEED = 10


def random_batch_reader():
    def _get_random_images_and_labels(image_shape, label_shape):
        np.random.seed(SEED)
        image = np.random.random(size=image_shape).astype('float32')
        label = np.random.random(size=label_shape).astype('int64')
        return image, label

    def __reader__():
        for _ in range(BATCH_NUM):
            batch_image, batch_label = _get_random_images_and_labels(
                [BATCH_SIZE, 784], [BATCH_SIZE, 1])
            yield batch_image, batch_label

    return __reader__


class LinearNet(fluid.dygraph.Layer):
    def __init__(self, in_size, out_size):
        super(LinearNet, self).__init__()
        self._linear = Linear(in_size, out_size)

    @declarative
    def forward(self, x):
        return self._linear(x)


class LinearNetNotDeclarative(fluid.dygraph.Layer):
    def __init__(self, in_size, out_size):
        super(LinearNetNotDeclarative, self).__init__()
        self._linear = Linear(in_size, out_size)

    def forward(self, x):
        return self._linear(x)


class LinearNetReturnLoss(fluid.dygraph.Layer):
    def __init__(self, in_size, out_size):
        super(LinearNetReturnLoss, self).__init__()
        self._linear = Linear(in_size, out_size)

    @declarative
    def forward(self, x):
        y = self._linear(x)
        z = self._linear(y)
        loss = fluid.layers.mean(z)
        return z, loss


def train(layer):
    # create optimizer
    adam = fluid.optimizer.SGDOptimizer(
        learning_rate=0.01, parameter_list=layer.parameters())
    # create data loader
    train_loader = fluid.io.DataLoader.from_generator(capacity=5)
    train_loader.set_batch_generator(random_batch_reader())
    # train
    for data in train_loader():
        img, label = data
        label.stop_gradient = True

        cost = layer(img)

        loss = fluid.layers.cross_entropy(cost, label)
        avg_loss = fluid.layers.mean(loss)

        avg_loss.backward()
        adam.minimize(avg_loss)
        layer.clear_gradients()
    return [img], layer, avg_loss


def infer(layer):
    x = fluid.dygraph.to_variable(np.random.random((1, 784)).astype('float32'))
    return layer(x)


class TestJitSaveLoad(unittest.TestCase):
    def setUp(self):
        self.model_path = "model.test_jit_save_load"
        # enable dygraph mode
        fluid.enable_dygraph()
        # config seed
        fluid.default_main_program().random_seed = SEED

    def train_and_save_model(self, model_path=None, configs=None):
        layer = LinearNet(784, 1)
        example_inputs, layer, _ = train(layer)
        final_model_path = model_path if model_path else self.model_path
        orig_input_types = [type(x) for x in example_inputs]
        fluid.dygraph.jit.save(
            layer=layer,
            model_path=final_model_path,
            input_spec=example_inputs,
            configs=configs)
        new_input_types = [type(x) for x in example_inputs]
        self.assertEqual(orig_input_types, new_input_types)
        return layer

    def test_save_load(self):
        # train and save model
        train_layer = self.train_and_save_model()
        # load model
        program_translator = ProgramTranslator()
        program_translator.enable(False)
        loaded_layer = fluid.dygraph.jit.load(self.model_path)
        self.load_and_inference(train_layer, loaded_layer)
        self.load_dygraph_state_dict(train_layer)
        self.load_and_finetune(train_layer, loaded_layer)
        program_translator.enable(True)

    def load_and_inference(self, train_layer, infer_layer):
        train_layer.eval()
        infer_layer.eval()
        # inference & compare
        x = fluid.dygraph.to_variable(
            np.random.random((1, 784)).astype('float32'))
        self.assertTrue(
            np.array_equal(train_layer(x).numpy(), infer_layer(x).numpy()))

    def load_and_finetune(self, train_layer, load_train_layer):
        train_layer.train()
        load_train_layer.train()
        # train & compare
        _, _, train_loss = train(train_layer)
        _, _, load_train_loss = train(load_train_layer)
        self.assertTrue(
            np.array_equal(train_loss.numpy(), load_train_loss.numpy()))

    def load_dygraph_state_dict(self, train_layer):
        train_layer.eval()
        # construct new model
        new_layer = LinearNet(784, 1)
        model_dict, _ = fluid.dygraph.load_dygraph(self.model_path)
        new_layer.set_dict(model_dict)
        new_layer.eval()
        # inference & compare
        x = fluid.dygraph.to_variable(
            np.random.random((1, 784)).astype('float32'))
        self.assertTrue(
            np.array_equal(train_layer(x).numpy(), new_layer(x).numpy()))

    def test_save_get_program_failed(self):
        layer = LinearNetNotDeclarative(784, 1)
        example_inputs, layer, _ = train(layer)
        with self.assertRaises(RuntimeError):
            fluid.dygraph.jit.save(
                layer=layer,
                model_path=self.model_path,
                input_spec=example_inputs)

    def test_load_dygraph_no_path(self):
        model_path = "model.test_jit_save_load.no_path"
        new_layer = LinearNet(784, 1)
        with self.assertRaises(ValueError):
            model_dict, _ = fluid.dygraph.load_dygraph(model_path)

    def test_load_dygraph_no_var_info(self):
        model_path = "model.test_jit_save_load.no_var_info"
        self.train_and_save_model(model_path=model_path)
        # remove `__variables.info__`
        var_info_path = os.path.join(model_path, EXTRA_VAR_INFO_FILENAME)
        os.remove(var_info_path)
        new_layer = LinearNet(784, 1)
        with self.assertRaises(RuntimeError):
            model_dict, _ = fluid.dygraph.load_dygraph(model_path)

    def test_load_dygraph_not_var_file(self):
        model_path = "model.test_jit_save_load.no_var_file"
        configs = fluid.dygraph.jit.SaveLoadConfig()
        configs.params_filename = "__params__"
        self.train_and_save_model(model_path=model_path, configs=configs)
        new_layer = LinearNet(784, 1)
        with self.assertRaises(RuntimeError):
            model_dict, _ = fluid.dygraph.load_dygraph(model_path)


class TestJitSaveLoadConfig(unittest.TestCase):
    def setUp(self):
        # enable dygraph mode
        fluid.enable_dygraph()
        # config seed
        fluid.default_main_program().random_seed = SEED

    def basic_save_load(self, layer, model_path, configs):
        # 1. train & save
        example_inputs, train_layer, _ = train(layer)
        fluid.dygraph.jit.save(
            layer=train_layer,
            model_path=model_path,
            input_spec=example_inputs,
            configs=configs)
        # 2. load 
        infer_layer = fluid.dygraph.jit.load(model_path, configs=configs)
        train_layer.eval()
        # 3. inference & compare
        x = fluid.dygraph.to_variable(
            np.random.random((1, 784)).astype('float32'))
        self.assertTrue(
            np.array_equal(train_layer(x).numpy(), infer_layer(x).numpy()))

    def test_model_filename(self):
        layer = LinearNet(784, 1)
        model_path = "model.save_load_config.output_spec"
        configs = fluid.dygraph.jit.SaveLoadConfig()
        configs.model_filename = "__simplenet__"
        self.basic_save_load(layer, model_path, configs)

    def test_params_filename(self):
        layer = LinearNet(784, 1)
        model_path = "model.save_load_config.params_filename"
        configs = fluid.dygraph.jit.SaveLoadConfig()
        configs.params_filename = "__params__"
        self.basic_save_load(layer, model_path, configs)

    def test_separate_params(self):
        layer = LinearNet(784, 1)
        model_path = "model.save_load_config.separate_params"
        configs = fluid.dygraph.jit.SaveLoadConfig()
        configs.separate_params = True
        self.basic_save_load(layer, model_path, configs)

    def test_output_spec(self):
        train_layer = LinearNetReturnLoss(8, 8)
        adam = fluid.optimizer.AdamOptimizer(
            learning_rate=0.1, parameter_list=train_layer.parameters())
        x = fluid.dygraph.to_variable(
            np.random.random((4, 8)).astype('float32'))
        for i in range(10):
            out, loss = train_layer(x)
            loss.backward()
            adam.minimize(loss)
            train_layer.clear_gradients()

        model_path = "model.save_load_config.output_spec"
        configs = fluid.dygraph.jit.SaveLoadConfig()
        configs.output_spec = [out]
        fluid.dygraph.jit.save(
            layer=train_layer,
            model_path=model_path,
            input_spec=[x],
            configs=configs)

        train_layer.eval()
        infer_layer = fluid.dygraph.jit.load(model_path, configs=configs)
        x = fluid.dygraph.to_variable(
            np.random.random((4, 8)).astype('float32'))
        self.assertTrue(
            np.array_equal(train_layer(x)[0].numpy(), infer_layer(x).numpy()))


class LinearNetMultiInput(fluid.dygraph.Layer):
    def __init__(self, in_size, out_size):
        super(LinearNetMultiInput, self).__init__()
        self._linear1 = Linear(in_size, out_size)
        # self._linear2 = Linear(in_size, out_size)

    @declarative(input_spec=[
        TensorSpec(
            [None, 8], dtype='float32'), TensorSpec(
                [None, 8], dtype='float32')
    ])
    def forward(self, x, y):
        x_out = self._linear1(x)
        y_out = self._linear1(y)
        loss = fluid.layers.mean(x_out + y_out)
        return x_out, y_out, loss


class TestSaveLoadWithTensorSpec(unittest.TestCase):
    def setUp(self):
        # enable dygraph mode
        fluid.enable_dygraph()

    def test_with_input_spec(self):
        net = LinearNetReturnLoss(8, 8)
        # set x.shape = [None, 8]
        net.forward = declarative(
            net.forward, input_spec=[TensorSpec(
                [None, 8], name='x')])

        model_path = "model.input_spec.output_spec"
        configs = fluid.dygraph.jit.SaveLoadConfig()
        # check inputs and outputs
        self.assertTrue(len(net.forward.inputs) == 1)
        input_x = net.forward.inputs[0]
        self.assertTrue(input_x.shape == (-1, 8))
        self.assertTrue(input_x.name == 'x')

        # 1. prune loss
        configs.output_spec = net.forward.outputs[:1]
        fluid.dygraph.jit.save(net, model_path, configs=configs)

        # 2. load to infer
        infer_layer = fluid.dygraph.jit.load(model_path, configs=configs)
        x = fluid.dygraph.to_variable(
            np.random.random((4, 8)).astype('float32'))
        pred = infer_layer(x)

    def test_multi_in_out(self):
        net = LinearNetMultiInput(8, 8)

        model_path = "model.multi_inout.output_spec1"
        configs = fluid.dygraph.jit.SaveLoadConfig()
        # 1. check inputs and outputs
        self.assertTrue(len(net.forward.inputs) == 2)
        input_x = net.forward.inputs[0]
        input_y = net.forward.inputs[1]
        self.assertTrue(input_x.shape == (-1, 8))
        self.assertTrue(input_y.shape == (-1, 8))

        # 2. prune loss
        configs.output_spec = net.forward.outputs[:2]
        fluid.dygraph.jit.save(net, model_path, configs=configs)

        # 3. load to infer
        infer_layer = fluid.dygraph.jit.load(model_path, configs=configs)
        x = fluid.dygraph.to_variable(
            np.random.random((4, 8)).astype('float32'))
        y = fluid.dygraph.to_variable(
            np.random.random((4, 8)).astype('float32'))
        # 4. predict
        pred_x, pred_y = infer_layer(x, y)

        # 1. prune y and loss
        model_path = "model.multi_inout.output_spec2"
        configs.output_spec = net.forward.outputs[:1]
        fluid.dygraph.jit.save(net, model_path, [input_x], configs)
        # 2. load again
        infer_layer2 = fluid.dygraph.jit.load(model_path, configs=configs)
        # 3. predict
        pred_xx = infer_layer2(x)

        # 4. assert pred_x == pred_xx
        self.assertTrue(np.allclose(pred_x.numpy(), pred_xx.numpy()))


if __name__ == '__main__':
    unittest.main()
