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

import unittest
import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Linear
from paddle.fluid.dygraph import declarative

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
    adam = fluid.optimizer.AdamOptimizer(
        learning_rate=0.1, parameter_list=layer.parameters())
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

    def train_and_save_model(self):
        layer = LinearNet(784, 1)
        example_inputs, layer, _ = train(layer)
        orig_input_types = [type(x) for x in example_inputs]
        fluid.dygraph.jit.save(
            layer=layer, model_path=self.model_path, input_spec=example_inputs)
        new_input_types = [type(x) for x in example_inputs]
        self.assertEqual(orig_input_types, new_input_types)
        return layer

    def test_save(self):
        # train and save model
        self.train_and_save_model()

    def test_load_infernece(self):
        # train and save model
        train_layer = self.train_and_save_model()
        # load model
        infer_layer = fluid.dygraph.jit.load(self.model_path)
        train_layer.eval()
        # inference & compare
        x = fluid.dygraph.to_variable(
            np.random.random((1, 784)).astype('float32'))
        self.assertTrue(
            np.array_equal(train_layer(x).numpy(), infer_layer(x).numpy()))

    def test_load_finetune(self):
        # train and save model
        train_layer = self.train_and_save_model()
        # load model
        load_train_layer = fluid.dygraph.jit.load(self.model_path)
        load_train_layer.train()
        # train & compare
        _, _, train_loss = train(train_layer)
        _, _, load_train_loss = train(load_train_layer)
        self.assertTrue(
            np.array_equal(train_loss.numpy(), load_train_loss.numpy()))

    def test_save_get_program_failed(self):
        layer = LinearNetNotDeclarative(784, 1)
        example_inputs, layer, _ = train(layer)
        with self.assertRaises(RuntimeError):
            fluid.dygraph.jit.save(
                layer=layer,
                model_path=self.model_path,
                input_spec=example_inputs)


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


if __name__ == '__main__':
    import paddle
    paddle.enable_static()
    print(paddle.in_dynamic_mode())
    unittest.main()
