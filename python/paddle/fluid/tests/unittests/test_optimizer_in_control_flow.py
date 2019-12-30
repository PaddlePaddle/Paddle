# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import unittest

import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.optimizer as optimizer
from paddle.fluid.framework import Program, program_guard

BATCH_SIZE = 1
INPUT_SIZE = 784

CLASS_NUM = 10
FC_SIZE = 40
EPOCH_NUM = 5
SWITCH_ID = 3
SEED = 123


def random_input(seed,
                 image_shape=[BATCH_SIZE, INPUT_SIZE],
                 label_shape=[BATCH_SIZE, 1]):
    np.random.seed(seed)
    image_np = np.random.random(size=image_shape).astype('float32')
    np.random.seed(seed)
    label_np = np.random.random_integers(
        low=0, high=CLASS_NUM - 1, size=label_shape).astype('int64')
    return image_np, label_np


def random_param(size, seed=100):
    np.random.seed(seed)
    np_param = np.random.random(size=size).astype('float32')
    return np_param


def static():
    def simple_fc_net(image):
        hidden = layers.fc(
            image,
            size=FC_SIZE,
            act='relu',
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(value=0.99)),
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(value=0.5)),
            name="hidden")

        prediction = layers.fc(
            hidden,
            size=CLASS_NUM,
            act='softmax',
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(value=1.2)),
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(value=0.8)),
            name="prediction")
        return hidden, prediction

    main_program = Program()
    main_program.random_seed = SEED
    startup_program = Program()
    startup_program.random_seed = SEED
    with program_guard(main_program, startup_program):
        image = fluid.data(
            name='image', shape=[BATCH_SIZE, INPUT_SIZE], dtype='float32')
        label = fluid.data(name='label', shape=[BATCH_SIZE, 1], dtype='int64')
        id = fluid.data(name='id', shape=[1], dtype='int32')

        two = layers.fill_constant(shape=[1], dtype='int32', value=2)
        hidden, prediction = simple_fc_net(image)

        adam = optimizer.Adam(learning_rate=0.001)
        adagrad = optimizer.Adagrad(learning_rate=0.001)

        def fn_1():
            cross_entropy_loss = layers.cross_entropy(
                input=prediction, label=label)
            mean_cross_entropy_loss = layers.mean(
                cross_entropy_loss, name="mean_cross_entropy_loss")
            adam.minimize(mean_cross_entropy_loss)
            return mean_cross_entropy_loss

        def fn_2():
            softmax_loss = layers.softmax_with_cross_entropy(
                logits=prediction, label=label)
            mean_softmax_loss = layers.mean(
                softmax_loss, name='mean_softmax_loss')
            adagrad.minimize(mean_softmax_loss)
            return mean_softmax_loss

        cond = layers.elementwise_mod(id, two) == 0
        avg_loss = layers.case([(cond, fn_2)], fn_1)

        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(fluid.default_startup_program())

        for epoch in range(EPOCH_NUM):
            feed_image, feed_label = random_input(epoch)
            out = exe.run(main_program,
                          feed={
                              'image': feed_image,
                              'label': feed_label,
                              'id': np.array([epoch]).astype('int32')
                          },
                          fetch_list=[
                              hidden,
                              prediction,
                              avg_loss,
                          ])
            out_hidden, out_prediction, loss = out
            print(epoch)
            print(out_prediction)
            # print(loss)
    return out_hidden, out_prediction, loss


class DygraphLayer(fluid.dygraph.Layer):
    def __init__(self):
        super(DygraphLayer, self).__init__()
        self.fc0 = fluid.dygraph.nn.Linear(
            INPUT_SIZE,
            FC_SIZE,
            act='relu',
            param_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(
                value=0.99)),
            bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(
                value=0.5)), )

        self.pre = fluid.dygraph.nn.Linear(
            FC_SIZE,
            CLASS_NUM,
            act='softmax',
            param_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(
                value=1.2)),
            bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(
                value=0.8)))

    def forward(self, inputs):
        h_0 = self.fc0(inputs)
        prediction = self.pre(h_0)
        return h_0, prediction


def dynamic():
    with fluid.dygraph.guard():
        fluid.default_startup_program().random_seed = SEED
        fluid.default_main_program().random_seed = SEED
        my_layer = DygraphLayer()
        adam = fluid.optimizer.Adam(
            learning_rate=0.001, parameter_list=my_layer.parameters())
        adagrad = fluid.optimizer.Adagrad(
            learning_rate=0.002, parameter_list=my_layer.parameters())
        print("--- liyamei: num of param", len(my_layer.parameters()))
        for epoch in range(EPOCH_NUM):
            image_data, label = random_input(epoch)
            var_input = fluid.dygraph.to_variable(image_data)
            var_lable = fluid.dygraph.to_variable(label)
            hidden, prediction = my_layer(var_input)

            if epoch % 2 != 0:
                cross_entropy_loss = layers.cross_entropy(prediction, var_lable)
                loss = layers.mean(cross_entropy_loss)
                loss.backward()
                adam.minimize(loss)
            else:
                softmax_loss = layers.softmax_with_cross_entropy(prediction,
                                                                 var_lable)
                loss = layers.mean(softmax_loss)
                loss.backward()
                adagrad.minimize(loss)

            my_layer.clear_gradients()
            print(epoch)
            print(prediction.numpy())
        return hidden.numpy(), prediction.numpy(), loss.numpy()


class TestMultiTask(unittest.TestCase):
    def test_1(self):
        print("-" * 20, " static ", "-" * 20)
        hidden_1, pre_1, loss_1 = static()

        print("-" * 20, " dynamic ", "-" * 20)
        hidden_2, pre_2, loss_2 = dynamic()

        # self.assertTrue(
        #     np.allclose(hidden_1, hidden_2),
        #     msg='static hidden is {} \n dynamic hidden is {}'.format(hidden_1,
        #                                                              hidden_2))
        self.assertTrue(
            np.allclose(pre_1, pre_2),
            msg='static prediction is {} \n dynamic prediction is {}'.format(
                pre_1, pre_2))
        # self.assertTrue(
        #     np.allclose(loss_1, loss_2),
        #     msg='static loss is {} \n dynamic loss is {}'.format(loss_1,
        #                                                          loss_2))


if __name__ == '__main__':
    unittest.main()
