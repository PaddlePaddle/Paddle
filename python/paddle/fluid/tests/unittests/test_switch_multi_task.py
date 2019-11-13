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

import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.optimizer as optimizer
import numpy as np

BATCH_SIZE = 1
CLASS_NUM = 10
FC_SIZE = 50
EPOCH_NUM = 10


def random_input(image_shape=[BATCH_SIZE, 784], label_shape=[BATCH_SIZE, 1]):
    image_np = np.random.random(size=image_shape).astype('float32')
    label_np = np.random.random_integers(
        low=0, high=CLASS_NUM - 1, size=label_shape).astype('int64')
    return image_np, label_np


def static():
    def simple_fc_net(image):
        hidden = layers.fc(
            image,
            size=FC_SIZE,
            act='relu',
            param_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(
                value=0.99)),
            bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(
                value=0.5)))

        prediction = layers.fc(
            hidden,
            size=CLASS_NUM,
            act='softmax',
            param_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(
                value=1.2)),
            bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(
                value=0.8)))
        return prediction

    image = fluid.data(name='image', shape=[BATCH_SIZE, 784], dtype='float32')
    label = fluid.data(name='label', shape=[BATCH_SIZE, 1], dtype='int64')
    switch_id = fluid.data(name='switch_id', shape=[1], dtype='int32')

    one = layers.fill_constant(shape=[1], dtype='int32', value=1)
    prediction = simple_fc_net(image)

    cross_entropy_loss = layers.cross_entropy(input=prediction, label=label)
    mean_cross_entropy_loss = layers.mean(cross_entropy_loss)

    softmax_loss = layers.softmax_with_cross_entropy(
        logits=prediction, label=label)
    mean_softmax_loss = layers.mean(softmax_loss)
    adam = optimizer.Adam(learning_rate=0.001)
    adagrad = optimizer.Adagrad(learning_rate=0.001)

    # different optimizer and loss
    def fn_1():
        _, params_grads = adam.minimize(mean_cross_entropy_loss)
        return params_grads

    def fn_2():
        _, params_grads = adagrad.minimize(mean_softmax_loss)
        return params_grads

    params_grads = layers.case([(switch_id == one, fn_1)])

    # normal
    # adagrad.minimize(mean_softmax_loss)

    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())
    # param_list = fluid.framework.default_main_program().block(0).all_parameters()
    # print(param_list)
    for epoch in range(EPOCH_NUM):
        np.random.seed(epoch)
        feed_image, feed_label = random_input()
        out = exe.run(
            fluid.default_main_program(),
            feed={
                'image': feed_image,
                'label': feed_label,
                'switch_id': np.array([epoch]).astype('int32')
            },
            fetch_list=[
                prediction.name + "@GRAD"
                #prediction
            ])
        print(epoch)
        print(out[0])
        #print(params_grads)


class MyLayer(fluid.dygraph.Layer):
    def __init__(self, name_scope):
        super(MyLayer, self).__init__(name_scope)
        self.fc0 = fluid.dygraph.nn.FC(
            self.full_name(),
            size=FC_SIZE,
            act='relu',
            param_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(
                value=0.99)),
            bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(
                value=0.5)))

        self.pre = fluid.dygraph.nn.FC(
            self.full_name(),
            size=CLASS_NUM,
            act='softmax',
            param_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(
                value=1.2)),
            bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(
                value=0.8)))

    def forward(self, inputs):
        h_0 = self.fc0(inputs)
        prediction = self.pre(h_0)
        return prediction


def dynamic():
    with fluid.dygraph.guard():
        adam = fluid.optimizer.AdamOptimizer(learning_rate=0.001)
        adagrad = fluid.optimizer.Adagrad(learning_rate=0.001)
        my_layer = MyLayer("my_layer")
        for epoch in range(EPOCH_NUM):
            np.random.seed(epoch)
            image_data, label = random_input()
            var_input = fluid.dygraph.to_variable(image_data)
            var_lable = fluid.dygraph.to_variable(label)
            prediction = my_layer(var_input)
            cross_entropy_loss = layers.cross_entropy(prediction, var_lable)
            softmax_loss = layers.softmax_with_cross_entropy(
                logits=prediction, label=var_lable)
            if epoch == -1:
                avg_loss = layers.mean(cross_entropy_loss)
                avg_loss.backward()
                _, params_grads = adam.minimize(avg_loss)
            else:
                avg_loss = layers.mean(softmax_loss)
                avg_loss.backward()
                _, params_grads = adagrad.minimize(avg_loss)
            my_layer.clear_gradients()
            print(epoch)
            print(prediction.gradient())

            # print(params_grads)
            # print prediction.numpy()


if __name__ == '__main__':
    print("-" * 20, " dynamic ", "-" * 20)
    dynamic()
    print("-" * 20, " static ", "-" * 20)
    static()
