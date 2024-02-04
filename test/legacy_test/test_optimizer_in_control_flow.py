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

import unittest

import numpy as np

import paddle
from paddle import base
from paddle.base import core
from paddle.base.framework import Program, program_guard

BATCH_SIZE = 1
INPUT_SIZE = 784
CLASS_NUM = 10
FC_SIZE = 40
EPOCH_NUM = 5
LR = 0.001
SEED = 2020

paddle.enable_static()


def static(
    train_data, loss_in_switch=True, use_cuda=False, use_parallel_exe=False
):
    startup_program = Program()
    main_program = Program()
    paddle.seed(SEED)

    with program_guard(main_program, startup_program):

        def double_fc_net(image):
            hidden = paddle.static.nn.fc(
                image,
                size=FC_SIZE,
                activation='relu',
                weight_attr=base.ParamAttr(
                    initializer=paddle.nn.initializer.Constant(value=0.99)
                ),
                bias_attr=base.ParamAttr(
                    initializer=paddle.nn.initializer.Constant(value=0.5)
                ),
                name="hidden",
            )

            prediction = paddle.static.nn.fc(
                hidden,
                size=CLASS_NUM,
                activation='softmax',
                weight_attr=base.ParamAttr(
                    initializer=paddle.nn.initializer.Constant(value=1.2)
                ),
                bias_attr=base.ParamAttr(
                    initializer=paddle.nn.initializer.Constant(value=0.8)
                ),
                name="prediction",
            )
            return hidden, prediction

        def fn_1(opt, avg_loss=None, pred=None, label=None):
            if avg_loss is None:
                loss = paddle.nn.functional.cross_entropy(
                    input=pred, label=label, reduction='none', use_softmax=False
                )
                avg_loss = paddle.mean(loss, name='mean_cross_entropy_loss')
            opt.minimize(avg_loss)
            return avg_loss

        def fn_2(opt, avg_loss=None, pred=None, label=None):
            if avg_loss is None:
                loss = paddle.nn.functional.softmax_with_cross_entropy(
                    logits=pred, label=label
                )
                avg_loss = paddle.mean(loss, name='mean_softmax_loss')
            opt.minimize(avg_loss)
            return avg_loss

        image = paddle.static.data('image', [BATCH_SIZE, INPUT_SIZE], 'float32')
        label = paddle.static.data('label', [BATCH_SIZE, 1], 'int64')
        hidden, prediction = double_fc_net(image)

        adam = paddle.optimizer.Adam(learning_rate=LR)
        sgd = paddle.optimizer.SGD(learning_rate=LR)

        id = paddle.static.data('id', [1], 'int32')
        two = paddle.tensor.fill_constant([1], 'int32', 2)
        mod_two = paddle.remainder(id, two) == 0

        if loss_in_switch:
            avg_loss = paddle.static.nn.case(
                [(mod_two, lambda: fn_1(adam, None, prediction, label))],
                lambda: fn_2(sgd, None, prediction, label),
            )
        else:
            loss_1 = paddle.nn.functional.cross_entropy(
                input=prediction,
                label=label,
                reduction='none',
                use_softmax=False,
            )
            avg_loss_1 = paddle.mean(loss_1)
            loss_2 = paddle.nn.functional.softmax_with_cross_entropy(
                logits=prediction, label=label
            )
            avg_loss_2 = paddle.mean(loss_2)
            avg_loss = paddle.static.nn.case(
                [(mod_two, lambda: fn_1(adam, avg_loss_1))],
                lambda: fn_2(sgd, avg_loss_2),
            )

    place = base.CUDAPlace(0) if use_cuda else base.CPUPlace()
    exe = base.Executor(place)
    exe.run(startup_program)

    for epoch in range(EPOCH_NUM):
        feed_image, feed_label = train_data[epoch]
        fetch_list = [hidden, prediction, avg_loss]
        feed = {
            'image': feed_image,
            'label': feed_label,
            'id': np.array([epoch]).astype('int32'),
        }
        out = exe.run(main_program, feed=feed, fetch_list=fetch_list)
        out_hidden, out_pred, loss = out

    return out_hidden, out_pred, loss


class DygraphLayer(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.fc_1 = paddle.nn.Linear(
            INPUT_SIZE,
            FC_SIZE,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.99)
            ),
            bias_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.5)
            ),
        )
        self.act_1 = paddle.nn.ReLU()
        self.fc_2 = paddle.nn.Linear(
            FC_SIZE,
            CLASS_NUM,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1.2)
            ),
            bias_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=0.8)
            ),
        )

        self.act_2 = paddle.nn.Softmax()

    def forward(self, inputs):
        hidden = self.fc_1(inputs)
        prediction = self.fc_2(hidden)
        return self.act_1(hidden), self.act_2(prediction)


def dynamic(train_data, use_cuda=False, use_parallel_exe=False):
    place = base.CUDAPlace(0) if use_cuda else base.CPUPlace()
    with base.dygraph.guard(place):
        paddle.seed(SEED)
        dy_layer = DygraphLayer()
        adam = paddle.optimizer.Adam(
            learning_rate=LR, parameters=dy_layer.parameters()
        )
        sgd = paddle.optimizer.SGD(
            learning_rate=LR, parameters=dy_layer.parameters()
        )

        for epoch in range(EPOCH_NUM):
            image_data, label = train_data[epoch]
            var_input = paddle.to_tensor(image_data)
            var_label = paddle.to_tensor(label)
            hidden, prediction = dy_layer(var_input)

            if epoch % 2 == 0:
                cross_entropy_loss = paddle.nn.functional.cross_entropy(
                    prediction, var_label, reduction='none', use_softmax=False
                )
                loss = paddle.mean(cross_entropy_loss)
                loss.backward()
                adam.minimize(loss)
            else:
                softmax_loss = paddle.nn.functional.softmax_with_cross_entropy(
                    prediction, var_label
                )
                loss = paddle.mean(softmax_loss)
                loss.backward()
                sgd.minimize(loss)

            dy_layer.clear_gradients()
        return hidden.numpy(), prediction.numpy(), loss.numpy()


class TestMultiTask(unittest.TestCase):
    '''
    Compare results of static graph and dynamic graph.
    Todo(liym27): add parallel GPU train.
    '''

    def random_input(
        self,
        seed,
        image_shape=[BATCH_SIZE, INPUT_SIZE],
        label_shape=[BATCH_SIZE, 1],
    ):
        np.random.seed(seed)
        image_np = np.random.random(size=image_shape).astype('float32')
        np.random.seed(seed)
        label_np = np.random.randint(
            low=0, high=CLASS_NUM - 1, size=label_shape
        ).astype('int64')
        return image_np, label_np

    def init_train_data(self):
        self.train_data = []
        for epoch in range(EPOCH_NUM):
            self.train_data.append(self.random_input(epoch))

    def test_optimizer_in_switch(self):
        self.init_train_data()
        use_cuda = core.is_compiled_with_cuda()
        hidden_2, pre_2, loss_2 = dynamic(self.train_data, use_cuda)
        for loss_in_switch in [True, False]:
            hidden_1, pre_1, loss_1 = static(
                self.train_data, loss_in_switch, use_cuda
            )
            np.testing.assert_allclose(hidden_1, hidden_2, rtol=1e-05)
            np.testing.assert_allclose(pre_1, pre_2, rtol=1e-05)
            np.testing.assert_allclose(loss_1, loss_2, rtol=1e-05)


if __name__ == '__main__':
    unittest.main()
