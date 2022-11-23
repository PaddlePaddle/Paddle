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

import os
import unittest

import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.optimizer as optimizer
from paddle.fluid.framework import Program, program_guard
import paddle.fluid.core as core

BATCH_SIZE = 1
INPUT_SIZE = 784
CLASS_NUM = 10
FC_SIZE = 40
EPOCH_NUM = 5
LR = 0.001
SEED = 2020

paddle.enable_static()


def static(train_data,
           loss_in_switch=True,
           use_cuda=False,
           use_parallel_exe=False):
    startup_program = Program()
    main_program = Program()
    startup_program.random_seed = SEED
    main_program.random_seed = SEED

    with program_guard(main_program, startup_program):

        def double_fc_net(image):
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

        def fn_1(opt, avg_loss=None, pred=None, label=None):
            if avg_loss is None:
                loss = layers.cross_entropy(input=pred, label=label)
                avg_loss = paddle.mean(loss, name='mean_cross_entropy_loss')
            opt.minimize(avg_loss)
            return avg_loss

        def fn_2(opt, avg_loss=None, pred=None, label=None):
            if avg_loss is None:
                loss = layers.softmax_with_cross_entropy(logits=pred,
                                                         label=label)
                avg_loss = paddle.mean(loss, name='mean_softmax_loss')
            opt.minimize(avg_loss)
            return avg_loss

        image = fluid.data('image', [BATCH_SIZE, INPUT_SIZE], 'float32')
        label = fluid.data('label', [BATCH_SIZE, 1], 'int64')
        hidden, prediction = double_fc_net(image)

        adam = optimizer.Adam(learning_rate=LR)
        sgd = optimizer.SGD(learning_rate=LR)

        id = fluid.data('id', [1], 'int32')
        two = layers.fill_constant([1], 'int32', 2)
        mod_two = layers.elementwise_mod(id, two) == 0

        if loss_in_switch:
            avg_loss = layers.case(
                [(mod_two, lambda: fn_1(adam, None, prediction, label))],
                lambda: fn_2(sgd, None, prediction, label))
        else:
            loss_1 = layers.cross_entropy(input=prediction, label=label)
            avg_loss_1 = paddle.mean(loss_1)
            loss_2 = layers.softmax_with_cross_entropy(logits=prediction,
                                                       label=label)
            avg_loss_2 = paddle.mean(loss_2)
            avg_loss = layers.case([(mod_two, lambda: fn_1(adam, avg_loss_1))],
                                   lambda: fn_2(sgd, avg_loss_2))

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_program)

    for epoch in range(EPOCH_NUM):
        feed_image, feed_label = train_data[epoch]
        fetch_list = [hidden, prediction, avg_loss]
        feed = {
            'image': feed_image,
            'label': feed_label,
            'id': np.array([epoch]).astype('int32')
        }
        out = exe.run(main_program, feed=feed, fetch_list=fetch_list)
        out_hidden, out_pred, loss = out

    return out_hidden, out_pred, loss


class DygraphLayer(fluid.dygraph.Layer):

    def __init__(self):
        super(DygraphLayer, self).__init__()
        self.fc_1 = fluid.dygraph.nn.Linear(
            INPUT_SIZE,
            FC_SIZE,
            act='relu',
            param_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(
                value=0.99)),
            bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(
                value=0.5)),
        )

        self.fc_2 = fluid.dygraph.nn.Linear(
            FC_SIZE,
            CLASS_NUM,
            act='softmax',
            param_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(
                value=1.2)),
            bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Constant(
                value=0.8)))

    def forward(self, inputs):
        hidden = self.fc_1(inputs)
        prediction = self.fc_2(hidden)
        return hidden, prediction


def dynamic(train_data, use_cuda=False, use_parallel_exe=False):
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        fluid.default_startup_program().random_seed = SEED
        fluid.default_main_program().random_seed = SEED
        dy_layer = DygraphLayer()
        adam = fluid.optimizer.Adam(learning_rate=LR,
                                    parameter_list=dy_layer.parameters())
        sgd = fluid.optimizer.SGD(learning_rate=LR,
                                  parameter_list=dy_layer.parameters())

        for epoch in range(EPOCH_NUM):
            image_data, label = train_data[epoch]
            var_input = fluid.dygraph.to_variable(image_data)
            var_label = fluid.dygraph.to_variable(label)
            hidden, prediction = dy_layer(var_input)

            if epoch % 2 == 0:
                cross_entropy_loss = layers.cross_entropy(prediction, var_label)
                loss = paddle.mean(cross_entropy_loss)
                loss.backward()
                adam.minimize(loss)
            else:
                softmax_loss = layers.softmax_with_cross_entropy(
                    prediction, var_label)
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

    def random_input(self,
                     seed,
                     image_shape=[BATCH_SIZE, INPUT_SIZE],
                     label_shape=[BATCH_SIZE, 1]):
        np.random.seed(seed)
        image_np = np.random.random(size=image_shape).astype('float32')
        np.random.seed(seed)
        label_np = np.random.randint(low=0,
                                     high=CLASS_NUM - 1,
                                     size=label_shape).astype('int64')
        return image_np, label_np

    def init_train_data(self):
        self.train_data = []
        for epoch in range(EPOCH_NUM):
            self.train_data.append(self.random_input(epoch))

    def test_optimzier_in_switch(self):
        self.init_train_data()
        use_cuda = core.is_compiled_with_cuda()
        hidden_2, pre_2, loss_2 = dynamic(self.train_data, use_cuda)
        for loss_in_switch in [True, False]:
            hidden_1, pre_1, loss_1 = static(self.train_data, loss_in_switch,
                                             use_cuda)
            np.testing.assert_allclose(hidden_1, hidden_2, rtol=1e-05)
            np.testing.assert_allclose(pre_1, pre_2, rtol=1e-05)
            np.testing.assert_allclose(loss_1, loss_2, rtol=1e-05)


class TestMultiOptimizersMultiCardsError(unittest.TestCase):

    def test_error(self):
        startup_program = Program()
        main_program = Program()
        use_cuda = core.is_compiled_with_cuda()
        with program_guard(main_program, startup_program):

            def fn_1(opt, avg_loss):
                opt.minimize(avg_loss)

            def fn_2(opt, avg_loss):
                opt.minimize(avg_loss)

            x = fluid.layers.data("X", [10], 'float32')
            hidden = layers.fc(x, 5)
            avg_loss = paddle.mean(hidden)

            adam = optimizer.Adam(learning_rate=LR)
            sgd = optimizer.SGD(learning_rate=LR)

            cond = layers.fill_constant([1], 'bool', True)

            layers.case([(cond, lambda: fn_1(adam, avg_loss))],
                        lambda: fn_2(sgd, avg_loss))

        cpu_place = fluid.CPUPlace()
        cuda_place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

        for place in [cpu_place, cuda_place]:

            exe = fluid.Executor(place)
            exe.run(startup_program)

            np.random.seed(SEED)

            # NOTE(liym27):
            # This test needs to run in multi cards to test NotImplementedError.
            # Here, move this test from RUN_TYPE=DIST in tests/unittests/CMakeList.txt,
            # to use multi cards ** only on CPU ** not GPU to reduce CI time.
            os.environ['CPU_NUM'] = str(2)

            pe_exe = fluid.ParallelExecutor(use_cuda=use_cuda,
                                            main_program=main_program,
                                            loss_name=avg_loss.name)
            num_devices = pe_exe.device_count

            def not_implemented_error():
                pe_exe.run(feed={
                    'X':
                    np.random.random(size=[64, 10]).astype('float32'),
                },
                           fetch_list=[avg_loss.name])

            if num_devices > 1:
                self.assertRaises(NotImplementedError, not_implemented_error)


if __name__ == '__main__':
    unittest.main()
