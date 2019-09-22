# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
from paddle.fluid.optimizer import SGDOptimizer
from paddle.fluid import Conv2D, Pool2D, FC, core
from paddle.fluid.dygraph.base import to_variable
from paddle.fluid.dygraph.learning_rate_scheduler import LearningRateDecay


class SimpleImgConvPool(fluid.Layer):
    def __init__(self,
                 name_scope,
                 num_filters,
                 filter_size,
                 pool_size,
                 pool_stride,
                 pool_padding=0,
                 pool_type='max',
                 global_pooling=False,
                 conv_stride=1,
                 conv_padding=0,
                 conv_dilation=1,
                 conv_groups=1,
                 act=None,
                 use_cudnn=False,
                 param_attr=None,
                 bias_attr=None):
        super(SimpleImgConvPool, self).__init__(name_scope)

        self._conv2d = Conv2D(
            self.full_name(),
            num_filters=num_filters,
            filter_size=filter_size,
            stride=conv_stride,
            padding=conv_padding,
            dilation=conv_dilation,
            groups=conv_groups,
            param_attr=None,
            bias_attr=None,
            use_cudnn=use_cudnn)

        self._pool2d = Pool2D(
            self.full_name(),
            pool_size=pool_size,
            pool_type=pool_type,
            pool_stride=pool_stride,
            pool_padding=pool_padding,
            global_pooling=global_pooling,
            use_cudnn=use_cudnn)

    def forward(self, inputs):
        x = self._conv2d(inputs)
        x = self._pool2d(x)
        return x


class MNIST(fluid.Layer):
    def __init__(self, name_scope):
        super(MNIST, self).__init__(name_scope)

        self._simple_img_conv_pool_1 = SimpleImgConvPool(
            self.full_name(), 20, 5, 2, 2, act="relu")

        self._simple_img_conv_pool_2 = SimpleImgConvPool(
            self.full_name(), 50, 5, 2, 2, act="relu")

        pool_2_shape = 50 * 4 * 4
        SIZE = 10
        scale = (2.0 / (pool_2_shape**2 * SIZE))**0.5
        self._fc = FC(self.full_name(),
                      10,
                      param_attr=fluid.param_attr.ParamAttr(
                          initializer=fluid.initializer.NormalInitializer(
                              loc=0.0, scale=scale)),
                      act="softmax")

    def forward(self, inputs):
        x = self._simple_img_conv_pool_1(inputs)
        x = self._simple_img_conv_pool_2(x)
        x = self._fc(x)
        return x


class TestDygraphCheckpoint(unittest.TestCase):
    def reader_decorator(self, reader):
        def _reader_imple():
            for item in reader():
                image = np.array(item[0]).reshape(1, 28, 28)
                label = np.array(item[1]).astype('int64').reshape(1)
                yield image, label

        return _reader_imple

    def test_save_load_persistables(self):
        seed = 90
        epoch_num = 1
        batch_size = 128

        with fluid.dygraph.guard():
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed

            mnist = MNIST("mnist")
            sgd = SGDOptimizer(learning_rate=1e-3)

            batch_py_reader = fluid.io.PyReader(capacity=1)
            batch_py_reader.decorate_sample_list_generator(
                paddle.batch(
                    self.reader_decorator(paddle.dataset.mnist.train()),
                    batch_size=batch_size,
                    drop_last=True),
                places=fluid.CPUPlace())

            dy_param_init_value = {}

            for epoch in range(epoch_num):
                for batch_id, data in enumerate(batch_py_reader()):
                    img = data[0]
                    label = data[1]
                    label.stop_gradient = True

                    cost = mnist(img)
                    loss = fluid.layers.cross_entropy(cost, label)
                    avg_loss = fluid.layers.mean(loss)

                    dy_out = avg_loss.numpy()

                    avg_loss.backward()
                    sgd.minimize(avg_loss)
                    fluid.dygraph.save_persistables(mnist.state_dict(),
                                                    "save_dir")
                    mnist.clear_gradients()

                    for param in mnist.parameters():
                        dy_param_init_value[param.name] = param.numpy()

                    restore, _ = fluid.dygraph.load_persistables("save_dir")

                    self.assertRaises(IOError, fluid.dygraph.load_persistables,
                                      "not_exist_dir")

                    mnist.load_dict(restore)

                    self.assertEqual(len(dy_param_init_value), len(restore))
                    for ky, value in restore.items():
                        self.assertTrue(
                            np.allclose(value.numpy(), dy_param_init_value[
                                value.name]))
                        self.assertTrue(np.isfinite(value.numpy().all()))
                        self.assertFalse(np.isnan(value.numpy().any()))

                    if batch_id > 10:
                        break


class TestDygraphIO(unittest.TestCase):
    def setUp(self):
        self._opt = None
        self.train_reader = paddle.batch(
            paddle.dataset.mnist.train(), batch_size=128, drop_last=True)

    def compare_acc_states(self, first, second):
        ret = True
        if sorted(first.keys()) != sorted(second.keys()):
            print("      ", "first level keys ")
            ret = False
        for key1 in first:
            if sorted(first[key1].keys()) != sorted(second[key1].keys()):
                print(first[key1].keys())
                print(second[key1].keys())
                print("      ", "second level keys ")
                ret = False
            for key2 in first[key1].keys():
                if not np.allclose(first[key1][key2], second[key1][key2]):
                    print("      ", "values not equal ")
                    ret = False
        return ret

    def compare_lr(self, opt_lr, opt_lr_after_loaded):
        if isinstance(opt_lr, LearningRateDecay):
            return opt_lr.__dict__ == opt_lr_after_loaded.__dict__
        else:
            return np.allclose(opt_lr, opt_lr_after_loaded)

    def get_opt(self, opt_type):
        if opt_type == 'sgd':
            return SGDOptimizer(learning_rate=fluid.layers.natural_exp_decay(
                learning_rate=0.1,
                decay_steps=1,
                decay_rate=0.5,
                staircase=True))
        elif opt_type == 'moment':
            return fluid.optimizer.MomentumOptimizer(
                learning_rate=0.001, momentum=0.9, use_nesterov=True)
        elif opt_type == 'dgc_moment':
            return fluid.optimizer.DGCMomentumOptimizer(
                learning_rate=0.0001,
                momentum=0.9,
                rampup_step=1000,
                rampup_begin_step=1252,
                sparsity=[0.999, 0.999])
        elif opt_type == 'lars_moment':
            return fluid.optimizer.LarsMomentum(
                learning_rate=0.001, momentum=0.1, lars_weight_decay=0.001)

        elif opt_type == 'adagrad':
            return fluid.optimizer.Adagrad(learning_rate=0.2)

        elif opt_type == 'adam':
            return fluid.optimizer.AdamOptimizer(0.01)
        elif opt_type == 'decay_adagrad':
            return fluid.optimizer.DecayedAdagrad(learning_rate=0.2)

        elif opt_type == 'adadelta':
            return fluid.optimizer.Adadelta(
                learning_rate=0.0003, epsilon=1.0e-6, rho=0.95)

        elif opt_type == 'rms':
            return fluid.optimizer.RMSProp(learning_rate=0.1)

        elif opt_type == 'ftrl':
            return fluid.optimizer.Ftrl(learning_rate=0.1)

        elif opt_type == 'lamb':
            return fluid.optimizer.Lamb(learning_rate=0.002)

    def test_save_load(self):
        with fluid.dygraph.guard():
            mnist = MNIST("mnist")
            opt_names = [
                'sgd', 'moment', 'dgc_moment', 'lars_moment', 'adagrad', 'adam',
                'decay_adagrad', 'adadelta', 'rms', 'ftrl', 'lamb'
            ]
            batch_size = 128

            for opt_name in opt_names:
                self._opt = self.get_opt(opt_name)
                for batch_id, data in enumerate(self.train_reader()):
                    dy_x_data = np.array(
                        [x[0].reshape(1, 28, 28)
                         for x in data]).astype('float32')
                    y_data = np.array(
                        [x[1] for x in data]).astype('int64').reshape(128, 1)

                    img = to_variable(dy_x_data)
                    label = to_variable(y_data)
                    label._stop_gradient = True

                    cost = mnist(img)
                    loss = fluid.layers.cross_entropy(cost, label)
                    avg_loss = fluid.layers.mean(loss)
                    avg_loss.backward()

                    self._opt.minimize(avg_loss)
                    mnist.clear_gradients()
                    if batch_id > 10:
                        break

                fluid.dygraph.save_parameter(
                    mnist.state_dict(), save_dir="save_dir")
                para_dict = fluid.dygraph.load_parameter(load_dir="save_dir")
                mnist.set_dict(para_dict)

                opt_dict = self._opt.state_dict()
                fluid.dygraph.save_optimizer(opt_dict, "save_dir")
                opt_dict_loaded = fluid.dygraph.load_optimizer("save_dir")

                opt_loaded = self.get_opt(opt_name)
                opt_loaded_dict = fluid.dygraph.load_optimizer("save_dir")
                opt_loaded.set_dict(opt_loaded_dict)
                self.assertTrue(
                    self.compare_lr(
                        opt_dict.get("learning_rate"),
                        opt_dict_loaded.get("learning_rate")))
                self.assertTrue(
                    self.compare_acc_states(
                        opt_dict.get("accumulator_states"),
                        opt_dict_loaded.get("accumulator_states")))


if __name__ == '__main__':
    unittest.main()
