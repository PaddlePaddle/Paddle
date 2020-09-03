#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from time import time

import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.base import switch_to_static_graph
from paddle.fluid.dygraph import to_variable
from paddle.fluid.dygraph.nn import Conv2D, Linear, Pool2D
from paddle.fluid.optimizer import AdamOptimizer
from paddle.fluid.dygraph.jit import declarative
from paddle.fluid.dygraph.dygraph_to_static import ProgramTranslator

SEED = 2020


class SimpleImgConvPool(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
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
        super(SimpleImgConvPool, self).__init__()

        self._conv2d = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=conv_stride,
            padding=conv_padding,
            dilation=conv_dilation,
            groups=conv_groups,
            param_attr=None,
            bias_attr=None,
            act=act,
            use_cudnn=use_cudnn)

        self._pool2d = Pool2D(
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


class MNIST(fluid.dygraph.Layer):
    def __init__(self):
        super(MNIST, self).__init__()

        self._simple_img_conv_pool_1 = SimpleImgConvPool(
            1, 20, 5, 2, 2, act="relu")

        self._simple_img_conv_pool_2 = SimpleImgConvPool(
            20, 50, 5, 2, 2, act="relu")

        self.pool_2_shape = 50 * 4 * 4
        SIZE = 10
        scale = (2.0 / (self.pool_2_shape**2 * SIZE))**0.5
        self._fc = Linear(
            self.pool_2_shape,
            10,
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.NormalInitializer(
                    loc=0.0, scale=scale)),
            act="softmax")

    @declarative
    def forward(self, inputs, label=None):
        x = self.inference(inputs)
        if label is not None:
            acc = fluid.layers.accuracy(input=x, label=label)
            loss = fluid.layers.cross_entropy(x, label)
            avg_loss = fluid.layers.mean(loss)
            return x, acc, avg_loss
        else:
            return x

    def inference(self, inputs):
        x = self._simple_img_conv_pool_1(inputs)
        x = self._simple_img_conv_pool_2(x)
        x = fluid.layers.reshape(x, shape=[-1, self.pool_2_shape])
        x = self._fc(x)
        return x


class TestMNIST(unittest.TestCase):
    def setUp(self):
        self.epoch_num = 1
        self.batch_size = 64
        self.place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        self.train_reader = paddle.batch(
            paddle.dataset.mnist.train(),
            batch_size=self.batch_size,
            drop_last=True)


class TestMNISTWithDeclarative(TestMNIST):
    """
    Tests model if doesn't change the layers while decorated
    by `dygraph_to_static_output`. In this case, everything should
    still works if model is trained in dygraph mode.
    """

    def train_static(self):
        return self.train(to_static=True)

    def train_dygraph(self):
        return self.train(to_static=False)

    def test_mnist_declarative(self):
        dygraph_loss = self.train_dygraph()
        static_loss = self.train_static()
        self.assertTrue(
            np.allclose(dygraph_loss, static_loss),
            msg='dygraph is {}\n static_res is \n{}'.format(dygraph_loss,
                                                            static_loss))

    def train(self, to_static=False):
        prog_trans = ProgramTranslator()
        prog_trans.enable(to_static)

        loss_data = []
        with fluid.dygraph.guard(self.place):
            fluid.default_main_program().random_seed = SEED
            fluid.default_startup_program().random_seed = SEED
            mnist = MNIST()
            adam = AdamOptimizer(
                learning_rate=0.001, parameter_list=mnist.parameters())

            for epoch in range(self.epoch_num):
                start = time()
                for batch_id, data in enumerate(self.train_reader()):
                    dy_x_data = np.array(
                        [x[0].reshape(1, 28, 28)
                         for x in data]).astype('float32')
                    y_data = np.array(
                        [x[1] for x in data]).astype('int64').reshape(-1, 1)

                    img = to_variable(dy_x_data)
                    label = to_variable(y_data)

                    label.stop_gradient = True
                    prediction, acc, avg_loss = mnist(img, label=label)
                    avg_loss.backward()

                    adam.minimize(avg_loss)
                    loss_data.append(avg_loss.numpy()[0])
                    # save checkpoint
                    mnist.clear_gradients()
                    if batch_id % 10 == 0:
                        print(
                            "Loss at epoch {} step {}: loss: {:}, acc: {}, cost: {}"
                            .format(epoch, batch_id,
                                    avg_loss.numpy(),
                                    acc.numpy(), time() - start))
                        start = time()
                    if batch_id == 50:
                        mnist.eval()
                        prediction, acc, avg_loss = mnist(img, label)
                        loss_data.append(avg_loss.numpy()[0])
                        self.check_save_inference_model([dy_x_data, y_data],
                                                        prog_trans, to_static,
                                                        prediction)
                        break
        return loss_data

    @switch_to_static_graph
    def check_save_inference_model(self, inputs, prog_trans, to_static, gt_out):
        if to_static:
            infer_model_path = "./test_mnist_inference_model"
            prog_trans.save_inference_model(infer_model_path)
            infer_out = self.load_and_run_inference(infer_model_path, inputs)
            self.assertTrue(np.allclose(gt_out.numpy(), infer_out))

    def load_and_run_inference(self, model_path, inputs):
        exe = fluid.Executor(self.place)
        [inference_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(
             dirname=model_path, executor=exe)
        assert len(inputs) == len(feed_target_names)
        results = exe.run(inference_program,
                          feed=dict(zip(feed_target_names, inputs)),
                          fetch_list=fetch_targets)

        return np.array(results[0])


if __name__ == "__main__":
    unittest.main()
