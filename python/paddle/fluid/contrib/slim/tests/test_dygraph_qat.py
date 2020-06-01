#   copyright (c) 2018 paddlepaddle authors. all rights reserved.
#
# licensed under the apache license, version 2.0 (the "license");
# you may not use this file except in compliance with the license.
# you may obtain a copy of the license at
#
#     http://www.apache.org/licenses/license-2.0
#
# unless required by applicable law or agreed to in writing, software
# distributed under the license is distributed on an "as is" basis,
# without warranties or conditions of any kind, either express or implied.
# see the license for the specific language governing permissions and
# limitations under the license.

from __future__ import print_function

import os
import numpy as np
import unittest
import paddle
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid.contrib.slim.quantization import DygraphQuantAware
from paddle.fluid.optimizer import AdamOptimizer
from paddle.fluid.dygraph import declarative

os.environ["CPU_NUM"] = "8"


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

        self._conv2d = fluid.dygraph.Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=conv_stride,
            padding=conv_padding,
            dilation=conv_dilation,
            groups=conv_groups,
            param_attr=param_attr,
            bias_attr=bias_attr,
            act=act,
            use_cudnn=use_cudnn)

        self._pool2d = fluid.dygraph.Pool2D(
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


class Mnist(fluid.dygraph.Layer):
    def __init__(self):
        super(Mnist, self).__init__()

        self._simple_img_conv_pool_1 = SimpleImgConvPool(
            1, 20, 5, 2, 2, act="relu")

        self._simple_img_conv_pool_2 = SimpleImgConvPool(
            20, 50, 5, 2, 2, act="relu")

        self.pool_2_shape = 50 * 4 * 4
        SIZE = 10
        scale = (2.0 / (self.pool_2_shape**2 * SIZE))**0.5
        self._fc = fluid.dygraph.Linear(
            self.pool_2_shape,
            10,
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.NormalInitializer(
                    loc=0.0, scale=scale)),
            act="softmax")

    @declarative
    def forward(self, inputs):
        x = self._simple_img_conv_pool_1(inputs)
        x = self._simple_img_conv_pool_2(x)
        x = fluid.layers.reshape(x, shape=[-1, self.pool_2_shape])
        x = self._fc(x)
        return x


class TestDygraphQat(unittest.TestCase):
    """
    QAT = quantization-aware training
    """

    def test_qat(self):
        dygraph_qat = DygraphQuantAware()
        dygraph_qat.prepare()

        with fluid.dygraph.guard():
            mnist = Mnist()
            dygraph_qat.quantize(mnist)
            adam = AdamOptimizer(
                learning_rate=0.001, parameter_list=mnist.parameters())
            train_reader = paddle.batch(
                paddle.dataset.mnist.train(), batch_size=32, drop_last=True)
            test_reader = paddle.batch(
                paddle.dataset.mnist.test(), batch_size=32)

            epoch_num = 2
            for epoch in range(epoch_num):
                mnist.train()
                for batch_id, data in enumerate(train_reader()):
                    x_data = np.array([x[0].reshape(1, 28, 28)
                                       for x in data]).astype('float32')
                    y_data = np.array(
                        [x[1] for x in data]).astype('int64').reshape(-1, 1)

                    img = fluid.dygraph.to_variable(x_data)
                    label = fluid.dygraph.to_variable(y_data)

                    out = mnist(img)
                    acc = fluid.layers.accuracy(out, label)
                    loss = fluid.layers.cross_entropy(out, label)
                    avg_loss = fluid.layers.mean(loss)
                    avg_loss.backward()
                    adam.minimize(avg_loss)
                    mnist.clear_gradients()
                    if batch_id % 100 == 0:
                        print(
                            "Train | At epoch {} step {}: loss = {:}, acc= {:}".
                            format(epoch, batch_id,
                                   avg_loss.numpy(), acc.numpy()))

                mnist.eval()
                for batch_id, data in enumerate(test_reader()):
                    x_data = np.array([x[0].reshape(1, 28, 28)
                                       for x in data]).astype('float32')
                    y_data = np.array(
                        [x[1] for x in data]).astype('int64').reshape(-1, 1)

                    img = fluid.dygraph.to_variable(x_data)
                    label = fluid.dygraph.to_variable(y_data)

                    out = mnist(img)
                    acc_top1 = fluid.layers.accuracy(
                        input=out, label=label, k=1)
                    acc_top5 = fluid.layers.accuracy(
                        input=out, label=label, k=5)

                    if batch_id % 100 == 0:
                        print(
                            "Test | At epoch {} step {}: acc1 = {:}, acc5 = {:}".
                            format(epoch, batch_id,
                                   acc_top1.numpy(), acc_top5.numpy()))

            # save weights
            model_dict = mnist.state_dict()
            fluid.save_dygraph(model_dict, "save_temp")

            # test the correctness of `save_infer_quant_model`
            data = next(test_reader())
            test_data = np.array([x[0].reshape(1, 28, 28)
                                  for x in data]).astype('float32')
            test_img = fluid.dygraph.to_variable(test_data)
            mnist.eval()
            before_save = mnist(test_img)

        # save inference quantized model
        path = "./mnist_infer_model"
        dygraph_qat.save_infer_quant_model(
            dirname=path,
            model=mnist,
            input_shape=(1, 28, 28),
            input_dtype='float32',
            feed=[0],
            fetch=[0])
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()
        exe = fluid.Executor(place)
        [inference_program, feed_target_names, fetch_targets] = (
            fluid.io.load_inference_model(
                dirname=path, executor=exe))
        after_save, = exe.run(inference_program,
                              feed={feed_target_names[0]: test_data},
                              fetch_list=fetch_targets)

        self.assertTrue(
            np.allclose(after_save, before_save.numpy()),
            msg='Failed to save the inference quantized model.')


if __name__ == '__main__':
    unittest.main()
