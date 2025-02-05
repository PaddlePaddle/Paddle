#   copyright (c) 2022 paddlepaddle authors. all rights reserved.
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

import logging
import os
import unittest

import numpy as np
from imperative_test_utils import fix_model_dict

import paddle
from paddle.framework import core, set_flags
from paddle.nn import (
    BatchNorm2D,
    Conv2D,
    LeakyReLU,
    Linear,
    MaxPool2D,
    PReLU,
    ReLU,
    Sequential,
    Sigmoid,
    Softmax,
)
from paddle.quantization import ImperativeQuantAware
from paddle.static.log_helper import get_logger

paddle.enable_static()

os.environ["CPU_NUM"] = "1"
if core.is_compiled_with_cuda():
    set_flags({"FLAGS_cudnn_deterministic": True})

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s'
)


class ImperativeLenet(paddle.nn.Layer):
    def __init__(self, num_classes=10):
        super().__init__()
        conv2d_w1_attr = paddle.ParamAttr(name="conv2d_w_1")
        conv2d_w2_attr = paddle.ParamAttr(name="conv2d_w_2")
        fc_w1_attr = paddle.ParamAttr(name="fc_w_1")
        fc_w2_attr = paddle.ParamAttr(name="fc_w_2")
        fc_w3_attr = paddle.ParamAttr(name="fc_w_3")
        conv2d_b2_attr = paddle.ParamAttr(name="conv2d_b_2")
        fc_b1_attr = paddle.ParamAttr(name="fc_b_1")
        fc_b2_attr = paddle.ParamAttr(name="fc_b_2")
        fc_b3_attr = paddle.ParamAttr(name="fc_b_3")
        self.features = Sequential(
            Conv2D(
                in_channels=1,
                out_channels=6,
                kernel_size=3,
                stride=1,
                padding=1,
                weight_attr=conv2d_w1_attr,
                bias_attr=False,
            ),
            BatchNorm2D(6),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),
            Conv2D(
                in_channels=6,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=0,
                weight_attr=conv2d_w2_attr,
                bias_attr=conv2d_b2_attr,
            ),
            BatchNorm2D(16),
            PReLU(),
            MaxPool2D(kernel_size=2, stride=2),
        )

        self.fc = Sequential(
            Linear(
                in_features=400,
                out_features=120,
                weight_attr=fc_w1_attr,
                bias_attr=fc_b1_attr,
            ),
            LeakyReLU(),
            Linear(
                in_features=120,
                out_features=84,
                weight_attr=fc_w2_attr,
                bias_attr=fc_b2_attr,
            ),
            Sigmoid(),
            Linear(
                in_features=84,
                out_features=num_classes,
                weight_attr=fc_w3_attr,
                bias_attr=fc_b3_attr,
            ),
            Softmax(),
        )

    def forward(self, inputs):
        x = self.features(inputs)
        x = paddle.flatten(x, 1)
        x = self.fc(x)
        return x


class TestImperativeQatLSQ(unittest.TestCase):
    def set_vars(self):
        self.weight_quantize_type = 'channel_wise_lsq_weight'
        self.activation_quantize_type = 'lsq_act'
        self.onnx_format = False
        self.fuse_conv_bn = False

    def func_qat(self):
        self.set_vars()

        imperative_qat = ImperativeQuantAware(
            weight_quantize_type=self.weight_quantize_type,
            activation_quantize_type=self.activation_quantize_type,
            fuse_conv_bn=self.fuse_conv_bn,
        )

        seed = 100
        np.random.seed(seed)
        paddle.seed(seed)
        paddle.disable_static()
        lenet = ImperativeLenet()
        lenet = fix_model_dict(lenet)
        imperative_qat.quantize(lenet)
        optimizer = paddle.optimizer.Momentum(
            learning_rate=0.1, parameters=lenet.parameters(), momentum=0.9
        )

        train_reader = paddle.batch(
            paddle.dataset.mnist.train(), batch_size=64, drop_last=True
        )
        test_reader = paddle.batch(paddle.dataset.mnist.test(), batch_size=32)
        epoch_num = 2
        for epoch in range(epoch_num):
            lenet.train()
            for batch_id, data in enumerate(train_reader()):
                x_data = np.array(
                    [x[0].reshape(1, 28, 28) for x in data]
                ).astype('float32')
                y_data = (
                    np.array([x[1] for x in data])
                    .astype('int64')
                    .reshape(-1, 1)
                )

                img = paddle.to_tensor(x_data)
                label = paddle.to_tensor(y_data)
                out = lenet(img)
                acc = paddle.metric.accuracy(out, label)
                loss = paddle.nn.functional.cross_entropy(
                    out, label, reduction='none', use_softmax=False
                )
                avg_loss = paddle.mean(loss)

                avg_loss.backward()
                optimizer.minimize(avg_loss)
                lenet.clear_gradients()

                if batch_id % 100 == 0:
                    _logger.info(
                        f"Train | At epoch {epoch} step {batch_id}: loss = {avg_loss.numpy()}, acc= {acc.numpy()}"
                    )

            lenet.eval()
            eval_acc_top1_list = []
            with paddle.no_grad():
                for batch_id, data in enumerate(test_reader()):
                    x_data = np.array(
                        [x[0].reshape(1, 28, 28) for x in data]
                    ).astype('float32')
                    y_data = (
                        np.array([x[1] for x in data])
                        .astype('int64')
                        .reshape(-1, 1)
                    )
                    img = paddle.to_tensor(x_data)
                    label = paddle.to_tensor(y_data)

                    out = lenet(img)
                    acc_top1 = paddle.metric.accuracy(
                        input=out, label=label, k=1
                    )
                    acc_top5 = paddle.metric.accuracy(
                        input=out, label=label, k=5
                    )

                    if batch_id % 100 == 0:
                        eval_acc_top1_list.append(float(acc_top1.numpy()))
                        _logger.info(
                            f"Test | At epoch {epoch} step {batch_id}: acc1 = {acc_top1.numpy()}, acc5 = {acc_top5.numpy()}"
                        )

            # check eval acc
            eval_acc_top1 = sum(eval_acc_top1_list) / len(eval_acc_top1_list)
            print('eval_acc_top1', eval_acc_top1)
        self.assertTrue(
            eval_acc_top1 > 0.9,
            msg=f"The test acc {{{eval_acc_top1:f}}} is less than 0.9.",
        )

    def test_qat(self):
        self.func_qat()


if __name__ == '__main__':
    unittest.main()
