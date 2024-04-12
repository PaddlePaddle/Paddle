#   copyright (c) 2021 paddlepaddle authors. all rights reserved.
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

import numpy as np

import paddle
from paddle.framework import ParamAttr
from paddle.nn import (
    BatchNorm1D,
    BatchNorm2D,
    Conv2D,
    LeakyReLU,
    Linear,
    MaxPool2D,
    PReLU,
    ReLU,
    ReLU6,
    Sequential,
    Sigmoid,
    Softmax,
)
from paddle.static.log_helper import get_logger

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s'
)


def fix_model_dict(model):
    fixed_state = {}
    for name, param in model.named_parameters():
        p_shape = param.numpy().shape
        p_value = param.numpy()
        if name.endswith("bias"):
            value = np.zeros_like(p_value).astype('float32')
        else:
            value = (
                np.random.normal(loc=0.0, scale=0.01, size=np.prod(p_shape))
                .reshape(p_shape)
                .astype('float32')
            )
        fixed_state[name] = value
    model.set_dict(fixed_state)
    return model


def pre_hook(layer, input):
    input_return = input[0] * 2
    return input_return


def post_hook(layer, input, output):
    return output * 2


def train_lenet(lenet, reader, optimizer):
    loss_list = []
    lenet.train()

    for batch_id, data in enumerate(reader()):
        x_data = np.array([x[0].reshape(1, 28, 28) for x in data]).astype(
            'float32'
        )
        y_data = np.array([x[1] for x in data]).astype('int64').reshape(-1, 1)

        img = paddle.to_tensor(x_data)
        label = paddle.to_tensor(y_data)

        out = lenet(img)
        loss = paddle.nn.functional.cross_entropy(
            out, label, reduction='none', use_softmax=False
        )
        avg_loss = paddle.mean(loss)
        avg_loss.backward()

        optimizer.minimize(avg_loss)
        lenet.clear_gradients()

        if batch_id % 100 == 0:
            loss_list.append(float(avg_loss))
            _logger.info('{}: {}'.format('loss', float(avg_loss)))

    return loss_list


class ImperativeLenet(paddle.nn.Layer):
    def __init__(self, num_classes=10):
        super().__init__()
        conv2d_w1_attr = ParamAttr(name="conv2d_w_1")
        conv2d_w2_attr = ParamAttr(name="conv2d_w_2")
        fc_w1_attr = ParamAttr(name="fc_w_1")
        fc_w2_attr = ParamAttr(name="fc_w_2")
        fc_w3_attr = ParamAttr(name="fc_w_3")
        conv2d_b2_attr = ParamAttr(name="conv2d_b_2")
        fc_b1_attr = ParamAttr(name="fc_b_1")
        fc_b2_attr = ParamAttr(name="fc_b_2")
        fc_b3_attr = ParamAttr(name="fc_b_3")
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
        self.add = paddle.nn.quant.add()
        self.quant_stub = paddle.nn.quant.QuantStub()

    def forward(self, inputs):
        x = self.quant_stub(inputs)
        x = self.features(x)

        x = paddle.flatten(x, 1)
        x = self.add(x, paddle.to_tensor([0.0]))  # For CI
        x = self.fc(x)
        return x


class ImperativeLenetWithSkipQuant(paddle.nn.Layer):
    def __init__(self, num_classes=10):
        super().__init__()

        conv2d_w1_attr = ParamAttr(name="conv2d_w_1")
        conv2d_w2_attr = ParamAttr(name="conv2d_w_2")
        fc_w1_attr = ParamAttr(name="fc_w_1")
        fc_w2_attr = ParamAttr(name="fc_w_2")
        fc_w3_attr = ParamAttr(name="fc_w_3")
        conv2d_b1_attr = ParamAttr(name="conv2d_b_1")
        conv2d_b2_attr = ParamAttr(name="conv2d_b_2")
        fc_b1_attr = ParamAttr(name="fc_b_1")
        fc_b2_attr = ParamAttr(name="fc_b_2")
        fc_b3_attr = ParamAttr(name="fc_b_3")
        self.conv2d_0 = Conv2D(
            in_channels=1,
            out_channels=6,
            kernel_size=3,
            stride=1,
            padding=1,
            weight_attr=conv2d_w1_attr,
            bias_attr=conv2d_b1_attr,
        )
        self.conv2d_0.skip_quant = True

        self.batch_norm_0 = BatchNorm2D(6)
        self.relu_0 = ReLU()
        self.pool2d_0 = MaxPool2D(kernel_size=2, stride=2)
        self.conv2d_1 = Conv2D(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=0,
            weight_attr=conv2d_w2_attr,
            bias_attr=conv2d_b2_attr,
        )
        self.conv2d_1.skip_quant = False

        self.batch_norm_1 = BatchNorm2D(16)
        self.relu6_0 = ReLU6()
        self.pool2d_1 = MaxPool2D(kernel_size=2, stride=2)
        self.linear_0 = Linear(
            in_features=400,
            out_features=120,
            weight_attr=fc_w1_attr,
            bias_attr=fc_b1_attr,
        )
        self.linear_0.skip_quant = True

        self.leaky_relu_0 = LeakyReLU()
        self.linear_1 = Linear(
            in_features=120,
            out_features=84,
            weight_attr=fc_w2_attr,
            bias_attr=fc_b2_attr,
        )
        self.linear_1.skip_quant = False

        self.sigmoid_0 = Sigmoid()
        self.linear_2 = Linear(
            in_features=84,
            out_features=num_classes,
            weight_attr=fc_w3_attr,
            bias_attr=fc_b3_attr,
        )
        self.linear_2.skip_quant = False
        self.softmax_0 = Softmax()

    def forward(self, inputs):
        x = self.conv2d_0(inputs)
        x = self.batch_norm_0(x)
        x = self.relu_0(x)
        x = self.pool2d_0(x)
        x = self.conv2d_1(x)
        x = self.batch_norm_1(x)
        x = self.relu6_0(x)
        x = self.pool2d_1(x)

        x = paddle.flatten(x, 1)
        x = self.linear_0(x)
        x = self.leaky_relu_0(x)
        x = self.linear_1(x)
        x = self.sigmoid_0(x)
        x = self.linear_2(x)
        x = self.softmax_0(x)

        return x


class ImperativeLinearBn(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

        fc_w_attr = paddle.ParamAttr(
            name="fc_weight",
            initializer=paddle.nn.initializer.Constant(value=0.5),
        )
        fc_b_attr = paddle.ParamAttr(
            name="fc_bias",
            initializer=paddle.nn.initializer.Constant(value=1.0),
        )
        bn_w_attr = paddle.ParamAttr(
            name="bn_weight",
            initializer=paddle.nn.initializer.Constant(value=0.5),
        )

        self.linear = Linear(
            in_features=10,
            out_features=10,
            weight_attr=fc_w_attr,
            bias_attr=fc_b_attr,
        )
        self.bn = BatchNorm1D(10, weight_attr=bn_w_attr)

    def forward(self, inputs):
        x = self.linear(inputs)
        x = self.bn(x)

        return x


class ImperativeLinearBn_hook(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

        fc_w_attr = paddle.ParamAttr(
            name="linear_weight",
            initializer=paddle.nn.initializer.Constant(value=0.5),
        )

        self.linear = Linear(
            in_features=10, out_features=10, weight_attr=fc_w_attr
        )
        self.bn = BatchNorm1D(10)

        forward_pre = self.linear.register_forward_pre_hook(pre_hook)
        forward_post = self.bn.register_forward_post_hook(post_hook)

    def forward(self, inputs):
        x = self.linear(inputs)
        x = self.bn(x)

        return x
