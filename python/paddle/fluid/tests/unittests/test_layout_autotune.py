# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import unittest
import numpy
import paddle.nn.functional as F


class SimpleNet(paddle.nn.Layer):
    def __init__(self, data_format="NCHW", class_num=2):
        super(SimpleNet, self).__init__()
        self.conv = paddle.nn.Conv2D(3, 8, (3, 3))
        self.bn = paddle.nn.BatchNorm(num_channels=8)
        self.relu = paddle.nn.ReLU()
        self.pool = paddle.nn.AvgPool2D(kernel_size=2, stride=2)
        self.flatten = paddle.nn.Flatten()
        self.fc = paddle.nn.Linear(392, class_num)

    def forward(self, image):
        conv_out = self.conv(image)
        bn_out = self.bn(conv_out)
        out = self.relu(bn_out)
        out = self.pool(out)
        out = self.flatten(out)
        out = self.fc(out)
        return conv_out, out


class LayoutAutoTune(unittest.TestCase):
    def use_autoune(self):
        if paddle.is_compiled_with_cuda():
            paddle.fluid.core.enable_layout_autotune()
            return paddle.fluid.core.use_layout_autotune()
        else:
            paddle.fluid.core.disable_layout_autotune()
            return paddle.fluid.core.use_layout_autotune()

    def train(self, data_format):
        model = SimpleNet(data_format="NCHW", class_num=2)
        data = paddle.rand([1, 3, 16, 16])
        if (data_format == "NHWC"):
            data = paddle.rand([1, 16, 16, 3])
        label_data = paddle.randint(0, 1, shape=[1, 1], dtype="int64")
        optimizer = paddle.optimizer.SGD(learning_rate=0.0001,
                                         parameters=model.parameters())
        scaler = paddle.amp.GradScaler()
        for i in range(2):
            with paddle.amp.auto_cast(level="O2"):
                conv_out, predict = model(data)
                loss = F.cross_entropy(predict, label=label_data)
                loss = loss.mean()

            scaled = scaler.scale(loss)
            scaled.backward()
            scaler.minimize(optimizer, scaled)
        return conv_out, predict

    def test_enable_autotune(self):
        if self.use_autoune():
            conv_out, predict = self.train(data_format="NCHW")
            self.assertEqual(conv_out.shape, [1, 14, 14, 8])
            self.assertEqual(predict.shape, [1, 2])
        else:
            conv_out, predict = self.train(data_format="NCHW")
            self.assertEqual(conv_out.shape, [1, 8, 14, 14])
            self.assertEqual(predict.shape, [1, 2])

    def test_transpose_op_transposer(self):
        if not self.use_autoune():
            return
        conv = paddle.nn.Conv2D(3, 8, (3, 3))
        data = paddle.rand([1, 3, 16, 14])
        label_data = paddle.randint(0, 1, shape=[1, 1], dtype="int64")
        optimizer = paddle.optimizer.SGD(learning_rate=0.0001,
                                         parameters=conv.parameters())
        scaler = paddle.amp.GradScaler()
        with paddle.amp.auto_cast(level="O2"):
            conv_out = conv(data)
            # conv_out.shape = [1, 14, 12, 8] with NHWC
            # layout tuner will transpose conv_out to 
            # [1, 8, 14, 12] with NCHW before the following transpose op.
            out = paddle.transpose(conv_out, perm=[0, 3, 1, 2])
            loss = out.mean()
        scaled = scaler.scale(loss)
        scaled.backward()
        scaler.minimize(optimizer, scaled)

        self.assertEqual(conv_out.shape, [1, 14, 12, 8])
        self.assertEqual(out.shape, [1, 12, 8, 14])

    def test_flatten_op_transposer(self):
        if not self.use_autoune():
            return
        paddle.fluid.core.enable_layout_autotune()
        conv = paddle.nn.Conv2D(3, 8, (3, 3))
        flatten = paddle.nn.Flatten(start_axis=1, stop_axis=2)
        data = paddle.rand([1, 3, 16, 14])
        with paddle.amp.auto_cast(level="O2"):
            conv_out = conv(data)
            # conv_out.shape = [1, 14, 12, 8] with NHWC
            # layout tuner will transpose conv_out to
            # [1, 8, 14, 12] with NCHW before the following flatten op
            # because it flatten the C and H dimensions.
            out = flatten(conv_out)

        self.assertEqual(conv_out.shape, [1, 14, 12, 8])
        self.assertEqual(out.shape, [1, 112, 12])


if __name__ == '__main__':
    unittest.main()
