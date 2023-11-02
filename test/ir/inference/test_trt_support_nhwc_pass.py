# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import shutil
import tempfile
import unittest

import numpy as np

import paddle
from paddle import inference, nn, static

paddle.enable_static()


class SimpleNet(nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2D(
            in_channels=4,
            out_channels=4,
            kernel_size=3,
            stride=2,
            padding=0,
            data_format='NHWC',
        )
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2D(
            in_channels=4,
            out_channels=2,
            kernel_size=3,
            stride=2,
            padding=0,
            data_format='NHWC',
        )
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2D(
            in_channels=2,
            out_channels=1,
            kernel_size=3,
            stride=2,
            padding=0,
            data_format='NHWC',
        )
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2D(
            in_channels=2,
            out_channels=1,
            kernel_size=3,
            stride=2,
            padding=0,
            data_format='NHWC',
        )
        self.relu4 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(729, 10)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        res = x
        x = self.conv3(x)
        x = self.relu3(x)
        res = self.conv4(res)
        res = self.relu4(res)
        x = x + res
        x = self.flatten(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x


class TRTNHWCConvertTest(unittest.TestCase):
    def setUp(self):
        self.place = paddle.CUDAPlace(0)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(
            self.temp_dir.name, 'inference_pass', 'nhwc_converter', ''
        )
        self.model_prefix = self.path + 'infer_model'
        self.set_args()

    def set_args(self):
        self.precision_mode = inference.PrecisionType.Float32

    def create_model(self):
        image = static.data(
            name='img', shape=[None, 224, 224, 4], dtype='float32'
        )
        predict = SimpleNet()(image)
        exe = paddle.static.Executor(self.place)
        exe.run(paddle.static.default_startup_program())
        paddle.static.save_inference_model(
            self.model_prefix, [image], [predict], exe
        )

    def create_predictor(self):
        config = paddle.inference.Config(
            self.model_prefix + '.pdmodel', self.model_prefix + '.pdiparams'
        )
        config.enable_memory_optim()
        config.enable_use_gpu(100, 0)
        config.enable_tensorrt_engine(
            workspace_size=1 << 30,
            max_batch_size=1,
            min_subgraph_size=3,
            precision_mode=self.precision_mode,
            use_static=False,
            use_calib_mode=False,
        )
        predictor = inference.create_predictor(config)
        return predictor

    def infer(self, predictor, img):
        input_names = predictor.get_input_names()
        for i, name in enumerate(input_names):
            input_tensor = predictor.get_input_handle(name)
            input_tensor.reshape(img[i].shape)
            input_tensor.copy_from_cpu(img[i].copy())
        predictor.run()
        results = []
        output_names = predictor.get_output_names()
        for i, name in enumerate(output_names):
            output_tensor = predictor.get_output_handle(name)
            output_data = output_tensor.copy_to_cpu()
            results.append(output_data)
        return results

    def test_nhwc_convert(self):
        self.create_model()
        predictor = self.create_predictor()
        img = np.ones((1, 224, 224, 4), dtype=np.float32)
        result = self.infer(predictor, img=[img])

    def tearDown(self):
        shutil.rmtree(self.path)


class TRTNHWCConvertAMPTest(TRTNHWCConvertTest):
    def set_args(self):
        self.precision_mode = inference.PrecisionType.Half

    def create_model(self):
        train_prog = paddle.static.Program()
        with paddle.static.program_guard(train_prog):
            with paddle.static.amp.fp16_guard():
                image = paddle.static.data(
                    name='image', shape=[None, 224, 224, 4], dtype='float32'
                )
                label = paddle.static.data(
                    name='label', shape=[None, 1], dtype='int64'
                )
                predict = SimpleNet()(image)
            cost = paddle.nn.functional.loss.cross_entropy(
                input=predict, label=label
            )
            avg_cost = paddle.mean(x=cost)
            optimizer = paddle.optimizer.Momentum(
                momentum=0.9,
                learning_rate=0.01,
                weight_decay=paddle.regularizer.L2Decay(4e-5),
            )
            optimizer = paddle.static.amp.decorate(
                optimizer,
                use_dynamic_loss_scaling=False,
                use_pure_fp16=False,
            )
            optimizer.minimize(avg_cost)
        val_prog = train_prog.clone(for_test=True)

        exe = paddle.static.Executor(self.place)
        exe.run(paddle.static.default_startup_program())
        paddle.static.save_inference_model(
            self.model_prefix, [image], [predict], exe, program=val_prog
        )


if __name__ == '__main__':
    unittest.main()
