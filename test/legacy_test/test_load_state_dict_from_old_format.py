# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import sys
import tempfile
import unittest

sys.path.append("../../legacy_test")
import nets
import numpy as np
from test_imperative_base import new_program_scope

import paddle
from paddle import base
from paddle.base import core
from paddle.framework import in_pir_mode


def convolutional_neural_network(img):
    conv_pool_1 = nets.simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=20,
        pool_size=2,
        pool_stride=2,
        act="relu",
    )
    conv_pool_1 = paddle.static.nn.batch_norm(conv_pool_1)
    conv_pool_2 = nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act="relu",
    )
    prediction = paddle.static.nn.fc(
        x=conv_pool_2, size=10, activation='softmax'
    )
    return prediction


def simple_img_conv_pool(
    input,
    in_channels,
    out_channels,
    kernel_size,
    pool_size,
    pool_stride,
    pool_padding=0,
    pool_type='max',
    global_pooling=False,
    conv_stride=1,
    conv_padding=0,
    conv_dilation=1,
    conv_groups=1,
    param_attr=None,
    bias_attr=None,
    act=None,
    use_cudnn=True,
):
    conv = paddle.nn.Conv2D(
        in_channels,
        out_channels,
        kernel_size,
        stride=conv_stride,
        padding=conv_padding,
        dilation=conv_dilation,
        groups=conv_groups,
        bias_attr=bias_attr,
    )
    conv_out = conv(input)
    if pool_type == 'max':
        pool_out = paddle.nn.functional.max_pool2d(
            x=conv_out,
            kernel_size=pool_size,
            stride=pool_stride,
            padding=pool_padding,
        )
    else:
        pool_out = paddle.nn.functional.avg_pool2d(
            x=conv_out,
            kernel_size=pool_size,
            stride=pool_stride,
            padding=pool_padding,
        )
    return pool_out


def convolutional_neural_network_pir(img):
    conv_pool_1 = simple_img_conv_pool(
        input=img,
        in_channels=1,
        out_channels=20,
        kernel_size=5,
        pool_size=2,
        pool_stride=2,
    )
    conv_pool_1 = paddle.nn.BatchNorm(20)(conv_pool_1)
    conv_pool_2 = simple_img_conv_pool(
        input=conv_pool_1,
        in_channels=20,
        out_channels=50,
        kernel_size=5,
        pool_size=2,
        pool_stride=2,
    )
    prediction = paddle.static.nn.fc(x=conv_pool_2, size=10, activation='relu')
    return prediction


def static_train_net(img, label):
    if in_pir_mode():
        prediction = convolutional_neural_network_pir(img)
    else:
        prediction = convolutional_neural_network(img)

    loss = paddle.nn.functional.cross_entropy(
        input=prediction, label=label, reduction='none', use_softmax=False
    )
    avg_loss = paddle.mean(loss)

    optimizer = paddle.optimizer.SGD(learning_rate=0.001)
    optimizer.minimize(avg_loss)

    return prediction, avg_loss


class TestLoadStateDictFromSaveInferenceModel(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.seed = 90
        self.epoch_num = 1
        self.batch_size = 128
        self.batch_num = 10
        # enable static graph mode
        paddle.enable_static()

    def tearDown(self):
        self.temp_dir.cleanup()

    def train_and_save_model(self):
        with new_program_scope():
            startup_program = paddle.static.default_startup_program()
            main_program = paddle.static.default_main_program()

            img = paddle.static.data(
                name='img', shape=[None, 1, 28, 28], dtype='float32'
            )
            label = paddle.static.data(
                name='label', shape=[None, 1], dtype='int64'
            )

            prediction, avg_loss = static_train_net(img, label)

            place = (
                base.CUDAPlace(0)
                if core.is_compiled_with_cuda()
                else base.CPUPlace()
            )

            exe = base.Executor(place)

            feeder = base.DataFeeder(feed_list=[img, label], place=place)
            exe.run(startup_program)

            train_reader = paddle.batch(
                paddle.reader.shuffle(
                    paddle.dataset.mnist.train(), buf_size=100
                ),
                batch_size=self.batch_size,
            )

            for _ in range(0, self.epoch_num):
                for batch_id, data in enumerate(train_reader()):
                    exe.run(
                        main_program,
                        feed=feeder.feed(data),
                        fetch_list=[avg_loss],
                    )

                    if batch_id > self.batch_num:
                        break

            static_param_dict = {}
            for param in main_program.global_block().all_parameters():
                static_param_dict[param.name] = base.executor._fetch_var(
                    param.name
                )

            paddle.static.io.save_inference_model(
                self.save_dirname,
                [img],
                [prediction],
                exe,
                program=main_program,
            )

        return static_param_dict

    def check_load_state_dict(self, orig_dict, load_dict):
        for var_name, value in orig_dict.items():
            np.testing.assert_array_equal(value, load_dict[var_name])

    def test_load_default(self):
        with paddle.base.unique_name.guard():
            self.save_dirname = os.path.join(
                self.temp_dir.name, "static_mnist.load_state_dict.default"
            )
            self.model_filename = None
            self.params_filename = None
            orig_param_dict = self.train_and_save_model()

            new_load_param_dict = paddle.load(self.save_dirname)
            self.check_load_state_dict(orig_param_dict, new_load_param_dict)

    def test_load_with_model_filename(self):
        with paddle.base.unique_name.guard():
            self.save_dirname = os.path.join(
                self.temp_dir.name,
                "static_mnist.load_state_dict.model_filename",
            )
            self.model_filename = "static_mnist.model"
            self.params_filename = None
            orig_param_dict = self.train_and_save_model()

            new_load_param_dict = paddle.load(
                self.save_dirname, model_filename=self.model_filename
            )
            self.check_load_state_dict(orig_param_dict, new_load_param_dict)

    def test_load_with_param_filename(self):
        with paddle.base.unique_name.guard():
            self.save_dirname = os.path.join(
                self.temp_dir.name,
                "static_mnist.load_state_dict.param_filename",
            )
            self.model_filename = None
            self.params_filename = "static_mnist.params"
            orig_param_dict = self.train_and_save_model()

            new_load_param_dict = paddle.load(
                self.save_dirname, params_filename=self.params_filename
            )
            self.check_load_state_dict(orig_param_dict, new_load_param_dict)

    def test_load_with_model_and_param_filename(self):
        with paddle.base.unique_name.guard():
            self.save_dirname = os.path.join(
                self.temp_dir.name,
                "static_mnist.load_state_dict.model_and_param_filename",
            )
            self.model_filename = "static_mnist.model"
            self.params_filename = "static_mnist.params"
            orig_param_dict = self.train_and_save_model()

            new_load_param_dict = paddle.load(
                self.save_dirname,
                params_filename=self.params_filename,
                model_filename=self.model_filename,
            )
            self.check_load_state_dict(orig_param_dict, new_load_param_dict)


if __name__ == '__main__':
    unittest.main()
