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

<<<<<<< HEAD
import os
import tempfile
import unittest

import numpy as np
from test_imperative_base import new_program_scope
=======
from __future__ import print_function

import os
import six
import unittest
import numpy as np
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

import paddle
import paddle.fluid as fluid
from paddle.fluid import core
<<<<<<< HEAD


def convolutional_neural_network(img):
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=20,
        pool_size=2,
        pool_stride=2,
        act="relu",
    )
    conv_pool_1 = paddle.static.nn.batch_norm(conv_pool_1)
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
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
=======
from test_imperative_base import new_program_scope
import tempfile


def convolutional_neural_network(img):
    conv_pool_1 = fluid.nets.simple_img_conv_pool(input=img,
                                                  filter_size=5,
                                                  num_filters=20,
                                                  pool_size=2,
                                                  pool_stride=2,
                                                  act="relu")
    conv_pool_1 = fluid.layers.batch_norm(conv_pool_1)
    conv_pool_2 = fluid.nets.simple_img_conv_pool(input=conv_pool_1,
                                                  filter_size=5,
                                                  num_filters=50,
                                                  pool_size=2,
                                                  pool_stride=2,
                                                  act="relu")
    prediction = fluid.layers.fc(input=conv_pool_2, size=10, act='softmax')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    return prediction


def static_train_net(img, label):
    prediction = convolutional_neural_network(img)

<<<<<<< HEAD
    loss = paddle.nn.functional.cross_entropy(
        input=prediction, label=label, reduction='none', use_softmax=False
    )
=======
    loss = fluid.layers.cross_entropy(input=prediction, label=label)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    avg_loss = paddle.mean(loss)

    optimizer = fluid.optimizer.SGD(learning_rate=0.001)
    optimizer.minimize(avg_loss)

    return prediction, avg_loss


class TestLoadStateDictFromSaveInferenceModel(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.seed = 90
        self.epoch_num = 1
        self.batch_size = 128
        self.batch_num = 10
<<<<<<< HEAD
        # enable static graph mode
=======
        # enable static mode
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        paddle.enable_static()

    def tearDown(self):
        self.temp_dir.cleanup()

    def train_and_save_model(self, only_params=False):
        with new_program_scope():
            startup_program = fluid.default_startup_program()
            main_program = fluid.default_main_program()

<<<<<<< HEAD
            img = fluid.data(
                name='img', shape=[None, 1, 28, 28], dtype='float32'
            )
=======
            img = fluid.data(name='img',
                             shape=[None, 1, 28, 28],
                             dtype='float32')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            label = fluid.data(name='label', shape=[None, 1], dtype='int64')

            prediction, avg_loss = static_train_net(img, label)

<<<<<<< HEAD
            place = (
                fluid.CUDAPlace(0)
                if core.is_compiled_with_cuda()
                else fluid.CPUPlace()
            )
=======
            place = fluid.CUDAPlace(
                0) if core.is_compiled_with_cuda() else fluid.CPUPlace()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            exe = fluid.Executor(place)

            feeder = fluid.DataFeeder(feed_list=[img, label], place=place)
            exe.run(startup_program)

<<<<<<< HEAD
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
=======
            train_reader = paddle.batch(paddle.reader.shuffle(
                paddle.dataset.mnist.train(), buf_size=100),
                                        batch_size=self.batch_size)

            for _ in range(0, self.epoch_num):
                for batch_id, data in enumerate(train_reader()):
                    exe.run(main_program,
                            feed=feeder.feed(data),
                            fetch_list=[avg_loss])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

                    if batch_id > self.batch_num:
                        break

            static_param_dict = {}
            for param in fluid.default_main_program().all_parameters():
                static_param_dict[param.name] = fluid.executor._fetch_var(
<<<<<<< HEAD
                    param.name
                )

            if only_params:
                fluid.io.save_params(
                    exe, self.save_dirname, filename=self.params_filename
                )
            else:
                fluid.io.save_inference_model(
                    self.save_dirname,
                    ["img"],
                    [prediction],
                    exe,
                    model_filename=self.model_filename,
                    params_filename=self.params_filename,
                )
=======
                    param.name)

            if only_params:
                fluid.io.save_params(exe,
                                     self.save_dirname,
                                     filename=self.params_filename)
            else:
                fluid.io.save_inference_model(
                    self.save_dirname, ["img"], [prediction],
                    exe,
                    model_filename=self.model_filename,
                    params_filename=self.params_filename)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        return static_param_dict

    def check_load_state_dict(self, orig_dict, load_dict):
<<<<<<< HEAD
        for var_name, value in orig_dict.items():
=======
        for var_name, value in six.iteritems(orig_dict):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            np.testing.assert_array_equal(value, load_dict[var_name])

    def test_load_default(self):
        self.save_dirname = os.path.join(
<<<<<<< HEAD
            self.temp_dir.name, "static_mnist.load_state_dict.default"
        )
=======
            self.temp_dir.name, "static_mnist.load_state_dict.default")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.model_filename = None
        self.params_filename = None
        orig_param_dict = self.train_and_save_model()

<<<<<<< HEAD
=======
        load_param_dict, _ = fluid.load_dygraph(self.save_dirname)
        self.check_load_state_dict(orig_param_dict, load_param_dict)

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        new_load_param_dict = paddle.load(self.save_dirname)
        self.check_load_state_dict(orig_param_dict, new_load_param_dict)

    def test_load_with_model_filename(self):
        self.save_dirname = os.path.join(
<<<<<<< HEAD
            self.temp_dir.name, "static_mnist.load_state_dict.model_filename"
        )
=======
            self.temp_dir.name, "static_mnist.load_state_dict.model_filename")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.model_filename = "static_mnist.model"
        self.params_filename = None
        orig_param_dict = self.train_and_save_model()

<<<<<<< HEAD
        new_load_param_dict = paddle.load(
            self.save_dirname, model_filename=self.model_filename
        )
=======
        load_param_dict, _ = fluid.load_dygraph(
            self.save_dirname, model_filename=self.model_filename)
        self.check_load_state_dict(orig_param_dict, load_param_dict)

        new_load_param_dict = paddle.load(self.save_dirname,
                                          model_filename=self.model_filename)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.check_load_state_dict(orig_param_dict, new_load_param_dict)

    def test_load_with_param_filename(self):
        self.save_dirname = os.path.join(
<<<<<<< HEAD
            self.temp_dir.name, "static_mnist.load_state_dict.param_filename"
        )
=======
            self.temp_dir.name, "static_mnist.load_state_dict.param_filename")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.model_filename = None
        self.params_filename = "static_mnist.params"
        orig_param_dict = self.train_and_save_model()

<<<<<<< HEAD
        new_load_param_dict = paddle.load(
            self.save_dirname, params_filename=self.params_filename
        )
=======
        load_param_dict, _ = fluid.load_dygraph(
            self.save_dirname, params_filename=self.params_filename)
        self.check_load_state_dict(orig_param_dict, load_param_dict)

        new_load_param_dict = paddle.load(self.save_dirname,
                                          params_filename=self.params_filename)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.check_load_state_dict(orig_param_dict, new_load_param_dict)

    def test_load_with_model_and_param_filename(self):
        self.save_dirname = os.path.join(
            self.temp_dir.name,
<<<<<<< HEAD
            "static_mnist.load_state_dict.model_and_param_filename",
        )
=======
            "static_mnist.load_state_dict.model_and_param_filename")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.model_filename = "static_mnist.model"
        self.params_filename = "static_mnist.params"
        orig_param_dict = self.train_and_save_model()

<<<<<<< HEAD
        new_load_param_dict = paddle.load(
            self.save_dirname,
            params_filename=self.params_filename,
            model_filename=self.model_filename,
        )
=======
        load_param_dict, _ = fluid.load_dygraph(
            self.save_dirname,
            params_filename=self.params_filename,
            model_filename=self.model_filename)
        self.check_load_state_dict(orig_param_dict, load_param_dict)

        new_load_param_dict = paddle.load(self.save_dirname,
                                          params_filename=self.params_filename,
                                          model_filename=self.model_filename)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.check_load_state_dict(orig_param_dict, new_load_param_dict)

    def test_load_state_dict_from_save_params(self):
        self.save_dirname = os.path.join(
<<<<<<< HEAD
            self.temp_dir.name, "static_mnist.load_state_dict.save_params"
        )
        self.params_filename = None
        orig_param_dict = self.train_and_save_model(True)

=======
            self.temp_dir.name, "static_mnist.load_state_dict.save_params")
        self.params_filename = None
        orig_param_dict = self.train_and_save_model(True)

        load_param_dict, _ = fluid.load_dygraph(self.save_dirname)
        self.check_load_state_dict(orig_param_dict, load_param_dict)

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        new_load_param_dict = paddle.load(self.save_dirname)
        self.check_load_state_dict(orig_param_dict, new_load_param_dict)


if __name__ == '__main__':
    unittest.main()
