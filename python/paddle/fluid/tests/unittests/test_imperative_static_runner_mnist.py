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

import unittest

import numpy as np
from test_imperative_base import new_program_scope

import paddle
import paddle.fluid as fluid
from paddle.fluid import core

LOADED_VAR_SUFFIX = ".load_0"


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
    prediction = fluid.layers.fc(input=conv_pool_2, size=10, act='softmax')
    return prediction


def static_train_net(img, label):
    prediction = convolutional_neural_network(img)

    loss = paddle.nn.functional.cross_entropy(
        input=prediction, label=label, reduction='none', use_softmax=False
    )
    avg_loss = paddle.mean(loss)

    optimizer = fluid.optimizer.SGD(learning_rate=0.001)
    optimizer.minimize(avg_loss)

    return prediction, avg_loss


class TestImperativeStaticModelRunnerMnist(unittest.TestCase):
    def setUp(self):
        self.seed = 90
        self.epoch_num = 1
        self.batch_size = 128
        self.batch_num = 50

    def reader_decorator(self, reader):
        def _reader_impl():
            for item in reader():
                image = np.array(item[0]).reshape(1, 28, 28)
                label = np.array(item[1]).astype('int64').reshape(1)
                yield image, label

        return _reader_impl

    def train_and_save_model(self):
        with new_program_scope():
            startup_program = fluid.default_startup_program()
            main_program = fluid.default_main_program()

            img = fluid.data(
                name='img', shape=[None, 1, 28, 28], dtype='float32'
            )
            label = fluid.data(name='label', shape=[None, 1], dtype='int64')

            prediction, avg_loss = static_train_net(img, label)

            place = (
                fluid.CUDAPlace(0)
                if core.is_compiled_with_cuda()
                else fluid.CPUPlace()
            )

            exe = fluid.Executor(place)

            feeder = fluid.DataFeeder(feed_list=[img, label], place=place)
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

            fluid.io.save_inference_model(
                self.save_dirname,
                ["img"],
                [prediction],
                exe,
                model_filename=self.model_filename,
                params_filename=self.params_filename,
                clip_extra=False,
            )

    def load_and_train_dygraph(self):
        place = (
            fluid.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else fluid.CPUPlace()
        )
        with fluid.dygraph.guard(place):
            fluid.default_startup_program().random_seed = self.seed
            fluid.default_main_program().random_seed = self.seed
            fluid.set_flags({'FLAGS_sort_sum_gradient': True})

            mnist = fluid.dygraph.static_runner.StaticModelRunner(
                model_dir=self.save_dirname,
                model_filename=self.model_filename,
                params_filename=self.params_filename,
            )

            suffix_varname_dict = mnist._program_holder_dict[
                'forward'
            ]._suffix_varname_dict
            dict_old_new = {v: k for k, v in suffix_varname_dict.items()}
            dy_param_init_value = {}
            for param in mnist.parameters():
                dy_param_init_value[param.name] = param.numpy()

            sgd = fluid.optimizer.SGD(
                learning_rate=0.001, parameter_list=mnist.parameters()
            )

            train_reader = paddle.batch(
                self.reader_decorator(paddle.dataset.mnist.train()),
                batch_size=self.batch_size,
                drop_last=True,
            )
            train_loader = fluid.io.DataLoader.from_generator(capacity=10)
            train_loader.set_sample_list_generator(train_reader, places=place)

            mnist.train()

            for epoch in range(self.epoch_num):
                for batch_id, data in enumerate(train_loader()):
                    img = data[0]
                    label = data[1]
                    label.stop_gradient = True

                    cost = mnist(img)

                    loss = paddle.nn.functional.cross_entropy(
                        cost, label, reduction='none', use_softmax=False
                    )
                    avg_loss = paddle.mean(loss)

                    avg_loss.backward()
                    sgd.minimize(avg_loss)
                    mnist.clear_gradients()

                    if batch_id >= self.batch_num:
                        break

            dy_x_data = img.numpy()
            dy_out = avg_loss.numpy()

            dy_param_value = {}
            for param in mnist.parameters():
                dy_param_value[param.name] = param.numpy()

        return (
            dy_x_data,
            dy_out,
            dy_param_init_value,
            dy_param_value,
            dict_old_new,
        )

    def load_and_train_static(self):
        with new_program_scope():
            fluid.default_startup_program().random_seed = self.seed
            fluid.default_main_program().random_seed = self.seed

            img = fluid.data(
                name='img', shape=[None, 1, 28, 28], dtype='float32'
            )
            label = fluid.data(name='label', shape=[None, 1], dtype='int64')

            prediction, avg_loss = static_train_net(img, label)

            place = (
                fluid.CUDAPlace(0)
                if core.is_compiled_with_cuda()
                else fluid.CPUPlace()
            )

            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())

            fluid.io.load_params(
                exe,
                self.save_dirname,
                main_program=fluid.default_main_program(),
                filename=self.params_filename,
            )

            static_param_init_value = {}
            static_param_name_list = []
            for param in fluid.default_main_program().all_parameters():
                static_param_name_list.append(param.name)
                static_param_init_value[param.name] = fluid.executor._fetch_var(
                    param.name
                )

            train_reader = paddle.batch(
                self.reader_decorator(paddle.dataset.mnist.train()),
                batch_size=self.batch_size,
                drop_last=True,
            )

            for epoch in range(self.epoch_num):
                for batch_id, data in enumerate(train_reader()):
                    static_x_data = np.array([x[0] for x in data])
                    y_data = np.array([x[1] for x in data]).reshape(
                        [self.batch_size, 1]
                    )

                    fetch_list = [avg_loss.name]
                    fetch_list.extend(static_param_name_list)

                    out = exe.run(
                        fluid.default_main_program(),
                        feed={"img": static_x_data, "label": y_data},
                        fetch_list=fetch_list,
                    )

                    if batch_id >= self.batch_num:
                        break

            static_param_value = {}
            static_out = out[0]
            for i in range(1, len(out)):
                static_param_value[static_param_name_list[i - 1]] = out[i]

        return (
            static_x_data,
            static_out,
            static_param_init_value,
            static_param_value,
        )

    def load_and_infer_dygraph(self):
        place = (
            fluid.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else fluid.CPUPlace()
        )
        with fluid.dygraph.guard(place):
            fluid.default_main_program().random_seed = self.seed

            mnist = fluid.dygraph.static_runner.StaticModelRunner(
                model_dir=self.save_dirname, model_filename=self.model_filename
            )

            train_reader = paddle.batch(
                self.reader_decorator(paddle.dataset.mnist.test()),
                batch_size=self.batch_size,
                drop_last=True,
            )
            train_loader = fluid.io.DataLoader.from_generator(capacity=10)
            train_loader.set_sample_list_generator(train_reader, places=place)

            mnist.eval()

            for batch_id, data in enumerate(train_loader()):
                img = data[0]
                cost = mnist(img)

                if batch_id >= 1:
                    break

            dy_x_data = img.numpy()
            dy_out = cost.numpy()

        return dy_x_data, dy_out

    def load_and_infer_static(self):
        with new_program_scope():
            place = (
                fluid.CUDAPlace(0)
                if core.is_compiled_with_cuda()
                else fluid.CPUPlace()
            )

            exe = fluid.Executor(place)
            [
                infer_program,
                feed_target_names,
                fetch_targets,
            ] = fluid.io.load_inference_model(self.save_dirname, exe)
            infer_program.random_seed = self.seed

            train_reader = paddle.batch(
                self.reader_decorator(paddle.dataset.mnist.test()),
                batch_size=self.batch_size,
                drop_last=True,
            )

            for batch_id, data in enumerate(train_reader()):
                static_x_data = np.array([x[0] for x in data])
                out = exe.run(
                    infer_program,
                    feed={feed_target_names[0]: static_x_data},
                    fetch_list=fetch_targets,
                )

                if batch_id >= 1:
                    break

            static_param_value = {}
            static_out = out[0]

        return static_x_data, static_out

    def test_mnist_train_no_params_filename(self):
        self.save_dirname = "mnist.inference.model.noname"
        self.model_filename = None
        self.params_filename = None
        # Phase 1. run and save static model
        self.train_and_save_model()

        # Phase 2. load model & train dygraph

        (
            dy_x_data,
            dy_out,
            dy_param_init_value,
            dy_param_value,
            dict_old_new_init,
        ) = self.load_and_train_dygraph()

        (
            static_x_data,
            static_out,
            static_param_init_value,
            static_param_value,
        ) = self.load_and_train_static()

        # Phase 3. compare
        np.testing.assert_array_equal(static_x_data, dy_x_data)

        for key, value in static_param_init_value.items():
            key = dict_old_new_init[key]
            np.testing.assert_array_equal(value, dy_param_init_value[key])

        # np.testing.assert_array_almost_equal(static_out, dy_out)
        np.testing.assert_allclose(static_out, dy_out, rtol=1e-05, atol=1e-4)

        for key, value in static_param_value.items():
            key = dict_old_new_init[key]
            np.testing.assert_allclose(
                value, dy_param_value[key], rtol=1e-05, atol=1e-4
            )

    def test_mnist_train_with_params_filename(self):
        self.save_dirname = "mnist.inference.model"
        self.model_filename = "mnist.model"
        self.params_filename = "mnist.params"
        # Phase 1. run and save static model
        self.train_and_save_model()

        # Phase 2. load model & train dygraph
        (
            dy_x_data,
            dy_out,
            dy_param_init_value,
            dy_param_value,
            dict_old_new_init,
        ) = self.load_and_train_dygraph()

        (
            static_x_data,
            static_out,
            static_param_init_value,
            static_param_value,
        ) = self.load_and_train_static()

        # Phase 3. compare
        np.testing.assert_array_equal(static_x_data, dy_x_data)
        for key, value in static_param_init_value.items():
            key = dict_old_new_init[key]
            np.testing.assert_array_equal(value, dy_param_init_value[key])

        # np.testing.assert_array_almost_equal(static_out, dy_out)
        np.testing.assert_allclose(static_out, dy_out, rtol=1e-05, atol=1e-4)

        for key, value in static_param_value.items():
            key = dict_old_new_init[key]
            np.testing.assert_allclose(
                value, dy_param_value[key], rtol=1e-05, atol=1e-4
            )

    def test_mnist_infer_no_params_filename(self):
        self.save_dirname = "mnist.inference.model.noname"
        self.model_filename = None
        self.params_filename = None
        # Phase 1. run and save static model
        self.train_and_save_model()

        # Phase 2. load model & train dygraph
        dy_x_data, dy_out = self.load_and_infer_dygraph()

        static_x_data, static_out = self.load_and_infer_static()

        # Phase 3. compare
        np.testing.assert_array_equal(static_x_data, dy_x_data)

        np.testing.assert_array_almost_equal(static_out, dy_out)
        np.testing.assert_allclose(static_out, dy_out, rtol=1e-05, atol=1e-4)


if __name__ == '__main__':
    unittest.main()
