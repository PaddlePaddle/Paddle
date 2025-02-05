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

import os
import tempfile
import unittest
from time import time

import numpy as np
from dygraph_to_static_utils import (
    Dy2StTestBase,
    test_default_and_pir,
)
from predictor_utils import PredictorTools

import paddle
from paddle import base
from paddle.framework import use_pir_api
from paddle.jit.pir_translated_layer import PIR_INFER_MODEL_SUFFIX
from paddle.jit.translated_layer import INFER_MODEL_SUFFIX, INFER_PARAMS_SUFFIX
from paddle.nn import Linear
from paddle.optimizer import Adam

SEED = 2020

if paddle.is_compiled_with_cuda():
    paddle.set_flags({'FLAGS_cudnn_deterministic': True})


class SimpleImgConvPool(paddle.nn.Layer):
    def __init__(
        self,
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
        use_cudnn=True,
        param_attr=None,
        bias_attr=None,
    ):
        super().__init__()

        self._conv2d = paddle.nn.Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=conv_stride,
            padding=conv_padding,
            dilation=conv_dilation,
            groups=conv_groups,
            weight_attr=None,
            bias_attr=None,
        )

        self._pool2d = paddle.nn.MaxPool2D(
            kernel_size=pool_size,
            stride=pool_stride,
            padding=pool_padding,
        )

    def forward(self, inputs):
        x = self._conv2d(inputs)
        x = self._pool2d(x)
        return x


class MNIST(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

        self._simple_img_conv_pool_1 = SimpleImgConvPool(
            1, 20, 5, 2, 2, act="relu"
        )

        self._simple_img_conv_pool_2 = SimpleImgConvPool(
            20, 50, 5, 2, 2, act="relu"
        )

        self.pool_2_shape = 50 * 4 * 4
        SIZE = 10
        scale = (2.0 / (self.pool_2_shape**2 * SIZE)) ** 0.5
        self._fc = Linear(
            self.pool_2_shape,
            10,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Normal(mean=0.0, std=scale)
            ),
        )

    def forward(self, inputs, label=None):
        x = self.inference(inputs)
        if label is not None:
            acc = paddle.static.accuracy(input=x, label=label)
            loss = paddle.nn.functional.cross_entropy(
                x, label, reduction='none', use_softmax=False
            )
            avg_loss = paddle.mean(loss)

            return x, acc, avg_loss
        else:
            return x

    def inference(self, inputs):
        x = self._simple_img_conv_pool_1(inputs)
        x = self._simple_img_conv_pool_2(x)
        x = paddle.reshape(x, shape=[-1, self.pool_2_shape])
        x = self._fc(x)
        x = paddle.nn.functional.softmax(x)
        return x


class TestMNIST(Dy2StTestBase):
    def setUp(self):
        self.epoch_num = 1
        self.batch_size = 64
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )
        self.train_reader = paddle.batch(
            paddle.dataset.mnist.train(),
            batch_size=self.batch_size,
            drop_last=True,
        )
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()


class TestMNISTWithToStatic(TestMNIST):
    """
    Tests model if doesn't change the layers while decorated
    by `dygraph_to_static_output`. In this case, everything should
    still works if model is trained in dygraph mode.
    """

    def train_static(self):
        return self.train(to_static=True)

    def train_dygraph(self):
        return self.train(to_static=False)

    @test_default_and_pir
    def test_mnist_to_static(self):
        dygraph_loss = self.train_dygraph()
        static_loss = self.train_static()
        np.testing.assert_allclose(
            dygraph_loss,
            static_loss,
            rtol=1e-05,
            err_msg=f'dygraph is {dygraph_loss}\n static_res is \n{static_loss}',
        )

    @test_default_and_pir
    def test_mnist_declarative_cpu_vs_mkldnn(self):
        dygraph_loss_cpu = self.train_dygraph()
        paddle.set_flags({'FLAGS_use_mkldnn': True})
        try:
            dygraph_loss_mkldnn = self.train_dygraph()
        finally:
            paddle.set_flags({'FLAGS_use_mkldnn': False})
        np.testing.assert_allclose(
            dygraph_loss_cpu,
            dygraph_loss_mkldnn,
            rtol=1e-05,
            err_msg=f'cpu dygraph is {dygraph_loss_cpu}\n mkldnn dygraph is \n{dygraph_loss_mkldnn}',
        )

    def train(self, to_static=False):
        loss_data = []
        paddle.seed(SEED)
        mnist = MNIST()
        if to_static:
            mnist = paddle.jit.to_static(mnist, full_graph=True)
        adam = Adam(learning_rate=0.001, parameters=mnist.parameters())

        for epoch in range(self.epoch_num):
            start = time()
            for batch_id, data in enumerate(self.train_reader()):
                dy_x_data = np.array(
                    [x[0].reshape(1, 28, 28) for x in data]
                ).astype('float32')
                y_data = (
                    np.array([x[1] for x in data])
                    .astype('int64')
                    .reshape(-1, 1)
                )

                img = paddle.to_tensor(dy_x_data)
                label = paddle.to_tensor(y_data)

                label.stop_gradient = True
                prediction, acc, avg_loss = mnist(img, label=label)
                avg_loss.backward()

                adam.minimize(avg_loss)
                loss_data.append(float(avg_loss))
                # save checkpoint
                mnist.clear_gradients()
                if batch_id % 10 == 0:
                    print(
                        f"Loss at epoch {epoch} step {batch_id}: loss: {avg_loss.numpy()}, acc: {acc.numpy()}, cost: {time() - start}"
                    )
                    start = time()
                if batch_id == 50:
                    mnist.eval()
                    prediction, acc, avg_loss = mnist(img, label)
                    loss_data.append(float(avg_loss))
                    # new save load check
                    self.check_jit_save_load(
                        mnist,
                        [dy_x_data],
                        [img, label],
                        to_static,
                        prediction,
                        0,
                        [img.name],
                    )
                    break
        return loss_data

    def check_jit_save_load(
        self,
        model,
        inputs,
        input_spec,
        to_static,
        gt_out,
        gt_out_index,
        input_names_after_prune,
    ):
        if to_static:
            infer_model_path = os.path.join(
                self.temp_dir.name, 'test_mnist_inference_model_by_jit_save'
            )
            model_save_dir = os.path.join(self.temp_dir.name, 'inference')
            model_save_prefix = os.path.join(model_save_dir, 'mnist')
            MODEL_SUFFIX = (
                PIR_INFER_MODEL_SUFFIX if use_pir_api() else INFER_MODEL_SUFFIX
            )
            model_filename = "mnist" + MODEL_SUFFIX
            params_filename = "mnist" + INFER_PARAMS_SUFFIX
            paddle.jit.save(
                layer=model,
                path=model_save_prefix,
                input_spec=input_spec,
                output_spec=[gt_out_index] if use_pir_api() else [gt_out],
                input_names_after_prune=input_names_after_prune,
            )
            # load in static graph mode
            static_infer_out = self.jit_load_and_run_inference_static(
                model_save_dir, model_filename, params_filename, inputs
            )
            np.testing.assert_allclose(
                gt_out.numpy(), static_infer_out, rtol=1e-05
            )
            # load in dygraph mode
            dygraph_infer_out = self.jit_load_and_run_inference_dygraph(
                model_save_prefix, inputs
            )
            np.testing.assert_allclose(
                gt_out.numpy(), dygraph_infer_out, rtol=1e-05
            )

            # load in Paddle-Inference
            predictor_infer_out = (
                self.predictor_load_and_run_inference_analysis(
                    model_save_dir, model_filename, params_filename, inputs
                )
            )
            np.testing.assert_allclose(
                gt_out.numpy(), predictor_infer_out, rtol=1e-05
            )

    def jit_load_and_run_inference_static(
        self, model_path, model_filename, params_filename, inputs
    ):
        paddle.enable_static()
        exe = base.Executor(self.place)
        [
            inference_program,
            feed_target_names,
            fetch_targets,
        ] = paddle.static.io.load_inference_model(
            path_prefix=model_path,
            executor=exe,
            model_filename=model_filename,
            params_filename=params_filename,
        )
        assert len(inputs) == len(feed_target_names)
        results = exe.run(
            inference_program,
            feed=dict(zip(feed_target_names, inputs)),
            fetch_list=fetch_targets,
        )
        paddle.disable_static()

        return np.array(results[0])

    def jit_load_and_run_inference_dygraph(self, model_path, inputs):
        infer_net = paddle.jit.load(model_path)
        pred = infer_net(inputs[0])
        return pred.numpy()

    def predictor_load_and_run_inference_analysis(
        self, model_path, model_filename, params_filename, inputs
    ):
        output = PredictorTools(
            model_path, model_filename, params_filename, inputs
        )
        (out,) = output()
        return out


if __name__ == "__main__":
    unittest.main()
