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

import logging
import math
import os
import tempfile
import time
import unittest

import numpy as np
from dygraph_to_static_utils import (
    Dy2StTestBase,
    enable_to_static_guard,
    test_default_and_pir,
)
from predictor_utils import PredictorTools

import paddle
from paddle import base
from paddle.framework import use_pir_api
from paddle.jit.pir_translated_layer import PIR_INFER_MODEL_SUFFIX
from paddle.jit.translated_layer import INFER_MODEL_SUFFIX, INFER_PARAMS_SUFFIX
from paddle.nn import BatchNorm, Linear
from paddle.static import InputSpec

SEED = 2020
np.random.seed(SEED)

BATCH_SIZE = 8
EPOCH_NUM = 1
PRINT_STEP = 2
STEP_NUM = 10

place = (
    paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() else paddle.CPUPlace()
)

# Note: Set True to eliminate randomness.
#     1. For one operation, cuDNN has several algorithms,
#        some algorithm results are non-deterministic, like convolution algorithms.
if paddle.is_compiled_with_cuda():
    paddle.set_flags({'FLAGS_cudnn_deterministic': True})

train_parameters = {
    "learning_strategy": {
        "name": "cosine_decay",
        "batch_size": BATCH_SIZE,
        "epochs": [40, 80, 100],
        "steps": [0.1, 0.01, 0.001, 0.0001],
    },
    "lr": 0.0125,
    "total_images": 6149,
    "momentum_rate": 0.9,
    "l2_decay": 1.2e-4,
    "num_epochs": 1,
}


def optimizer_setting(params, parameter_list):
    ls = params["learning_strategy"]
    if "total_images" not in params:
        total_images = 6149
    else:
        total_images = params["total_images"]

    batch_size = ls["batch_size"]
    l2_decay = params["l2_decay"]
    momentum_rate = params["momentum_rate"]

    step = int(math.ceil(float(total_images) / batch_size))
    bd = [step * e for e in ls["epochs"]]
    lr = params["lr"]
    num_epochs = params["num_epochs"]
    optimizer = paddle.optimizer.Momentum(
        learning_rate=paddle.optimizer.lr.CosineAnnealingDecay(
            learning_rate=lr, T_max=num_epochs
        ),
        momentum=momentum_rate,
        weight_decay=paddle.regularizer.L2Decay(l2_decay),
        parameters=parameter_list,
    )

    return optimizer


class ConvBNLayer(paddle.nn.Layer):
    def __init__(
        self,
        num_channels,
        num_filters,
        filter_size,
        stride=1,
        groups=1,
        act=None,
    ):
        super().__init__()

        self._conv = paddle.nn.Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            bias_attr=False,
        )

        self._batch_norm = BatchNorm(num_filters, act=act)

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)

        return y


class SqueezeExcitation(paddle.nn.Layer):
    def __init__(self, num_channels, reduction_ratio):
        super().__init__()
        self._num_channels = num_channels
        self._pool = paddle.nn.AdaptiveAvgPool2D(1)
        stdv = 1.0 / math.sqrt(num_channels * 1.0)
        self._fc = Linear(
            num_channels,
            num_channels // reduction_ratio,
            weight_attr=base.ParamAttr(
                initializer=paddle.nn.initializer.Uniform(-stdv, stdv)
            ),
        )
        stdv = 1.0 / math.sqrt(num_channels / 16.0 * 1.0)
        self._excitation = Linear(
            num_channels // reduction_ratio,
            num_channels,
            weight_attr=base.ParamAttr(
                initializer=paddle.nn.initializer.Uniform(-stdv, stdv)
            ),
        )

    def forward(self, input):
        y = self._pool(input)
        y = paddle.reshape(y, shape=[-1, self._num_channels])
        y = self._fc(y)
        y = paddle.nn.functional.relu(y)
        y = self._excitation(y)
        y = paddle.nn.functional.sigmoid(y)
        y = paddle.tensor.math._multiply_with_axis(x=input, y=y, axis=0)
        return y


class BottleneckBlock(paddle.nn.Layer):
    def __init__(
        self,
        num_channels,
        num_filters,
        stride,
        cardinality,
        reduction_ratio,
        shortcut=True,
    ):
        super().__init__()

        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1,
            act="relu",
        )
        self.conv1 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            groups=cardinality,
            act="relu",
        )
        self.conv2 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters * 2,
            filter_size=1,
            act=None,
        )

        self.scale = SqueezeExcitation(
            num_channels=num_filters * 2, reduction_ratio=reduction_ratio
        )

        if not shortcut:
            self.short = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters * 2,
                filter_size=1,
                stride=stride,
            )

        self.shortcut = shortcut

        self._num_channels_out = num_filters * 2

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)
        scale = self.scale(conv2)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        y = paddle.nn.functional.relu(paddle.add(x=short, y=scale))
        return y


class SeResNeXt(paddle.nn.Layer):
    def __init__(self, layers=50, class_dim=102):
        super().__init__()

        self.layers = layers
        supported_layers = [50, 101, 152]
        assert (
            layers in supported_layers
        ), f"supported layers are {supported_layers} but input layer is {layers}"

        if layers == 50:
            cardinality = 32
            reduction_ratio = 16
            depth = [3, 4, 6, 3]
            num_filters = [128, 256, 512, 1024]
            self.conv0 = ConvBNLayer(
                num_channels=3,
                num_filters=64,
                filter_size=7,
                stride=2,
                act='relu',
            )
            self.pool = paddle.nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        elif layers == 101:
            cardinality = 32
            reduction_ratio = 16
            depth = [3, 4, 23, 3]
            num_filters = [128, 256, 512, 1024]
            self.conv0 = ConvBNLayer(
                num_channels=3,
                num_filters=64,
                filter_size=7,
                stride=2,
                act='relu',
            )
            self.pool = paddle.nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        elif layers == 152:
            cardinality = 64
            reduction_ratio = 16
            depth = [3, 8, 36, 3]
            num_filters = [128, 256, 512, 1024]
            self.conv0 = ConvBNLayer(
                num_channels=3,
                num_filters=64,
                filter_size=3,
                stride=2,
                act='relu',
            )
            self.conv1 = ConvBNLayer(
                num_channels=64,
                num_filters=64,
                filter_size=3,
                stride=1,
                act='relu',
            )
            self.conv2 = ConvBNLayer(
                num_channels=64,
                num_filters=128,
                filter_size=3,
                stride=1,
                act='relu',
            )
            self.pool = paddle.nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

        self.bottleneck_block_list = []
        num_channels = 64
        if layers == 152:
            num_channels = 128
        for block in range(len(depth)):
            shortcut = False
            for i in range(depth[block]):
                bottleneck_block = self.add_sublayer(
                    f'bb_{block}_{i}',
                    BottleneckBlock(
                        num_channels=num_channels,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        cardinality=cardinality,
                        reduction_ratio=reduction_ratio,
                        shortcut=shortcut,
                    ),
                )
                num_channels = bottleneck_block._num_channels_out
                self.bottleneck_block_list.append(bottleneck_block)
                shortcut = True
        self.pool2d_avg = paddle.nn.AdaptiveAvgPool2D(1)
        stdv = 1.0 / math.sqrt(2048 * 1.0)

        self.pool2d_avg_output = num_filters[len(num_filters) - 1] * 2 * 1 * 1

        self.dropout = paddle.nn.Dropout(p=0.5, mode="downscale_in_infer")

        self.out = Linear(
            self.pool2d_avg_output,
            class_dim,
            weight_attr=base.param_attr.ParamAttr(
                initializer=paddle.nn.initializer.Uniform(-stdv, stdv)
            ),
        )

    def forward(self, inputs, label):
        if self.layers == 50 or self.layers == 101:
            y = self.conv0(inputs)
            y = self.pool(y)
        elif self.layers == 152:
            y = self.conv0(inputs)
            y = self.conv1(y)
            y = self.conv2(y)
            y = self.pool(y)

        for bottleneck_block in self.bottleneck_block_list:
            y = bottleneck_block(y)

        y = self.pool2d_avg(y)
        y = self.dropout(y)
        y = paddle.reshape(y, shape=[-1, self.pool2d_avg_output])
        out = self.out(y)

        softmax_out = paddle.nn.functional.softmax(out)
        loss = paddle.nn.functional.cross_entropy(
            input=softmax_out, label=label, reduction='none', use_softmax=False
        )
        avg_loss = paddle.mean(x=loss)

        acc_top1 = paddle.static.accuracy(input=softmax_out, label=label, k=1)
        acc_top5 = paddle.static.accuracy(input=softmax_out, label=label, k=5)
        return out, avg_loss, acc_top1, acc_top5


class TestSeResnet(Dy2StTestBase):
    def setUp(self):
        self.train_reader = paddle.batch(
            paddle.dataset.flowers.train(use_xmap=False, cycle=True),
            batch_size=BATCH_SIZE,
            drop_last=True,
        )
        self.temp_dir = tempfile.TemporaryDirectory()

        self.model_save_dir = os.path.join(self.temp_dir.name, "inference")
        self.model_save_prefix = os.path.join(
            self.temp_dir.name, "inference/se_resnet"
        )
        self.model_filename = "se_resnet" + INFER_MODEL_SUFFIX
        self.pir_model_filename = "se_resnet" + PIR_INFER_MODEL_SUFFIX
        self.params_filename = "se_resnet" + INFER_PARAMS_SUFFIX
        self.dy_state_dict_save_path = os.path.join(
            self.temp_dir.name, "se_resnet.dygraph"
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def train(self, train_reader, to_static):
        np.random.seed(SEED)

        with base.dygraph.guard(place):
            paddle.seed(SEED)
            paddle.framework.random._manual_program_seed(SEED)
            se_resnext = SeResNeXt()
            se_resnext = paddle.jit.to_static(se_resnext, full_graph=True)
            optimizer = optimizer_setting(
                train_parameters, se_resnext.parameters()
            )

            for epoch_id in range(EPOCH_NUM):
                total_loss = 0.0
                total_acc1 = 0.0
                total_acc5 = 0.0
                total_sample = 0
                step_idx = 0
                speed_list = []
                for step_id, data in enumerate(train_reader()):
                    dy_x_data = np.array(
                        [x[0].reshape(3, 224, 224) for x in data]
                    ).astype('float32')
                    y_data = (
                        np.array([x[1] for x in data])
                        .astype('int64')
                        .reshape(BATCH_SIZE, 1)
                    )

                    img = paddle.to_tensor(dy_x_data)
                    label = paddle.to_tensor(y_data)
                    label.stop_gradient = True

                    pred, avg_loss, acc_top1, acc_top5 = se_resnext(img, label)

                    dy_out = avg_loss.numpy()
                    avg_loss.backward()

                    optimizer.minimize(avg_loss)
                    se_resnext.clear_gradients()

                    lr = optimizer._global_learning_rate().numpy()
                    total_loss += dy_out
                    total_acc1 += acc_top1.numpy()
                    total_acc5 += acc_top5.numpy()
                    total_sample += 1
                    if step_id % PRINT_STEP == 0:
                        if step_id == 0:
                            logging.info(
                                f"epoch {epoch_id} | step {step_id}, "
                                f"loss {total_loss / total_sample:0.3f}, "
                                f"acc1 {total_acc1 / total_sample:0.3f}, "
                                f"acc5 {total_acc5 / total_sample:0.3f}"
                            )
                            avg_batch_time = time.time()
                        else:
                            speed = PRINT_STEP / (time.time() - avg_batch_time)
                            speed_list.append(speed)
                            logging.info(
                                f"epoch {epoch_id} | step {step_id}, "
                                f"loss {total_loss / total_sample:0.3f}, "
                                f"acc1 {total_acc1 / total_sample:0.3f}, "
                                f"acc5 {total_acc5 / total_sample:0.3f}, "
                                f"speed {speed:.3f} steps/s"
                            )
                            avg_batch_time = time.time()

                    step_idx += 1
                    if step_idx == STEP_NUM:
                        if to_static:
                            if use_pir_api():
                                output_spec = [0]
                            else:
                                output_spec = [pred]

                            paddle.jit.save(
                                se_resnext,
                                self.model_save_prefix,
                                output_spec=output_spec,
                                input_names_after_prune=['x'],
                                input_spec=[
                                    InputSpec(
                                        shape=[None, 3, 224, 224], name='x'
                                    ),
                                    InputSpec(shape=[None, 1], name='y'),
                                ],
                                clip_extra=False,
                            )
                        else:
                            paddle.save(
                                se_resnext.state_dict(),
                                self.dy_state_dict_save_path + '.pdparams',
                            )
                        break
            return (
                pred.numpy(),
                avg_loss.numpy(),
                acc_top1.numpy(),
                acc_top5.numpy(),
            )

    def predict_dygraph(self, data):
        with enable_to_static_guard(False):
            with base.dygraph.guard(place):
                se_resnext = SeResNeXt()

                model_dict = paddle.load(
                    self.dy_state_dict_save_path + '.pdparams'
                )
                se_resnext.set_dict(model_dict)
                se_resnext.eval()

                label = np.random.random([1, 1]).astype("int64")
                img = paddle.to_tensor(data)
                label = paddle.to_tensor(label)
                pred_res, _, _, _ = se_resnext(img, label)

                return pred_res.numpy()

    def predict_static(self, data):
        paddle.enable_static()
        if use_pir_api():
            model_filename = self.pir_model_filename
        else:
            model_filename = self.model_filename

        exe = base.Executor(place)
        [
            inference_program,
            feed_target_names,
            fetch_targets,
        ] = paddle.static.io.load_inference_model(
            self.model_save_dir,
            executor=exe,
            model_filename=model_filename,
            params_filename=self.params_filename,
        )

        pred_res = exe.run(
            inference_program,
            feed={feed_target_names[0]: data},
            fetch_list=fetch_targets,
        )

        return pred_res[0]

    def predict_dygraph_jit(self, data):
        with base.dygraph.guard(place):
            se_resnext = paddle.jit.load(self.model_save_prefix)
            se_resnext.eval()

            pred_res = se_resnext(data)

            return pred_res.numpy()

    def predict_analysis_inference(self, data):
        if use_pir_api():
            model_filename = self.pir_model_filename
        else:
            model_filename = self.model_filename
        output = PredictorTools(
            self.model_save_dir,
            model_filename,
            self.params_filename,
            [data],
        )
        (out,) = output()
        return out

    def verify_predict(self):
        image = np.random.random([1, 3, 224, 224]).astype('float32')
        dy_pre = self.predict_dygraph(image)
        st_pre = self.predict_static(image)
        dy_jit_pre = self.predict_dygraph_jit(image)

        np.testing.assert_allclose(
            dy_pre,
            st_pre,
            rtol=1e-05,
            err_msg=f'dy_pre:\n {dy_pre}\n, st_pre: \n{st_pre}.',
        )
        np.testing.assert_allclose(
            dy_jit_pre,
            st_pre,
            rtol=1e-05,
            err_msg=f'dy_jit_pre:\n {dy_jit_pre}\n, st_pre: \n{st_pre}.',
        )

        predictor_pre = self.predict_analysis_inference(image)
        flat_st_pre = st_pre.flatten()
        flat_predictor_pre = np.array(predictor_pre).flatten()
        for i in range(len(flat_predictor_pre)):
            # modify precision to 1e-6, avoid unittest failed
            self.assertAlmostEqual(
                flat_predictor_pre[i],
                flat_st_pre[i],
                delta=1e-5,
                msg=f"predictor_pre:\n {flat_predictor_pre[i]}\n, st_pre: \n{flat_st_pre[i]}.",
            )

    @test_default_and_pir
    def test_check_result(self):
        with enable_to_static_guard(False):
            pred_1, loss_1, acc1_1, acc5_1 = self.train(
                self.train_reader, to_static=False
            )
        pred_2, loss_2, acc1_2, acc5_2 = self.train(
            self.train_reader, to_static=True
        )

        np.testing.assert_allclose(
            pred_1,
            pred_2,
            rtol=1e-05,
            err_msg=f'static pred: {pred_1} \ndygraph pred: {pred_2}',
        )
        np.testing.assert_allclose(
            loss_1,
            loss_2,
            rtol=1e-05,
            err_msg=f'static loss: {loss_1} \ndygraph loss: {loss_2}',
        )
        np.testing.assert_allclose(
            acc1_1,
            acc1_2,
            rtol=1e-05,
            err_msg=f'static acc1: {acc1_1} \ndygraph acc1: {acc1_2}',
        )
        np.testing.assert_allclose(
            acc5_1,
            acc5_2,
            rtol=1e-05,
            err_msg=f'static acc5: {acc5_1} \ndygraph acc5: {acc5_2}',
        )

        self.verify_predict()


if __name__ == '__main__':
    unittest.main()
