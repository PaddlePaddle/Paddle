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

import logging
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.append("../../quantization")
from imperative_test_utils import ImperativeLenet, fix_model_dict

import paddle
from paddle import base
from paddle.framework import core, set_flags
from paddle.nn import Conv2D, Conv2DTranspose
from paddle.nn.quant.quant_layers import (
    QuantizedConv2D,
    QuantizedConv2DTranspose,
)
from paddle.optimizer import Adam
from paddle.quantization import ImperativeQuantAware
from paddle.static.log_helper import get_logger

INFER_MODEL_SUFFIX = ".pdmodel"
INFER_PARAMS_SUFFIX = ".pdiparams"

paddle.enable_static()

os.environ["CPU_NUM"] = "1"
if core.is_compiled_with_cuda():
    set_flags({"FLAGS_cudnn_deterministic": True})

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s'
)


class TestImperativeQat(unittest.TestCase):
    """
    QAT = quantization-aware training
    """

    def set_vars(self):
        self.weight_quantize_type = 'abs_max'
        self.activation_quantize_type = 'moving_average_abs_max'
        self.onnx_format = False
        self.check_export_model_accuracy = True
        # The original model and quantized model may have different prediction.
        # There are 32 test data and we allow at most one is different.
        # Hence, the diff_threshold is 1 / 32 = 0.03125
        self.diff_threshold = 0.03125
        self.fuse_conv_bn = False

    def test_qat(self):
        self.set_vars()

        imperative_qat = ImperativeQuantAware(
            weight_quantize_type=self.weight_quantize_type,
            activation_quantize_type=self.activation_quantize_type,
            fuse_conv_bn=self.fuse_conv_bn,
            onnx_format=self.onnx_format,
        )

        with base.dygraph.guard():
            # For CI coverage
            conv1 = Conv2D(
                in_channels=3,
                out_channels=2,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode='replicate',
            )
            quant_conv1 = QuantizedConv2D(conv1)
            data = np.random.uniform(-1, 1, [10, 3, 32, 32]).astype('float32')
            quant_conv1(paddle.to_tensor(data))

            conv_transpose = Conv2DTranspose(4, 6, (3, 3))
            quant_conv_transpose = QuantizedConv2DTranspose(conv_transpose)
            x_var = paddle.uniform(
                (2, 4, 8, 8), dtype='float32', min=-1.0, max=1.0
            )
            quant_conv_transpose(x_var)

            seed = 1
            np.random.seed(seed)
            paddle.seed(seed)

            lenet = ImperativeLenet()
            lenet = fix_model_dict(lenet)
            imperative_qat.quantize(lenet)
            adam = Adam(learning_rate=0.001, parameters=lenet.parameters())

            train_reader = paddle.batch(
                paddle.dataset.mnist.train(), batch_size=32, drop_last=True
            )
            test_reader = paddle.batch(
                paddle.dataset.mnist.test(), batch_size=32
            )

            epoch_num = 1
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
                    adam.minimize(avg_loss)
                    lenet.clear_gradients()
                    if batch_id % 100 == 0:
                        _logger.info(
                            f"Train | At epoch {epoch} step {batch_id}: loss = {avg_loss.numpy()}, acc= {acc.numpy()}"
                        )
                    if batch_id == 500:  # For shortening CI time
                        break

                lenet.eval()
                eval_acc_top1_list = []
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
                eval_acc_top1 = sum(eval_acc_top1_list) / len(
                    eval_acc_top1_list
                )
                print('eval_acc_top1', eval_acc_top1)
                self.assertTrue(
                    eval_acc_top1 > 0.9,
                    msg=f"The test acc {{{eval_acc_top1:f}}} is less than 0.9.",
                )

            # test the correctness of `paddle.jit.save`
            data = next(test_reader())
            test_data = np.array(
                [x[0].reshape(1, 28, 28) for x in data]
            ).astype('float32')
            y_data = (
                np.array([x[1] for x in data]).astype('int64').reshape(-1, 1)
            )
            test_img = paddle.to_tensor(test_data)
            label = paddle.to_tensor(y_data)
            lenet.eval()
            fp32_out = lenet(test_img)
            fp32_acc = paddle.metric.accuracy(fp32_out, label).numpy()

        with tempfile.TemporaryDirectory(prefix="qat_save_path_") as tmpdir:
            # save inference quantized model
            imperative_qat.save_quantized_model(
                layer=lenet,
                path=os.path.join(tmpdir, "lenet"),
                input_spec=[
                    paddle.static.InputSpec(
                        shape=[None, 1, 28, 28], dtype='float32'
                    )
                ],
            )
            print(f'Quantized model saved in {tmpdir}')

            if core.is_compiled_with_cuda():
                place = core.CUDAPlace(0)
            else:
                place = core.CPUPlace()
            exe = paddle.static.Executor(place)
            with paddle.pir_utils.OldIrGuard():
                [
                    inference_program,
                    feed_target_names,
                    fetch_targets,
                ] = paddle.static.load_inference_model(
                    tmpdir,
                    executor=exe,
                    model_filename="lenet" + INFER_MODEL_SUFFIX,
                    params_filename="lenet" + INFER_PARAMS_SUFFIX,
                )
                (quant_out,) = exe.run(
                    inference_program,
                    feed={feed_target_names[0]: test_data},
                    fetch_list=fetch_targets,
                )
            paddle.disable_static()
            quant_out = paddle.to_tensor(quant_out)
            quant_acc = paddle.metric.accuracy(quant_out, label).numpy()
            paddle.enable_static()
            delta_value = fp32_acc - quant_acc
            self.assertLessEqual(delta_value, self.diff_threshold)


class TestImperativeQatONNXFormat(unittest.TestCase):
    def set_vars(self):
        self.weight_quantize_type = 'abs_max'
        self.activation_quantize_type = 'moving_average_abs_max'
        self.onnx_format = True
        self.diff_threshold = 0.03125
        self.fuse_conv_bn = False


if __name__ == '__main__':
    unittest.main()
