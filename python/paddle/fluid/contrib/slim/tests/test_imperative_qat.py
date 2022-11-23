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

import os
import numpy as np
import random
import time
import tempfile
import unittest
import logging

import paddle
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid.optimizer import AdamOptimizer
from paddle.fluid.contrib.slim.quantization import ImperativeQuantAware
from paddle.nn import Sequential
from paddle.nn import Linear, Conv2D, Softmax, Conv2DTranspose
from paddle.fluid.log_helper import get_logger
from paddle.fluid.dygraph.io import INFER_MODEL_SUFFIX, INFER_PARAMS_SUFFIX
from paddle.nn.quant.quant_layers import (
    QuantizedConv2D,
    QuantizedConv2DTranspose,
)
from paddle.fluid.framework import _test_eager_guard
from imperative_test_utils import fix_model_dict, ImperativeLenet

paddle.enable_static()

os.environ["CPU_NUM"] = "1"
if core.is_compiled_with_cuda():
    fluid.set_flags({"FLAGS_cudnn_deterministic": True})

<<<<<<< HEAD
_logger = get_logger(__name__,
                     logging.INFO,
                     fmt='%(asctime)s-%(levelname)s: %(message)s')
=======
_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s'
)
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91


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

    def func_qat(self):
        self.set_vars()

        imperative_qat = ImperativeQuantAware(
            weight_quantize_type=self.weight_quantize_type,
            activation_quantize_type=self.activation_quantize_type,
            fuse_conv_bn=self.fuse_conv_bn,
            onnx_format=self.onnx_format,
        )

        with fluid.dygraph.guard():
            # For CI coverage
<<<<<<< HEAD
            conv1 = Conv2D(in_channels=3,
                           out_channels=2,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           padding_mode='replicate')
=======
            conv1 = Conv2D(
                in_channels=3,
                out_channels=2,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode='replicate',
            )
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
            quant_conv1 = QuantizedConv2D(conv1)
            data = np.random.uniform(-1, 1, [10, 3, 32, 32]).astype('float32')
            quant_conv1(fluid.dygraph.to_variable(data))

            conv_transpose = Conv2DTranspose(4, 6, (3, 3))
            quant_conv_transpose = QuantizedConv2DTranspose(conv_transpose)
<<<<<<< HEAD
            x_var = paddle.uniform((2, 4, 8, 8),
                                   dtype='float32',
                                   min=-1.0,
                                   max=1.0)
=======
            x_var = paddle.uniform(
                (2, 4, 8, 8), dtype='float32', min=-1.0, max=1.0
            )
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
            quant_conv_transpose(x_var)

            seed = 1
            np.random.seed(seed)
            fluid.default_main_program().random_seed = seed
            fluid.default_startup_program().random_seed = seed

            lenet = ImperativeLenet()
            lenet = fix_model_dict(lenet)
            imperative_qat.quantize(lenet)
<<<<<<< HEAD
            adam = AdamOptimizer(learning_rate=0.001,
                                 parameter_list=lenet.parameters())

            train_reader = paddle.batch(paddle.dataset.mnist.train(),
                                        batch_size=32,
                                        drop_last=True)
            test_reader = paddle.batch(paddle.dataset.mnist.test(),
                                       batch_size=32)
=======
            adam = AdamOptimizer(
                learning_rate=0.001, parameter_list=lenet.parameters()
            )

            train_reader = paddle.batch(
                paddle.dataset.mnist.train(), batch_size=32, drop_last=True
            )
            test_reader = paddle.batch(
                paddle.dataset.mnist.test(), batch_size=32
            )
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91

            epoch_num = 1
            for epoch in range(epoch_num):
                lenet.train()
                for batch_id, data in enumerate(train_reader()):
<<<<<<< HEAD
                    x_data = np.array([x[0].reshape(1, 28, 28)
                                       for x in data]).astype('float32')
                    y_data = np.array([x[1] for x in data
                                       ]).astype('int64').reshape(-1, 1)
=======
                    x_data = np.array(
                        [x[0].reshape(1, 28, 28) for x in data]
                    ).astype('float32')
                    y_data = (
                        np.array([x[1] for x in data])
                        .astype('int64')
                        .reshape(-1, 1)
                    )
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91

                    img = fluid.dygraph.to_variable(x_data)
                    label = fluid.dygraph.to_variable(y_data)
                    out = lenet(img)
                    acc = fluid.layers.accuracy(out, label)
                    loss = fluid.layers.cross_entropy(out, label)
                    avg_loss = paddle.mean(loss)
                    avg_loss.backward()
                    adam.minimize(avg_loss)
                    lenet.clear_gradients()
                    if batch_id % 100 == 0:
                        _logger.info(
<<<<<<< HEAD
                            "Train | At epoch {} step {}: loss = {:}, acc= {:}".
                            format(epoch, batch_id, avg_loss.numpy(),
                                   acc.numpy()))
=======
                            "Train | At epoch {} step {}: loss = {:}, acc= {:}".format(
                                epoch, batch_id, avg_loss.numpy(), acc.numpy()
                            )
                        )
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
                    if batch_id == 500:  # For shortening CI time
                        break

                lenet.eval()
                eval_acc_top1_list = []
                for batch_id, data in enumerate(test_reader()):
<<<<<<< HEAD
                    x_data = np.array([x[0].reshape(1, 28, 28)
                                       for x in data]).astype('float32')
                    y_data = np.array([x[1] for x in data
                                       ]).astype('int64').reshape(-1, 1)
=======
                    x_data = np.array(
                        [x[0].reshape(1, 28, 28) for x in data]
                    ).astype('float32')
                    y_data = (
                        np.array([x[1] for x in data])
                        .astype('int64')
                        .reshape(-1, 1)
                    )
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91

                    img = fluid.dygraph.to_variable(x_data)
                    label = fluid.dygraph.to_variable(y_data)

                    out = lenet(img)
<<<<<<< HEAD
                    acc_top1 = fluid.layers.accuracy(input=out,
                                                     label=label,
                                                     k=1)
                    acc_top5 = fluid.layers.accuracy(input=out,
                                                     label=label,
                                                     k=5)
=======
                    acc_top1 = fluid.layers.accuracy(
                        input=out, label=label, k=1
                    )
                    acc_top5 = fluid.layers.accuracy(
                        input=out, label=label, k=5
                    )
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91

                    if batch_id % 100 == 0:
                        eval_acc_top1_list.append(float(acc_top1.numpy()))
                        _logger.info(
<<<<<<< HEAD
                            "Test | At epoch {} step {}: acc1 = {:}, acc5 = {:}"
                            .format(epoch, batch_id, acc_top1.numpy(),
                                    acc_top5.numpy()))
=======
                            "Test | At epoch {} step {}: acc1 = {:}, acc5 = {:}".format(
                                epoch,
                                batch_id,
                                acc_top1.numpy(),
                                acc_top5.numpy(),
                            )
                        )
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91

                # check eval acc
                eval_acc_top1 = sum(eval_acc_top1_list) / len(
                    eval_acc_top1_list
                )
                print('eval_acc_top1', eval_acc_top1)
<<<<<<< HEAD
                self.assertTrue(eval_acc_top1 > 0.9,
                                msg="The test acc {%f} is less than 0.9." %
                                eval_acc_top1)

            # test the correctness of `paddle.jit.save`
            data = next(test_reader())
            test_data = np.array([x[0].reshape(1, 28, 28)
                                  for x in data]).astype('float32')
            y_data = np.array([x[1]
                               for x in data]).astype('int64').reshape(-1, 1)
=======
                self.assertTrue(
                    eval_acc_top1 > 0.9,
                    msg="The test acc {%f} is less than 0.9." % eval_acc_top1,
                )

            # test the correctness of `paddle.jit.save`
            data = next(test_reader())
            test_data = np.array(
                [x[0].reshape(1, 28, 28) for x in data]
            ).astype('float32')
            y_data = (
                np.array([x[1] for x in data]).astype('int64').reshape(-1, 1)
            )
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
            test_img = fluid.dygraph.to_variable(test_data)
            label = fluid.dygraph.to_variable(y_data)
            lenet.eval()
            fp32_out = lenet(test_img)
            fp32_acc = fluid.layers.accuracy(fp32_out, label).numpy()

        with tempfile.TemporaryDirectory(prefix="qat_save_path_") as tmpdir:
            # save inference quantized model
            imperative_qat.save_quantized_model(
                layer=lenet,
                path=os.path.join(tmpdir, "lenet"),
                input_spec=[
<<<<<<< HEAD
                    paddle.static.InputSpec(shape=[None, 1, 28, 28],
                                            dtype='float32')
=======
                    paddle.static.InputSpec(
                        shape=[None, 1, 28, 28], dtype='float32'
                    )
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
                ],
            )
            print('Quantized model saved in %s' % tmpdir)

            if core.is_compiled_with_cuda():
                place = core.CUDAPlace(0)
            else:
                place = core.CPUPlace()
            exe = fluid.Executor(place)
            [
                inference_program,
                feed_target_names,
                fetch_targets,
            ] = fluid.io.load_inference_model(
                dirname=tmpdir,
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
            quant_out = fluid.dygraph.to_variable(quant_out)
            quant_acc = fluid.layers.accuracy(quant_out, label).numpy()
            paddle.enable_static()
            delta_value = fp32_acc - quant_acc
            self.assertLessEqual(delta_value, self.diff_threshold)

    def test_qat(self):
        with _test_eager_guard():
            self.func_qat()
        self.func_qat()


class TestImperativeQatONNXFormat(unittest.TestCase):

    def set_vars(self):
        self.weight_quantize_type = 'abs_max'
        self.activation_quantize_type = 'moving_average_abs_max'
        self.onnx_format = True
        self.diff_threshold = 0.03125
        self.fuse_conv_bn = False


if __name__ == '__main__':
    unittest.main()
