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

from __future__ import print_function

import os
import numpy as np
import random
import unittest
import logging
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid import core
from paddle.fluid.optimizer import AdamOptimizer
from paddle.fluid.framework import IrGraph
from paddle.fluid.contrib.slim.quantization import ImperativeQuantAware
from paddle.fluid.contrib.slim.quantization import OutScaleForTrainingPass, OutScaleForInferencePass, QuantizationTransformPass
from paddle.fluid.dygraph.container import Sequential
from paddle.fluid.dygraph.io import INFER_MODEL_SUFFIX, INFER_PARAMS_SUFFIX
from paddle.nn.layer import ReLU, LeakyReLU, Sigmoid, Softmax, ReLU6
from paddle.nn import Linear, Conv2D, Softmax, BatchNorm
from paddle.fluid.dygraph.nn import Pool2D
from paddle.fluid.log_helper import get_logger

paddle.enable_static()

os.environ["CPU_NUM"] = "1"
if core.is_compiled_with_cuda():
    fluid.set_flags({"FLAGS_cudnn_deterministic": True})

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')


def StaticLenet(data, num_classes=10, classifier_activation='softmax'):
    conv2d_w1_attr = fluid.ParamAttr(name="conv2d_w_1")
    conv2d_w2_attr = fluid.ParamAttr(name="conv2d_w_2")
    fc_w1_attr = fluid.ParamAttr(name="fc_w_1")
    fc_w2_attr = fluid.ParamAttr(name="fc_w_2")
    fc_w3_attr = fluid.ParamAttr(name="fc_w_3")
    conv2d_b1_attr = fluid.ParamAttr(name="conv2d_b_1")
    conv2d_b2_attr = fluid.ParamAttr(name="conv2d_b_2")
    fc_b1_attr = fluid.ParamAttr(name="fc_b_1")
    fc_b2_attr = fluid.ParamAttr(name="fc_b_2")
    fc_b3_attr = fluid.ParamAttr(name="fc_b_3")
    conv1 = fluid.layers.conv2d(
        data,
        num_filters=6,
        filter_size=3,
        stride=1,
        padding=1,
        param_attr=conv2d_w1_attr,
        bias_attr=conv2d_b1_attr)
    batch_norm1 = layers.batch_norm(conv1)
    relu1 = layers.relu(batch_norm1)
    pool1 = fluid.layers.pool2d(
        relu1, pool_size=2, pool_type='max', pool_stride=2)
    conv2 = fluid.layers.conv2d(
        pool1,
        num_filters=16,
        filter_size=5,
        stride=1,
        padding=0,
        param_attr=conv2d_w2_attr,
        bias_attr=conv2d_b2_attr)
    batch_norm2 = layers.batch_norm(conv2)
    relu6_1 = layers.relu6(batch_norm2)
    pool2 = fluid.layers.pool2d(
        relu6_1, pool_size=2, pool_type='max', pool_stride=2)

    fc1 = fluid.layers.fc(input=pool2,
                          size=120,
                          param_attr=fc_w1_attr,
                          bias_attr=fc_b1_attr)
    leaky_relu1 = layers.leaky_relu(fc1, alpha=0.01)
    fc2 = fluid.layers.fc(input=leaky_relu1,
                          size=84,
                          param_attr=fc_w2_attr,
                          bias_attr=fc_b2_attr)
    sigmoid1 = layers.sigmoid(fc2)
    fc3 = fluid.layers.fc(input=sigmoid1,
                          size=num_classes,
                          param_attr=fc_w3_attr,
                          bias_attr=fc_b3_attr)
    softmax1 = layers.softmax(fc3, use_cudnn=True)
    return softmax1


class ImperativeLenet(fluid.dygraph.Layer):
    def __init__(self, num_classes=10, classifier_activation='softmax'):
        super(ImperativeLenet, self).__init__()
        conv2d_w1_attr = fluid.ParamAttr(name="conv2d_w_1")
        conv2d_w2_attr = fluid.ParamAttr(name="conv2d_w_2")
        fc_w1_attr = fluid.ParamAttr(name="fc_w_1")
        fc_w2_attr = fluid.ParamAttr(name="fc_w_2")
        fc_w3_attr = fluid.ParamAttr(name="fc_w_3")
        conv2d_b1_attr = fluid.ParamAttr(name="conv2d_b_1")
        conv2d_b2_attr = fluid.ParamAttr(name="conv2d_b_2")
        fc_b1_attr = fluid.ParamAttr(name="fc_b_1")
        fc_b2_attr = fluid.ParamAttr(name="fc_b_2")
        fc_b3_attr = fluid.ParamAttr(name="fc_b_3")
        self.features = Sequential(
            Conv2D(
                in_channels=1,
                out_channels=6,
                kernel_size=3,
                stride=1,
                padding=1,
                weight_attr=conv2d_w1_attr,
                bias_attr=conv2d_b1_attr),
            BatchNorm(6),
            ReLU(),
            Pool2D(
                pool_size=2, pool_type='max', pool_stride=2),
            Conv2D(
                in_channels=6,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=0,
                weight_attr=conv2d_w2_attr,
                bias_attr=conv2d_b2_attr),
            BatchNorm(16),
            ReLU6(),
            Pool2D(
                pool_size=2, pool_type='max', pool_stride=2))

        self.fc = Sequential(
            Linear(
                in_features=400,
                out_features=120,
                weight_attr=fc_w1_attr,
                bias_attr=fc_b1_attr),
            LeakyReLU(),
            Linear(
                in_features=120,
                out_features=84,
                weight_attr=fc_w2_attr,
                bias_attr=fc_b2_attr),
            Sigmoid(),
            Linear(
                in_features=84,
                out_features=num_classes,
                weight_attr=fc_w3_attr,
                bias_attr=fc_b3_attr),
            Softmax())

    def forward(self, inputs):
        x = self.features(inputs)

        x = fluid.layers.flatten(x, 1)
        x = self.fc(x)
        return x


class TestImperativeOutSclae(unittest.TestCase):
    def test_out_scale_acc(self):
        def _build_static_lenet(main, startup, is_test=False, seed=1000):
            with fluid.unique_name.guard():
                with fluid.program_guard(main, startup):
                    main.random_seed = seed
                    startup.random_seed = seed
                    img = fluid.layers.data(
                        name='image', shape=[1, 28, 28], dtype='float32')
                    label = fluid.layers.data(
                        name='label', shape=[1], dtype='int64')
                    prediction = StaticLenet(img)
                    if not is_test:
                        loss = fluid.layers.cross_entropy(
                            input=prediction, label=label)
                        avg_loss = fluid.layers.mean(loss)
                    else:
                        avg_loss = prediction
            return img, label, avg_loss

        reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=32, drop_last=True)
        weight_quantize_type = 'abs_max'
        activation_quant_type = 'moving_average_abs_max'
        param_init_map = {}
        seed = 1000
        lr = 0.1
        dynamic_out_scale_list = []
        static_out_scale_list = []

        # imperative train
        _logger.info(
            "--------------------------dynamic graph qat--------------------------"
        )
        imperative_out_scale = ImperativeQuantAware()

        with fluid.dygraph.guard():
            np.random.seed(seed)
            fluid.default_main_program().random_seed = seed
            fluid.default_startup_program().random_seed = seed
            lenet = ImperativeLenet()
            fixed_state = {}
            for name, param in lenet.named_parameters():
                p_shape = param.numpy().shape
                p_value = param.numpy()
                if name.endswith("bias"):
                    value = np.zeros_like(p_value).astype('float32')
                else:
                    value = np.random.normal(
                        loc=0.0, scale=0.01, size=np.product(p_shape)).reshape(
                            p_shape).astype('float32')
                fixed_state[name] = value
                param_init_map[param.name] = value
            lenet.set_dict(fixed_state)
            imperative_out_scale.quantize(lenet)
            adam = AdamOptimizer(
                learning_rate=lr, parameter_list=lenet.parameters())
            dynamic_loss_rec = []
            lenet.train()
            for batch_id, data in enumerate(reader()):
                x_data = np.array([x[0].reshape(1, 28, 28)
                                   for x in data]).astype('float32')
                y_data = np.array(
                    [x[1] for x in data]).astype('int64').reshape(-1, 1)

                img = fluid.dygraph.to_variable(x_data)
                label = fluid.dygraph.to_variable(y_data)

                out = lenet(img)
                loss = fluid.layers.cross_entropy(out, label)
                avg_loss = fluid.layers.mean(loss)
                avg_loss.backward()
                adam.minimize(avg_loss)
                lenet.clear_gradients()
                dynamic_loss_rec.append(avg_loss.numpy()[0])
                if batch_id % 100 == 0:
                    _logger.info('{}: {}'.format('loss', avg_loss.numpy()))

            lenet.eval()

        path = "./dynamic_outscale_infer_model/lenet"
        dynamic_save_dir = "./dynamic_outscale_infer_model"

        imperative_out_scale.save_quantized_model(
            layer=lenet,
            path=path,
            input_spec=[
                paddle.static.InputSpec(
                    shape=[None, 1, 28, 28], dtype='float32')
            ])

        _logger.info(
            "--------------------------static graph qat--------------------------"
        )
        static_loss_rec = []
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()
        exe = fluid.Executor(place)

        main = fluid.Program()
        infer = fluid.Program()
        startup = fluid.Program()
        static_img, static_label, static_loss = _build_static_lenet(
            main, startup, False, seed)
        infer_img, _, infer_pre = _build_static_lenet(infer, startup, True,
                                                      seed)
        with fluid.unique_name.guard():
            with fluid.program_guard(main, startup):
                opt = AdamOptimizer(learning_rate=lr)
                opt.minimize(static_loss)

        scope = core.Scope()
        with fluid.scope_guard(scope):
            exe.run(startup)
        for param in main.all_parameters():
            param_tensor = scope.var(param.name).get_tensor()
            param_tensor.set(param_init_map[param.name], place)
        main_graph = IrGraph(core.Graph(main.desc), for_test=False)
        infer_graph = IrGraph(core.Graph(infer.desc), for_test=True)
        transform_pass = QuantizationTransformPass(
            scope=scope,
            place=place,
            activation_quantize_type=activation_quant_type,
            weight_quantize_type=weight_quantize_type,
            quantizable_op_type=['conv2d', 'depthwise_conv2d', 'mul'])
        transform_pass.apply(main_graph)
        transform_pass.apply(infer_graph)
        outscale_pass = OutScaleForTrainingPass(scope=scope, place=place)
        outscale_pass.apply(main_graph)
        build_strategy = fluid.BuildStrategy()
        build_strategy.fuse_all_reduce_ops = False
        binary = fluid.CompiledProgram(main_graph.graph).with_data_parallel(
            loss_name=static_loss.name, build_strategy=build_strategy)

        feeder = fluid.DataFeeder(
            feed_list=[static_img, static_label], place=place)
        with fluid.scope_guard(scope):
            for batch_id, data in enumerate(reader()):
                loss_v, = exe.run(binary,
                                  feed=feeder.feed(data),
                                  fetch_list=[static_loss])
                static_loss_rec.append(loss_v[0])
                if batch_id % 100 == 0:
                    _logger.info('{}: {}'.format('loss', loss_v))
        scale_inference_pass = OutScaleForInferencePass(scope=scope)
        scale_inference_pass.apply(infer_graph)

        save_program = infer_graph.to_program()
        static_save_dir = "./static_outscale_infer_model"
        with fluid.scope_guard(scope):
            fluid.io.save_inference_model(
                dirname=static_save_dir,
                feeded_var_names=[infer_img.name],
                target_vars=[infer_pre],
                executor=exe,
                main_program=save_program,
                model_filename="lenet" + INFER_MODEL_SUFFIX,
                params_filename="lenet" + INFER_PARAMS_SUFFIX)

        rtol = 1e-05
        atol = 1e-08
        for i, (loss_d,
                loss_s) in enumerate(zip(dynamic_loss_rec, static_loss_rec)):
            diff = np.abs(loss_d - loss_s)
            if diff > (atol + rtol * np.abs(loss_s)):
                _logger.info(
                    "diff({}) at {}, dynamic loss = {}, static loss = {}".
                    format(diff, i, loss_d, loss_s))
                break

        self.assertTrue(
            np.allclose(
                np.array(dynamic_loss_rec),
                np.array(static_loss_rec),
                rtol=rtol,
                atol=atol,
                equal_nan=True),
            msg='Failed to do the imperative qat.')

        # load dynamic model
        [dynamic_inference_program, feed_target_names, fetch_targets] = (
            fluid.io.load_inference_model(
                dirname=dynamic_save_dir,
                executor=exe,
                model_filename="lenet" + INFER_MODEL_SUFFIX,
                params_filename="lenet" + INFER_PARAMS_SUFFIX))
        # load static model
        [static_inference_program, feed_target_names, fetch_targets] = (
            fluid.io.load_inference_model(
                dirname=static_save_dir,
                executor=exe,
                model_filename="lenet" + INFER_MODEL_SUFFIX,
                params_filename="lenet" + INFER_PARAMS_SUFFIX))

        dynamic_ops = dynamic_inference_program.global_block().ops
        static_ops = static_inference_program.global_block().ops

        for op in dynamic_ops[:]:
            if op.type == "flatten2" or 'fake' in op.type:
                dynamic_ops.remove(op)

        for op in static_ops[:]:
            if 'fake' in op.type:
                static_ops.remove(op)

        for i in range(len(dynamic_ops)):
            if dynamic_ops[i].has_attr("out_threshold"):
                self.assertTrue(dynamic_ops[i].type == static_ops[i].type)
                self.assertTrue(dynamic_ops[i].attr("out_threshold") ==
                                static_ops[i].attr("out_threshold"))


if __name__ == '__main__':
    unittest.main()
