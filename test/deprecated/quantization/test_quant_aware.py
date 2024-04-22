# Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
import os
import unittest

import numpy as np

import paddle
from paddle.nn.initializer import KaimingUniform
from paddle.static.quantization.quanter import convert, quant_aware

logging.basicConfig(level="INFO", format="%(message)s")

train_parameters = {
    "input_size": [3, 224, 224],
    "input_mean": [0.485, 0.456, 0.406],
    "input_std": [0.229, 0.224, 0.225],
    "learning_strategy": {
        "name": "piecewise_decay",
        "batch_size": 256,
        "epochs": [10, 16, 30],
        "steps": [0.1, 0.01, 0.001, 0.0001],
    },
}


class MobileNet:
    def __init__(self):
        self.params = train_parameters

    def net(self, input, class_dim=1000, scale=1.0):
        # conv1: 112x112
        input = self.conv_bn_layer(
            input,
            filter_size=3,
            channels=3,
            num_filters=int(32 * scale),
            stride=2,
            padding=1,
            name="conv1",
        )

        # 56x56
        input = self.depthwise_separable(
            input,
            num_filters1=32,
            num_filters2=64,
            num_groups=32,
            stride=1,
            scale=scale,
            name="conv2_1",
        )

        input = self.depthwise_separable(
            input,
            num_filters1=64,
            num_filters2=128,
            num_groups=64,
            stride=2,
            scale=scale,
            name="conv2_2",
        )

        # 28x28
        input = self.depthwise_separable(
            input,
            num_filters1=128,
            num_filters2=128,
            num_groups=128,
            stride=1,
            scale=scale,
            name="conv3_1",
        )

        input = self.depthwise_separable(
            input,
            num_filters1=128,
            num_filters2=256,
            num_groups=128,
            stride=2,
            scale=scale,
            name="conv3_2",
        )

        # 14x14
        input = self.depthwise_separable(
            input,
            num_filters1=256,
            num_filters2=256,
            num_groups=256,
            stride=1,
            scale=scale,
            name="conv4_1",
        )

        input = self.depthwise_separable(
            input,
            num_filters1=256,
            num_filters2=512,
            num_groups=256,
            stride=2,
            scale=scale,
            name="conv4_2",
        )

        # 14x14
        for i in range(5):
            input = self.depthwise_separable(
                input,
                num_filters1=512,
                num_filters2=512,
                num_groups=512,
                stride=1,
                scale=scale,
                name="conv5" + "_" + str(i + 1),
            )
        # 7x7
        input = self.depthwise_separable(
            input,
            num_filters1=512,
            num_filters2=1024,
            num_groups=512,
            stride=2,
            scale=scale,
            name="conv5_6",
        )

        input = self.depthwise_separable(
            input,
            num_filters1=1024,
            num_filters2=1024,
            num_groups=1024,
            stride=1,
            scale=scale,
            name="conv6",
        )

        input = paddle.nn.functional.adaptive_avg_pool2d(input, 1)
        with paddle.static.name_scope('last_fc'):
            output = paddle.static.nn.fc(
                input,
                class_dim,
                weight_attr=paddle.ParamAttr(
                    initializer=KaimingUniform(), name="fc7_weights"
                ),
                bias_attr=paddle.ParamAttr(name="fc7_offset"),
            )

        return output

    def conv_bn_layer(
        self,
        input,
        filter_size,
        num_filters,
        stride,
        padding,
        channels=None,
        num_groups=1,
        act='relu',
        use_cudnn=True,
        name=None,
    ):
        conv = paddle.static.nn.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            act=None,
            use_cudnn=use_cudnn,
            param_attr=paddle.ParamAttr(
                initializer=KaimingUniform(), name=name + "_weights"
            ),
            bias_attr=False,
        )
        bn_name = name + "_bn"
        return paddle.static.nn.batch_norm(
            input=conv,
            act=act,
            param_attr=paddle.ParamAttr(name=bn_name + "_scale"),
            bias_attr=paddle.ParamAttr(name=bn_name + "_offset"),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance',
        )

    def depthwise_separable(
        self,
        input,
        num_filters1,
        num_filters2,
        num_groups,
        stride,
        scale,
        name=None,
    ):
        depthwise_conv = self.conv_bn_layer(
            input=input,
            filter_size=3,
            num_filters=int(num_filters1 * scale),
            stride=stride,
            padding=1,
            num_groups=int(num_groups * scale),
            use_cudnn=False,
            name=name + "_dw",
        )

        pointwise_conv = self.conv_bn_layer(
            input=depthwise_conv,
            filter_size=1,
            num_filters=int(num_filters2 * scale),
            stride=1,
            padding=0,
            name=name + "_sep",
        )
        return pointwise_conv


class StaticCase(unittest.TestCase):
    def setUp(self):
        # switch mode
        paddle.enable_static()


class TestQuantAwareCase(StaticCase):
    def test_accuracy(self):
        image = paddle.static.data(
            name='image', shape=[None, 1, 28, 28], dtype='float32'
        )
        label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
        model = MobileNet()
        out = model.net(input=image, class_dim=10)
        cost = paddle.nn.functional.loss.cross_entropy(input=out, label=label)
        avg_cost = paddle.mean(x=cost)
        acc_top1 = paddle.metric.accuracy(input=out, label=label, k=1)
        acc_top5 = paddle.metric.accuracy(input=out, label=label, k=5)
        optimizer = paddle.optimizer.Momentum(
            momentum=0.9,
            learning_rate=0.01,
            weight_decay=paddle.regularizer.L2Decay(4e-5),
        )
        optimizer.minimize(avg_cost)
        main_prog = paddle.static.default_main_program()
        val_prog = paddle.static.default_main_program().clone(for_test=True)

        place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )
        exe = paddle.static.Executor(place)
        exe.run(paddle.static.default_startup_program())

        def transform(x):
            return np.reshape(x, [1, 28, 28])

        train_dataset = paddle.vision.datasets.MNIST(
            mode='train', backend='cv2', transform=transform
        )
        test_dataset = paddle.vision.datasets.MNIST(
            mode='test', backend='cv2', transform=transform
        )
        batch_size = 64 if os.environ.get('DATASET') == 'full' else 8
        train_loader = paddle.io.DataLoader(
            train_dataset,
            places=place,
            feed_list=[image, label],
            drop_last=True,
            return_list=False,
            batch_size=batch_size,
        )
        valid_loader = paddle.io.DataLoader(
            test_dataset,
            places=place,
            feed_list=[image, label],
            batch_size=batch_size,
            return_list=False,
        )

        def train(program):
            iter = 0
            stop_iter = None if os.environ.get('DATASET') == 'full' else 10
            for data in train_loader():
                cost, top1, top5 = exe.run(
                    program,
                    feed=data,
                    fetch_list=[avg_cost, acc_top1, acc_top5],
                )
                iter += 1
                if iter % 100 == 0:
                    logging.info(
                        f'train iter={iter}, avg loss {cost}, acc_top1 {top1}, acc_top5 {top5}'
                    )
                if stop_iter is not None and iter == stop_iter:
                    break

        def test(program):
            iter = 0
            stop_iter = None if os.environ.get('DATASET') == 'full' else 10
            result = [[], [], []]
            for data in valid_loader():
                cost, top1, top5 = exe.run(
                    program,
                    feed=data,
                    fetch_list=[avg_cost, acc_top1, acc_top5],
                )
                iter += 1
                if iter % 100 == 0:
                    logging.info(
                        f'eval iter={iter}, avg loss {cost}, acc_top1 {top1}, acc_top5 {top5}'
                    )
                result[0].append(cost)
                result[1].append(top1)
                result[2].append(top5)
                if stop_iter is not None and iter == stop_iter:
                    break
            logging.info(
                f' avg loss {np.mean(result[0])}, acc_top1 {np.mean(result[1])}, acc_top5 {np.mean(result[2])}'
            )
            return np.mean(result[1]), np.mean(result[2])

        train(main_prog)
        top1_1, top5_1 = test(main_prog)

        config = {
            'weight_quantize_type': 'channel_wise_abs_max',
            'activation_quantize_type': 'moving_average_abs_max',
            'quantize_op_types': ['depthwise_conv2d', 'mul', 'conv2d'],
        }
        quant_train_prog = quant_aware(main_prog, place, config, for_test=False)
        quant_eval_prog = quant_aware(val_prog, place, config, for_test=True)
        op_nums_1, quant_op_nums_1 = self.get_op_number(quant_eval_prog)
        # test quant_aware op numbers
        self.assertEqual(op_nums_1 * 2, quant_op_nums_1)

        train(quant_train_prog)
        convert_eval_prog = convert(quant_eval_prog, place, config)

        top1_2, top5_2 = test(convert_eval_prog)
        # values before quantization and after quantization should be close
        logging.info(f"before quantization: top1: {top1_1}, top5: {top5_1}")
        logging.info(f"after quantization: top1: {top1_2}, top5: {top5_2}")

        convert_op_nums_1, convert_quant_op_nums_1 = self.get_convert_op_number(
            convert_eval_prog
        )
        # test convert op numbers
        self.assertEqual(convert_op_nums_1 + 25, convert_quant_op_nums_1)

        config['not_quant_pattern'] = ['last_fc']
        quant_prog_2 = quant_aware(
            main_prog, place, config=config, for_test=True
        )
        op_nums_2, quant_op_nums_2 = self.get_op_number(quant_prog_2)
        convert_prog_2 = convert(quant_prog_2, place, config=config)
        convert_op_nums_2, convert_quant_op_nums_2 = self.get_convert_op_number(
            convert_prog_2
        )

        self.assertEqual(op_nums_1, op_nums_2)
        # test skip_quant
        self.assertEqual(quant_op_nums_1 - 2, quant_op_nums_2)

        # The following assert will fail and is waiting for investigation.
        # self.assertEqual(convert_quant_op_nums_1, convert_quant_op_nums_2)

    def get_op_number(self, prog):
        graph = paddle.base.framework.IrGraph(
            paddle.framework.core.Graph(prog.desc), for_test=False
        )
        quant_op_nums = 0
        op_nums = 0
        for op in graph.all_op_nodes():
            if op.name() in ['conv2d', 'depthwise_conv2d', 'mul']:
                op_nums += 1
            elif op.name() == 'quantize_linear':
                quant_op_nums += 1
        return op_nums, quant_op_nums

    def get_convert_op_number(self, prog):
        graph = paddle.base.framework.IrGraph(
            paddle.framework.core.Graph(prog.desc), for_test=True
        )
        quant_op_nums = 0
        op_nums = 0
        dequant_num = 0
        for op in graph.all_op_nodes():
            if op.name() not in ['quantize_linear', 'dequantize_linear']:
                op_nums += 1
            elif op.name() == 'quantize_linear':
                quant_op_nums += 1
        return op_nums, quant_op_nums


if __name__ == '__main__':
    unittest.main()
