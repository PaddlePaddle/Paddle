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

import sys
sys.path.append("../")
import unittest
import paddle
from paddleslim.quant import quant_aware, convert
from static_case import StaticCase
sys.path.append("../demo")
from models import MobileNet
from layers import conv_bn_layer
import numpy as np


class TestQuantAwareCase(StaticCase):
    def test_accuracy(self):
        image = paddle.static.data(
            name='image', shape=[None, 1, 28, 28], dtype='float32')
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
            weight_decay=paddle.regularizer.L2Decay(4e-5))
        optimizer.minimize(avg_cost)
        main_prog = paddle.static.default_main_program()
        val_prog = paddle.static.default_main_program().clone(for_test=True)

        place = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda(
        ) else paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        exe.run(paddle.static.default_startup_program())

        def transform(x):
            return np.reshape(x, [1, 28, 28])

        train_dataset = paddle.vision.datasets.MNIST(
            mode='train', backend='cv2', transform=transform)
        test_dataset = paddle.vision.datasets.MNIST(
            mode='test', backend='cv2', transform=transform)
        train_loader = paddle.io.DataLoader(
            train_dataset,
            places=place,
            feed_list=[image, label],
            drop_last=True,
            return_list=False,
            batch_size=64)
        valid_loader = paddle.io.DataLoader(
            test_dataset,
            places=place,
            feed_list=[image, label],
            batch_size=64,
            return_list=False)

        def train(program):
            iter = 0
            for data in train_loader():
                cost, top1, top5 = exe.run(
                    program,
                    feed=data,
                    fetch_list=[avg_cost, acc_top1, acc_top5])
                iter += 1
                if iter % 100 == 0:
                    print(
                        'train iter={}, avg loss {}, acc_top1 {}, acc_top5 {}'.
                        format(iter, cost, top1, top5))

        def test(program):
            iter = 0
            result = [[], [], []]
            for data in valid_loader():
                cost, top1, top5 = exe.run(
                    program,
                    feed=data,
                    fetch_list=[avg_cost, acc_top1, acc_top5])
                iter += 1
                if iter % 100 == 0:
                    print('eval iter={}, avg loss {}, acc_top1 {}, acc_top5 {}'.
                          format(iter, cost, top1, top5))
                result[0].append(cost)
                result[1].append(top1)
                result[2].append(top5)
            print(' avg loss {}, acc_top1 {}, acc_top5 {}'.format(
                np.mean(result[0]), np.mean(result[1]), np.mean(result[2])))
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
        self.assertTrue(op_nums_1 * 2 == quant_op_nums_1)

        train(quant_train_prog)
        convert_eval_prog = convert(quant_eval_prog, place, config)

        top1_2, top5_2 = test(convert_eval_prog)
        # values before quantization and after quantization should be close
        print("before quantization: top1: {}, top5: {}".format(top1_1, top5_1))
        print("after quantization: top1: {}, top5: {}".format(top1_2, top5_2))

        convert_op_nums_1, convert_quant_op_nums_1 = self.get_convert_op_number(
            convert_eval_prog)
        # test convert op numbers
        self.assertTrue(convert_op_nums_1 + 25 == convert_quant_op_nums_1)

        config['not_quant_pattern'] = ['last_fc']
        quant_prog_2 = quant_aware(
            main_prog, place, config=config, for_test=True)
        op_nums_2, quant_op_nums_2 = self.get_op_number(quant_prog_2)
        convert_prog_2 = convert(quant_prog_2, place, config=config)
        convert_op_nums_2, convert_quant_op_nums_2 = self.get_convert_op_number(
            convert_prog_2)

        self.assertTrue(op_nums_1 == op_nums_2)
        # test skip_quant
        self.assertTrue(quant_op_nums_1 - 2 == quant_op_nums_2)
        self.assertTrue(convert_quant_op_nums_1 == convert_quant_op_nums_2)

    def get_op_number(self, prog):
        graph = paddle.fluid.framework.IrGraph(
            paddle.framework.core.Graph(prog.desc), for_test=False)
        quant_op_nums = 0
        op_nums = 0
        for op in graph.all_op_nodes():
            if op.name() in ['conv2d', 'depthwise_conv2d', 'mul']:
                op_nums += 1
            elif op.name() == 'quantize_linear':
                quant_op_nums += 1
        return op_nums, quant_op_nums

    def get_convert_op_number(self, prog):
        graph = paddle.fluid.framework.IrGraph(
            paddle.framework.core.Graph(prog.desc), for_test=True)
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

