# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import random
import sys
import unittest

sys.path.append(".")
import numpy as np
from test_quant_aware_deprecated import StaticCase

import paddle
from paddle.static.quantization.quanter import convert, quant_aware

np.random.seed(0)
random.seed(0)
paddle.seed(0)
logging.basicConfig(level="INFO", format="%(message)s")


class RandomDataset(paddle.io.Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        enc_input = np.random.random([4, 128]).astype('float32')
        attn_mask = np.random.random([2, 4, 4]).astype('float32')
        label = np.random.randint(0, 2, (1,)).astype('int64')
        return enc_input, attn_mask, label

    def __len__(self):
        return self.num_samples


class TestQuantPostQuantAwareCase1(StaticCase):
    def test_accuracy(self):
        def simple_transformer(enc_input, attn_mask):
            encoder_layer = paddle.nn.TransformerEncoderLayer(128, 2, 512)
            encoder = paddle.nn.TransformerEncoder(encoder_layer, 2)
            encoder_output = encoder(enc_input, attn_mask)
            first_token = encoder_output[:, 0]
            bias = paddle.full(shape=[1, 128], fill_value=1e-6)
            linear = paddle.nn.Linear(128, 2)
            logits = linear(first_token + bias)
            return logits

        enc_input = paddle.static.data(
            name='enc_input', shape=[None, 4, 128], dtype='float32'
        )
        attn_mask = paddle.static.data(
            name='attn_mask', shape=[None, 2, 4, 4], dtype='float32'
        )
        label = paddle.static.data(name='label', shape=[None, 1], dtype='int64')
        out = simple_transformer(enc_input, attn_mask)
        cost = paddle.nn.functional.loss.cross_entropy(input=out, label=label)
        avg_cost = paddle.mean(x=cost)
        acc_top1 = paddle.metric.accuracy(input=out, label=label, k=1)
        optimizer = paddle.optimizer.Momentum(
            momentum=0.9,
            learning_rate=0.01,
            weight_decay=paddle.regularizer.L2Decay(4e-5),
        )
        optimizer.minimize(avg_cost)
        main_prog = paddle.static.default_main_program()
        val_prog = main_prog.clone(for_test=True)

        place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )
        exe = paddle.static.Executor(place)
        exe.run(paddle.static.default_startup_program())

        train_dataset = RandomDataset(100)
        test_dataset = RandomDataset(50)
        train_loader = paddle.io.DataLoader(
            train_dataset,
            places=place,
            feed_list=[enc_input, attn_mask, label],
            drop_last=True,
            return_list=False,
            batch_size=10,
        )
        valid_loader = paddle.io.DataLoader(
            test_dataset,
            places=place,
            feed_list=[enc_input, attn_mask, label],
            batch_size=10,
            return_list=False,
        )

        def train(program):
            iter = 0
            for data in train_loader():
                cost, top1 = exe.run(
                    program, feed=data, fetch_list=[avg_cost, acc_top1]
                )
                iter += 1
                if iter % 100 == 0:
                    logging.info(
                        f'train iter={iter}, avg loss {cost}, acc_top1 {top1}'
                    )

        def test(program):
            iter = 0
            result = [[], []]
            for data in valid_loader():
                cost, top1 = exe.run(
                    program, feed=data, fetch_list=[avg_cost, acc_top1]
                )
                iter += 1
                if iter % 100 == 0:
                    logging.info(
                        f'eval iter={iter}, avg loss {cost}, acc_top1 {top1}'
                    )
                result[0].append(cost)
                result[1].append(top1)
            logging.info(
                f' avg loss {np.mean(result[0])}, acc_top1 {np.mean(result[1])}'
            )
            return np.mean(result[1])

        train(main_prog)
        top1_1 = test(main_prog)

        config = {
            'weight_quantize_type': 'channel_wise_abs_max',
            'activation_quantize_type': 'moving_average_abs_max',
            'quantize_op_types': [
                'conv2d',
                'depthwise_conv2d',
                'mul',
                'matmul',
                'elementwise_add',
            ],
            'quant_post_first': True,
            'scale_trainable': True,
        }
        calib_config = {
            'data_loader': valid_loader,
            'algo': 'abs_max',
            'feed_list': ['enc_input', 'attn_mask', 'label'],
            'fetch_list': [avg_cost, acc_top1],
        }
        quant_eval_prog, scale_dict, _, _ = quant_aware(
            val_prog,
            place,
            config,
            for_test=True,
            calib_config=calib_config,
            model_type='transformer',
            return_scale_dict=True,
        )
        quant_train_prog = quant_aware(
            main_prog,
            place,
            config,
            for_test=False,
            calib_config=calib_config,
            return_program=True,
            scale_dict=scale_dict,
            model_type='transformer',
        )
        train(quant_train_prog)
        quant_eval_prog = convert(quant_eval_prog, place, config)
        top1_2 = test(quant_eval_prog)
        # values before quantization and after quantization should be close
        logging.info(f"before quantization: top1: {top1_1}")
        logging.info(f"after quantization: top1: {top1_2}")


if __name__ == '__main__':
    unittest.main()
