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

from __future__ import division
from __future__ import print_function

import argparse
import os

import time
import math
import numpy as np

import reader
import models
import paddle
import paddle.fluid as fluid
from paddle.fluid.contrib.slim.quantization import DygraphQuantAware
from paddle.fluid.optimizer import AdamOptimizer, SGD


def make_optimizer(step_per_epoch, parameter_list=None):
    base_lr = FLAGS.lr
    lr_scheduler = FLAGS.lr_scheduler
    momentum = FLAGS.momentum
    weight_decay = FLAGS.weight_decay

    if lr_scheduler == 'piecewise':
        milestones = FLAGS.milestones
        boundaries = [step_per_epoch * e for e in milestones]
        values = [base_lr * (0.1**i) for i in range(len(boundaries) + 1)]
        print("lr value:" + str(values))
        learning_rate = fluid.layers.piecewise_decay(
            boundaries=boundaries, values=values)
    elif lr_scheduler == 'cosine':
        learning_rate = fluid.layers.cosine_decay(base_lr, step_per_epoch,
                                                  FLAGS.epoch)
    else:
        raise ValueError(
            "Expected lr_scheduler in ['piecewise', 'cosine'], but got {}".
            format(lr_scheduler))

    learning_rate = fluid.layers.linear_lr_warmup(
        learning_rate=learning_rate,
        warmup_steps=5 * step_per_epoch,
        start_lr=0.,
        end_lr=base_lr)

    # optimizer = fluid.optimizer.Momentum(
    #     learning_rate=learning_rate,
    #     momentum=momentum,
    #     regularization=fluid.regularizer.L2Decay(weight_decay),
    #     parameter_list=parameter_list)

    optimizer = fluid.optimizer.AdamOptimizer(
        learning_rate=0.001, parameter_list=parameter_list)

    return optimizer


def main():
    dygraph_qat = DygraphQuantAware()
    paddle.enable_imperative()

    print("Load model ...")
    model_list = [x for x in models.__dict__["__all__"]]
    assert FLAGS.model_name in model_list, "Expected FLAGS.model_name in {}, but received {}".format(
        model_list, FLAGS.model_name)
    model = models.__dict__[FLAGS.model_name](pretrained=True)  # load weights

    print("Quantize model ...")
    dygraph_qat.quantize(model)

    print("Prepare train ...")
    adam = SGD(learning_rate=0.1, parameter_list=model.parameters())
    train_reader = paddle.batch(
        reader.train(data_dir=FLAGS.data_path),
        batch_size=FLAGS.batch_size,
        drop_last=True)
    test_reader = paddle.batch(
        reader.val(data_dir=FLAGS.data_path), batch_size=128)

    print("Train and test ...")
    for epoch in range(FLAGS.epoch):
        if not FLAGS.action_only_eval:
            # Train
            model.train()
            for batch_id, data in enumerate(test_reader()):
                x_data = np.array(
                    [x[0].reshape(3, 224, 224) for x in data]).astype('float32')
                # x_data = np.ones_like(np.array(x_data)) * batch_id
                # print(x_data[0][0][0][:10])
                y_data = np.array([x[1] for x in data]).astype('int64').reshape(
                    -1, 1)
                for p in model.parameters():
                    if p.name == 'conv_bn_layer_0_weights':
                        # print("weight check----------------", p.numpy()[0][0][0][:10])
                        pass
                img = fluid.dygraph.to_variable(x_data)
                label = fluid.dygraph.to_variable(y_data)
                out = model(img)
                acc = fluid.layers.accuracy(out, label)
                loss = fluid.layers.cross_entropy(out, label)
                avg_loss = fluid.layers.mean(loss)
                avg_loss.backward()
                adam.minimize(avg_loss)
                model.clear_gradients()
                if batch_id % 1 == 0:
                    print("Train | At epoch {} step {}: loss = {:}, acc= {:}".
                          format(epoch, batch_id, avg_loss.numpy(), acc.numpy(
                          )))
                if FLAGS.action_fast_test and batch_id > 20:
                    break

        # Test
        model.eval()
        all_acc_top1 = 0
        all_acc_top5 = 0
        for batch_id, data in enumerate(test_reader()):
            x_data = np.array([x[0].reshape(3, 224, 224)
                               for x in data]).astype('float32')
            y_data = np.array(
                [x[1] for x in data]).astype('int64').reshape(-1, 1)

            img = fluid.dygraph.to_variable(x_data)
            label = fluid.dygraph.to_variable(y_data)

            out = model(img)
            acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
            acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)
            all_acc_top1 += acc_top1.numpy()
            all_acc_top5 += acc_top5.numpy()

            if batch_id % 10 == 0:
                print(
                    "Test | At epoch {} step {}: avg_acc1 = {:}, avg_acc5 = {:}".
                    format(epoch, batch_id, all_acc_top1 / (batch_id + 1),
                           all_acc_top5 / (batch_id + 1)))
            if FLAGS.action_fast_test and batch_id > 20:
                break
        print(
            "Finish Test | At epoch {} step {}: avg_acc1 = {:}, avg_acc5 = {:}".
            format(epoch, batch_id, all_acc_top1 / (batch_id + 1), all_acc_top5
                   / (batch_id + 1)))

        # save inference quantized model
        print("Save quantized model ...")
        output_dir = os.path.join(FLAGS.output_dir, FLAGS.model_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_path = output_dir + "_epoch" + str(epoch)
        dygraph_qat.save_infer_quant_model(
            dirname=save_path,
            model=model,
            input_shape=[(3, 224, 224)],
            input_dtype=['float32'],
            feed=[0],
            fetch=[0])
        print("Finish quantization, and save quantized model to " + save_path +
              "\n\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Training on ImageNet")
    parser.add_argument(
        '--data_path',
        default="/work/datasets/ILSVRC2012/",
        help='path to dataset '
        '(should have subdirectories named "train" and "val"')
    parser.add_argument(
        "--output_dir", type=str, default='output', help="save dir")
    parser.add_argument(
        "--model_name", type=str, default='mobilenet_v1', help="model name")
    parser.add_argument(
        "--device", type=str, default='gpu', help="device to run, cpu or gpu")
    parser.add_argument(
        "-e", "--epoch", default=3, type=int, help="number of epoch")
    parser.add_argument(
        "-b", "--batch_size", default=64, type=int, help="batch size")
    parser.add_argument(
        "--action_only_eval", action="store_true", help="not train, only eval")
    parser.add_argument(
        "--action_fast_test",
        action="store_true",
        help="fast train and test a model")

    parser.add_argument(
        "--image_size", default=224, type=int, help="intput image size")
    parser.add_argument(
        '--lr',
        '--learning_rate',
        default=0.0001,
        type=float,
        metavar='LR',
        help='initial learning rate')
    parser.add_argument(
        "--lr_scheduler",
        default='piecewise',
        type=str,
        help="learning rate scheduler")
    parser.add_argument(
        "--milestones",
        nargs='+',
        type=int,
        default=[30, 60, 80],
        help="piecewise decay milestones")
    parser.add_argument(
        "--weight_decay", default=1e-4, type=float, help="weight decay")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")

    FLAGS = parser.parse_args()
    print("Input params:")
    print(FLAGS)
    main()
