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
import contextlib
import os
import sys
sys.path.append('../')

import time
import math
import numpy as np

import paddle.fluid as fluid
from paddle.fluid.dygraph.parallel import ParallelEnv
from paddle.io import BatchSampler, DataLoader

from hapi.model import CrossEntropy, Input, set_device
from hapi.distributed import DistributedBatchSampler
from hapi.metrics import Accuracy
import hapi.vision.models as models

from imagenet_dataset import ImageNetDataset


def make_optimizer(step_per_epoch, parameter_list=None):
    base_lr = FLAGS.lr
    lr_scheduler = FLAGS.lr_scheduler
    momentum = FLAGS.momentum
    weight_decay = FLAGS.weight_decay

    if lr_scheduler == 'piecewise':
        milestones = FLAGS.milestones
        boundaries = [step_per_epoch * e for e in milestones]
        values = [base_lr * (0.1**i) for i in range(len(boundaries) + 1)]
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

    optimizer = fluid.optimizer.Momentum(
        learning_rate=learning_rate,
        momentum=momentum,
        regularization=fluid.regularizer.L2Decay(weight_decay),
        parameter_list=parameter_list)

    return optimizer


def main():
    device = set_device(FLAGS.device)
    fluid.enable_dygraph(device) if FLAGS.dynamic else None

    model_list = [x for x in models.__dict__["__all__"]]
    assert FLAGS.arch in model_list, "Expected FLAGS.arch in {}, but received {}".format(
        model_list, FLAGS.arch)
    model = models.__dict__[FLAGS.arch](pretrained=FLAGS.eval_only and
                                        not FLAGS.resume)

    if FLAGS.resume is not None:
        model.load(FLAGS.resume)

    inputs = [Input([None, 3, 224, 224], 'float32', name='image')]
    labels = [Input([None, 1], 'int64', name='label')]

    train_dataset = ImageNetDataset(
        os.path.join(FLAGS.data, 'train'), mode='train')
    val_dataset = ImageNetDataset(os.path.join(FLAGS.data, 'val'), mode='val')

    optim = make_optimizer(
        np.ceil(
            len(train_dataset) * 1. / FLAGS.batch_size / ParallelEnv().nranks),
        parameter_list=model.parameters())

    model.prepare(
        optim,
        CrossEntropy(),
        Accuracy(topk=(1, 5)),
        inputs,
        labels,
        FLAGS.device)

    if FLAGS.eval_only:
        model.evaluate(
            val_dataset,
            batch_size=FLAGS.batch_size,
            num_workers=FLAGS.num_workers)
        return

    output_dir = os.path.join(FLAGS.output_dir, FLAGS.arch,
                              time.strftime('%Y-%m-%d-%H-%M',
                                            time.localtime()))
    if ParallelEnv().local_rank == 0 and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model.fit(train_dataset,
              val_dataset,
              batch_size=FLAGS.batch_size,
              epochs=FLAGS.epoch,
              save_dir=output_dir,
              num_workers=FLAGS.num_workers)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Resnet Training on ImageNet")
    parser.add_argument(
        'data',
        metavar='DIR',
        help='path to dataset '
        '(should have subdirectories named "train" and "val"')
    parser.add_argument(
        "--arch", type=str, default='resnet50', help="model name")
    parser.add_argument(
        "--device", type=str, default='gpu', help="device to run, cpu or gpu")
    parser.add_argument(
        "-d", "--dynamic", action='store_true', help="enable dygraph mode")
    parser.add_argument(
        "-e", "--epoch", default=90, type=int, help="number of epoch")
    parser.add_argument(
        '--lr',
        '--learning-rate',
        default=0.1,
        type=float,
        metavar='LR',
        help='initial learning rate')
    parser.add_argument(
        "-b", "--batch-size", default=64, type=int, help="batch size")
    parser.add_argument(
        "-n", "--num-workers", default=4, type=int, help="dataloader workers")
    parser.add_argument(
        "--output-dir", type=str, default='output', help="save dir")
    parser.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="checkpoint path to resume")
    parser.add_argument(
        "--eval-only", action='store_true', help="only evaluate the model")
    parser.add_argument(
        "--lr-scheduler",
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
        "--weight-decay", default=1e-4, type=float, help="weight decay")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    FLAGS = parser.parse_args()
    assert FLAGS.data, "error: must provide data path"
    main()
