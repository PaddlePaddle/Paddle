# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from model import CrossEntropy, Input
from utils import AverageMeter, accuracy, ImageNetDataset
from distributed import prepare_context, all_gather, Env, get_nranks, get_local_rank, DistributedBatchSampler
from models import resnet50
from metrics import Accuracy
from paddle.fluid.io import BatchSampler, DataLoader


def make_optimizer(parameter_list=None):
    total_images = 1281167
    base_lr = FLAGS.lr
    momentum = 0.9
    weight_decay = 1e-4
    step_per_epoch = int(math.floor(float(total_images) / FLAGS.batch_size))
    boundaries = [step_per_epoch * e for e in [30, 60, 90]]
    values = [base_lr * (0.1**i) for i in range(len(boundaries) + 1)]
    learning_rate = fluid.layers.piecewise_decay(
        boundaries=boundaries, values=values)
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


def run(model, loader, mode='train'):
    total_loss = 0
    total_time = 0.0
    local_rank = get_local_rank()
    start = time.time()
    start_time = time.time()
    for idx, batch in enumerate(loader()):
        if not fluid.in_dygraph_mode():
            batch = batch[0]

        losses, metrics = getattr(model, mode)(
            batch[0], batch[1])

        if idx > 1:  # skip first two steps
            total_time += time.time() - start
        total_loss += np.sum(losses)
        if idx % 10 == 0 and local_rank == 0:
            print("{:04d}: loss {:0.3f} top1: {:0.3f}% top5: {:0.3f}% time: {:0.3f} samples: {}".format(
                idx, total_loss / (idx + 1), metrics[0][0] * 100, metrics[0][1] * 100, total_time / max(1, (idx - 1)), model._metrics[0].count[0]))
        start = time.time()
    eval_time = time.time() - start_time
    for metric in model._metrics:
        res = metric.accumulate()
        if local_rank == 0 and mode == 'eval':
            print("[EVAL END]: top1: {:0.3f}%, top5: {:0.3f} total samples: {} total time: {:.3f}".format(res[0] * 100, res[1] * 100, model._metrics[0].count[0], eval_time))
        metric.reset()

def main():
    @contextlib.contextmanager
    def null_guard():
        yield

    epoch = FLAGS.epoch
    place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id) \
        if fluid.dygraph.parallel.Env().nranks > 1 else fluid.CUDAPlace(0)
    guard = fluid.dygraph.guard(place) if FLAGS.dynamic else null_guard()
    if fluid.dygraph.parallel.Env().nranks > 1:
        prepare_context(place)

    if not os.path.exists('resnet_checkpoints'):
        os.mkdir('resnet_checkpoints')

    with guard:
        model = resnet50()
        optim = make_optimizer(parameter_list=model.parameters())
        
        inputs = [Input([None, 3, 224, 224], 'float32', name='image')]
        labels = [Input([None, 1], 'int64', name='label')]

        if fluid.in_dygraph_mode():
            feed_list = None
        else:
            feed_list = [x.forward() for x in inputs + labels]
        
        train_dataset = ImageNetDataset(os.path.join(FLAGS.data, 'val'), mode='train')
        val_dataset = ImageNetDataset(os.path.join(FLAGS.data, 'val'), mode='val')
        if get_nranks() > 1:
            train_sampler = DistributedBatchSampler(train_dataset, batch_size=FLAGS.batch_size, shuffle=True)
            train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, places=place, 
                                    feed_list=feed_list, num_workers=0, return_list=True)
            val_sampler = DistributedBatchSampler(val_dataset, batch_size=FLAGS.batch_size)
            val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, places=place, 
                                    feed_list=feed_list, num_workers=0, return_list=True)
        else:
            train_loader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, places=place, 
                                    feed_list=feed_list, num_workers=0, return_list=True)
            val_loader = DataLoader(val_dataset, batch_size=FLAGS.batch_size, places=place, 
                                    feed_list=feed_list, num_workers=0, return_list=True)

        model.prepare(optim, CrossEntropy(), Accuracy(topk=(1, 5)), inputs, labels, val_dataset)
        if FLAGS.resume is not None:
            model.load(FLAGS.resume)

        for e in range(epoch):
            if get_local_rank() == 0:
                print("======== train epoch {} ========".format(e))
            run(model, train_loader)
            model.save('resnet_checkpoints/{:02d}'.format(e))
            if get_local_rank() == 0:
                print("======== eval epoch {} ========".format(e))
            run(model, val_loader, mode='eval')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Resnet Training on ImageNet")
    parser.add_argument('data', metavar='DIR', help='path to dataset '
                        '(should have subdirectories named "train" and "val"')
    parser.add_argument(
        "-d", "--dynamic", action='store_true', help="enable dygraph mode")
    parser.add_argument(
        "-e", "--epoch", default=120, type=int, help="number of epoch")
    parser.add_argument(
        '--lr', '--learning-rate', default=0.1, type=float, metavar='LR',
        help='initial learning rate')
    parser.add_argument(
        "-b", "--batch_size", default=256, type=int, help="batch size")
    parser.add_argument(
        "-n", "--num_devices", default=1, type=int, help="number of devices")
    parser.add_argument(
        "-r", "--resume", default=None, type=str,
        help="checkpoint path to resume")
    FLAGS = parser.parse_args()
    assert FLAGS.data, "error: must provide data path"
    main()
