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

import numpy as np

from paddle import fluid
from paddle.fluid.optimizer import Momentum
from paddle.fluid.io import DataLoader

from model import Model, Input, set_device
from distributed import DistributedBatchSampler
from models import yolov3_darknet53, YoloLoss

from coco_metric import COCOMetric
from coco import COCODataset
from transforms import *

NUM_MAX_BOXES = 50


def make_optimizer(step_per_epoch, parameter_list=None):
    base_lr = FLAGS.lr
    warm_up_iter = 1000
    momentum = 0.9
    weight_decay = 5e-4
    boundaries = [step_per_epoch * e for e in [200, 250]]
    values = [base_lr * (0.1 ** i) for i in range(len(boundaries) + 1)]
    learning_rate = fluid.layers.piecewise_decay(
        boundaries=boundaries,
        values=values)
    learning_rate = fluid.layers.linear_lr_warmup(
        learning_rate=learning_rate,
        warmup_steps=warm_up_iter,
        start_lr=0.0,
        end_lr=base_lr)
    optimizer = fluid.optimizer.Momentum(
        learning_rate=learning_rate,
        regularization=fluid.regularizer.L2Decay(weight_decay),
        momentum=momentum,
        parameter_list=parameter_list)
    return optimizer


def main():
    device = set_device(FLAGS.device)
    fluid.enable_dygraph(device) if FLAGS.dynamic else None
    
    inputs = [Input([None, 1], 'int64', name='img_id'),
              Input([None, 2], 'int32', name='img_shape'),
              Input([None, 3, None, None], 'float32', name='image')]
    labels = [Input([None, NUM_MAX_BOXES, 4], 'float32', name='gt_bbox'),
	      Input([None, NUM_MAX_BOXES], 'int32', name='gt_label'),
	      Input([None, NUM_MAX_BOXES], 'float32', name='gt_score')]

    if not FLAGS.eval_only: # training mode
        train_transform = Compose([ColorDistort(),
                                   RandomExpand(),
                                   RandomCrop(),
                                   RandomFlip(),
                                   NormalizeBox(),
                                   PadBox(),
                                   BboxXYXY2XYWH()])
        train_collate_fn = BatchCompose([RandomShape(), NormalizeImage()])
        dataset = COCODataset(dataset_dir=FLAGS.data,
                              anno_path='annotations/instances_train2017.json',
                              image_dir='train2017',
                              with_background=False,
                              mixup=True,
                              transform=train_transform)
        batch_sampler = DistributedBatchSampler(dataset,
                                                batch_size=FLAGS.batch_size,
                                                shuffle=True,
                                                drop_last=True)
        loader = DataLoader(dataset,
                            batch_sampler=batch_sampler,
                            places=device,
                            num_workers=FLAGS.num_workers,
                            return_list=True,
                            collate_fn=train_collate_fn)
    else: # evaluation mode
        eval_transform = Compose([ResizeImage(target_size=608),
                                  NormalizeBox(),
                                  PadBox(),
                                  BboxXYXY2XYWH()])
        eval_collate_fn = BatchCompose([NormalizeImage()])
        dataset = COCODataset(dataset_dir=FLAGS.data,
                              anno_path='annotations/instances_val2017.json',
                              image_dir='val2017',
                              with_background=False,
                              transform=eval_transform)
        # batch_size can only be 1 in evaluation for YOLOv3
        # prediction bbox is a LoDTensor
        batch_sampler = DistributedBatchSampler(dataset,
                                                batch_size=1,
                                                shuffle=False,
                                                drop_last=False)
        loader = DataLoader(dataset,
                            batch_sampler=batch_sampler,
                            places=device,
                            num_workers=FLAGS.num_workers,
                            return_list=True,
                            collate_fn=eval_collate_fn)

    pretrained = FLAGS.eval_only and FLAGS.weights is None
    model = yolov3_darknet53(num_classes=dataset.num_classes,
                   model_mode='eval' if FLAGS.eval_only else 'train',
                   pretrained=pretrained)

    if FLAGS.pretrain_weights is not None:
        model.load(FLAGS.pretrain_weights, skip_mismatch=True, reset_optimizer=True)

    optim = make_optimizer(len(batch_sampler), parameter_list=model.parameters())

    model.prepare(optim,
                  YoloLoss(num_classes=dataset.num_classes),
                  inputs=inputs, labels=labels,
                  device=FLAGS.device)

    # NOTE: we implement COCO metric of YOLOv3 model here, separately
    # from 'prepare' and 'fit' framework for follwing reason:
    # 1. YOLOv3 network structure is different between 'train' and
    # 'eval' mode, in 'eval' mode, output prediction bbox is not the
    # feature map used for YoloLoss calculating
    # 2. COCO metric behavior is also different from defined Metric
    # for COCO metric should not perform accumulate in each iteration
    # but only accumulate at the end of an epoch
    if FLAGS.eval_only:
        if FLAGS.weights is not None:
            model.load(FLAGS.weights, reset_optimizer=True)
        preds = model.predict(loader, stack_outputs=False)
        _, _, _, img_ids, bboxes = preds

        anno_path = os.path.join(FLAGS.data, 'annotations/instances_val2017.json')
        coco_metric = COCOMetric(anno_path=anno_path, with_background=False)
        for img_id, bbox in zip(img_ids, bboxes):
            coco_metric.update(img_id, bbox)
        coco_metric.accumulate()
        coco_metric.reset()
        return

    if FLAGS.resume is not None:
        model.load(FLAGS.resume)

    model.fit(train_data=loader,
              epochs=FLAGS.epoch - FLAGS.no_mixup_epoch,
              save_dir="yolo_checkpoint/mixup",
              save_freq=10)

    # do not use image mixup transfrom in laste FLAGS.no_mixup_epoch epoches
    dataset.mixup = False
    model.fit(train_data=loader,
              epochs=FLAGS.no_mixup_epoch,
              save_dir="yolo_checkpoint/no_mixup",
              save_freq=5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Yolov3 Training on VOC")
    parser.add_argument(
        "--data", type=str, default='dataset/voc',
        help="path to dataset directory")
    parser.add_argument(
        "--device", type=str, default='gpu', help="device to use, gpu or cpu")
    parser.add_argument(
        "-d", "--dynamic", action='store_true', help="enable dygraph mode")
    parser.add_argument(
        "--eval_only", action='store_true', help="run evaluation only")
    parser.add_argument(
        "-e", "--epoch", default=300, type=int, help="number of epoch")
    parser.add_argument(
        "--no_mixup_epoch", default=30, type=int,
        help="number of the last N epoch without image mixup")
    parser.add_argument(
        '--lr', '--learning-rate', default=0.001, type=float, metavar='LR',
        help='initial learning rate')
    parser.add_argument(
        "-b", "--batch_size", default=8, type=int, help="batch size")
    parser.add_argument(
        "-j", "--num_workers", default=4, type=int, help="reader worker number")
    parser.add_argument(
        "-p", "--pretrain_weights", default=None, type=str,
        help="path to pretrained weights")
    parser.add_argument(
        "-r", "--resume", default=None, type=str,
        help="path to model weights")
    parser.add_argument(
        "-w", "--weights", default=None, type=str,
        help="path to weights for evaluation")
    FLAGS = parser.parse_args()
    assert FLAGS.data, "error: must provide data path"
    main()
