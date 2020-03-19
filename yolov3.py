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
import random
import time

from functools import partial

import cv2
import numpy as np
from pycocotools.coco import COCO

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay

from model import Model, Loss, Input
from resnet import ResNet, ConvBNLayer

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


# XXX transfer learning
class ResNetBackBone(ResNet):
    def __init__(self, depth=50):
        super(ResNetBackBone, self).__init__(depth=depth)
        delattr(self, 'fc')

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.pool(x)
        outputs = []
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        return outputs


class YoloDetectionBlock(fluid.dygraph.Layer):
    def __init__(self, num_channels, num_filters):
        super(YoloDetectionBlock, self).__init__()

        assert num_filters % 2 == 0, \
            "num_filters {} cannot be divided by 2".format(num_filters)

        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1,
            act='leaky_relu')
        self.conv1 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters * 2,
            filter_size=3,
            act='leaky_relu')
        self.conv2 = ConvBNLayer(
            num_channels=num_filters * 2,
            num_filters=num_filters,
            filter_size=1,
            act='leaky_relu')
        self.conv3 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters * 2,
            filter_size=3,
            act='leaky_relu')
        self.route = ConvBNLayer(
            num_channels=num_filters * 2,
            num_filters=num_filters,
            filter_size=1,
            act='leaky_relu')
        self.tip = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters * 2,
            filter_size=3,
            act='leaky_relu')

    def forward(self, inputs):
        out = self.conv0(inputs)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        route = self.route(out)
        tip = self.tip(route)
        return route, tip


class YOLOv3(Model):
    def __init__(self, num_classes=80):
        super(YOLOv3, self).__init__()
        self.num_classes = num_classes
        self.anchors = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45,
                        59, 119, 116, 90, 156, 198, 373, 326]
        self.anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        self.valid_thresh = 0.005
        self.nms_thresh = 0.45
        self.nms_topk = 400
        self.nms_posk = 100
        self.draw_thresh = 0.5

        self.backbone = ResNetBackBone()
        self.block_outputs = []
        self.yolo_blocks = []
        self.route_blocks = []

        for idx, num_chan in enumerate([2048, 1280, 640]):
            yolo_block = self.add_sublayer(
                "detecton_block_{}".format(idx),
                YoloDetectionBlock(num_chan, num_filters=512 // (2**idx)))
            self.yolo_blocks.append(yolo_block)

            num_filters = len(self.anchor_masks[idx]) * (self.num_classes + 5)

            block_out = self.add_sublayer(
                "block_out_{}".format(idx),
                Conv2D(num_channels=1024 // (2**idx),
                       num_filters=num_filters,
                       filter_size=1,
                       param_attr=ParamAttr(
                           initializer=fluid.initializer.Normal(0., 0.02)),
                       bias_attr=ParamAttr(
                           initializer=fluid.initializer.Constant(0.0),
                           regularizer=L2Decay(0.))))
            self.block_outputs.append(block_out)
            if idx < 2:
                route = self.add_sublayer(
                    "route_{}".format(idx),
                    ConvBNLayer(num_channels=512 // (2**idx),
                                num_filters=256 // (2**idx),
                                filter_size=1,
                                act='leaky_relu'))
                self.route_blocks.append(route)

    def forward(self, inputs, img_info):
        outputs = []
        boxes = []
        scores = []
        downsample = 32

        feats = self.backbone(inputs)
        feats = feats[::-1][:len(self.anchor_masks)]
        route = None
        for idx, feat in enumerate(feats):
            if idx > 0:
                feat = fluid.layers.concat(input=[route, feat], axis=1)
            route, tip = self.yolo_blocks[idx](feat)
            block_out = self.block_outputs[idx](tip)
            outputs.append(block_out)

            if idx < 2:
                route = self.route_blocks[idx](route)
                route = fluid.layers.resize_nearest(route, scale=2)

            if self.mode == 'test':
                anchor_mask = self.anchor_masks[idx]
                mask_anchors = []
                for m in anchor_mask:
                    mask_anchors.append(self.anchors[2 * m])
                    mask_anchors.append(self.anchors[2 * m + 1])
                img_shape = fluid.layers.slice(img_info, axes=[1], starts=[1], ends=[3])
                img_id = fluid.layers.slice(img_info, axes=[1], starts=[0], ends=[1])
                b, s = fluid.layers.yolo_box(
                    x=block_out,
                    img_size=img_shape,
                    anchors=mask_anchors,
                    class_num=self.num_classes,
                    conf_thresh=self.valid_thresh,
                    downsample_ratio=downsample)

                boxes.append(b)
                scores.append(fluid.layers.transpose(s, perm=[0, 2, 1]))

            downsample //= 2

        if self.mode != 'test':
            return outputs

        return [img_id, fluid.layers.multiclass_nms(
            bboxes=fluid.layers.concat(boxes, axis=1),
            scores=fluid.layers.concat(scores, axis=2),
            score_threshold=self.valid_thresh,
            nms_top_k=self.nms_topk,
            keep_top_k=self.nms_posk,
            nms_threshold=self.nms_thresh,
            background_label=-1)]


class YoloLoss(Loss):
    def __init__(self, num_classes=80):
        super(YoloLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_thresh = 0.7
        self.anchors = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45,
                        59, 119, 116, 90, 156, 198, 373, 326]
        self.anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    def forward(self, outputs, labels):
        downsample = 32
        gt_box, gt_label, gt_score = labels
        losses = []

        for idx, out in enumerate(outputs):
            anchor_mask = self.anchor_masks[idx]
            loss = fluid.layers.yolov3_loss(
                x=out,
                gt_box=gt_box,
                gt_label=gt_label,
                gt_score=gt_score,
                anchor_mask=anchor_mask,
                downsample_ratio=downsample,
                anchors=self.anchors,
                class_num=self.num_classes,
                ignore_thresh=self.ignore_thresh,
                use_label_smooth=True)
            loss = fluid.layers.reduce_mean(loss)
            losses.append(loss)
            downsample //= 2
        return losses


def make_optimizer(parameter_list=None):
    base_lr = FLAGS.lr
    warm_up_iter = 4000
    momentum = 0.9
    weight_decay = 5e-4
    boundaries = [400000, 450000]
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


def _iou_matrix(a, b):
    tl_i = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    br_i = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])
    area_i = np.prod(br_i - tl_i, axis=2) * (tl_i < br_i).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    area_o = (area_a[:, np.newaxis] + area_b - area_i)
    return area_i / (area_o + 1e-10)


def _crop_box_with_center_constraint(box, crop):
    cropped_box = box.copy()
    cropped_box[:, :2] = np.maximum(box[:, :2], crop[:2])
    cropped_box[:, 2:] = np.minimum(box[:, 2:], crop[2:])
    cropped_box[:, :2] -= crop[:2]
    cropped_box[:, 2:] -= crop[:2]
    centers = (box[:, :2] + box[:, 2:]) / 2
    valid = np.logical_and(
        crop[:2] <= centers, centers < crop[2:]).all(axis=1)
    valid = np.logical_and(
        valid, (cropped_box[:, :2] < cropped_box[:, 2:]).all(axis=1))
    return cropped_box, np.where(valid)[0]


def random_crop(inputs):
    aspect_ratios = [.5, 2.]
    thresholds = [.0, .1, .3, .5, .7, .9]
    scaling = [.3, 1.]

    img, img_ids, gt_box, gt_label = inputs
    h, w = img.shape[:2]

    if len(gt_box) == 0:
        return inputs

    np.random.shuffle(thresholds)
    for thresh in thresholds:
        found = False
        for i in range(50):
            scale = np.random.uniform(*scaling)
            min_ar, max_ar = aspect_ratios
            ar = np.random.uniform(max(min_ar, scale**2),
                                   min(max_ar, scale**-2))
            crop_h = int(h * scale / np.sqrt(ar))
            crop_w = int(w * scale * np.sqrt(ar))
            crop_y = np.random.randint(0, h - crop_h)
            crop_x = np.random.randint(0, w - crop_w)
            crop_box = [crop_x, crop_y, crop_x + crop_w, crop_y + crop_h]
            iou = _iou_matrix(gt_box, np.array([crop_box], dtype=np.float32))
            if iou.max() < thresh:
                continue

            cropped_box, valid_ids = _crop_box_with_center_constraint(
                gt_box, np.array(crop_box, dtype=np.float32))
            if valid_ids.size > 0:
                found = True
                break

        if found:
            x1, y1, x2, y2 = crop_box
            img = img[y1:y2, x1:x2, :]
            gt_box = np.take(cropped_box, valid_ids, axis=0)
            gt_label = np.take(gt_label, valid_ids, axis=0)
            return img, img_ids, gt_box, gt_label

        return inputs


# XXX mix up, color distort and random expand are skipped for simplicity
def sample_transform(inputs, mode='train', num_max_boxes=50):
    if mode == 'train':
        img, img_id, gt_box, gt_label = random_crop(inputs)
    else:
        img, img_id, gt_box, gt_label = inputs

    h, w = img.shape[:2]
    # random flip
    if mode == 'train' and np.random.uniform(0., 1.) > .5:
        img = img[:, ::-1, :]
        if len(gt_box) > 0:
            swap = gt_box.copy()
            gt_box[:, 0] = w - swap[:, 2] - 1
            gt_box[:, 2] = w - swap[:, 0] - 1

    if len(gt_label) == 0:
        gt_box = np.zeros([num_max_boxes, 4], dtype=np.float32)
        gt_label = np.zeros([num_max_boxes], dtype=np.int32)
        return img, gt_box, gt_label

    gt_box = gt_box[:num_max_boxes, :]
    gt_label = gt_label[:num_max_boxes, 0]
    # normalize boxes
    gt_box /= np.array([w, h] * 2, dtype=np.float32)
    gt_box[:, 2:] = gt_box[:, 2:] - gt_box[:, :2]
    gt_box[:, :2] = gt_box[:, :2] + gt_box[:, 2:] / 2.

    pad = num_max_boxes - gt_label.size
    gt_box = np.pad(gt_box, ((0, pad), (0, 0)), mode='constant')
    gt_label = np.pad(gt_label, ((0, pad)), mode='constant')

    return img, img_id, gt_box, gt_label


def batch_transform(batch, mode='train'):
    if mode == 'train':
        d = np.random.choice(
            [320, 352, 384, 416, 448, 480, 512, 544, 576, 608])
        interp = np.random.choice(range(5))
    else:
        d = 608
        interp = cv2.INTER_CUBIC
    # transpose batch
    imgs, img_ids, gt_boxes, gt_labels = list(zip(*batch))
    img_shapes = np.array([[im.shape[0], im.shape[1]] for im in imgs]).astype('int32')
    imgs = np.array([cv2.resize(
        img, (d, d), interpolation=interp) for img in imgs])

    # transpose, permute and normalize
    imgs = imgs.astype(np.float32)[..., ::-1]
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.120, 57.375], dtype=np.float32)
    invstd = 1. / std
    imgs -= mean
    imgs *= invstd
    imgs = imgs.transpose((0, 3, 1, 2))

    img_ids = np.array(img_ids)
    img_info = np.concatenate([img_ids, img_shapes], axis=1)
    gt_boxes = np.array(gt_boxes)
    gt_labels = np.array(gt_labels)
    # XXX since mix up is not used, scores are all ones
    gt_scores = np.ones_like(gt_labels, dtype=np.float32)
    return [imgs, img_info], [gt_boxes, gt_labels, gt_scores]


def coco2017(root_dir, mode='train'):
    json_path = os.path.join(
        root_dir, 'annotations/instances_{}2017.json'.format(mode))
    coco = COCO(json_path)
    img_ids = coco.getImgIds()
    imgs = coco.loadImgs(img_ids)
    class_map = {v: i + 1 for i, v in enumerate(coco.getCatIds())}
    samples = []

    for img in imgs:
        img_path = os.path.join(
            root_dir, '{}2017'.format(mode), img['file_name'])
        file_path = img_path
        width = img['width']
        height = img['height']
        ann_ids = coco.getAnnIds(imgIds=img['id'], iscrowd=False)
        anns = coco.loadAnns(ann_ids)

        gt_box = []
        gt_label = []

        for ann in anns:
            x1, y1, w, h = ann['bbox']
            x2 = x1 + w - 1
            y2 = y1 + h - 1
            x1 = np.clip(x1, 0, width - 1)
            x2 = np.clip(x2, 0, width - 1)
            y1 = np.clip(y1, 0, height - 1)
            y2 = np.clip(y2, 0, height - 1)
            if ann['area'] <= 0 or x2 < x1 or y2 < y1:
                continue
            gt_label.append(ann['category_id'])
            gt_box.append([x1, y1, x2, y2])

        gt_box = np.array(gt_box, dtype=np.float32)
        gt_label = np.array([class_map[cls] for cls in gt_label],
                            dtype=np.int32)[:, np.newaxis]
        im_id = np.array([img['id']], dtype=np.int32)

        if gt_label.size == 0 and not mode == 'train':
            continue
        samples.append((file_path, im_id.copy(), gt_box.copy(), gt_label.copy()))

    def iterator():
        if mode == 'train':
            np.random.shuffle(samples)
        for file_path, im_id, gt_box, gt_label in samples:
            img = cv2.imread(file_path)
            yield img, im_id, gt_box, gt_label

    return iterator


# XXX coco metrics not included for simplicity
def run(model, loader, mode='train'):
    total_loss = 0.
    total_time = 0.
    device_ids = list(range(FLAGS.num_devices))
    start = time.time()

    for idx, batch in enumerate(loader()):
        losses = getattr(model, mode)(batch[0], batch[1])

        total_loss += np.sum(losses)
        if idx > 1:  # skip first two steps
            total_time += time.time() - start
        if idx % 10 == 0:
            logger.info("{:04d}: loss {:0.3f} time: {:0.3f}".format(
                idx, total_loss / (idx + 1), total_time / max(1, (idx - 1))))
        start = time.time()


def main():
    @contextlib.contextmanager
    def null_guard():
        yield

    epoch = FLAGS.epoch
    batch_size = FLAGS.batch_size
    guard = fluid.dygraph.guard() if FLAGS.dynamic else null_guard()

    train_loader = fluid.io.xmap_readers(
        batch_transform,
        paddle.batch(
            fluid.io.xmap_readers(
                sample_transform,
                coco2017(FLAGS.data, 'train'),
                process_num=8,
                buffer_size=4 * batch_size),
            batch_size=batch_size,
            drop_last=True),
        process_num=2, buffer_size=4)

    val_sample_transform = partial(sample_transform, mode='val')
    val_batch_transform = partial(batch_transform, mode='val')

    val_loader = fluid.io.xmap_readers(
        val_batch_transform,
        paddle.batch(
            fluid.io.xmap_readers(
                val_sample_transform,
                coco2017(FLAGS.data, 'val'),
                process_num=8,
                buffer_size=4 * batch_size),
            batch_size=1),
        process_num=2, buffer_size=4)

    if not os.path.exists('yolo_checkpoints'):
        os.mkdir('yolo_checkpoints')

    with guard:
        NUM_CLASSES = 7
        NUM_MAX_BOXES = 50
        model = YOLOv3(num_classes=NUM_CLASSES)
        # XXX transfer learning
        if FLAGS.pretrain_weights is not None:
            model.backbone.load(FLAGS.pretrain_weights)
        if FLAGS.weights is not None:
            model.load(FLAGS.weights)
        optim = make_optimizer(parameter_list=model.parameters())
        anno_path = os.path.join(FLAGS.data, 'annotations', 'instances_val2017.json')
        inputs = [Input([None, 3, None, None], 'float32', name='image'),
                  Input([None, 3], 'int32', name='img_info')]
        labels = [Input([None, NUM_MAX_BOXES, 4], 'float32', name='gt_bbox'),
                  Input([None, NUM_MAX_BOXES], 'int32', name='gt_label'),
                  Input([None, NUM_MAX_BOXES], 'float32', name='gt_score')]
        model.prepare(optim,
                      YoloLoss(num_classes=NUM_CLASSES),
                      # For YOLOv3, output variable in train/eval is different,
                      # which is not supported by metric, add by callback later?
                      # metrics=COCOMetric(anno_path, with_background=False)
                      inputs=inputs,
                      labels = labels)

        for e in range(epoch):
            logger.info("======== train epoch {} ========".format(e))
            run(model, train_loader)
            model.save('yolo_checkpoints/{:02d}'.format(e))
            logger.info("======== eval epoch {} ========".format(e))
            run(model, val_loader, mode='eval')
            # should be called in fit()
            for metric in model._metrics:
                metric.accumulate()
                metric.reset()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Yolov3 Training on COCO")
    parser.add_argument('data', metavar='DIR', help='path to COCO dataset')
    parser.add_argument(
        "-d", "--dynamic", action='store_true', help="enable dygraph mode")
    parser.add_argument(
        "-e", "--epoch", default=300, type=int, help="number of epoch")
    parser.add_argument(
        '--lr', '--learning-rate', default=0.001, type=float, metavar='LR',
        help='initial learning rate')
    parser.add_argument(
        "-b", "--batch_size", default=64, type=int, help="batch size")
    parser.add_argument(
        "-n", "--num_devices", default=8, type=int, help="number of devices")
    parser.add_argument(
        "-p", "--pretrain_weights", default=None, type=str,
        help="path to pretrained weights")
    parser.add_argument(
        "-w", "--weights", default=None, type=str,
        help="path to model weights")
    FLAGS = parser.parse_args()
    assert FLAGS.data, "error: must provide data path"
    main()
