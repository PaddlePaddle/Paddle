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

import argparse
import contextlib
import math
import os
import random

import cv2
import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear

from model import Model


def center_crop_resize(img):
    h, w = img.shape[:2]
    c = int(224 / 256 * min((h, w)))
    i = (h + 1 - c) // 2
    j = (w + 1 - c) // 2
    img = img[i: i + c, j: j + c, :]
    return cv2.resize(img, (224, 224), 0, 0, cv2.INTER_LINEAR)


def random_crop_resize(img):
    height, width = img.shape[:2]
    area = height * width

    for attempt in range(10):
        target_area = random.uniform(0.08, 1.) * area
        log_ratio = (math.log(3 / 4), math.log(4 / 3))
        aspect_ratio = math.exp(random.uniform(*log_ratio))

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if w <= width and h <= height:
            i = random.randint(0, height - h)
            j = random.randint(0, width - w)
            img = img[i: i + h, j: j + w, :]
            return cv2.resize(img, (224, 224), 0, 0, cv2.INTER_LINEAR)

    return center_crop_resize(img)


def random_flip(img):
    return img[:, ::-1, :]


def normalize_permute(img):
    # transpose and convert to RGB from BGR
    img = img.astype(np.float32).transpose((2, 0, 1))[::-1, ...]
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.120, 57.375], dtype=np.float32)
    invstd = 1. / std
    for v, m, s in zip(img, mean, invstd):
        v.__isub__(m).__imul__(s)
    return img


def compose(functions):
    def process(sample):
        img, label = sample
        for fn in functions:
            img = fn(img)
        return img, label
    return process


def image_folder(path, shuffle=False):
    valid_ext = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.webp')
    classes = [d for d in os.listdir(path) if
               os.path.isdir(os.path.join(path, d))]
    classes.sort()
    class_map = {cls: idx for idx, cls in enumerate(classes)}
    samples = []
    for dir in sorted(class_map.keys()):
        d = os.path.join(path, dir)
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                p = os.path.join(root, fname)
                if os.path.splitext(p)[1].lower() in valid_ext:
                    samples.append((p, class_map[dir]))
    if shuffle:
        random.shuffle(samples)

    def iterator():
        for s in samples:
            yield s

    return iterator


class ConvBNLayer(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act=None):
        super(ConvBNLayer, self).__init__()

        self._conv = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            bias_attr=False)

        self._batch_norm = BatchNorm(num_filters, act=act)

    def forward(self, inputs):
        x = self._conv(inputs)
        x = self._batch_norm(x)

        return x


class BottleneckBlock(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 shortcut=True):
        super(BottleneckBlock, self).__init__()

        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1,
            act='relu')
        self.conv1 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu')
        self.conv2 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters * 4,
            filter_size=1,
            act=None)

        if not shortcut:
            self.short = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters * 4,
                filter_size=1,
                stride=stride)

        self.shortcut = shortcut

        self._num_channels_out = num_filters * 4

    def forward(self, inputs):
        x = self.conv0(inputs)
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        x = fluid.layers.elementwise_add(x=short, y=conv2)

        layer_helper = LayerHelper(self.full_name(), act='relu')
        return layer_helper.append_activation(x)


class ResNet(Model):
    def __init__(self, depth=50, num_classes=1000):
        super(ResNet, self).__init__()

        layer_config = {
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3],
        }
        assert depth in layer_config.keys(), \
            "supported depth are {} but input layer is {}".format(
                layer_config.keys(), depth)

        layers = layer_config[depth]
        num_channels = [64, 256, 512, 1024]
        num_filters = [64, 128, 256, 512]

        self.conv = ConvBNLayer(
            num_channels=3,
            num_filters=64,
            filter_size=7,
            stride=2,
            act='relu')
        self.pool = Pool2D(
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max')

        self.blocks = []
        for b in range(len(layers)):
            shortcut = False
            for i in range(layers[b]):
                block = self.add_sublayer(
                    'layer_{}_{}'.format(b, i),
                    BottleneckBlock(
                        num_channels=num_channels[b]
                        if i == 0 else num_filters[b] * 4,
                        num_filters=num_filters[b],
                        stride=2 if i == 0 and b != 0 else 1,
                        shortcut=shortcut))
                self.blocks.append(block)
                shortcut = True

        self.global_pool = Pool2D(
            pool_size=7, pool_type='avg', global_pooling=True)

        stdv = 1.0 / math.sqrt(2048 * 1.0)
        self.fc_input_dim = num_filters[len(num_filters) - 1] * 4 * 1 * 1
        self.fc = Linear(self.fc_input_dim,
                         num_classes,
                         act='softmax',
                         param_attr=fluid.param_attr.ParamAttr(
                             initializer=fluid.initializer.Uniform(
                                 -stdv, stdv)))

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.pool(x)
        for block in self.blocks:
            x = block(x)
        x = self.global_pool(x)
        x = fluid.layers.reshape(x, shape=[-1, self.fc_input_dim])
        x = self.fc(x)
        return x


def make_optimizer(parameter_list=None):
    total_images = 1281167
    base_lr = 0.1
    momentum = 0.9
    l2_decay = 1e-4

    step_per_epoch = int(math.floor(float(total_images) / FLAGS.batch_size))
    boundaries = [step_per_epoch * e for e in [30, 60, 80]]

    lr = [base_lr * (0.1**i) for i in range(len(boundaries) + 1)]
    optimizer = fluid.optimizer.Momentum(
        learning_rate=fluid.layers.piecewise_decay(
            boundaries=boundaries, values=lr),
        momentum=momentum,
        regularization=fluid.regularizer.L2Decay(l2_decay),
        parameter_list=parameter_list)
    return optimizer


def accuracy(pred, label, topk=(1, )):
    maxk = max(topk)
    pred = np.argsort(pred)[:, ::-1][:, :maxk]
    correct = (pred == np.repeat(label, maxk, 1))

    batch_size = label.shape[0]
    res = []
    for k in topk:
        correct_k = correct[:, :k].sum()
        res.append(100.0 * correct_k / batch_size)
    return res


def run(model, loader, mode='train'):
    total_loss = 0.0
    total_acc1 = 0.0
    total_acc5 = 0.0
    num_steps = 0
    device_ids = list(range(FLAGS.num_devices))
    for idx, batch in enumerate(loader()):
        outputs, losses = getattr(model, mode)(
            batch[0], batch[1], device='gpu', device_ids=device_ids)
        top1, top5 = accuracy(outputs[0], batch[1], topk=(1, 5))

        total_loss += np.sum(losses)
        total_acc1 += top1
        total_acc5 += top5
        num_steps += 1
        if idx % 10 == 0:
            print("{:04d}: loss {:0.3f} top1: {:0.3f}% top5: {:0.3f}%".format(
                idx, total_loss / num_steps,
                total_acc1 / num_steps, total_acc5 / num_steps))
        num_steps += 1


def main():
    @contextlib.contextmanager
    def null_guard():
        yield

    epoch = FLAGS.epoch
    batch_size = FLAGS.batch_size
    if FLAGS.dynamic:
        guard = fluid.dygraph.guard()
    else:
        guard = null_guard()

    train_dir = os.path.join(FLAGS.data, 'train')
    val_dir = os.path.join(FLAGS.data, 'val')

    train_loader = fluid.io.xmap_readers(
        lambda batch: (np.array([b[0] for b in batch]),
                       np.array([b[1] for b in batch]).reshape(-1, 1)),
        paddle.batch(
            fluid.io.xmap_readers(
                compose([cv2.imread, random_crop_resize, random_flip,
                         normalize_permute]),
                image_folder(train_dir, shuffle=True),
                process_num=8,
                buffer_size=4 * batch_size),
            batch_size=batch_size,
            drop_last=True),
        process_num=2, buffer_size=4)

    val_loader = fluid.io.xmap_readers(
        lambda batch: (np.array([b[0] for b in batch]),
                       np.array([b[1] for b in batch]).reshape(-1, 1)),
        paddle.batch(
            fluid.io.xmap_readers(
                compose([cv2.imread, center_crop_resize, normalize_permute]),
                image_folder(val_dir),
                process_num=8,
                buffer_size=4 * batch_size),
            batch_size=batch_size),
        process_num=2, buffer_size=4)

    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')

    with guard:
        model = ResNet()
        sgd = make_optimizer(parameter_list=model.parameters())
        model.prepare(sgd, 'cross_entropy')

        for e in range(epoch):
            print("======== train epoch {} ========".format(e))
            run(model, train_loader)
            model.save('checkpoints/{:02d}'.format(e))
            print("======== eval epoch {} ========".format(e))
            run(model, val_loader, mode='eval')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Resnet Training on ImageNet")
    parser.add_argument('data', metavar='DIR', help='path to dataset '
                        '(should have subdirectories named "train" and "val"')
    parser.add_argument(
        "-e", "--epoch", default=90, type=int, help="number of epoch")
    parser.add_argument(
        "-b", "--batch_size", default=512, type=int, help="batch size")
    parser.add_argument(
        "-n", "--num_devices", default=4, type=int, help="number of devices")
    parser.add_argument(
        "-d", "--dynamic", action='store_true', help="enable dygraph mode")
    FLAGS = parser.parse_args()
    main()
