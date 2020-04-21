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

import os
import argparse
import numpy as np

from hapi.model import Input, set_device

from check import check_gpu, check_version
from modeling import tsm_resnet50
from kinetics_dataset import KineticsDataset
from transforms import *

import logging
logger = logging.getLogger(__name__)


def main():
    device = set_device(FLAGS.device)
    fluid.enable_dygraph(device) if FLAGS.dynamic else None

    transform = Compose([GroupScale(),
                         GroupCenterCrop(),
                         NormalizeImage()])
    dataset = KineticsDataset(
            pickle_file=FLAGS.infer_file,
            label_list=FLAGS.label_list,
            mode='test',
            transform=transform)
    labels = dataset.label_list

    model = tsm_resnet50(num_classes=len(labels),
                         pretrained=FLAGS.weights is None)

    inputs = [Input([None, 8, 3, 224, 224], 'float32', name='image')]

    model.prepare(inputs=inputs, device=FLAGS.device)

    if FLAGS.weights is not None:
        model.load(FLAGS.weights, reset_optimizer=True)

    imgs, label = dataset[0]
    pred = model.test([imgs[np.newaxis, :]])
    pred = labels[np.argmax(pred)]
    logger.info("Sample {} predict label: {}, ground truth label: {}" \
                .format(FLAGS.infer_file, pred, labels[int(label)]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("CNN training on TSM")
    parser.add_argument(
        "--data", type=str, default='dataset/kinetics',
        help="path to dataset root directory")
    parser.add_argument(
        "--device", type=str, default='gpu',
        help="device to use, gpu or cpu")
    parser.add_argument(
        "-d", "--dynamic", action='store_true',
        help="enable dygraph mode")
    parser.add_argument(
        "--label_list", type=str, default=None,
        help="path to category index label list file")
    parser.add_argument(
        "--infer_file", type=str, default=None,
        help="path to pickle file for inference")
    parser.add_argument(
        "-w",
        "--weights",
        default=None,
        type=str,
        help="weights path for evaluation")
    FLAGS = parser.parse_args()

    check_gpu(str.lower(FLAGS.device) == 'gpu')
    check_version()
    main()
