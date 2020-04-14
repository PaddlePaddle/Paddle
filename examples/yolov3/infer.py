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
from PIL import Image 

from paddle import fluid
from paddle.fluid.optimizer import Momentum
from paddle.io import DataLoader

from hapi.model import Model, Input, set_device
from hapi.vision.models import yolov3_darknet53, YoloLoss
from hapi.vision.transforms import *

from coco import COCODataset
from visualizer import draw_bbox

import logging
logger = logging.getLogger(__name__)

IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]


def get_save_image_name(output_dir, image_path):
    """
    Get save image name from source image path.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    image_name = os.path.split(image_path)[-1]
    name, ext = os.path.splitext(image_name)
    return os.path.join(output_dir, "{}".format(name)) + ext


def load_labels(label_list, with_background=True):
    idx = int(with_background)
    cat2name = {}
    with open(label_list) as f:
        for line in f.readlines():
            line = line.strip()
            if line:
                cat2name[idx] = line
                idx += 1
    return cat2name


def main():
    device = set_device(FLAGS.device)
    fluid.enable_dygraph(device) if FLAGS.dynamic else None
    
    inputs = [Input([None, 1], 'int64', name='img_id'),
              Input([None, 2], 'int32', name='img_shape'),
              Input([None, 3, None, None], 'float32', name='image')]

    cat2name = load_labels(FLAGS.label_list, with_background=False)

    model = yolov3_darknet53(num_classes=len(cat2name),
                             model_mode='test',
                             pretrained=FLAGS.weights is None)

    model.prepare(inputs=inputs, device=FLAGS.device)

    if FLAGS.weights is not None:
        model.load(FLAGS.weights, reset_optimizer=True)

    # image preprocess
    orig_img = Image.open(FLAGS.infer_image).convert('RGB')
    w, h  = orig_img.size
    img = orig_img.resize((608, 608), Image.BICUBIC)
    img = np.array(img).astype('float32') / 255.0
    img -= np.array(IMAGE_MEAN)
    img /= np.array(IMAGE_STD)
    img = img.transpose((2, 0, 1))[np.newaxis, :]
    img_id = np.array([0]).astype('int64')[np.newaxis, :]
    img_shape = np.array([h, w]).astype('int32')[np.newaxis, :]

    _, bboxes = model.test([img_id, img_shape, img])

    vis_img = draw_bbox(orig_img, cat2name, bboxes, FLAGS.draw_threshold)
    save_name = get_save_image_name(FLAGS.output_dir, FLAGS.infer_image)
    logger.info("Detection bbox results save in {}".format(save_name))
    vis_img.save(save_name, quality=95)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Yolov3 Training on VOC")
    parser.add_argument(
        "--device", type=str, default='gpu', help="device to use, gpu or cpu")
    parser.add_argument(
        "-d", "--dynamic", action='store_true', help="enable dygraph mode")
    parser.add_argument(
        "--label_list", type=str, default=None,
        help="path to category label list file")
    parser.add_argument(
        "-t", "--draw_threshold", type=float, default=0.5,
        help="threshold to reserve the result for visualization")
    parser.add_argument(
        "-i", "--infer_image", type=str, default=None,
        help="image path for inference")
    parser.add_argument(
        "-o", "--output_dir", type=str, default='output',
        help="directory to save inference result if --visualize is set")
    parser.add_argument(
        "-w", "--weights", default=None, type=str,
        help="path to weights for inference")
    FLAGS = parser.parse_args()
    assert os.path.isfile(FLAGS.infer_image), \
            "infer_image {} not a file".format(FLAGS.infer_image)
    assert os.path.isfile(FLAGS.label_list), \
            "label_list {} not a file".format(FLAGS.label_list)
    main()
