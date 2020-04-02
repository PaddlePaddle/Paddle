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

import numpy as np
from PIL import Image, ImageDraw

import logging
logger = logging.getLogger(__name__)

__all__ = ['draw_bbox']


def color_map(num_classes):
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = np.array(color_map).reshape(-1, 3)
    return color_map


def draw_bbox(image, catid2name, bboxes, threshold):
    """
    Draw bbox on image
    """
    bboxes = np.array(bboxes)
    if bboxes.shape[1] != 6:
        logger.info("No bbox detect")
        return image

    draw = ImageDraw.Draw(image)

    catid2color = {}
    color_list = color_map(len(catid2name))
    for bbox in bboxes:
        catid, score, xmin, ymin, xmax, ymax = bbox

        if score < threshold:
            continue

        if catid not in catid2color:
            idx = np.random.randint(len(color_list))
            catid2color[catid] = color_list[idx]
        color = tuple(catid2color[catid])

        # draw bbox
        draw.line(
            [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
             (xmin, ymin)],
            width=2,
            fill=color)
        logger.info("detect {} at {} score: {:.2f}".format(
            catid2name[int(catid)], [xmin, ymin, xmax, ymax], score))

        # draw label
        text = "{} {:.2f}".format(catid2name[catid], score)
        tw, th = draw.textsize(text)
        draw.rectangle(
            [(xmin + 1, ymin - th), (xmin + tw + 1, ymin)], fill=color)
        draw.text((xmin + 1, ymin - th), text, fill=(255, 255, 255))

    return image
