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

import numpy as np
import pytest
from api_base import ApiBase

import paddle


@pytest.mark.yolo_box
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_yolo_box_0():
    anchors = [10, 13, 16, 30, 33, 23]
    an_num = int(len(anchors) // 2)
    batch_size = 32
    class_num = 2
    conf_thresh = 0.5
    downsample = 32
    clip_bbox = True
    x_shape = [batch_size, (an_num * (5 + class_num)), 13, 13]
    imgsize_shape = [batch_size, 2]
    scale_x_y = 1.0
    iou_aware = False
    iou_aware_factor = 0.5
    test = ApiBase(
        func=paddle.vision.ops.yolo_box,
        feed_names=['x', 'img_size'],
        feed_shapes=[x_shape, imgsize_shape],
        feed_dtypes=['float32', 'int32'],
        input_is_list=False,
        is_train=False,
    )

    np.random.seed(1)
    x = np.random.random(x_shape).astype('float32')
    img_size = np.random.randint(10, 20, imgsize_shape).astype('int32')
    test.run(
        feed=[x, img_size],
        anchors=anchors,
        class_num=class_num,
        conf_thresh=conf_thresh,
        downsample_ratio=downsample,
        clip_bbox=clip_bbox,
        scale_x_y=scale_x_y,
        iou_aware=iou_aware,
        iou_aware_factor=iou_aware_factor,
    )


@pytest.mark.yolo_box
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_yolo_box_1():
    anchors = [10, 13, 16, 30, 33, 23]
    an_num = int(len(anchors) // 2)
    batch_size = 32
    class_num = 2
    conf_thresh = 0.5
    downsample = 32
    clip_bbox = False
    x_shape = [batch_size, (an_num * (5 + class_num)), 13, 13]
    imgsize_shape = [batch_size, 2]
    scale_x_y = 1.0
    iou_aware = False
    iou_aware_factor = 0.5
    test = ApiBase(
        func=paddle.vision.ops.yolo_box,
        feed_names=['x', 'img_size'],
        feed_shapes=[x_shape, imgsize_shape],
        feed_dtypes=['float32', 'int32'],
        input_is_list=False,
        is_train=False,
    )

    np.random.seed(1)
    x = np.random.random(x_shape).astype('float32')
    img_size = np.random.randint(10, 20, imgsize_shape).astype('int32')
    test.run(
        feed=[x, img_size],
        anchors=anchors,
        class_num=class_num,
        conf_thresh=conf_thresh,
        downsample_ratio=downsample,
        clip_bbox=clip_bbox,
        scale_x_y=scale_x_y,
        iou_aware=iou_aware,
        iou_aware_factor=iou_aware_factor,
    )


@pytest.mark.yolo_box
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_yolo_box_2():
    anchors = [10, 13, 16, 30, 33, 23]
    an_num = int(len(anchors) // 2)
    batch_size = 32
    class_num = 2
    conf_thresh = 0.5
    downsample = 32
    clip_bbox = True
    x_shape = [batch_size, (an_num * (5 + class_num)), 13, 13]
    imgsize_shape = [batch_size, 2]
    scale_x_y = 1.2
    iou_aware = False
    iou_aware_factor = 0.5
    test = ApiBase(
        func=paddle.vision.ops.yolo_box,
        feed_names=['x', 'img_size'],
        feed_shapes=[x_shape, imgsize_shape],
        feed_dtypes=['float32', 'int32'],
        input_is_list=False,
        is_train=False,
    )

    np.random.seed(1)
    x = np.random.random(x_shape).astype('float32')
    img_size = np.random.randint(10, 20, imgsize_shape).astype('int32')
    test.run(
        feed=[x, img_size],
        anchors=anchors,
        class_num=class_num,
        conf_thresh=conf_thresh,
        downsample_ratio=downsample,
        clip_bbox=clip_bbox,
        scale_x_y=scale_x_y,
        iou_aware=iou_aware,
        iou_aware_factor=iou_aware_factor,
    )


@pytest.mark.yolo_box
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_yolo_box_3():
    anchors = [10, 13, 16, 30, 33, 23]
    an_num = int(len(anchors) // 2)
    batch_size = 32
    class_num = 2
    conf_thresh = 0.5
    downsample = 32
    clip_bbox = True
    x_shape = [batch_size, (an_num * (6 + class_num)), 13, 13]
    imgsize_shape = [batch_size, 2]
    scale_x_y = 1.0
    iou_aware = True
    iou_aware_factor = 0.5
    test = ApiBase(
        func=paddle.vision.ops.yolo_box,
        feed_names=['x', 'img_size'],
        feed_shapes=[x_shape, imgsize_shape],
        feed_dtypes=['float32', 'int32'],
        input_is_list=False,
        is_train=False,
    )

    np.random.seed(1)
    x = np.random.random(x_shape).astype('float32')
    img_size = np.random.randint(10, 20, imgsize_shape).astype('int32')
    test.run(
        feed=[x, img_size],
        anchors=anchors,
        class_num=class_num,
        conf_thresh=conf_thresh,
        downsample_ratio=downsample,
        clip_bbox=clip_bbox,
        scale_x_y=scale_x_y,
        iou_aware=iou_aware,
        iou_aware_factor=iou_aware_factor,
    )


@pytest.mark.yolo_box
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_yolo_box_4():
    anchors = [10, 13, 16, 30, 33, 23]
    an_num = int(len(anchors) // 2)
    batch_size = 32
    class_num = 2
    conf_thresh = 0.5
    downsample = 32
    clip_bbox = False
    x_shape = [batch_size, (an_num * (5 + class_num)), 13, 9]
    imgsize_shape = [batch_size, 2]
    scale_x_y = 1.0
    iou_aware = False
    iou_aware_factor = 0.5
    test = ApiBase(
        func=paddle.vision.ops.yolo_box,
        feed_names=['x', 'img_size'],
        feed_shapes=[x_shape, imgsize_shape],
        feed_dtypes=['float32', 'int32'],
        input_is_list=False,
        is_train=False,
    )

    np.random.seed(1)
    x = np.random.random(x_shape).astype('float32')
    img_size = np.random.randint(10, 20, imgsize_shape).astype('int32')
    test.run(
        feed=[x, img_size],
        anchors=anchors,
        class_num=class_num,
        conf_thresh=conf_thresh,
        downsample_ratio=downsample,
        clip_bbox=clip_bbox,
        scale_x_y=scale_x_y,
        iou_aware=iou_aware,
        iou_aware_factor=iou_aware_factor,
    )
