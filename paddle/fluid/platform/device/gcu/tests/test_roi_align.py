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
from paddle import fluid


# paddle.vision.ops.roi_align
def make_rois(
    batch_size,
    input_height,
    input_width,
    pooled_height,
    pooled_width,
    spatial_scale,
):
    rois = []
    rois_lod = [[]]
    for bno in range(batch_size):
        rois_lod[0].append(bno + 1)
        for _ in range(bno + 1):
            x1 = np.random.random_integers(
                0, input_width // spatial_scale - pooled_width
            )
            y1 = np.random.random_integers(
                0, input_height // spatial_scale - pooled_height
            )

            x2 = np.random.random_integers(
                x1 + pooled_width, input_width // spatial_scale
            )
            y2 = np.random.random_integers(
                y1 + pooled_height, input_height // spatial_scale
            )

            roi = [bno, x1, y1, x2, y2]
            rois.append(roi)
    rois = np.array(rois).astype("float32")
    boxes_num = np.array([bno + 1 for bno in range(batch_size)]).astype('int32')
    return rois, rois_lod, boxes_num


@pytest.mark.roi_align
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_roi_align():
    np.random.seed(1)
    data = np.random.random(size=[4, 128, 16, 16]).astype('float32')
    rois, rois_lod, boxes_num = make_rois(4, 16, 16, 1, 1, 0.1)
    roi = paddle.fluid.create_lod_tensor(
        rois[:, 1:5], rois_lod, fluid.CPUPlace()
    )

    # TODO(ALL TBD): to test backward, we need first modify api_base.py as below:
    # g = paddle.static.gradients(loss, inputs[0])
    # for grad in g:
    #     if hasattr(grad, 'name'):
    #         fetch_list.append(grad.name)
    test = ApiBase(
        func=paddle.vision.ops.roi_align,
        feed_names=['data', 'boxes', 'boxes_num'],
        is_train=False,
        feed_shapes=[[4, 128, 16, 16], [len(rois), 4], [4]],
        feed_dtypes=['float32', 'float32', 'int32'],
    )

    test.run(
        feed=[data, roi, boxes_num],
        output_size=1,
        spatial_scale=0.1,
        sampling_ratio=1,
    )


@pytest.mark.roi_align
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_roi_align_2():
    input_dim = [3, 3, 8, 6]
    spatial_scale = 1.0 / 2.0
    pooled_height = 2
    pooled_width = 2
    sampling_ratio = -1
    rois, rois_lod, boxes_num = make_rois(
        input_dim[0],
        input_dim[2],
        input_dim[3],
        pooled_height,
        pooled_width,
        spatial_scale,
    )

    # TODO(ALL TBD): to test backward, we need first modify api_base.py as below:
    # g = paddle.static.gradients(loss, inputs[0])
    # for grad in g:
    #     if hasattr(grad, 'name'):
    #         fetch_list.append(grad.name)
    test = ApiBase(
        func=paddle.vision.ops.roi_align,
        feed_names=['data', 'boxes', 'boxes_num'],
        is_train=False,
        feed_shapes=[input_dim, [len(rois), 4], [3]],
        feed_dtypes=['float32', 'float32', 'int32'],
    )

    np.random.seed(1)
    data = np.random.random(size=input_dim).astype('float32')
    roi = paddle.fluid.create_lod_tensor(
        rois[:, 1:5], rois_lod, fluid.CPUPlace()
    )

    test.run(
        feed=[data, roi, boxes_num],
        output_size=(pooled_height, pooled_width),
        spatial_scale=spatial_scale,
        sampling_ratio=sampling_ratio,
    )
