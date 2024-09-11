# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import numpy as np
from tensorrt_test_base import TensorRTBaseTest

from paddle import _C_ops


def multiclass_nms3(
    bboxes,
    scores,
    rois_num=None,
    score_threshold=0.3,
    nms_top_k=4,
    keep_top_k=1,
    nms_threshold=0.3,
    normalized=True,
    nms_eta=1.5,
    background_label=-1,
    return_index=False,
    return_rois_num=True,
    name=None,
):
    attrs = (
        score_threshold,
        nms_top_k,
        keep_top_k,
        nms_threshold,
        normalized,
        nms_eta,
        background_label,
    )
    output, index, nms_rois_num = _C_ops.multiclass_nms3(
        bboxes, scores, rois_num, *attrs
    )
    if not return_index:
        index = None
    return output, nms_rois_num, index


class TestMulticlassNMS3TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = multiclass_nms3
        self.api_args = {
            "bboxes": np.random.randn(2, 5, 4).astype(np.float32),
            "scores": np.random.randn(2, 4, 5).astype(np.float32),
        }
        self.program_config = {"feed_list": ["bboxes", "scores"]}
        self.min_shape = {"bboxes": [1, 5, 4], "scores": [1, 4, 5]}
        self.max_shape = {"bboxes": [3, 5, 4], "scores": [3, 4, 5]}

    def test_trt_result(self):
        self.check_trt_result()


class TestMulticlassNMS3Marker(TensorRTBaseTest):
    def setUp(self):
        self.python_api = multiclass_nms3
        self.api_args = {
            "bboxes": np.random.randn(2, 5, 4, 1).astype(np.float32),
            "scores": np.random.randn(2, 4, 5, 1).astype(np.float32),
        }
        self.program_config = {"feed_list": ["bboxes", "scores"]}
        self.target_marker_op = "pd_op.multiclass_nms3"

    def test_trt_result(self):
        self.check_marker(expected_result=False)


if __name__ == '__main__':
    unittest.main()
