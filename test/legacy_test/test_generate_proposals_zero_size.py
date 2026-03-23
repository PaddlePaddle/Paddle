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
from op_test import get_device_place, is_custom_device

import paddle
from paddle.base import core


def make_tensor(shape, dtype, place):
    np_dtype = {
        "float32": np.float32,
        "int32": np.int32,
    }[dtype]
    return paddle.to_tensor(np.zeros(shape, dtype=np_dtype), place=place)


@unittest.skipIf(
    not (core.is_compiled_with_cuda() or is_custom_device())
    or core.is_compiled_with_rocm(),
    "generate_proposals zero-size regressions require CUDA",
)
class TestGenerateProposalsZeroSize(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.place = get_device_place()

    def tearDown(self):
        paddle.enable_static()

    def test_empty_im_shape_returns_true_empty_outputs(self):
        rois, probs, rois_num = paddle.vision.ops.generate_proposals(
            make_tensor([2, 3, 4, 4], "float32", self.place),
            make_tensor([2, 12, 4, 4], "float32", self.place),
            make_tensor([2, 0], "float32", self.place),
            make_tensor([4, 4, 3, 4], "float32", self.place),
            make_tensor([4, 4, 3, 4], "float32", self.place),
            pre_nms_top_n=10,
            post_nms_top_n=5,
            return_rois_num=True,
        )
        self.assertEqual(list(rois.shape), [0, 4])
        self.assertEqual(list(probs.shape), [0, 1])
        np.testing.assert_array_equal(
            rois_num.numpy(), np.array([0, 0], dtype=np.int32)
        )

    def test_keep_num_zero_does_not_fabricate_dummy_boxes(self):
        rois, probs, rois_num = paddle.vision.ops.generate_proposals(
            make_tensor([1, 3, 4, 4], "float32", self.place),
            make_tensor([1, 12, 4, 4], "float32", self.place),
            paddle.to_tensor(
                [[1.0, 1.0, 1.0]], dtype="float32", place=self.place
            ),
            make_tensor([4, 4, 3, 4], "float32", self.place),
            make_tensor([4, 4, 3, 4], "float32", self.place),
            pre_nms_top_n=10,
            post_nms_top_n=5,
            min_size=1000.0,
            eta=1.0,
            pixel_offset=True,
            return_rois_num=True,
        )
        self.assertEqual(list(rois.shape), [0, 4])
        self.assertEqual(list(probs.shape), [0, 1])
        np.testing.assert_array_equal(rois_num.numpy(), np.array([0], "int32"))


if __name__ == "__main__":
    unittest.main()
