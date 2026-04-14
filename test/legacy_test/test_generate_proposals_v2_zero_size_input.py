#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

"""
Unit tests for generate_proposals with zero-size inputs.

Covers two crash scenarios discovered in a.config:
  1. im_shape has zero second dimension (shape [N, 0], numel=0):
       -> ClipTiledBoxes / FilterBoxes would dereference empty data pointer -> SIGSEGV
  2. bbox_deltas has zero batch dimension (shape [0, C, H, W], numel=0)
       while scores batch > 0:
       -> rpn_rois pre-allocated as [0,4], AppendProposals writes to nullptr -> SIGSEGV

Fix: GenerateProposalsKernel now returns empty tensors immediately when
     bbox_deltas.numel() == 0 or im_shape.numel() == 0.
"""

import unittest

import numpy as np

import paddle

# Force CPU execution: these tests target the CPU kernel fix.
paddle.device.set_device('cpu')


def _call_generate_proposals(
    scores_shape,
    bbox_deltas_shape,
    im_shape_val,
    anchors_shape,
    variances_shape,
    pre_nms_top_n,
    post_nms_top_n,
    nms_thresh=0.7,
    min_size=0.0,
    eta=1.0,
    pixel_offset=False,
    return_rois_num=True,
):
    """Helper: construct tensors on CPU and call generate_proposals."""
    cpu = paddle.CPUPlace()
    np.random.seed(0)
    scores = paddle.to_tensor(
        np.random.rand(*scores_shape).astype('float32'), place=cpu
    )
    bbox_deltas = paddle.to_tensor(
        np.random.rand(*bbox_deltas_shape).astype('float32')
        if 0 not in bbox_deltas_shape
        else np.empty(bbox_deltas_shape, dtype='float32'),
        place=cpu,
    )
    im_shape = paddle.to_tensor(
        np.array(im_shape_val, dtype='float32'), place=cpu
    )
    anchors = paddle.to_tensor(
        np.random.rand(*anchors_shape).astype('float32'), place=cpu
    )
    variances = paddle.to_tensor(
        np.ones(variances_shape, dtype='float32'), place=cpu
    )

    return paddle.vision.ops.generate_proposals(
        scores,
        bbox_deltas,
        im_shape,
        anchors,
        variances,
        pre_nms_top_n=pre_nms_top_n,
        post_nms_top_n=post_nms_top_n,
        nms_thresh=nms_thresh,
        min_size=min_size,
        eta=eta,
        pixel_offset=pixel_offset,
        return_rois_num=return_rois_num,
    )


class TestGenerateProposalsZeroImShape(unittest.TestCase):
    """
    im_shape second dimension = 0  (e.g. shape [1, 0] or [2, 0]).

    Corresponds to a.config lines with Tensor([1, 0],"float32") or
    Tensor([2, 0],"float32") as im_shape argument.
    Expected: no crash; rpn_rois shape [0,4], rpn_roi_probs shape [0,1].
    """

    def setUp(self):
        paddle.disable_static()

    def _check_zero_output(self, result, return_rois_num):
        if return_rois_num:
            rpn_rois, rpn_roi_probs, _ = result
        else:
            rpn_rois, rpn_roi_probs = result
        self.assertEqual(
            list(rpn_rois.shape),
            [0, 4],
            f"rpn_rois shape {rpn_rois.shape} != [0, 4]",
        )
        self.assertEqual(
            list(rpn_roi_probs.shape),
            [0, 1],
            f"rpn_roi_probs shape {rpn_roi_probs.shape} != [0, 1]",
        )

    def test_im_shape_zero_dim1_batch1_c15_h40_w60(self):
        """a.config line 2: scores[1,15,40,60], bbox_deltas[1,60,40,60], im_shape[1,0]"""
        result = _call_generate_proposals(
            scores_shape=[1, 15, 40, 60],
            bbox_deltas_shape=[1, 60, 40, 60],
            im_shape_val=np.empty((1, 0), dtype='float32'),
            anchors_shape=[36000, 4],
            variances_shape=[36000, 4],
            pre_nms_top_n=12000,
            post_nms_top_n=2000,
        )
        self._check_zero_output(result, return_rois_num=True)

    def test_im_shape_zero_dim1_batch1_c15_h42_w63(self):
        """a.config line 4: scores[1,15,42,63], bbox_deltas[1,60,42,63], im_shape[1,0]"""
        result = _call_generate_proposals(
            scores_shape=[1, 15, 42, 63],
            bbox_deltas_shape=[1, 60, 42, 63],
            im_shape_val=np.empty((1, 0), dtype='float32'),
            anchors_shape=[39690, 4],
            variances_shape=[39690, 4],
            pre_nms_top_n=12000,
            post_nms_top_n=2000,
        )
        self._check_zero_output(result, return_rois_num=True)

    def test_im_shape_zero_dim1_batch1_c3_h10_w14(self):
        """a.config line 6: scores[1,3,10,14], bbox_deltas[1,12,10,14], im_shape[1,0]"""
        result = _call_generate_proposals(
            scores_shape=[1, 3, 10, 14],
            bbox_deltas_shape=[1, 12, 10, 14],
            im_shape_val=np.empty((1, 0), dtype='float32'),
            anchors_shape=[420, 4],
            variances_shape=[420, 4],
            pre_nms_top_n=2000,
            post_nms_top_n=2000,
        )
        self._check_zero_output(result, return_rois_num=True)

    def test_im_shape_zero_dim1_batch1_c3_h10_w15(self):
        """a.config line 8: scores[1,3,10,15], bbox_deltas[1,12,10,15], im_shape[1,0]"""
        result = _call_generate_proposals(
            scores_shape=[1, 3, 10, 15],
            bbox_deltas_shape=[1, 12, 10, 15],
            im_shape_val=np.empty((1, 0), dtype='float32'),
            anchors_shape=[450, 4],
            variances_shape=[450, 4],
            pre_nms_top_n=2000,
            post_nms_top_n=2000,
        )
        self._check_zero_output(result, return_rois_num=True)

    def test_im_shape_zero_dim1_batch1_c4_h16_w16_pixel_offset(self):
        """a.config line 10: scores[1,4,16,16], bbox_deltas[1,16,16,16], im_shape[1,0], pixel_offset=True"""
        result = _call_generate_proposals(
            scores_shape=[1, 4, 16, 16],
            bbox_deltas_shape=[1, 16, 16, 16],
            im_shape_val=np.empty((1, 0), dtype='float32'),
            anchors_shape=[16, 16, 4, 4],
            variances_shape=[16, 16, 4, 4],
            pre_nms_top_n=12000,
            post_nms_top_n=5000,
            nms_thresh=0.7,
            min_size=3.0,
            pixel_offset=True,
        )
        self._check_zero_output(result, return_rois_num=True)

    def test_im_shape_zero_dim1_batch1_c9_h10_w8(self):
        """a.config line 12: scores[1,9,10,8], bbox_deltas[1,36,10,8], im_shape[1,0]"""
        result = _call_generate_proposals(
            scores_shape=[1, 9, 10, 8],
            bbox_deltas_shape=[1, 36, 10, 8],
            im_shape_val=np.empty((1, 0), dtype='float32'),
            anchors_shape=[720, 4],
            variances_shape=[720, 4],
            pre_nms_top_n=4000,
            post_nms_top_n=4000,
        )
        self._check_zero_output(result, return_rois_num=True)

    def test_im_shape_zero_dim1_batch1_c9_h12_w9(self):
        """a.config line 14: scores[1,9,12,9], bbox_deltas[1,36,12,9], im_shape[1,0]"""
        result = _call_generate_proposals(
            scores_shape=[1, 9, 12, 9],
            bbox_deltas_shape=[1, 36, 12, 9],
            im_shape_val=np.empty((1, 0), dtype='float32'),
            anchors_shape=[972, 4],
            variances_shape=[972, 4],
            pre_nms_top_n=4000,
            post_nms_top_n=4000,
        )
        self._check_zero_output(result, return_rois_num=True)

    def test_im_shape_zero_dim1_batch2_c3_h4_w4(self):
        """a.config line 16: scores[2,3,4,4], bbox_deltas[2,12,4,4], im_shape[2,0]"""
        result = _call_generate_proposals(
            scores_shape=[2, 3, 4, 4],
            bbox_deltas_shape=[2, 12, 4, 4],
            im_shape_val=np.empty((2, 0), dtype='float32'),
            anchors_shape=[4, 4, 3, 4],
            variances_shape=[4, 4, 3, 4],
            pre_nms_top_n=10,
            post_nms_top_n=5,
        )
        self._check_zero_output(result, return_rois_num=True)


class TestGenerateProposalsZeroBboxDeltas(unittest.TestCase):
    """
    bbox_deltas batch dimension = 0 (e.g. shape [0, C, H, W], numel=0)
    while scores batch > 0.

    Corresponds to a.config lines with Tensor([0, C, H, W],"float32") as
    bbox_deltas argument (odd-numbered pairs in a.config).
    Expected: no crash; rpn_rois shape [0,4], rpn_roi_probs shape [0,1].
    """

    def setUp(self):
        paddle.disable_static()

    def _check_zero_output(self, result, return_rois_num):
        if return_rois_num:
            rpn_rois, rpn_roi_probs, _ = result
        else:
            rpn_rois, rpn_roi_probs = result
        self.assertEqual(
            list(rpn_rois.shape),
            [0, 4],
            f"rpn_rois shape {rpn_rois.shape} != [0, 4]",
        )
        self.assertEqual(
            list(rpn_roi_probs.shape),
            [0, 1],
            f"rpn_roi_probs shape {rpn_roi_probs.shape} != [0, 1]",
        )

    def test_bbox_deltas_zero_batch_c15_h40_w60(self):
        """a.config line 1: scores[1,15,40,60], bbox_deltas[0,60,40,60], im_shape[1,2]"""
        result = _call_generate_proposals(
            scores_shape=[1, 15, 40, 60],
            bbox_deltas_shape=[0, 60, 40, 60],
            im_shape_val=[[60.0, 40.0]],
            anchors_shape=[36000, 4],
            variances_shape=[36000, 4],
            pre_nms_top_n=12000,
            post_nms_top_n=2000,
        )
        self._check_zero_output(result, return_rois_num=True)

    def test_bbox_deltas_zero_batch_c15_h42_w63(self):
        """a.config line 3: scores[1,15,42,63], bbox_deltas[0,60,42,63], im_shape[1,2]"""
        result = _call_generate_proposals(
            scores_shape=[1, 15, 42, 63],
            bbox_deltas_shape=[0, 60, 42, 63],
            im_shape_val=[[60.0, 42.0]],
            anchors_shape=[39690, 4],
            variances_shape=[39690, 4],
            pre_nms_top_n=12000,
            post_nms_top_n=2000,
        )
        self._check_zero_output(result, return_rois_num=True)

    def test_bbox_deltas_zero_batch_c3_h10_w14(self):
        """a.config line 5: scores[1,3,10,14], bbox_deltas[0,12,10,14], im_shape[1,2]"""
        result = _call_generate_proposals(
            scores_shape=[1, 3, 10, 14],
            bbox_deltas_shape=[0, 12, 10, 14],
            im_shape_val=[[12.0, 10.0]],
            anchors_shape=[420, 4],
            variances_shape=[420, 4],
            pre_nms_top_n=2000,
            post_nms_top_n=2000,
        )
        self._check_zero_output(result, return_rois_num=True)

    def test_bbox_deltas_zero_batch_c3_h10_w15(self):
        """a.config line 7: scores[1,3,10,15], bbox_deltas[0,12,10,15], im_shape[1,2]"""
        result = _call_generate_proposals(
            scores_shape=[1, 3, 10, 15],
            bbox_deltas_shape=[0, 12, 10, 15],
            im_shape_val=[[12.0, 10.0]],
            anchors_shape=[450, 4],
            variances_shape=[450, 4],
            pre_nms_top_n=2000,
            post_nms_top_n=2000,
        )
        self._check_zero_output(result, return_rois_num=True)

    def test_bbox_deltas_zero_batch_c4_h16_w16_pixel_offset(self):
        """a.config line 9: scores[1,4,16,16], bbox_deltas[0,16,16,16], im_shape[1,2], pixel_offset=True"""
        result = _call_generate_proposals(
            scores_shape=[1, 4, 16, 16],
            bbox_deltas_shape=[0, 16, 16, 16],
            im_shape_val=[[16.0, 16.0]],
            anchors_shape=[16, 16, 4, 4],
            variances_shape=[16, 16, 4, 4],
            pre_nms_top_n=12000,
            post_nms_top_n=5000,
            nms_thresh=0.7,
            min_size=3.0,
            pixel_offset=True,
        )
        self._check_zero_output(result, return_rois_num=True)

    def test_bbox_deltas_zero_batch_c9_h10_w8(self):
        """a.config line 11: scores[1,9,10,8], bbox_deltas[0,36,10,8], im_shape[1,2]"""
        result = _call_generate_proposals(
            scores_shape=[1, 9, 10, 8],
            bbox_deltas_shape=[0, 36, 10, 8],
            im_shape_val=[[36.0, 10.0]],
            anchors_shape=[720, 4],
            variances_shape=[720, 4],
            pre_nms_top_n=4000,
            post_nms_top_n=4000,
        )
        self._check_zero_output(result, return_rois_num=True)

    def test_bbox_deltas_zero_batch_c9_h12_w9(self):
        """a.config line 13: scores[1,9,12,9], bbox_deltas[0,36,12,9], im_shape[1,2]"""
        result = _call_generate_proposals(
            scores_shape=[1, 9, 12, 9],
            bbox_deltas_shape=[0, 36, 12, 9],
            im_shape_val=[[36.0, 12.0]],
            anchors_shape=[972, 4],
            variances_shape=[972, 4],
            pre_nms_top_n=4000,
            post_nms_top_n=4000,
        )
        self._check_zero_output(result, return_rois_num=True)

    def test_bbox_deltas_zero_batch_multibatch_c3_h4_w4(self):
        """a.config line 15: scores[2,3,4,4], bbox_deltas[0,12,4,4], im_shape[2,3]"""
        result = _call_generate_proposals(
            scores_shape=[2, 3, 4, 4],
            bbox_deltas_shape=[0, 12, 4, 4],
            im_shape_val=[[4.0, 4.0, 1.0], [4.0, 4.0, 1.0]],
            anchors_shape=[4, 4, 3, 4],
            variances_shape=[4, 4, 3, 4],
            pre_nms_top_n=10,
            post_nms_top_n=5,
        )
        self._check_zero_output(result, return_rois_num=True)


if __name__ == '__main__':
    paddle.disable_static()
    unittest.main()
