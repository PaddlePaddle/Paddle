# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.append("../../legacy_test")
from test_nms_op import nms

import paddle


def _find(condition):
    """
    Find the indices of elements saticfied the condition.

    Args:
        condition(Tensor[N] or np.ndarray([N,])): Element should be bool type.

    Returns:
        Tensor: Indices of True element.
    """
    res = []
    for i in range(condition.shape[0]):
        if condition[i]:
            res.append(i)
    return np.array(res)


def multiclass_nms(boxes, scores, category_idxs, iou_threshold, top_k):
    mask = np.zeros_like(scores)

    for category_id in np.unique(category_idxs):
        cur_category_boxes_idxs = _find(category_idxs == category_id)
        cur_category_boxes = boxes[cur_category_boxes_idxs]
        cur_category_scores = scores[cur_category_boxes_idxs]
        cur_category_sorted_indices = np.argsort(-cur_category_scores)
        cur_category_sorted_boxes = cur_category_boxes[
            cur_category_sorted_indices
        ]

        cur_category_keep_boxes_sub_idxs = cur_category_sorted_indices[
            nms(cur_category_sorted_boxes, iou_threshold)
        ]

        mask[cur_category_boxes_idxs[cur_category_keep_boxes_sub_idxs]] = True

    keep_boxes_idxs = _find(mask)
    topK_sub_indices = np.argsort(-scores[keep_boxes_idxs])[:top_k]
    return keep_boxes_idxs[topK_sub_indices]


def gen_args(num_boxes, dtype):
    boxes = np.random.rand(num_boxes, 4).astype(dtype)
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

    scores = np.random.rand(num_boxes).astype(dtype)

    categories = [0, 1, 2, 3]
    category_idxs = np.random.choice(categories, num_boxes)

    return boxes, scores, category_idxs, categories


class TestOpsNMS(unittest.TestCase):
    def setUp(self):
        self.num_boxes = 64
        self.threshold = 0.5
        self.topk = 20
        self.dtypes = ['float32']
        self.devices = ['cpu']
        if paddle.is_compiled_with_cuda():
            self.devices.append('gpu')
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.temp_dir.name, './net')

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_nms(self):
        for device in self.devices:
            for dtype in self.dtypes:
                boxes, scores, category_idxs, categories = gen_args(
                    self.num_boxes, dtype
                )
                paddle.set_device(device)
                out = paddle.vision.ops.nms(
                    paddle.to_tensor(boxes),
                    self.threshold,
                    paddle.to_tensor(scores),
                )
                out = paddle.vision.ops.nms(
                    paddle.to_tensor(boxes), self.threshold
                )
                out_py = nms(boxes, self.threshold)

                np.testing.assert_array_equal(
                    out.numpy(),
                    out_py,
                    err_msg=f'paddle out: {out}\n py out: {out_py}\n',
                )

    def test_multiclass_nms_dynamic(self):
        for device in self.devices:
            for dtype in self.dtypes:
                boxes, scores, category_idxs, categories = gen_args(
                    self.num_boxes, dtype
                )
                paddle.set_device(device)
                out = paddle.vision.ops.nms(
                    paddle.to_tensor(boxes),
                    self.threshold,
                    paddle.to_tensor(scores),
                    paddle.to_tensor(category_idxs),
                    categories,
                    self.topk,
                )
                out_py = multiclass_nms(
                    boxes, scores, category_idxs, self.threshold, self.topk
                )

                np.testing.assert_array_equal(
                    out.numpy(),
                    out_py,
                    err_msg=f'paddle out: {out}\n py out: {out_py}\n',
                )

    def test_multiclass_nms_static(self):
        for device in self.devices:
            for dtype in self.dtypes:
                with paddle.static.program_guard(
                    paddle.static.Program(), paddle.static.Program()
                ):
                    paddle.enable_static()
                    boxes, scores, category_idxs, categories = gen_args(
                        self.num_boxes, dtype
                    )
                    boxes_static = paddle.static.data(
                        shape=boxes.shape, dtype=boxes.dtype, name="boxes"
                    )
                    scores_static = paddle.static.data(
                        shape=scores.shape, dtype=scores.dtype, name="scores"
                    )
                    category_idxs_static = paddle.static.data(
                        shape=category_idxs.shape,
                        dtype=category_idxs.dtype,
                        name="category_idxs",
                    )
                    out = paddle.vision.ops.nms(
                        boxes_static,
                        self.threshold,
                        scores_static,
                        category_idxs_static,
                        categories,
                        self.topk,
                    )
                    place = paddle.CPUPlace()
                    if device == 'gpu':
                        place = paddle.CUDAPlace(0)
                    exe = paddle.static.Executor(place)
                    out = exe.run(
                        paddle.static.default_main_program(),
                        feed={
                            'boxes': boxes,
                            'scores': scores,
                            'category_idxs': category_idxs,
                        },
                        fetch_list=[out],
                    )
                    paddle.disable_static()
                    out_py = multiclass_nms(
                        boxes, scores, category_idxs, self.threshold, self.topk
                    )
                    out = np.array(out)
                    out = np.squeeze(out)
                    np.testing.assert_array_equal(
                        out,
                        out_py,
                        err_msg=f'paddle out: {out}\n py out: {out_py}\n',
                    )

    def test_multiclass_nms_dynamic_to_static(self):
        for device in self.devices:
            for dtype in self.dtypes:
                paddle.set_device(device)

                def fun(x):
                    scores = np.arange(0, 64).astype('float32')
                    categories = np.array([0, 1, 2, 3])
                    category_idxs = categories.repeat(16)
                    out = paddle.vision.ops.nms(
                        x,
                        0.1,
                        paddle.to_tensor(scores),
                        paddle.to_tensor(category_idxs),
                        categories,
                        10,
                    )
                    return out

                boxes = np.random.rand(64, 4).astype('float32')
                boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
                boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

                origin = fun(paddle.to_tensor(boxes))
                paddle.jit.save(
                    fun,
                    self.path,
                    input_spec=[
                        paddle.static.InputSpec(
                            shape=[None, 4], dtype='float32', name='x'
                        )
                    ],
                )
                load_func = paddle.jit.load(self.path)
                res = load_func(paddle.to_tensor(boxes))
                np.testing.assert_array_equal(
                    origin,
                    res,
                    err_msg=f'origin out: {origin}\n inference model out: {res}\n',
                )

    def test_matrix_nms_dynamic(self):
        for device in self.devices:
            for dtype in self.dtypes:
                boxes, scores, category_idxs, categories = gen_args(
                    self.num_boxes, dtype
                )
                scores = np.random.rand(1, 4, self.num_boxes).astype(dtype)
                paddle.set_device(device)
                out = paddle.vision.ops.matrix_nms(
                    paddle.to_tensor(boxes).unsqueeze(0),
                    paddle.to_tensor(scores),
                    self.threshold,
                    post_threshold=0.0,
                    nms_top_k=400,
                    keep_top_k=100,
                )


if __name__ == '__main__':
    unittest.main()
