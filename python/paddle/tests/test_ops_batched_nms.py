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

import unittest
import numpy as np

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


def iou(box_a, box_b):
    """Apply intersection-over-union overlap between box_a and box_b
    """
    xmin_a = min(box_a[0], box_a[2])
    ymin_a = min(box_a[1], box_a[3])
    xmax_a = max(box_a[0], box_a[2])
    ymax_a = max(box_a[1], box_a[3])

    xmin_b = min(box_b[0], box_b[2])
    ymin_b = min(box_b[1], box_b[3])
    xmax_b = max(box_b[0], box_b[2])
    ymax_b = max(box_b[1], box_b[3])

    area_a = (ymax_a - ymin_a) * (xmax_a - xmin_a)
    area_b = (ymax_b - ymin_b) * (xmax_b - xmin_b)
    if area_a <= 0 and area_b <= 0:
        return 0.0

    xa = max(xmin_a, xmin_b)
    ya = max(ymin_a, ymin_b)
    xb = min(xmax_a, xmax_b)
    yb = min(ymax_a, ymax_b)

    inter_area = max(xb - xa, 0.0) * max(yb - ya, 0.0)

    iou_ratio = inter_area / (area_a + area_b - inter_area)
    return iou_ratio


def nms(boxes, nms_threshold):
    selected_indices = np.zeros(boxes.shape[0], dtype=np.int64)
    keep = np.ones(boxes.shape[0], dtype=int)
    io_ratio = np.ones((boxes.shape[0], boxes.shape[0]), dtype=np.float64)
    cnt = 0
    for i in range(boxes.shape[0]):
        if keep[i] == 0:
            continue
        selected_indices[cnt] = i
        cnt += 1
        for j in range(0, boxes.shape[0]):
            io_ratio[i][j] = iou(boxes[i], boxes[j])
            if keep[j]:
                overlap = iou(boxes[i], boxes[j])
                keep[j] = 1 if overlap <= nms_threshold else 0
            else:
                continue

    return selected_indices


def batched_nms(boxes, scores, category_idxs, iou_threshold, top_k):
    mask = np.zeros_like(scores)

    for category_id in np.unique(category_idxs):
        cur_category_boxes_idxs = _find(category_idxs == category_id)
        cur_category_boxes = boxes[cur_category_boxes_idxs]
        cur_category_scores = scores[cur_category_boxes_idxs]
        cur_category_sorted_indices = np.argsort(-cur_category_scores)
        cur_category_sorted_boxes = cur_category_boxes[
            cur_category_sorted_indices]

        cur_category_keep_boxes_sub_idxs = cur_category_sorted_indices[nms(
            cur_category_sorted_boxes, iou_threshold)]

        mask[cur_category_boxes_idxs[cur_category_keep_boxes_sub_idxs]] = True

    keep_boxes_idxs = _find(mask == True)
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


class TestBatchedNMS(unittest.TestCase):
    def setUp(self):
        self.num_boxes = 640
        self.threshold = 0.5
        self.topk = 20
        self.dtypes = ['float32']
        self.devices = ['cpu']
        if paddle.is_compiled_with_cuda():
            self.devices.append('gpu')

    def test_batched_nms_dynamic(self):
        for device in self.devices:
            for dtype in self.dtypes:
                if device == 'cpu' and dtype == 'float16':
                    continue
                boxes, scores, category_idxs, categories = gen_args(
                    self.num_boxes, dtype)
                paddle.set_device(device)
                out = paddle.vision.ops.batched_nms(
                    paddle.to_tensor(boxes),
                    paddle.to_tensor(scores),
                    paddle.to_tensor(category_idxs), categories, self.threshold,
                    self.topk)
                out_py = batched_nms(boxes, scores, category_idxs,
                                     self.threshold, self.topk)

                self.assertTrue(
                    np.array_equal(out.numpy(), out_py),
                    "paddle out: {}\n py out: {}\n".format(out, out_py))

    def test_batched_nms_static(self):
        for device in self.devices:
            for dtype in self.dtypes:
                if device == 'cpu' and dtype == 'float16':
                    continue
                paddle.enable_static()
                paddle.set_device(device)
                boxes, scores, category_idxs, categories = gen_args(
                    self.num_boxes, dtype)
                boxes_static = paddle.static.data(
                    shape=boxes.shape, dtype=boxes.dtype, name="boxes")
                scores_static = paddle.static.data(
                    shape=scores.shape, dtype=scores.dtype, name="scores")
                category_idxs_static = paddle.static.data(
                    shape=category_idxs.shape,
                    dtype=category_idxs.dtype,
                    name="category_idxs")
                out = paddle.vision.ops.batched_nms(
                    boxes_static, scores_static, category_idxs_static,
                    categories, self.threshold, self.topk)
                place = paddle.CPUPlace()
                exe = paddle.static.Executor(place)
                out = exe.run(paddle.static.default_main_program(),
                              feed={
                                  'boxes': boxes,
                                  'scores': scores,
                                  'category_idxs': category_idxs
                              },
                              fetch_list=[out])
                paddle.disable_static()
                out_py = batched_nms(boxes, scores, category_idxs,
                                     self.threshold, self.topk)
                out = np.array(out)
                out = np.squeeze(out)
                self.assertTrue(
                    np.array_equal(out, out_py),
                    "paddle out: {}\n py out: {}\n".format(out, out_py))


if __name__ == '__main__':
    unittest.main()
