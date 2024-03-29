// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/kernels/matrix_nms_kernel.h"

#include "paddle/common/ddim.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <class T>
static inline T BBoxArea(const T* box, const bool normalized) {
  if (box[2] < box[0] || box[3] < box[1]) {
    // If coordinate values are invalid
    // (e.g. xmax < xmin or ymax < ymin), return 0.
    return static_cast<T>(0.);
  } else {
    const T w = box[2] - box[0];
    const T h = box[3] - box[1];
    if (normalized) {
      return w * h;
    } else {
      // If coordinate values are not within range [0, 1].
      return (w + 1) * (h + 1);
    }
  }
}

template <class T>
static inline T JaccardOverlap(const T* box1,
                               const T* box2,
                               const bool normalized) {
  if (box2[0] > box1[2] || box2[2] < box1[0] || box2[1] > box1[3] ||
      box2[3] < box1[1]) {
    return static_cast<T>(0.);
  } else {
    const T inter_xmin = std::max(box1[0], box2[0]);
    const T inter_ymin = std::max(box1[1], box2[1]);
    const T inter_xmax = std::min(box1[2], box2[2]);
    const T inter_ymax = std::min(box1[3], box2[3]);
    T norm = normalized ? static_cast<T>(0.) : static_cast<T>(1.);
    T inter_w = inter_xmax - inter_xmin + norm;
    T inter_h = inter_ymax - inter_ymin + norm;
    const T inter_area = inter_w * inter_h;
    const T bbox1_area = BBoxArea<T>(box1, normalized);
    const T bbox2_area = BBoxArea<T>(box2, normalized);
    return inter_area / (bbox1_area + bbox2_area - inter_area);
  }
}

template <typename T, bool gaussian>
struct decay_score;

template <typename T>
struct decay_score<T, true> {
  T operator()(T iou, T max_iou, T sigma) {
    return std::exp((max_iou * max_iou - iou * iou) * sigma);
  }
};

template <typename T>
struct decay_score<T, false> {
  T operator()(T iou, T max_iou, T sigma UNUSED) {
    return (1. - iou) / (1. - max_iou);
  }
};

template <typename T, bool gaussian>
void NMSMatrix(const DenseTensor& bbox,
               const DenseTensor& scores,
               const T score_threshold,
               const T post_threshold,
               const float sigma,
               const int64_t top_k,
               const bool normalized,
               std::vector<int>* selected_indices,
               std::vector<T>* decayed_scores) {
  int64_t num_boxes = bbox.dims()[0];
  int64_t box_size = bbox.dims()[1];

  auto score_ptr = scores.data<T>();
  auto bbox_ptr = bbox.data<T>();

  std::vector<int32_t> perm(num_boxes);
  std::iota(perm.begin(), perm.end(), 0);
  auto end = std::remove_if(
      perm.begin(), perm.end(), [&score_ptr, score_threshold](int32_t idx) {
        return score_ptr[idx] <= score_threshold;
      });

  auto sort_fn = [&score_ptr](int32_t lhs, int32_t rhs) {
    return score_ptr[lhs] > score_ptr[rhs];
  };

  int64_t num_pre = std::distance(perm.begin(), end);
  if (num_pre <= 0) {
    return;
  }
  if (top_k > -1 && num_pre > top_k) {
    num_pre = top_k;
  }
  std::partial_sort(perm.begin(), perm.begin() + num_pre, end, sort_fn);

  std::vector<T> iou_matrix((num_pre * (num_pre - 1)) >> 1);
  std::vector<T> iou_max(num_pre);

  iou_max[0] = 0.;
  for (int64_t i = 1; i < num_pre; i++) {
    T max_iou = 0.;
    auto idx_a = perm[i];
    for (int64_t j = 0; j < i; j++) {
      auto idx_b = perm[j];
      auto iou = JaccardOverlap<T>(
          bbox_ptr + idx_a * box_size, bbox_ptr + idx_b * box_size, normalized);
      max_iou = std::max(max_iou, iou);
      iou_matrix[i * (i - 1) / 2 + j] = iou;
    }
    iou_max[i] = max_iou;
  }

  if (score_ptr[perm[0]] > post_threshold) {
    selected_indices->push_back(perm[0]);
    decayed_scores->push_back(score_ptr[perm[0]]);
  }

  decay_score<T, gaussian> decay_fn;
  for (int64_t i = 1; i < num_pre; i++) {
    T min_decay = 1.;
    for (int64_t j = 0; j < i; j++) {
      auto max_iou = iou_max[j];
      auto iou = iou_matrix[i * (i - 1) / 2 + j];
      auto decay = decay_fn(iou, max_iou, sigma);
      min_decay = std::min(min_decay, decay);
    }
    auto ds = min_decay * score_ptr[perm[i]];
    if (ds <= post_threshold) continue;
    selected_indices->push_back(perm[i]);
    decayed_scores->push_back(ds);
  }
}

template <typename T>
size_t MultiClassMatrixNMS(const DenseTensor& scores,
                           const DenseTensor& bboxes,
                           std::vector<T>* out,
                           std::vector<int>* indices,
                           int start,
                           int64_t background_label,
                           int64_t nms_top_k,
                           int64_t keep_top_k,
                           bool normalized,
                           T score_threshold,
                           T post_threshold,
                           bool use_gaussian,
                           float gaussian_sigma) {
  std::vector<int> all_indices;
  std::vector<T> all_scores;
  std::vector<T> all_classes;
  all_indices.reserve(scores.numel());
  all_scores.reserve(scores.numel());
  all_classes.reserve(scores.numel());

  size_t num_det = 0;
  auto class_num = scores.dims()[0];
  DenseTensor score_slice;
  for (int64_t c = 0; c < class_num; ++c) {
    if (c == background_label) continue;
    score_slice = scores.Slice(c, c + 1);
    if (use_gaussian) {
      NMSMatrix<T, true>(bboxes,
                         score_slice,
                         score_threshold,
                         post_threshold,
                         gaussian_sigma,
                         nms_top_k,
                         normalized,
                         &all_indices,
                         &all_scores);
    } else {
      NMSMatrix<T, false>(bboxes,
                          score_slice,
                          score_threshold,
                          post_threshold,
                          gaussian_sigma,
                          nms_top_k,
                          normalized,
                          &all_indices,
                          &all_scores);
    }
    for (size_t i = 0; i < all_indices.size() - num_det; i++) {
      all_classes.push_back(static_cast<T>(c));
    }
    num_det = all_indices.size();
  }

  if (num_det <= 0) {
    return num_det;
  }

  if (keep_top_k > -1) {
    auto k = static_cast<size_t>(keep_top_k);
    if (num_det > k) num_det = k;
  }

  std::vector<int32_t> perm(all_indices.size());
  std::iota(perm.begin(), perm.end(), 0);

  std::partial_sort(perm.begin(),
                    perm.begin() + num_det,  // NOLINT
                    perm.end(),
                    [&all_scores](int lhs, int rhs) {
                      return all_scores[lhs] > all_scores[rhs];
                    });

  for (size_t i = 0; i < num_det; i++) {
    auto p = perm[i];
    auto idx = all_indices[p];
    auto cls = all_classes[p];
    auto score = all_scores[p];
    auto bbox = bboxes.data<T>() + idx * bboxes.dims()[1];
    (*indices).push_back(start + idx);
    (*out).push_back(cls);
    (*out).push_back(score);
    for (int j = 0; j < bboxes.dims()[1]; j++) {
      (*out).push_back(bbox[j]);
    }
  }

  return num_det;
}

template <typename T, typename Context>
void MatrixNMSKernel(const Context& ctx,
                     const DenseTensor& bboxes,
                     const DenseTensor& scores,
                     float score_threshold,
                     int nms_top_k,
                     int keep_top_k,
                     float post_threshold,
                     bool use_gaussian,
                     float gaussian_sigma,
                     int background_label,
                     bool normalized,
                     DenseTensor* out,
                     DenseTensor* index,
                     DenseTensor* roisnum) {
  auto score_dims = common::vectorize<int>(scores.dims());
  auto batch_size = score_dims[0];
  auto num_boxes = score_dims[2];
  auto box_dim = bboxes.dims()[2];
  auto out_dim = box_dim + 2;

  DenseTensor boxes_slice, scores_slice;
  size_t num_out = 0;
  std::vector<size_t> offsets = {0};
  std::vector<T> detections;
  std::vector<int> indices;
  std::vector<int> num_per_batch;
  detections.reserve(out_dim * num_boxes * batch_size);
  indices.reserve(num_boxes * batch_size);
  num_per_batch.reserve(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    scores_slice = scores.Slice(i, i + 1);
    scores_slice.Resize({score_dims[1], score_dims[2]});
    boxes_slice = bboxes.Slice(i, i + 1);
    boxes_slice.Resize({score_dims[2], box_dim});
    int start = i * score_dims[2];
    num_out = MultiClassMatrixNMS(scores_slice,
                                  boxes_slice,
                                  &detections,
                                  &indices,
                                  start,
                                  background_label,
                                  nms_top_k,
                                  keep_top_k,
                                  normalized,
                                  static_cast<T>(score_threshold),
                                  static_cast<T>(post_threshold),
                                  use_gaussian,
                                  gaussian_sigma);
    offsets.push_back(offsets.back() + num_out);
    num_per_batch.emplace_back(num_out);
  }

  int64_t num_kept = static_cast<int64_t>(offsets.back());
  if (num_kept == 0) {
    out->Resize(common::make_ddim({0, out_dim}));
    ctx.template Alloc<T>(out);
    index->Resize(common::make_ddim({0, 1}));
    ctx.template Alloc<int>(index);
  } else {
    out->Resize(common::make_ddim({num_kept, out_dim}));
    ctx.template Alloc<T>(out);
    index->Resize(common::make_ddim({num_kept, 1}));
    ctx.template Alloc<int>(index);
    std::copy(detections.begin(), detections.end(), out->data<T>());
    std::copy(indices.begin(), indices.end(), index->data<int>());
  }

  if (roisnum != nullptr) {
    roisnum->Resize(common::make_ddim({batch_size}));
    ctx.template Alloc<int>(roisnum);
    std::copy(num_per_batch.begin(), num_per_batch.end(), roisnum->data<int>());
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    matrix_nms, CPU, ALL_LAYOUT, phi::MatrixNMSKernel, float, double) {
  kernel->OutputAt(1).SetDataType(phi::DataType::INT32);
  kernel->OutputAt(2).SetDataType(phi::DataType::INT32);
}
