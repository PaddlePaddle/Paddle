// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/detection/nms_util.h"

namespace phi {

template <typename T>
void SliceOneClass(const phi::DeviceContext& dev_ctx,
                   const phi::DenseTensor& items,
                   const int class_id,
                   phi::DenseTensor* one_class_item) {
  T* item_data = dev_ctx.template Alloc<T>(one_class_item);
  const T* items_data = items.data<T>();
  const int64_t num_item = items.dims()[0];
  const int class_num = static_cast<int>(items.dims()[1]);
  if (items.dims().size() == 3) {
    int item_size = static_cast<int>(items.dims()[2]);
    for (int i = 0; i < num_item; ++i) {
      std::memcpy(item_data + i * item_size,
                  items_data + i * class_num * item_size + class_id * item_size,
                  sizeof(T) * item_size);
    }
  } else {
    for (int i = 0; i < num_item; ++i) {
      item_data[i] = items_data[i * class_num + class_id];
    }
  }
}

template <typename T>
void NMSFast(const phi::DenseTensor& bbox,
             const phi::DenseTensor& scores,
             const T score_threshold,
             const T nms_threshold,
             const T eta,
             const int64_t top_k,
             std::vector<int>* selected_indices,
             const bool normalized) {
  // The total boxes for each instance.
  int64_t num_boxes = bbox.dims()[0];
  // 4: [xmin ymin xmax ymax]
  // 8: [x1 y1 x2 y2 x3 y3 x4 y4]
  // 16, 24, or 32: [x1 y1 x2 y2 ...  xn yn], n = 8, 12 or 16
  int64_t box_size = bbox.dims()[1];

  std::vector<T> scores_data(num_boxes);
  std::copy_n(scores.data<T>(), num_boxes, scores_data.begin());
  std::vector<std::pair<T, int>> sorted_indices;
  phi::funcs::GetMaxScoreIndex(
      scores_data, score_threshold, top_k, &sorted_indices);

  selected_indices->clear();
  T adaptive_threshold = nms_threshold;
  const T* bbox_data = bbox.data<T>();

  while (!sorted_indices.empty()) {
    const int idx = sorted_indices.front().second;
    bool keep = true;
    for (const auto kept_idx : *selected_indices) {
      if (keep) {
        T overlap = T(0.);
        // 4: [xmin ymin xmax ymax]
        if (box_size == 4) {
          overlap =
              phi::funcs::JaccardOverlap<T>(bbox_data + idx * box_size,
                                            bbox_data + kept_idx * box_size,
                                            normalized);
        }
        // 8: [x1 y1 x2 y2 x3 y3 x4 y4] or 16, 24, 32
        if (box_size == 8 || box_size == 16 || box_size == 24 ||
            box_size == 32) {
          overlap = phi::funcs::PolyIoU<T>(bbox_data + idx * box_size,
                                           bbox_data + kept_idx * box_size,
                                           box_size,
                                           normalized);
        }
        keep = overlap <= adaptive_threshold;
      } else {
        break;
      }
    }
    if (keep) {
      selected_indices->push_back(idx);
    }
    sorted_indices.erase(sorted_indices.begin());
    if (keep && eta < 1 && adaptive_threshold > 0.5) {
      adaptive_threshold *= eta;
    }
  }
}

template <typename T, typename Context>
void MultiClassNMS(const Context& dev_ctx,
                   const phi::DenseTensor& scores,
                   const phi::DenseTensor& bboxes,
                   const int scores_size,
                   std::map<int, std::vector<int>>* indices,
                   int* num_nmsed_out,
                   float score_threshold_in,
                   int nms_top_k_in,
                   float nms_threshold_in,
                   float nms_eta_in,
                   int keep_top_k_in,
                   bool normalized_in,
                   int background_label_in) {
  int64_t background_label = background_label_in;
  int64_t nms_top_k = nms_top_k_in;
  int64_t keep_top_k = keep_top_k_in;
  bool normalized = normalized_in;
  T nms_threshold = static_cast<T>(nms_threshold_in);
  T nms_eta = static_cast<T>(nms_eta_in);
  T score_threshold = static_cast<T>(score_threshold_in);

  int num_det = 0;

  int64_t class_num = scores_size == 3 ? scores.dims()[0] : scores.dims()[1];
  phi::DenseTensor bbox_slice, score_slice;
  for (int64_t c = 0; c < class_num; ++c) {
    if (c == background_label) continue;
    if (scores_size == 3) {
      score_slice = scores.Slice(c, c + 1);
      bbox_slice = bboxes;
    } else {
      score_slice.Resize({scores.dims()[0], 1});
      bbox_slice.Resize({scores.dims()[0], 4});
      SliceOneClass<T>(dev_ctx, scores, c, &score_slice);
      SliceOneClass<T>(dev_ctx, bboxes, c, &bbox_slice);
    }
    NMSFast<T>(bbox_slice,
               score_slice,
               score_threshold,
               nms_threshold,
               nms_eta,
               nms_top_k,
               &((*indices)[c]),  // NOLINT
               normalized);
    if (scores_size == 2) {
      std::stable_sort((*indices)[c].begin(), (*indices)[c].end());  // NOLINT
    }
    num_det += (*indices)[c].size();  // NOLINT
  }

  *num_nmsed_out = num_det;
  const T* scores_data = scores.data<T>();
  if (keep_top_k > -1 && num_det > keep_top_k) {
    const T* sdata = nullptr;
    std::vector<std::pair<float, std::pair<int, int>>> score_index_pairs;
    for (const auto& it : *indices) {
      int label = it.first;
      if (scores_size == 3) {
        sdata = scores_data + label * scores.dims()[1];
      } else {
        score_slice.Resize({scores.dims()[0], 1});
        SliceOneClass<T>(dev_ctx, scores, label, &score_slice);
        sdata = score_slice.data<T>();
      }
      const std::vector<int>& label_indices = it.second;
      for (auto idx : label_indices) {
        score_index_pairs.push_back(
            std::make_pair(sdata[idx], std::make_pair(label, idx)));
      }
    }
    // Keep top k results per image.
    std::stable_sort(score_index_pairs.begin(),
                     score_index_pairs.end(),
                     phi::funcs::SortScorePairDescend<std::pair<int, int>>);
    score_index_pairs.resize(keep_top_k);

    // Store the new indices.
    std::map<int, std::vector<int>> new_indices;
    for (auto& score_index_pair : score_index_pairs) {
      int label = score_index_pair.second.first;
      int idx = score_index_pair.second.second;
      new_indices[label].push_back(idx);
    }
    if (scores_size == 2) {
      for (const auto& it : new_indices) {
        int label = it.first;
        std::stable_sort(new_indices[label].begin(), new_indices[label].end());
      }
    }
    new_indices.swap(*indices);
    *num_nmsed_out = keep_top_k;  // NOLINT
  }
}

template <typename T, typename Context>
void MultiClassOutput(const Context& dev_ctx,
                      const phi::DenseTensor& scores,
                      const phi::DenseTensor& bboxes,
                      const std::map<int, std::vector<int>>& selected_indices,
                      const int scores_size,
                      phi::DenseTensor* outs,
                      int* oindices = nullptr,
                      const int offset = 0) {
  int64_t class_num = scores.dims()[1];
  int64_t predict_dim = scores.dims()[1];
  int64_t box_size = bboxes.dims()[1];
  if (scores_size == 2) {
    box_size = bboxes.dims()[2];
  }
  int64_t out_dim = box_size + 2;
  auto* scores_data = scores.data<T>();
  auto* bboxes_data = bboxes.data<T>();
  auto* odata = outs->data<T>();
  const T* sdata = nullptr;
  phi::DenseTensor bbox;
  bbox.Resize({scores.dims()[0], box_size});
  int count = 0;
  for (const auto& it : selected_indices) {
    int label = it.first;
    const std::vector<int>& indices = it.second;
    if (scores_size == 2) {
      SliceOneClass<T>(dev_ctx, bboxes, label, &bbox);
    } else {
      sdata = scores_data + label * predict_dim;
    }

    for (auto idx : indices) {
      odata[count * out_dim] = label;  // label
      const T* bdata = nullptr;
      if (scores_size == 3) {
        bdata = bboxes_data + idx * box_size;
        odata[count * out_dim + 1] = sdata[idx];  // score
        if (oindices != nullptr) {
          oindices[count] = offset + idx;
        }
      } else {
        bdata = bbox.data<T>() + idx * box_size;
        odata[count * out_dim + 1] = *(scores_data + idx * class_num + label);
        if (oindices != nullptr) {
          oindices[count] = static_cast<int>(offset + idx * class_num + label);
        }
      }
      // xmin, ymin, xmax, ymax or multi-points coordinates
      std::memcpy(odata + count * out_dim + 2, bdata, box_size * sizeof(T));
      count++;
    }
  }
}

template <typename T, typename Context>
void MulticlassNMSv1Kernel(const Context& dev_ctx,
                           const DenseTensor& bboxes_in,
                           const DenseTensor& scores_in,
                           float score_threshold,
                           int nms_top_k,
                           int keep_top_k,
                           float nms_threshold,
                           float nms_eta,
                           bool normalized,
                           int background_label,
                           DenseTensor* out) {
  auto* boxes = &bboxes_in;
  auto* scores = &scores_in;
  auto* outs = out;

  auto score_dims = common::vectorize<int>(scores->dims());
  auto score_size = score_dims.size();

  std::vector<std::map<int, std::vector<int>>> all_indices;
  std::vector<size_t> batch_starts = {0};
  int64_t batch_size = score_dims[0];
  int64_t box_dim = boxes->dims()[2];
  int64_t out_dim = box_dim + 2;
  int num_nmsed_out = 0;
  phi::DenseTensor boxes_slice, scores_slice;
  int n = 0;

  n = static_cast<int>(score_size == 3 ? batch_size
                                       : boxes->lod().back().size() - 1);

  for (int i = 0; i < n; ++i) {
    std::map<int, std::vector<int>> indices;
    if (score_size == 3) {
      scores_slice = scores->Slice(i, i + 1);
      scores_slice.Resize({score_dims[1], score_dims[2]});
      boxes_slice = boxes->Slice(i, i + 1);
      boxes_slice.Resize({score_dims[2], box_dim});
    } else {
      std::vector<size_t> boxes_lod;
      boxes_lod = boxes->lod().back();
      if (boxes_lod[i] == boxes_lod[i + 1]) {
        all_indices.push_back(indices);
        batch_starts.push_back(batch_starts.back());
        continue;
      }
      scores_slice = scores->Slice(static_cast<int64_t>(boxes_lod[i]),
                                   static_cast<int64_t>(boxes_lod[i + 1]));
      boxes_slice = boxes->Slice(static_cast<int64_t>(boxes_lod[i]),
                                 static_cast<int64_t>(boxes_lod[i + 1]));
    }
    MultiClassNMS<T, Context>(dev_ctx,
                              scores_slice,
                              boxes_slice,
                              score_size,
                              &indices,
                              &num_nmsed_out,
                              score_threshold,
                              nms_top_k,
                              nms_threshold,
                              nms_eta,
                              keep_top_k,
                              normalized,
                              background_label);
    all_indices.push_back(indices);
    batch_starts.push_back(batch_starts.back() + num_nmsed_out);
  }

  int num_kept = static_cast<int>(batch_starts.back());
  if (num_kept == 0) {
    outs->Resize({1, 1});
    T* od = dev_ctx.template Alloc<T>(outs);
    od[0] = -1;
    batch_starts = {0, 1};
  } else {
    outs->Resize({num_kept, out_dim});
    dev_ctx.template Alloc<T>(outs);
    int offset = 0;
    int* oindices = nullptr;
    for (int i = 0; i < n; ++i) {
      if (score_size == 3) {
        scores_slice = scores->Slice(i, i + 1);
        boxes_slice = boxes->Slice(i, i + 1);
        scores_slice.Resize({score_dims[1], score_dims[2]});
        boxes_slice.Resize({score_dims[2], box_dim});
      } else {
        std::vector<size_t> boxes_lod;

        boxes_lod = boxes->lod().back();

        if (boxes_lod[i] == boxes_lod[i + 1]) continue;
        scores_slice = scores->Slice(static_cast<int64_t>(boxes_lod[i]),
                                     static_cast<int64_t>(boxes_lod[i + 1]));
        boxes_slice = boxes->Slice(static_cast<int64_t>(boxes_lod[i]),
                                   static_cast<int64_t>(boxes_lod[i + 1]));
      }

      int64_t s = static_cast<int64_t>(batch_starts[i]);
      int64_t e = static_cast<int64_t>(batch_starts[i + 1]);
      if (e > s) {
        phi::DenseTensor out = outs->Slice(s, e);
        MultiClassOutput<T>(dev_ctx,
                            scores_slice,
                            boxes_slice,
                            all_indices[i],
                            score_dims.size(),
                            &out,
                            oindices,
                            offset);
      }
    }
  }

  phi::LegacyLoD lod;
  lod.emplace_back(batch_starts);
  outs->set_lod(lod);
}
}  // namespace phi

PD_REGISTER_KERNEL(multiclass_nms,
                   CPU,
                   ALL_LAYOUT,
                   phi::MulticlassNMSv1Kernel,
                   float,
                   double) {}
