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

#include "paddle/phi/kernels/multiclass_nms3_kernel.h"

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/gpc.h"

namespace phi {

template <class T>
class Point_ {
 public:
  // default constructor
  Point_() = default;
  Point_(T _x, T _y) {}
  Point_(const Point_& pt UNUSED) {}

  Point_& operator=(const Point_& pt);
  // conversion to another data type
  // template<typename _T> operator Point_<_T>() const;
  // conversion to the old-style C structures
  // operator Vec<T, 2>() const;

  // checks whether the point is inside the specified rectangle
  // bool inside(const Rect_<T>& r) const;
  T x;  //!< x coordinate of the point
  T y;  //!< y coordinate of the point
};

template <class T>
void Array2PointVec(const T* box,
                    const size_t box_size,
                    std::vector<Point_<T>>* vec) {
  size_t pts_num = box_size / 2;
  (*vec).resize(pts_num);
  for (size_t i = 0; i < pts_num; i++) {
    (*vec).at(i).x = box[2 * i];
    (*vec).at(i).y = box[2 * i + 1];
  }
}

template <class T>
void Array2Poly(const T* box,
                const size_t box_size,
                phi::funcs::gpc_polygon* poly) {
  size_t pts_num = box_size / 2;
  (*poly).num_contours = 1;
  (*poly).hole = reinterpret_cast<int*>(malloc(sizeof(int)));  // NOLINT
  (*poly).hole[0] = 0;
  (*poly).contour = (phi::funcs::gpc_vertex_list*)malloc(  // NOLINT
      sizeof(phi::funcs::gpc_vertex_list));
  (*poly).contour->num_vertices = static_cast<int>(pts_num);
  (*poly).contour->vertex = (phi::funcs::gpc_vertex*)malloc(  // NOLINT
      sizeof(phi::funcs::gpc_vertex) * pts_num);
  for (size_t i = 0; i < pts_num; ++i) {
    (*poly).contour->vertex[i].x = box[2 * i];
    (*poly).contour->vertex[i].y = box[2 * i + 1];
  }
}

template <class T>
void PointVec2Poly(const std::vector<Point_<T>>& vec,
                   phi::funcs::gpc_polygon* poly) {
  size_t pts_num = vec.size();
  (*poly).num_contours = 1;
  (*poly).hole = reinterpret_cast<int*>(malloc(sizeof(int)));  // NOLINT
  (*poly).hole[0] = 0;
  (*poly).contour = (phi::funcs::gpc_vertex_list*)malloc(  // NOLINT
      sizeof(phi::funcs::gpc_vertex_list));
  (*poly).contour->num_vertices = pts_num;
  (*poly).contour->vertex = (phi::funcs::gpc_vertex*)malloc(  // NOLINT
      sizeof(phi::funcs::gpc_vertex) * pts_num);
  for (size_t i = 0; i < pts_num; ++i) {
    (*poly).contour->vertex[i].x = vec[i].x;
    (*poly).contour->vertex[i].y = vec[i].y;
  }
}

template <class T>
void Poly2PointVec(const phi::funcs::gpc_vertex_list& contour,
                   std::vector<Point_<T>>* vec) {
  int pts_num = contour.num_vertices;
  (*vec).resize(pts_num);
  for (int i = 0; i < pts_num; i++) {
    (*vec).at(i).x = contour.vertex[i].x;
    (*vec).at(i).y = contour.vertex[i].y;
  }
}

template <class T>
T GetContourArea(const std::vector<Point_<T>>& vec) {
  size_t pts_num = vec.size();
  if (pts_num < 3) return T(0.);
  T area = T(0.);
  for (size_t i = 0; i < pts_num; ++i) {
    area += vec[i].x * vec[(i + 1) % pts_num].y -
            vec[i].y * vec[(i + 1) % pts_num].x;
  }
  return std::fabs(area / 2.0);
}

template <class T>
T PolyArea(const T* box, const size_t box_size, const bool normalized UNUSED) {
  // If coordinate values are is invalid
  // if area size <= 0,  return 0.
  std::vector<Point_<T>> vec;
  Array2PointVec<T>(box, box_size, &vec);
  return GetContourArea<T>(vec);
}

template <class T>
T PolyOverlapArea(const T* box1,
                  const T* box2,
                  const size_t box_size,
                  const bool normalized UNUSED) {
  phi::funcs::gpc_polygon poly1;
  phi::funcs::gpc_polygon poly2;
  Array2Poly<T>(box1, box_size, &poly1);
  Array2Poly<T>(box2, box_size, &poly2);
  phi::funcs::gpc_polygon respoly;
  phi::funcs::gpc_op op = phi::funcs::GPC_INT;
  phi::funcs::gpc_polygon_clip(op, &poly2, &poly1, &respoly);

  T inter_area = T(0.);
  int contour_num = respoly.num_contours;
  for (int i = 0; i < contour_num; ++i) {
    std::vector<Point_<T>> resvec;
    Poly2PointVec<T>(respoly.contour[i], &resvec);
    // inter_area += std::fabs(cv::contourArea(resvec)) + 0.5f *
    // (cv::arcLength(resvec, true));
    inter_area += GetContourArea<T>(resvec);
  }

  phi::funcs::gpc_free_polygon(&poly1);
  phi::funcs::gpc_free_polygon(&poly2);
  phi::funcs::gpc_free_polygon(&respoly);
  return inter_area;
}

template <class T>
bool SortScorePairDescend(const std::pair<float, T>& pair1,
                          const std::pair<float, T>& pair2) {
  return pair1.first > pair2.first;
}

template <class T>
static inline void GetMaxScoreIndex(
    const std::vector<T>& scores,
    const T threshold,
    int top_k,
    std::vector<std::pair<T, int>>* sorted_indices) {
  for (size_t i = 0; i < scores.size(); ++i) {
    if (scores[i] > threshold) {
      sorted_indices->push_back(std::make_pair(scores[i], i));
    }
  }
  // Sort the score pair according to the scores in descending order
  std::stable_sort(sorted_indices->begin(),
                   sorted_indices->end(),
                   SortScorePairDescend<int>);
  // Keep top_k scores if needed.
  if (top_k > -1 && top_k < static_cast<int>(sorted_indices->size())) {
    sorted_indices->resize(top_k);
  }
}

template <class T>
static inline T BBoxArea(const T* box, const bool normalized) {
  if (box[2] < box[0] || box[3] < box[1]) {
    // If coordinate values are is invalid
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

template <class T>
T PolyIoU(const T* box1,
          const T* box2,
          const size_t box_size,
          const bool normalized) {
  T bbox1_area = PolyArea<T>(box1, box_size, normalized);
  T bbox2_area = PolyArea<T>(box2, box_size, normalized);
  T inter_area = PolyOverlapArea<T>(box1, box2, box_size, normalized);
  if (bbox1_area == 0 || bbox2_area == 0 || inter_area == 0) {
    // If coordinate values are invalid
    // if area size <= 0,  return 0.
    return T(0.);
  } else {
    return inter_area / (bbox1_area + bbox2_area - inter_area);
  }
}

inline std::vector<size_t> GetNmsLodFromRoisNum(const DenseTensor* rois_num) {
  std::vector<size_t> rois_lod;
  if (rois_num->dtype() == phi::DataType::INT64) {
    auto* rois_num_data = rois_num->data<int64_t>();
    rois_lod.push_back(static_cast<size_t>(0));
    for (int64_t i = 0; i < rois_num->numel(); ++i) {
      rois_lod.push_back(rois_lod.back() +
                         static_cast<size_t>(rois_num_data[i]));
    }
  } else if (rois_num->dtype() == phi::DataType::INT32) {
    auto* rois_num_data = rois_num->data<int>();
    rois_lod.push_back(static_cast<size_t>(0));
    for (int i = 0; i < rois_num->numel(); ++i) {
      rois_lod.push_back(rois_lod.back() +
                         static_cast<size_t>(rois_num_data[i]));
    }
  }
  return rois_lod;
}

template <typename T, typename Context>
void SliceOneClass(const Context& ctx,
                   const DenseTensor& items,
                   const int class_id,
                   DenseTensor* one_class_item) {
  T* item_data = ctx.template Alloc<T>(one_class_item);
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
void NMSFast(const DenseTensor& bbox,
             const DenseTensor& scores,
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
  GetMaxScoreIndex<T>(scores_data, score_threshold, top_k, &sorted_indices);

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
          overlap = JaccardOverlap<T>(bbox_data + idx * box_size,
                                      bbox_data + kept_idx * box_size,
                                      normalized);
        }
        // 8: [x1 y1 x2 y2 x3 y3 x4 y4] or 16, 24, 32
        if (box_size == 8 || box_size == 16 || box_size == 24 ||
            box_size == 32) {
          overlap = PolyIoU<T>(bbox_data + idx * box_size,
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
void MultiClassNMS(const Context& ctx,
                   const DenseTensor& scores,
                   const DenseTensor& bboxes,
                   const int scores_size,
                   float scorethreshold,
                   int nms_top_k,
                   int keep_top_k,
                   float nmsthreshold,
                   bool normalized,
                   float nmseta,
                   int background_label,
                   std::map<int, std::vector<int>>* indices,
                   int* num_nmsed_out) {
  T nms_threshold = static_cast<T>(nmsthreshold);
  T nms_eta = static_cast<T>(nmseta);
  T score_threshold = static_cast<T>(scorethreshold);

  int num_det = 0;

  int class_num =
      static_cast<int>(scores_size == 3 ? scores.dims()[0] : scores.dims()[1]);
  DenseTensor bbox_slice, score_slice;
  for (int c = 0; c < class_num; ++c) {
    if (c == background_label) continue;
    if (scores_size == 3) {
      score_slice = scores.Slice(c, c + 1);
      bbox_slice = bboxes;
    } else {
      score_slice.Resize({scores.dims()[0], 1});
      bbox_slice.Resize({scores.dims()[0], 4});
      SliceOneClass<T, Context>(ctx, scores, c, &score_slice);
      SliceOneClass<T, Context>(ctx, bboxes, c, &bbox_slice);
    }
    NMSFast<T>(bbox_slice,
               score_slice,
               score_threshold,
               nms_threshold,
               nms_eta,
               nms_top_k,
               &((*indices)[c]),
               normalized);
    if (scores_size == 2) {
      std::stable_sort((*indices)[c].begin(), (*indices)[c].end());
    }
    num_det += static_cast<int>((*indices)[c].size());
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
        SliceOneClass<T, Context>(ctx, scores, label, &score_slice);
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
                     SortScorePairDescend<std::pair<int, int>>);
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
    *num_nmsed_out = keep_top_k;
  }
}

template <typename T, typename Context>
void MultiClassOutput(const Context& ctx,
                      const DenseTensor& scores,
                      const DenseTensor& bboxes,
                      const std::map<int, std::vector<int>>& selected_indices,
                      const int scores_size,
                      DenseTensor* out,
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
  auto* odata = out->data<T>();
  const T* sdata = nullptr;
  DenseTensor bbox;
  bbox.Resize({scores.dims()[0], box_size});
  int count = 0;
  for (const auto& it : selected_indices) {
    int label = it.first;
    const std::vector<int>& indices = it.second;
    if (scores_size == 2) {
      SliceOneClass<T, Context>(ctx, bboxes, label, &bbox);
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
          oindices[count] = offset + idx * class_num + label;  // NOLINT
        }
      }
      // xmin, ymin, xmax, ymax or multi-points coordinates
      std::memcpy(odata + count * out_dim + 2, bdata, box_size * sizeof(T));
      count++;
    }
  }
}

template <typename T, typename Context>
void MultiClassNMSKernel(const Context& ctx,
                         const DenseTensor& bboxes,
                         const DenseTensor& scores,
                         const paddle::optional<DenseTensor>& rois_num,
                         float score_threshold,
                         int nms_top_k,
                         int keep_top_k,
                         float nms_threshold,
                         bool normalized,
                         float nms_eta,
                         int background_label,
                         DenseTensor* out,
                         DenseTensor* index,
                         DenseTensor* nms_rois_num) {
  bool return_index = index != nullptr;
  bool has_roisnum = rois_num.get_ptr() != nullptr;
  auto score_dims = common::vectorize<int>(scores.dims());
  auto score_size = score_dims.size();

  std::vector<std::map<int, std::vector<int>>> all_indices;
  std::vector<size_t> batch_starts = {0};
  int64_t batch_size = score_dims[0];
  int64_t box_dim = bboxes.dims()[2];
  int64_t out_dim = box_dim + 2;
  int num_nmsed_out = 0;
  DenseTensor boxes_slice, scores_slice;
  int n = 0;
  if (has_roisnum) {
    n = static_cast<int>(score_size == 3 ? batch_size
                                         : rois_num.get_ptr()->numel());
  } else {
    n = static_cast<int>(score_size == 3 ? batch_size
                                         : bboxes.lod().back().size() - 1);
  }
  for (int i = 0; i < n; ++i) {
    std::map<int, std::vector<int>> indices;
    if (score_size == 3) {
      scores_slice = scores.Slice(i, i + 1);
      scores_slice.Resize({score_dims[1], score_dims[2]});
      boxes_slice = bboxes.Slice(i, i + 1);
      boxes_slice.Resize({score_dims[2], box_dim});
    } else {
      std::vector<size_t> boxes_lod;
      if (has_roisnum) {
        boxes_lod = GetNmsLodFromRoisNum(rois_num.get_ptr());
      } else {
        boxes_lod = bboxes.lod().back();
      }
      if (boxes_lod[i] == boxes_lod[i + 1]) {
        all_indices.push_back(indices);
        batch_starts.push_back(batch_starts.back());
        continue;
      }
      scores_slice = scores.Slice(boxes_lod[i], boxes_lod[i + 1]);  // NOLINT
      boxes_slice = bboxes.Slice(boxes_lod[i], boxes_lod[i + 1]);   // NOLINT
    }
    MultiClassNMS<T, Context>(ctx,
                              scores_slice,
                              boxes_slice,
                              score_size,
                              score_threshold,
                              nms_top_k,
                              keep_top_k,
                              nms_threshold,
                              normalized,
                              nms_eta,
                              background_label,
                              &indices,
                              &num_nmsed_out);
    all_indices.push_back(indices);
    batch_starts.push_back(batch_starts.back() + num_nmsed_out);
  }

  int num_kept = static_cast<int>(batch_starts.back());
  if (num_kept == 0) {
    if (return_index) {
      out->Resize({0, out_dim});
      ctx.template Alloc<T>(out);
      index->Resize({0, 1});
      ctx.template Alloc<int>(index);
    } else {
      out->Resize({1, 1});
      T* od = ctx.template Alloc<T>(out);
      od[0] = -1;
      batch_starts = {0, 1};
    }
  } else {
    out->Resize({num_kept, out_dim});
    ctx.template Alloc<T>(out);
    int offset = 0;
    int* oindices = nullptr;
    for (int i = 0; i < n; ++i) {
      if (score_size == 3) {
        scores_slice = scores.Slice(i, i + 1);
        boxes_slice = bboxes.Slice(i, i + 1);
        scores_slice.Resize({score_dims[1], score_dims[2]});
        boxes_slice.Resize({score_dims[2], box_dim});
        if (return_index) {
          offset = i * score_dims[2];
        }
      } else {
        std::vector<size_t> boxes_lod;
        if (has_roisnum) {
          boxes_lod = GetNmsLodFromRoisNum(rois_num.get_ptr());
        } else {
          boxes_lod = bboxes.lod().back();
        }
        if (boxes_lod[i] == boxes_lod[i + 1]) continue;
        scores_slice = scores.Slice(boxes_lod[i], boxes_lod[i + 1]);  // NOLINT
        boxes_slice = bboxes.Slice(boxes_lod[i], boxes_lod[i + 1]);   // NOLINT
        if (return_index) {
          offset = static_cast<int>(boxes_lod[i] * score_dims[1]);
        }
      }

      int64_t s = static_cast<int64_t>(batch_starts[i]);
      int64_t e = static_cast<int64_t>(batch_starts[i + 1]);
      if (e > s) {
        DenseTensor nout = out->Slice(s, e);
        if (return_index) {
          index->Resize({num_kept, 1});
          int* output_idx = ctx.template Alloc<int>(index);
          oindices = output_idx + s;
        }
        MultiClassOutput<T, Context>(ctx,
                                     scores_slice,
                                     boxes_slice,
                                     all_indices[i],
                                     score_dims.size(),
                                     &nout,
                                     oindices,
                                     offset);
      }
    }
  }
  if (nms_rois_num != nullptr) {
    nms_rois_num->Resize({n});
    ctx.template Alloc<int>(nms_rois_num);
    int* num_data = nms_rois_num->data<int>();
    for (int i = 1; i <= n; i++) {
      num_data[i - 1] = batch_starts[i] - batch_starts[i - 1];  // NOLINT
    }
    nms_rois_num->Resize({n});
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    multiclass_nms3, CPU, ALL_LAYOUT, phi::MultiClassNMSKernel, float, double) {
  kernel->OutputAt(1).SetDataType(phi::DataType::INT32);
  kernel->OutputAt(2).SetDataType(phi::DataType::INT32);
}
