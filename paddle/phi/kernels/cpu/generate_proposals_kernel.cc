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

#include "paddle/phi/kernels/generate_proposals_kernel.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/detection/nms_util.h"
#include "paddle/phi/kernels/funcs/gather.h"

namespace phi {

static const double kBBoxClipDefault = std::log(1000.0 / 16.0);

static void AppendProposals(DenseTensor* dst,
                            int64_t offset,
                            const DenseTensor& src) {
  auto* out_data = dst->data();
  auto* to_add_data = src.data();
  size_t size_of_t = SizeOf(src.dtype());
  offset *= static_cast<int64_t>(size_of_t);
  uintptr_t ptr = reinterpret_cast<uintptr_t>(out_data) + offset;
  std::memcpy(
      reinterpret_cast<void*>(ptr), to_add_data, src.numel() * size_of_t);
}

template <class T>
void ClipTiledBoxes(const phi::CPUContext& ctx,
                    const DenseTensor& im_info,
                    const DenseTensor& input_boxes,
                    DenseTensor* out,
                    bool is_scale = true,
                    bool pixel_offset = true) {
  T* out_data = ctx.template Alloc<T>(out);
  const T* im_info_data = im_info.data<T>();
  const T* input_boxes_data = input_boxes.data<T>();
  T offset = pixel_offset ? static_cast<T>(1.0) : 0;
  T zero(0);
  T im_w =
      is_scale ? round(im_info_data[1] / im_info_data[2]) : im_info_data[1];
  T im_h =
      is_scale ? round(im_info_data[0] / im_info_data[2]) : im_info_data[0];
  for (int64_t i = 0; i < input_boxes.numel(); ++i) {
    if ((i % 4 == 0) || (i % 4 == 2)) {
      out_data[i] =
          std::max(std::min(input_boxes_data[i], im_w - offset), zero);
    } else {
      out_data[i] =
          std::max(std::min(input_boxes_data[i], im_h - offset), zero);
    }
  }
}

// Filter the box with small area
template <class T>
void FilterBoxes(const phi::CPUContext& ctx,
                 const DenseTensor* boxes,
                 float min_size,
                 const DenseTensor& im_info,
                 bool is_scale,
                 DenseTensor* keep,
                 bool pixel_offset = true) {
  const T* im_info_data = im_info.data<T>();
  const T* boxes_data = boxes->data<T>();
  keep->Resize(common::make_ddim({boxes->dims()[0]}));
  min_size = std::max(min_size, 1.0f);
  int* keep_data = ctx.template Alloc<int>(keep);
  T offset = pixel_offset ? static_cast<T>(1.0) : 0;

  int keep_len = 0;
  for (int i = 0; i < boxes->dims()[0]; ++i) {
    T ws = boxes_data[4 * i + 2] - boxes_data[4 * i] + offset;
    T hs = boxes_data[4 * i + 3] - boxes_data[4 * i + 1] + offset;
    if (pixel_offset) {
      T x_ctr = boxes_data[4 * i] + ws / 2;
      T y_ctr = boxes_data[4 * i + 1] + hs / 2;

      if (is_scale) {
        ws = (boxes_data[4 * i + 2] - boxes_data[4 * i]) / im_info_data[2] + 1;
        hs = (boxes_data[4 * i + 3] - boxes_data[4 * i + 1]) / im_info_data[2] +
             1;
      }
      if (ws >= min_size && hs >= min_size && x_ctr <= im_info_data[1] &&
          y_ctr <= im_info_data[0]) {
        keep_data[keep_len++] = i;
      }
    } else {
      if (ws >= min_size && hs >= min_size) {
        keep_data[keep_len++] = i;
      }
    }
  }
  keep->Resize(common::make_ddim({keep_len}));
}

template <class T>
static void BoxCoder(const phi::CPUContext& ctx,
                     DenseTensor* all_anchors,
                     DenseTensor* bbox_deltas,
                     DenseTensor* variances,
                     DenseTensor* proposals,
                     const bool pixel_offset = true) {
  T* proposals_data = ctx.template Alloc<T>(proposals);

  int64_t row = all_anchors->dims()[0];
  int64_t len = all_anchors->dims()[1];

  auto* bbox_deltas_data = bbox_deltas->data<T>();
  auto* anchor_data = all_anchors->data<T>();
  const T* variances_data = nullptr;
  if (variances) {
    variances_data = variances->data<T>();
  }

  T offset = pixel_offset ? static_cast<T>(1.0) : 0;
  for (int64_t i = 0; i < row; ++i) {
    T anchor_width = anchor_data[i * len + 2] - anchor_data[i * len] + offset;
    T anchor_height =
        anchor_data[i * len + 3] - anchor_data[i * len + 1] + offset;

    T anchor_center_x = anchor_data[i * len] + 0.5 * anchor_width;
    T anchor_center_y = anchor_data[i * len + 1] + 0.5 * anchor_height;

    T bbox_center_x = 0, bbox_center_y = 0;
    T bbox_width = 0, bbox_height = 0;

    if (variances) {
      bbox_center_x =
          variances_data[i * len] * bbox_deltas_data[i * len] * anchor_width +
          anchor_center_x;
      bbox_center_y = variances_data[i * len + 1] *
                          bbox_deltas_data[i * len + 1] * anchor_height +
                      anchor_center_y;
      bbox_width = std::exp(std::min<T>(variances_data[i * len + 2] *
                                            bbox_deltas_data[i * len + 2],
                                        kBBoxClipDefault)) *
                   anchor_width;
      bbox_height = std::exp(std::min<T>(variances_data[i * len + 3] *
                                             bbox_deltas_data[i * len + 3],
                                         kBBoxClipDefault)) *
                    anchor_height;
    } else {
      bbox_center_x =
          bbox_deltas_data[i * len] * anchor_width + anchor_center_x;
      bbox_center_y =
          bbox_deltas_data[i * len + 1] * anchor_height + anchor_center_y;
      bbox_width = std::exp(std::min<T>(bbox_deltas_data[i * len + 2],
                                        kBBoxClipDefault)) *
                   anchor_width;
      bbox_height = std::exp(std::min<T>(bbox_deltas_data[i * len + 3],
                                         kBBoxClipDefault)) *
                    anchor_height;
    }

    proposals_data[i * len] = bbox_center_x - bbox_width / 2;
    proposals_data[i * len + 1] = bbox_center_y - bbox_height / 2;
    proposals_data[i * len + 2] = bbox_center_x + bbox_width / 2 - offset;
    proposals_data[i * len + 3] = bbox_center_y + bbox_height / 2 - offset;
  }
  // return proposals;
}

template <typename T>
std::pair<DenseTensor, DenseTensor> ProposalForOneImage(
    const phi::CPUContext& ctx,
    const DenseTensor& im_shape_slice,
    const DenseTensor& anchors,
    const DenseTensor& variances,
    const DenseTensor& bbox_deltas_slice,  // [M, 4]
    const DenseTensor& scores_slice,       // [N, 1]
    int pre_nms_top_n,
    int post_nms_top_n,
    float nms_thresh,
    float min_size,
    float eta,
    bool pixel_offset = true) {
  auto* scores_data = scores_slice.data<T>();

  // Sort index
  DenseTensor index_t;
  index_t.Resize(common::make_ddim({scores_slice.numel()}));
  int* index = ctx.template Alloc<int>(&index_t);
  for (int i = 0; i < scores_slice.numel(); ++i) {
    index[i] = i;
  }
  auto compare = [scores_data](const int64_t& i, const int64_t& j) {
    return scores_data[i] > scores_data[j];
  };

  if (pre_nms_top_n <= 0 || pre_nms_top_n >= scores_slice.numel()) {
    std::sort(index, index + scores_slice.numel(), compare);
  } else {
    std::nth_element(
        index, index + pre_nms_top_n, index + scores_slice.numel(), compare);
    index_t.Resize(common::make_ddim({pre_nms_top_n}));
  }

  DenseTensor scores_sel, bbox_sel, anchor_sel, var_sel;
  scores_sel.Resize(common::make_ddim({index_t.numel(), 1}));
  ctx.template Alloc<T>(&scores_sel);

  bbox_sel.Resize(common::make_ddim({index_t.numel(), 4}));
  ctx.template Alloc<T>(&bbox_sel);

  anchor_sel.Resize(common::make_ddim({index_t.numel(), 4}));
  ctx.template Alloc<T>(&anchor_sel);

  var_sel.Resize(common::make_ddim({index_t.numel(), 4}));
  ctx.template Alloc<T>(&var_sel);

  phi::funcs::CPUGather<T>(ctx, scores_slice, index_t, &scores_sel);
  phi::funcs::CPUGather<T>(ctx, bbox_deltas_slice, index_t, &bbox_sel);
  phi::funcs::CPUGather<T>(ctx, anchors, index_t, &anchor_sel);
  phi::funcs::CPUGather<T>(ctx, variances, index_t, &var_sel);

  DenseTensor proposals;
  proposals.Resize(common::make_ddim({index_t.numel(), 4}));
  ctx.template Alloc<T>(&proposals);

  BoxCoder<T>(ctx, &anchor_sel, &bbox_sel, &var_sel, &proposals, pixel_offset);

  ClipTiledBoxes<T>(
      ctx, im_shape_slice, proposals, &proposals, false, pixel_offset);

  DenseTensor keep;
  FilterBoxes<T>(
      ctx, &proposals, min_size, im_shape_slice, false, &keep, pixel_offset);
  // Handle the case when there is no keep index left
  if (keep.numel() == 0) {
    phi::funcs::SetConstant<phi::CPUContext, T> set_zero;
    bbox_sel.Resize(common::make_ddim({1, 4}));
    ctx.template Alloc<T>(&bbox_sel);
    set_zero(ctx, &bbox_sel, static_cast<T>(0));
    DenseTensor scores_filter;
    scores_filter.Resize(common::make_ddim({1, 1}));
    ctx.template Alloc<T>(&scores_filter);
    set_zero(ctx, &scores_filter, static_cast<T>(0));
    return std::make_pair(bbox_sel, scores_filter);
  }

  DenseTensor scores_filter;
  bbox_sel.Resize(common::make_ddim({keep.numel(), 4}));
  ctx.template Alloc<T>(&bbox_sel);
  scores_filter.Resize(common::make_ddim({keep.numel(), 1}));
  ctx.template Alloc<T>(&scores_filter);
  phi::funcs::CPUGather<T>(ctx, proposals, keep, &bbox_sel);
  phi::funcs::CPUGather<T>(ctx, scores_sel, keep, &scores_filter);
  if (nms_thresh <= 0) {
    return std::make_pair(bbox_sel, scores_filter);
  }

  DenseTensor keep_nms = phi::funcs::NMS<T>(
      ctx, &bbox_sel, &scores_filter, nms_thresh, eta, pixel_offset);

  if (post_nms_top_n > 0 && post_nms_top_n < keep_nms.numel()) {
    keep_nms.Resize(common::make_ddim({post_nms_top_n}));
  }

  proposals.Resize(common::make_ddim({keep_nms.numel(), 4}));
  ctx.template Alloc<T>(&proposals);
  scores_sel.Resize(common::make_ddim({keep_nms.numel(), 1}));
  ctx.template Alloc<T>(&scores_sel);
  phi::funcs::CPUGather<T>(ctx, bbox_sel, keep_nms, &proposals);
  phi::funcs::CPUGather<T>(ctx, scores_filter, keep_nms, &scores_sel);

  return std::make_pair(proposals, scores_sel);
}

template <typename T, typename Context>
void GenerateProposalsKernel(const Context& ctx,
                             const DenseTensor& scores,
                             const DenseTensor& bbox_deltas,
                             const DenseTensor& im_shape,
                             const DenseTensor& anchors,
                             const DenseTensor& variances,
                             int pre_nms_top_n,
                             int post_nms_top_n,
                             float nms_thresh,
                             float min_size,
                             float eta,
                             bool pixel_offset,
                             DenseTensor* rpn_rois,
                             DenseTensor* rpn_roi_probs,
                             DenseTensor* rpn_rois_num) {
  auto& scores_dim = scores.dims();
  int64_t num = scores_dim[0];
  int64_t c_score = scores_dim[1];
  int64_t h_score = scores_dim[2];
  int64_t w_score = scores_dim[3];

  auto& bbox_dim = bbox_deltas.dims();
  int64_t c_bbox = bbox_dim[1];
  int64_t h_bbox = bbox_dim[2];
  int64_t w_bbox = bbox_dim[3];

  rpn_rois->Resize(common::make_ddim({bbox_deltas.numel() / 4, 4}));
  ctx.template Alloc<T>(rpn_rois);

  rpn_roi_probs->Resize(common::make_ddim({scores.numel(), 1}));
  ctx.template Alloc<T>(rpn_roi_probs);

  DenseTensor bbox_deltas_swap, scores_swap;
  bbox_deltas_swap.Resize(common::make_ddim({num, h_bbox, w_bbox, c_bbox}));
  ctx.template Alloc<T>(&bbox_deltas_swap);

  scores_swap.Resize(common::make_ddim({num, h_score, w_score, c_score}));
  ctx.template Alloc<T>(&scores_swap);

  phi::funcs::Transpose<phi::CPUContext, T, 4> trans;
  std::vector<int> axis = {0, 2, 3, 1};
  trans(ctx, bbox_deltas, &bbox_deltas_swap, axis);
  trans(ctx, scores, &scores_swap, axis);

  phi::LegacyLoD lod;
  lod.resize(1);
  auto& lod0 = lod[0];
  lod0.push_back(0);
  DenseTensor tmp_anchors = anchors;
  DenseTensor tmp_variances = variances;
  tmp_anchors.Resize(common::make_ddim({tmp_anchors.numel() / 4, 4}));
  tmp_variances.Resize(common::make_ddim({tmp_variances.numel() / 4, 4}));
  std::vector<int> tmp_num;

  int64_t num_proposals = 0;
  for (int64_t i = 0; i < num; ++i) {
    DenseTensor im_shape_slice = im_shape.Slice(i, i + 1);
    DenseTensor bbox_deltas_slice = bbox_deltas_swap.Slice(i, i + 1);
    DenseTensor scores_slice = scores_swap.Slice(i, i + 1);

    bbox_deltas_slice.Resize(
        common::make_ddim({h_bbox * w_bbox * c_bbox / 4, 4}));
    scores_slice.Resize(common::make_ddim({h_score * w_score * c_score, 1}));

    std::pair<DenseTensor, DenseTensor> tensor_pair =
        ProposalForOneImage<T>(ctx,
                               im_shape_slice,
                               tmp_anchors,
                               tmp_variances,
                               bbox_deltas_slice,
                               scores_slice,
                               pre_nms_top_n,
                               post_nms_top_n,
                               nms_thresh,
                               min_size,
                               eta,
                               pixel_offset);
    DenseTensor& proposals = tensor_pair.first;
    DenseTensor& nscores = tensor_pair.second;

    AppendProposals(rpn_rois, 4 * num_proposals, proposals);
    AppendProposals(rpn_roi_probs, num_proposals, nscores);
    num_proposals += proposals.dims()[0];
    lod0.push_back(num_proposals);
    tmp_num.push_back(static_cast<int>(proposals.dims()[0]));
  }
  if (rpn_rois_num != nullptr) {
    rpn_rois_num->Resize(common::make_ddim({num}));
    ctx.template Alloc<int>(rpn_rois_num);
    int* num_data = rpn_rois_num->data<int>();
    for (int i = 0; i < num; i++) {
      num_data[i] = tmp_num[i];
    }
    rpn_rois_num->Resize(common::make_ddim({num}));
  }
  rpn_rois->Resize(common::make_ddim({num_proposals, 4}));
  rpn_roi_probs->Resize(common::make_ddim({num_proposals, 1}));
}

}  // namespace phi

PD_REGISTER_KERNEL(generate_proposals,
                   CPU,
                   ALL_LAYOUT,
                   phi::GenerateProposalsKernel,
                   float,
                   double) {
  kernel->OutputAt(2).SetDataType(phi::DataType::INT32);
}
