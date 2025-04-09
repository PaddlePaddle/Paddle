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

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/detection/bbox_util.h"
#include "paddle/phi/kernels/funcs/detection/nms_util.h"
#include "paddle/phi/kernels/funcs/gather.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T>
std::pair<phi::DenseTensor, phi::DenseTensor> ProposalForOneImage(
    const phi::CPUContext &ctx,
    const phi::DenseTensor &im_info_slice,
    const phi::DenseTensor &anchors,
    const phi::DenseTensor &variances,
    const phi::DenseTensor &bbox_deltas_slice,  // [M, 4]
    const phi::DenseTensor &scores_slice,       // [N, 1]
    int pre_nms_top_n,
    int post_nms_top_n,
    float nms_thresh,
    float min_size,
    float eta) {
  auto *scores_data = scores_slice.data<T>();

  // Sort index
  phi::DenseTensor index_t;
  index_t.Resize({scores_slice.numel()});
  int *index = ctx.Alloc<int>(&index_t);
  for (int i = 0; i < scores_slice.numel(); ++i) {
    index[i] = i;
  }
  auto compare = [scores_data](const int64_t &i, const int64_t &j) {
    return scores_data[i] > scores_data[j];
  };

  if (pre_nms_top_n <= 0 || pre_nms_top_n >= scores_slice.numel()) {
    std::sort(index, index + scores_slice.numel(), compare);
  } else {
    std::nth_element(
        index, index + pre_nms_top_n, index + scores_slice.numel(), compare);
    index_t.Resize({pre_nms_top_n});
  }

  phi::DenseTensor scores_sel, bbox_sel, anchor_sel, var_sel;
  scores_sel.Resize({index_t.numel(), 1});
  bbox_sel.Resize({index_t.numel(), 4});
  anchor_sel.Resize({index_t.numel(), 4});
  var_sel.Resize({index_t.numel(), 4});
  ctx.Alloc<T>(&scores_sel);
  ctx.Alloc<T>(&bbox_sel);
  ctx.Alloc<T>(&anchor_sel);
  ctx.Alloc<T>(&var_sel);

  phi::funcs::CPUGather<T>(ctx, scores_slice, index_t, &scores_sel);
  phi::funcs::CPUGather<T>(ctx, bbox_deltas_slice, index_t, &bbox_sel);
  phi::funcs::CPUGather<T>(ctx, anchors, index_t, &anchor_sel);
  phi::funcs::CPUGather<T>(ctx, variances, index_t, &var_sel);

  phi::DenseTensor proposals;
  proposals.Resize({index_t.numel(), 4});
  ctx.Alloc<T>(&proposals);
  phi::funcs::BoxCoder<T>(ctx, &anchor_sel, &bbox_sel, &var_sel, &proposals);

  phi::funcs::ClipTiledBoxes<T>(
      ctx, im_info_slice, proposals, &proposals, false);

  phi::DenseTensor keep;
  phi::funcs::FilterBoxes<T>(
      ctx, &proposals, min_size, im_info_slice, true, &keep);
  // Handle the case when there is no keep index left
  if (keep.numel() == 0) {
    phi::funcs::SetConstant<phi::CPUContext, T> set_zero;
    bbox_sel.Resize({1, 4});
    ctx.Alloc<T>(&bbox_sel);
    set_zero(ctx, &bbox_sel, static_cast<T>(0));
    phi::DenseTensor scores_filter;
    scores_filter.Resize({1, 1});
    ctx.Alloc<T>(&scores_filter);
    set_zero(ctx, &scores_filter, static_cast<T>(0));
    return std::make_pair(bbox_sel, scores_filter);
  }

  phi::DenseTensor scores_filter;
  bbox_sel.Resize({keep.numel(), 4});
  scores_filter.Resize({keep.numel(), 1});
  ctx.Alloc<T>(&bbox_sel);
  ctx.Alloc<T>(&scores_filter);
  phi::funcs::CPUGather<T>(ctx, proposals, keep, &bbox_sel);
  phi::funcs::CPUGather<T>(ctx, scores_sel, keep, &scores_filter);
  if (nms_thresh <= 0) {
    return std::make_pair(bbox_sel, scores_filter);
  }

  phi::DenseTensor keep_nms =
      phi::funcs::NMS<T>(ctx, &bbox_sel, &scores_filter, nms_thresh, eta);

  if (post_nms_top_n > 0 && post_nms_top_n < keep_nms.numel()) {
    keep_nms.Resize({post_nms_top_n});
  }

  proposals.Resize({keep_nms.numel(), 4});
  scores_sel.Resize({keep_nms.numel(), 1});
  ctx.Alloc<T>(&proposals);
  ctx.Alloc<T>(&scores_sel);
  phi::funcs::CPUGather<T>(ctx, bbox_sel, keep_nms, &proposals);
  phi::funcs::CPUGather<T>(ctx, scores_filter, keep_nms, &scores_sel);

  return std::make_pair(proposals, scores_sel);
}

template <typename T, typename Context>
void GenerateProposalsKernel(const Context &dev_ctx,
                             const DenseTensor &scores_in,
                             const DenseTensor &bbox_deltas_in,
                             const DenseTensor &im_info_in,
                             const DenseTensor &anchors_in,
                             const DenseTensor &variances_in,
                             int pre_nms_top_n,
                             int post_nms_top_n,
                             float nms_thresh,
                             float min_size,
                             float eta,
                             DenseTensor *rpn_rois,
                             DenseTensor *rpn_roi_probs,
                             DenseTensor *rpn_rois_num) {
  auto *scores = &scores_in;
  auto *bbox_deltas = &bbox_deltas_in;
  auto *im_info = &im_info_in;
  auto anchors = anchors_in;
  auto variances = variances_in;

  auto &scores_dim = scores->dims();
  int64_t num = scores_dim[0];
  int64_t c_score = scores_dim[1];
  int64_t h_score = scores_dim[2];
  int64_t w_score = scores_dim[3];

  auto &bbox_dim = bbox_deltas->dims();
  int64_t c_bbox = bbox_dim[1];
  int64_t h_bbox = bbox_dim[2];
  int64_t w_bbox = bbox_dim[3];

  rpn_rois->Resize({bbox_deltas->numel() / 4, 4});
  rpn_roi_probs->Resize({scores->numel(), 1});
  dev_ctx.template Alloc<T>(rpn_rois);
  dev_ctx.template Alloc<T>(rpn_roi_probs);

  phi::DenseTensor bbox_deltas_swap, scores_swap;
  bbox_deltas_swap.Resize({num, h_bbox, w_bbox, c_bbox});
  dev_ctx.template Alloc<T>(&bbox_deltas_swap);
  scores_swap.Resize({num, h_score, w_score, c_score});
  dev_ctx.template Alloc<T>(&scores_swap);

  phi::funcs::Transpose<phi::CPUContext, T, 4> trans;
  std::vector<int> axis = {0, 2, 3, 1};
  trans(dev_ctx, *bbox_deltas, &bbox_deltas_swap, axis);
  trans(dev_ctx, *scores, &scores_swap, axis);

  phi::LegacyLoD lod;
  lod.resize(1);
  auto &lod0 = lod[0];
  lod0.push_back(0);
  anchors.Resize({anchors.numel() / 4, 4});
  variances.Resize({variances.numel() / 4, 4});
  std::vector<int> tmp_num;

  int64_t num_proposals = 0;
  for (int64_t i = 0; i < num; ++i) {
    phi::DenseTensor im_info_slice = im_info->Slice(i, i + 1);
    phi::DenseTensor bbox_deltas_slice = bbox_deltas_swap.Slice(i, i + 1);
    phi::DenseTensor scores_slice = scores_swap.Slice(i, i + 1);

    bbox_deltas_slice.Resize({h_bbox * w_bbox * c_bbox / 4, 4});
    scores_slice.Resize({h_score * w_score * c_score, 1});

    std::pair<phi::DenseTensor, phi::DenseTensor> tensor_pair =
        ProposalForOneImage<T>(dev_ctx,
                               im_info_slice,
                               anchors,
                               variances,
                               bbox_deltas_slice,
                               scores_slice,
                               pre_nms_top_n,
                               post_nms_top_n,
                               nms_thresh,
                               min_size,
                               eta);
    phi::DenseTensor &proposals = tensor_pair.first;
    phi::DenseTensor &scores = tensor_pair.second;

    phi::funcs::AppendProposals(rpn_rois, 4 * num_proposals, proposals);
    phi::funcs::AppendProposals(rpn_roi_probs, num_proposals, scores);
    num_proposals += proposals.dims()[0];
    lod0.push_back(num_proposals);
    tmp_num.push_back(proposals.dims()[0]);  // NOLINT
  }
  if (rpn_rois_num != nullptr) {
    rpn_rois_num->Resize({num});
    dev_ctx.template Alloc<int>(rpn_rois_num);
    int *num_data = rpn_rois_num->data<int>();
    for (int i = 0; i < num; i++) {
      num_data[i] = tmp_num[i];
    }
    rpn_rois_num->Resize({num});
  }
  rpn_rois->set_lod(lod);
  rpn_roi_probs->set_lod(lod);
  rpn_rois->Resize({num_proposals, 4});
  rpn_roi_probs->Resize({num_proposals, 1});
}

}  // namespace phi

PD_REGISTER_KERNEL(legacy_generate_proposals,
                   CPU,
                   ALL_LAYOUT,
                   phi::GenerateProposalsKernel,
                   float,
                   double) {}
