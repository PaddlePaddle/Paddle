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

#include <stdio.h>

#include <string>
#include <vector>

#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/mixed_vector.h"
#include "paddle/phi/kernels/funcs/detection/bbox_util.cu.h"
#include "paddle/phi/kernels/funcs/gather.cu.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

namespace {
template <typename T>
static std::pair<phi::DenseTensor, phi::DenseTensor> ProposalForOneImage(
    const phi::GPUContext &ctx,
    const phi::DenseTensor &im_info,
    const phi::DenseTensor &anchors,
    const phi::DenseTensor &variances,
    const phi::DenseTensor &bbox_deltas,  // [M, 4]
    const phi::DenseTensor &scores,       // [N, 1]
    int pre_nms_top_n,
    int post_nms_top_n,
    float nms_thresh,
    float min_size,
    float eta) {
  // 1. pre nms
  phi::DenseTensor scores_sort, index_sort;
  phi::funcs::SortDescending<T>(ctx, scores, &scores_sort, &index_sort);
  int num = scores.numel();
  int pre_nms_num = (pre_nms_top_n <= 0 || pre_nms_top_n > num) ? scores.numel()
                                                                : pre_nms_top_n;
  scores_sort.Resize({pre_nms_num, 1});
  index_sort.Resize({pre_nms_num, 1});

  // 2. box decode and clipping
  phi::DenseTensor proposals;
  proposals.Resize({pre_nms_num, 4});
  ctx.Alloc<T>(&proposals);

  {
    phi::funcs::ForRange<phi::GPUContext> for_range(ctx, pre_nms_num);
    for_range(phi::funcs::BoxDecodeAndClipFunctor<T>{anchors.data<T>(),
                                                     bbox_deltas.data<T>(),
                                                     variances.data<T>(),
                                                     index_sort.data<int>(),
                                                     im_info.data<T>(),
                                                     proposals.data<T>()});
  }

  // 3. filter
  phi::DenseTensor keep_index, keep_num_t;
  keep_index.Resize({pre_nms_num});
  keep_num_t.Resize({1});
  ctx.Alloc<int>(&keep_index);
  ctx.Alloc<int>(&keep_num_t);
  min_size = std::max(min_size, 1.0f);
  auto stream = ctx.stream();
  phi::funcs::FilterBBoxes<T, 512>
      <<<1, 512, 0, stream>>>(proposals.data<T>(),
                              im_info.data<T>(),
                              min_size,
                              pre_nms_num,
                              keep_num_t.data<int>(),
                              keep_index.data<int>());
  int keep_num;
  const auto gpu_place = ctx.GetPlace();
  phi::memory_utils::Copy(phi::CPUPlace(),
                          &keep_num,
                          gpu_place,
                          keep_num_t.data<int>(),
                          sizeof(int),
                          ctx.stream());
  ctx.Wait();
  keep_index.Resize({keep_num});

  phi::DenseTensor scores_filter, proposals_filter;
  // Handle the case when there is no keep index left
  if (keep_num == 0) {
    phi::funcs::SetConstant<phi::GPUContext, T> set_zero;
    proposals_filter.Resize({1, 4});
    scores_filter.Resize({1, 1});
    ctx.Alloc<T>(&proposals_filter);
    ctx.Alloc<T>(&scores_filter);
    set_zero(ctx, &proposals_filter, static_cast<T>(0));
    set_zero(ctx, &scores_filter, static_cast<T>(0));
    return std::make_pair(proposals_filter, scores_filter);
  }
  proposals_filter.Resize({keep_num, 4});
  scores_filter.Resize({keep_num, 1});
  ctx.Alloc<T>(&proposals_filter);
  ctx.Alloc<T>(&scores_filter);
  phi::funcs::GPUGather<T>(ctx, proposals, keep_index, &proposals_filter);
  phi::funcs::GPUGather<T>(ctx, scores_sort, keep_index, &scores_filter);

  if (nms_thresh <= 0) {
    return std::make_pair(proposals_filter, scores_filter);
  }

  // 4. nms
  phi::DenseTensor keep_nms;
  phi::funcs::NMS<T>(ctx, proposals_filter, keep_index, nms_thresh, &keep_nms);
  if (post_nms_top_n > 0 && post_nms_top_n < keep_nms.numel()) {
    keep_nms.Resize({post_nms_top_n});
  }

  phi::DenseTensor scores_nms, proposals_nms;
  proposals_nms.Resize({keep_nms.numel(), 4});
  scores_nms.Resize({keep_nms.numel(), 1});
  ctx.Alloc<T>(&proposals_nms);
  ctx.Alloc<T>(&scores_nms);
  phi::funcs::GPUGather<T>(ctx, proposals_filter, keep_nms, &proposals_nms);
  phi::funcs::GPUGather<T>(ctx, scores_filter, keep_nms, &scores_nms);

  return std::make_pair(proposals_nms, scores_nms);
}
}  // namespace

template <typename T, typename Context>
void CUDAGenerateProposalsKernel(const Context &dev_ctx,
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

  PADDLE_ENFORCE_GE(eta,
                    1.,
                    common::errors::InvalidArgument(
                        "Not support adaptive NMS. The attribute 'eta' "
                        "should not less than 1. But received eta=[%d]",
                        eta));

  auto scores_dim = scores->dims();
  int64_t num = scores_dim[0];
  int64_t c_score = scores_dim[1];
  int64_t h_score = scores_dim[2];
  int64_t w_score = scores_dim[3];

  auto bbox_dim = bbox_deltas->dims();
  int64_t c_bbox = bbox_dim[1];
  int64_t h_bbox = bbox_dim[2];
  int64_t w_bbox = bbox_dim[3];

  phi::DenseTensor bbox_deltas_swap, scores_swap;
  bbox_deltas_swap.Resize({num, h_bbox, w_bbox, c_bbox});
  dev_ctx.template Alloc<T>(&bbox_deltas_swap);
  scores_swap.Resize({num, h_score, w_score, c_score});
  dev_ctx.template Alloc<T>(&scores_swap);

  phi::funcs::Transpose<Context, T, 4> trans;
  std::vector<int> axis = {0, 2, 3, 1};
  trans(dev_ctx, *bbox_deltas, &bbox_deltas_swap, axis);
  trans(dev_ctx, *scores, &scores_swap, axis);

  anchors.Resize({anchors.numel() / 4, 4});
  variances.Resize({variances.numel() / 4, 4});

  rpn_rois->Resize({bbox_deltas->numel() / 4, 4});
  rpn_roi_probs->Resize({scores->numel(), 1});
  dev_ctx.template Alloc<T>(rpn_rois);
  dev_ctx.template Alloc<T>(rpn_roi_probs);

  T *rpn_rois_data = rpn_rois->data<T>();
  T *rpn_roi_probs_data = rpn_roi_probs->data<T>();

  auto place = dev_ctx.GetPlace();
  auto cpu_place = phi::CPUPlace();

  int64_t num_proposals = 0;
  std::vector<size_t> offset(1, 0);
  std::vector<int> tmp_num;

  for (int64_t i = 0; i < num; ++i) {
    phi::DenseTensor im_info_slice = im_info->Slice(i, i + 1);
    phi::DenseTensor bbox_deltas_slice = bbox_deltas_swap.Slice(i, i + 1);
    phi::DenseTensor scores_slice = scores_swap.Slice(i, i + 1);

    bbox_deltas_slice.Resize({h_bbox * w_bbox * c_bbox / 4, 4});
    scores_slice.Resize({h_score * w_score * c_score, 1});

    std::pair<phi::DenseTensor, phi::DenseTensor> box_score_pair =
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

    phi::DenseTensor &proposals = box_score_pair.first;
    phi::DenseTensor &scores = box_score_pair.second;

    phi::memory_utils::Copy(place,
                            rpn_rois_data + num_proposals * 4,
                            place,
                            proposals.data<T>(),
                            sizeof(T) * proposals.numel(),
                            dev_ctx.stream());
    phi::memory_utils::Copy(place,
                            rpn_roi_probs_data + num_proposals,
                            place,
                            scores.data<T>(),
                            sizeof(T) * scores.numel(),
                            dev_ctx.stream());
    num_proposals += proposals.dims()[0];
    offset.emplace_back(num_proposals);
    tmp_num.push_back(proposals.dims()[0]);
  }
  if (rpn_rois_num != nullptr) {
    rpn_rois_num->Resize({num});
    dev_ctx.template Alloc<int>(rpn_rois_num);
    int *num_data = rpn_rois_num->data<int>();
    phi::memory_utils::Copy(place,
                            num_data,
                            cpu_place,
                            &tmp_num[0],
                            sizeof(int) * num,
                            dev_ctx.stream());
    rpn_rois_num->Resize({num});
  }
  phi::LegacyLoD lod;
  lod.emplace_back(offset);
  rpn_rois->set_lod(lod);
  rpn_roi_probs->set_lod(lod);
  rpn_rois->Resize({num_proposals, 4});
  rpn_roi_probs->Resize({num_proposals, 1});
}
}  // namespace phi

PD_REGISTER_KERNEL(legacy_generate_proposals,
                   GPU,
                   ALL_LAYOUT,
                   phi::CUDAGenerateProposalsKernel,
                   float) {}
