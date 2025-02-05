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

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function_impl.h"

#include "paddle/phi/common/memory_utils.h"

namespace phi {

template <typename T>
static void SortDescending(const XPUContext& dev_ctx,
                           const DenseTensor& value,
                           DenseTensor* index_out,
                           int pre_nms_top_n) {
  auto* value_data = value.data<T>();
  auto place = dev_ctx.GetPlace();
  auto cpu_place = phi::CPUPlace();

  DenseTensor scores_slice_cpu;
  scores_slice_cpu.Resize({value.numel()});
  T* scores_slice_cpu_data = dev_ctx.template HostAlloc<T>(&scores_slice_cpu);

  memory_utils::Copy(cpu_place,
                     scores_slice_cpu_data,
                     place,
                     value_data,
                     sizeof(T) * value.numel());
  // Sort index
  DenseTensor index_t;
  index_t.Resize({value.numel()});
  int* index = dev_ctx.template HostAlloc<int>(&index_t);
  for (int i = 0; i < value.numel(); ++i) {
    index[i] = i;
  }

  auto compare = [scores_slice_cpu_data](const int64_t& i, const int64_t& j) {
    return scores_slice_cpu_data[i] > scores_slice_cpu_data[j];
  };

  if (pre_nms_top_n <= 0 || pre_nms_top_n >= value.numel()) {
    std::sort(index, index + value.numel(), compare);
  } else {
    std::nth_element(
        index, index + pre_nms_top_n, index + value.numel(), compare);
    std::sort(index, index + pre_nms_top_n, compare);
    index_t.Resize({pre_nms_top_n});
  }

  index_out->Resize({index_t.numel()});
  int* idx_out = dev_ctx.template Alloc<int>(index_out);
  memory_utils::Copy(
      place, idx_out, cpu_place, index, sizeof(T) * index_t.numel());
}

template <typename T>
std::pair<DenseTensor, DenseTensor> ProposalForOneImage(
    const phi::XPUContext& dev_ctx,
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
  // 1. pre nms
  DenseTensor index_sort;
  SortDescending<T>(dev_ctx, scores_slice, &index_sort, pre_nms_top_n);

  DenseTensor scores_sel, bbox_sel, anchor_sel, var_sel;
  scores_sel.Resize(common::make_ddim({index_sort.numel(), 1}));
  dev_ctx.template Alloc<T>(&scores_sel);

  bbox_sel.Resize(common::make_ddim({index_sort.numel(), 4}));
  dev_ctx.template Alloc<T>(&bbox_sel);

  anchor_sel.Resize(common::make_ddim({index_sort.numel(), 4}));
  dev_ctx.template Alloc<T>(&anchor_sel);

  var_sel.Resize(common::make_ddim({index_sort.numel(), 4}));
  dev_ctx.template Alloc<T>(&var_sel);

  int r = xpu::paddle_gather<T>(dev_ctx.x_context(),
                                scores_slice.data<T>(),
                                index_sort.data<int>(),
                                scores_sel.data<T>(),
                                {static_cast<int>(scores_slice.numel()), 1},
                                index_sort.numel(),
                                0);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "paddle_gather");

  r = xpu::paddle_gather<T>(
      dev_ctx.x_context(),
      bbox_deltas_slice.data<T>(),
      index_sort.data<int>(),
      bbox_sel.data<T>(),
      {static_cast<int>(bbox_deltas_slice.numel()) / 4, 4},
      index_sort.numel(),
      0);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "paddle_gather");

  r = xpu::paddle_gather<T>(dev_ctx.x_context(),
                            anchors.data<T>(),
                            index_sort.data<int>(),
                            anchor_sel.data<T>(),
                            {static_cast<int>(anchors.numel()) / 4, 4},
                            index_sort.numel(),
                            0);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "paddle_gather");

  r = xpu::paddle_gather<T>(dev_ctx.x_context(),
                            variances.data<T>(),
                            index_sort.data<int>(),
                            var_sel.data<T>(),
                            {static_cast<int>(variances.numel()) / 4, 4},
                            index_sort.numel(),
                            0);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "paddle_gather");

  int num = scores_slice.numel();
  int pre_nms_num = (pre_nms_top_n <= 0 || pre_nms_top_n > num)
                        ? scores_slice.numel()
                        : pre_nms_top_n;
  scores_sel.Resize({pre_nms_num, 1});
  index_sort.Resize({pre_nms_num, 1});

  // 2. box decode and clipping
  DenseTensor proposals;
  proposals.Resize(common::make_ddim({index_sort.numel(), 4}));
  dev_ctx.template Alloc<T>(&proposals);

  r = xpu::box_decoder<T>(dev_ctx.x_context(),
                          anchor_sel.data<T>(),
                          var_sel.data<T>(),
                          bbox_sel.data<T>(),
                          proposals.data<T>(),
                          pre_nms_num,
                          !pixel_offset,
                          true,
                          im_shape_slice.data<T>());
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "box_decoder");

  // 3. filter
  DenseTensor keep_index, keep_num_t;
  keep_index.Resize(common::make_ddim({pre_nms_num}));
  dev_ctx.template Alloc<int>(&keep_index);

  keep_num_t.Resize(common::make_ddim({1}));
  dev_ctx.template Alloc<int>(&keep_num_t);
  min_size = std::max(min_size, 1.0f);
  r = xpu::remove_small_boxes<T>(dev_ctx.x_context(),
                                 proposals.data<T>(),
                                 im_shape_slice.data<T>(),
                                 keep_index.data<int>(),
                                 keep_num_t.data<int>(),
                                 pre_nms_num,
                                 min_size,
                                 false,
                                 pixel_offset);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "remove_small_boxes");

  int keep_num;
  const auto xpu_place = dev_ctx.GetPlace();
  memory_utils::Copy(phi::CPUPlace(),
                     &keep_num,
                     xpu_place,
                     keep_num_t.data<int>(),
                     sizeof(int));
  keep_index.Resize({keep_num});

  DenseTensor scores_filter, proposals_filter;
  // Handle the case when there is no keep index left
  if (keep_num == 0) {
    phi::funcs::SetConstant<phi::XPUContext, T> set_zero;
    proposals_filter.Resize(common::make_ddim({1, 4}));
    dev_ctx.template Alloc<T>(&proposals_filter);
    scores_filter.Resize(common::make_ddim({1, 1}));
    dev_ctx.template Alloc<T>(&scores_filter);
    set_zero(dev_ctx, &proposals_filter, static_cast<T>(0));
    set_zero(dev_ctx, &scores_filter, static_cast<T>(0));
    return std::make_pair(proposals_filter, scores_filter);
  }
  proposals_filter.Resize(common::make_ddim({keep_num, 4}));
  dev_ctx.template Alloc<T>(&proposals_filter);
  scores_filter.Resize(common::make_ddim({keep_num, 1}));
  dev_ctx.template Alloc<T>(&scores_filter);
  r = xpu::paddle_gather<T>(dev_ctx.x_context(),
                            proposals.data<T>(),
                            keep_index.data<int>(),
                            proposals_filter.data<T>(),
                            {pre_nms_num, 4},
                            keep_num,
                            0);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "paddle_gather");

  r = xpu::paddle_gather<T>(dev_ctx.x_context(),
                            scores_sel.data<T>(),
                            keep_index.data<int>(),
                            scores_filter.data<T>(),
                            {pre_nms_num, 1},
                            keep_num,
                            0);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "paddle_gather");

  if (nms_thresh <= 0) {
    if (dev_ctx.x_context()->xpu_stream) {
      dev_ctx.Wait();
    }
    return std::make_pair(proposals_filter, scores_filter);
  }

  // 4. nms
  int nms_keep_num = 0;
  r = xpu::sorted_nms<T>(dev_ctx.x_context(),
                         proposals_filter.data<T>(),
                         keep_index.data<int>(),
                         nms_keep_num,
                         keep_num,
                         nms_thresh,
                         pixel_offset);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "sorted_nms");
  if (post_nms_top_n > 0 && post_nms_top_n < nms_keep_num) {
    keep_index.Resize({post_nms_top_n});
  } else {
    keep_index.Resize({nms_keep_num});
  }

  DenseTensor scores_nms, proposals_nms;
  proposals_nms.Resize(common::make_ddim({keep_index.numel(), 4}));
  dev_ctx.template Alloc<T>(&proposals_nms);
  scores_nms.Resize(common::make_ddim({keep_index.numel(), 1}));
  dev_ctx.template Alloc<T>(&scores_nms);
  r = xpu::paddle_gather<T>(dev_ctx.x_context(),
                            proposals_filter.data<T>(),
                            keep_index.data<int>(),
                            proposals_nms.data<T>(),
                            {keep_num, 4},
                            keep_index.numel(),
                            0);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "paddle_gather");
  r = xpu::paddle_gather<T>(dev_ctx.x_context(),
                            scores_filter.data<T>(),
                            keep_index.data<int>(),
                            scores_nms.data<T>(),
                            {keep_num, 1},
                            keep_index.numel(),
                            0);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "paddle_gather");
  if (dev_ctx.x_context()->xpu_stream) {
    dev_ctx.Wait();
  }
  return std::make_pair(proposals_nms, scores_nms);
}

template <typename T, typename Context>
void GenerateProposalsKernel(const Context& dev_ctx,
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
  PADDLE_ENFORCE_GE(eta,
                    1.,
                    common::errors::InvalidArgument(
                        "Not support adaptive NMS. The attribute 'eta' "
                        "should not less than 1. But received eta=[%d]",
                        eta));

  auto& scores_dim = scores.dims();
  // the shape of bbox score
  int num = scores_dim[0];
  int c_score = scores_dim[1];
  int h_score = scores_dim[2];
  int w_score = scores_dim[3];

  auto& bbox_dim = bbox_deltas.dims();
  int c_bbox = bbox_dim[1];
  int h_bbox = bbox_dim[2];
  int w_bbox = bbox_dim[3];

  DenseTensor bbox_deltas_swap, scores_swap;
  bbox_deltas_swap.Resize(common::make_ddim({num, h_bbox, w_bbox, c_bbox}));
  dev_ctx.template Alloc<T>(&bbox_deltas_swap);

  scores_swap.Resize(common::make_ddim({num, h_score, w_score, c_score}));
  dev_ctx.template Alloc<T>(&scores_swap);

  std::vector<int> axis = {0, 2, 3, 1};
  int r = xpu::transpose<T>(dev_ctx.x_context(),
                            bbox_deltas.data<T>(),
                            bbox_deltas_swap.data<T>(),
                            {num, c_bbox, h_bbox, w_bbox},
                            axis);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");

  r = xpu::transpose<T>(dev_ctx.x_context(),
                        scores.data<T>(),
                        scores_swap.data<T>(),
                        {num, c_score, h_score, w_score},
                        axis);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");

  DenseTensor tmp_anchors = anchors;
  DenseTensor tmp_variances = variances;
  tmp_anchors.Resize(common::make_ddim({tmp_anchors.numel() / 4, 4}));
  tmp_variances.Resize(common::make_ddim({tmp_variances.numel() / 4, 4}));

  // output
  rpn_rois->Resize(common::make_ddim({bbox_deltas.numel() / 4, 4}));
  dev_ctx.template Alloc<T>(rpn_rois);

  rpn_roi_probs->Resize(common::make_ddim({scores.numel(), 1}));
  dev_ctx.template Alloc<T>(rpn_roi_probs);

  auto place = dev_ctx.GetPlace();
  auto cpu_place = phi::CPUPlace();

  int num_proposals = 0;
  std::vector<size_t> offset(1, 0);
  std::vector<int> tmp_num;

  for (int64_t i = 0; i < num; ++i) {
    DenseTensor im_shape_slice = im_shape.Slice(i, i + 1);
    DenseTensor bbox_deltas_slice = bbox_deltas_swap.Slice(i, i + 1);
    DenseTensor scores_slice = scores_swap.Slice(i, i + 1);

    bbox_deltas_slice.Resize(
        common::make_ddim({h_bbox * w_bbox * c_bbox / 4, 4}));
    scores_slice.Resize(common::make_ddim({h_score * w_score * c_score, 1}));

    std::pair<DenseTensor, DenseTensor> tensor_pair =
        ProposalForOneImage<T>(dev_ctx,
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

    r = xpu::copy(dev_ctx.x_context(),
                  proposals.data<T>(),
                  rpn_rois->data<T>() + num_proposals * 4,
                  proposals.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");
    r = xpu::copy(dev_ctx.x_context(),
                  nscores.data<T>(),
                  rpn_roi_probs->data<T>() + num_proposals,
                  nscores.numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");

    if (dev_ctx.x_context()->xpu_stream) {
      dev_ctx.Wait();
    }
    num_proposals += proposals.dims()[0];
    offset.emplace_back(num_proposals);
    tmp_num.push_back(proposals.dims()[0]);
  }

  if (rpn_rois_num != nullptr) {
    rpn_rois_num->Resize(common::make_ddim({num}));
    dev_ctx.template Alloc<int>(rpn_rois_num);
    int* num_data = rpn_rois_num->data<int>();
    memory_utils::Copy(
        place, num_data, cpu_place, &tmp_num[0], sizeof(int) * num);
  }

  phi::LegacyLoD lod;
  lod.emplace_back(offset);
  rpn_rois->set_lod(lod);
  rpn_roi_probs->set_lod(lod);
  rpn_rois->Resize(common::make_ddim({num_proposals, 4}));
  rpn_roi_probs->Resize(common::make_ddim({num_proposals, 1}));
}
}  // namespace phi

PD_REGISTER_KERNEL(
    generate_proposals, XPU, ALL_LAYOUT, phi::GenerateProposalsKernel, float) {
  kernel->OutputAt(2).SetDataType(phi::DataType::INT32);
}
