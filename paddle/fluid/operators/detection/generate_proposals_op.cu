/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <paddle/fluid/memory/allocation/allocator.h>
#include <stdio.h>
#include <string>
#include <vector>
#include "paddle/fluid/framework/mixed_vector.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/operators/detection/bbox_util.cu.h"
#include "paddle/phi/kernels/funcs/gather.cu.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

namespace {
template <typename T>
static std::pair<Tensor, Tensor> ProposalForOneImage(
    const platform::CUDADeviceContext &ctx, const Tensor &im_info,
    const Tensor &anchors, const Tensor &variances,
    const Tensor &bbox_deltas,  // [M, 4]
    const Tensor &scores,       // [N, 1]
    int pre_nms_top_n, int post_nms_top_n, float nms_thresh, float min_size,
    float eta) {
  // 1. pre nms
  Tensor scores_sort, index_sort;
  SortDescending<T>(ctx, scores, &scores_sort, &index_sort);
  int num = scores.numel();
  int pre_nms_num = (pre_nms_top_n <= 0 || pre_nms_top_n > num) ? scores.numel()
                                                                : pre_nms_top_n;
  scores_sort.Resize({pre_nms_num, 1});
  index_sort.Resize({pre_nms_num, 1});

  // 2. box decode and clipping
  Tensor proposals;
  proposals.mutable_data<T>({pre_nms_num, 4}, ctx.GetPlace());

  {
    platform::ForRange<platform::CUDADeviceContext> for_range(ctx, pre_nms_num);
    for_range(BoxDecodeAndClipFunctor<T>{
        anchors.data<T>(), bbox_deltas.data<T>(), variances.data<T>(),
        index_sort.data<int>(), im_info.data<T>(), proposals.data<T>()});
  }

  // 3. filter
  Tensor keep_index, keep_num_t;
  keep_index.mutable_data<int>({pre_nms_num}, ctx.GetPlace());
  keep_num_t.mutable_data<int>({1}, ctx.GetPlace());
  min_size = std::max(min_size, 1.0f);
  auto stream = ctx.stream();
  FilterBBoxes<T, 512><<<1, 512, 0, stream>>>(
      proposals.data<T>(), im_info.data<T>(), min_size, pre_nms_num,
      keep_num_t.data<int>(), keep_index.data<int>());
  int keep_num;
  const auto gpu_place = ctx.GetPlace();
  memory::Copy(platform::CPUPlace(), &keep_num, gpu_place,
               keep_num_t.data<int>(), sizeof(int), ctx.stream());
  ctx.Wait();
  keep_index.Resize({keep_num});

  Tensor scores_filter, proposals_filter;
  // Handle the case when there is no keep index left
  if (keep_num == 0) {
    phi::funcs::SetConstant<platform::CUDADeviceContext, T> set_zero;
    proposals_filter.mutable_data<T>({1, 4}, ctx.GetPlace());
    scores_filter.mutable_data<T>({1, 1}, ctx.GetPlace());
    set_zero(ctx, &proposals_filter, static_cast<T>(0));
    set_zero(ctx, &scores_filter, static_cast<T>(0));
    return std::make_pair(proposals_filter, scores_filter);
  }
  proposals_filter.mutable_data<T>({keep_num, 4}, ctx.GetPlace());
  scores_filter.mutable_data<T>({keep_num, 1}, ctx.GetPlace());
  phi::funcs::GPUGather<T>(ctx, proposals, keep_index, &proposals_filter);
  phi::funcs::GPUGather<T>(ctx, scores_sort, keep_index, &scores_filter);

  if (nms_thresh <= 0) {
    return std::make_pair(proposals_filter, scores_filter);
  }

  // 4. nms
  Tensor keep_nms;
  NMS<T>(ctx, proposals_filter, keep_index, nms_thresh, &keep_nms);
  if (post_nms_top_n > 0 && post_nms_top_n < keep_nms.numel()) {
    keep_nms.Resize({post_nms_top_n});
  }

  Tensor scores_nms, proposals_nms;
  proposals_nms.mutable_data<T>({keep_nms.numel(), 4}, ctx.GetPlace());
  scores_nms.mutable_data<T>({keep_nms.numel(), 1}, ctx.GetPlace());
  phi::funcs::GPUGather<T>(ctx, proposals_filter, keep_nms, &proposals_nms);
  phi::funcs::GPUGather<T>(ctx, scores_filter, keep_nms, &scores_nms);

  return std::make_pair(proposals_nms, scores_nms);
}
}  // namespace

template <typename DeviceContext, typename T>
class CUDAGenerateProposalsKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *scores = context.Input<Tensor>("Scores");
    auto *bbox_deltas = context.Input<Tensor>("BboxDeltas");
    auto *im_info = context.Input<Tensor>("ImInfo");
    auto anchors = GET_DATA_SAFELY(context.Input<Tensor>("Anchors"), "Input",
                                   "Anchors", "GenerateProposals");
    auto variances = GET_DATA_SAFELY(context.Input<Tensor>("Variances"),
                                     "Input", "Variances", "GenerateProposals");

    auto *rpn_rois = context.Output<LoDTensor>("RpnRois");
    auto *rpn_roi_probs = context.Output<LoDTensor>("RpnRoiProbs");

    int pre_nms_top_n = context.Attr<int>("pre_nms_topN");
    int post_nms_top_n = context.Attr<int>("post_nms_topN");
    float nms_thresh = context.Attr<float>("nms_thresh");
    float min_size = context.Attr<float>("min_size");
    float eta = context.Attr<float>("eta");
    PADDLE_ENFORCE_GE(eta, 1.,
                      platform::errors::InvalidArgument(
                          "Not support adaptive NMS. The attribute 'eta' "
                          "should not less than 1. But received eta=[%d]",
                          eta));

    auto &dev_ctx = context.template device_context<DeviceContext>();

    auto scores_dim = scores->dims();
    int64_t num = scores_dim[0];
    int64_t c_score = scores_dim[1];
    int64_t h_score = scores_dim[2];
    int64_t w_score = scores_dim[3];

    auto bbox_dim = bbox_deltas->dims();
    int64_t c_bbox = bbox_dim[1];
    int64_t h_bbox = bbox_dim[2];
    int64_t w_bbox = bbox_dim[3];

    Tensor bbox_deltas_swap, scores_swap;
    bbox_deltas_swap.mutable_data<T>({num, h_bbox, w_bbox, c_bbox},
                                     dev_ctx.GetPlace());
    scores_swap.mutable_data<T>({num, h_score, w_score, c_score},
                                dev_ctx.GetPlace());

    phi::funcs::Transpose<DeviceContext, T, 4> trans;
    std::vector<int> axis = {0, 2, 3, 1};
    trans(dev_ctx, *bbox_deltas, &bbox_deltas_swap, axis);
    trans(dev_ctx, *scores, &scores_swap, axis);

    anchors.Resize({anchors.numel() / 4, 4});
    variances.Resize({variances.numel() / 4, 4});

    rpn_rois->mutable_data<T>({bbox_deltas->numel() / 4, 4},
                              context.GetPlace());
    rpn_roi_probs->mutable_data<T>({scores->numel(), 1}, context.GetPlace());

    T *rpn_rois_data = rpn_rois->data<T>();
    T *rpn_roi_probs_data = rpn_roi_probs->data<T>();

    auto place = dev_ctx.GetPlace();
    auto cpu_place = platform::CPUPlace();

    int64_t num_proposals = 0;
    std::vector<size_t> offset(1, 0);
    std::vector<int> tmp_num;

    for (int64_t i = 0; i < num; ++i) {
      Tensor im_info_slice = im_info->Slice(i, i + 1);
      Tensor bbox_deltas_slice = bbox_deltas_swap.Slice(i, i + 1);
      Tensor scores_slice = scores_swap.Slice(i, i + 1);

      bbox_deltas_slice.Resize({h_bbox * w_bbox * c_bbox / 4, 4});
      scores_slice.Resize({h_score * w_score * c_score, 1});

      std::pair<Tensor, Tensor> box_score_pair =
          ProposalForOneImage<T>(dev_ctx, im_info_slice, anchors, variances,
                                 bbox_deltas_slice, scores_slice, pre_nms_top_n,
                                 post_nms_top_n, nms_thresh, min_size, eta);

      Tensor &proposals = box_score_pair.first;
      Tensor &scores = box_score_pair.second;

      memory::Copy(place, rpn_rois_data + num_proposals * 4, place,
                   proposals.data<T>(), sizeof(T) * proposals.numel(),
                   dev_ctx.stream());
      memory::Copy(place, rpn_roi_probs_data + num_proposals, place,
                   scores.data<T>(), sizeof(T) * scores.numel(),
                   dev_ctx.stream());
      num_proposals += proposals.dims()[0];
      offset.emplace_back(num_proposals);
      tmp_num.push_back(proposals.dims()[0]);
    }
    if (context.HasOutput("RpnRoisNum")) {
      auto *rpn_rois_num = context.Output<Tensor>("RpnRoisNum");
      rpn_rois_num->mutable_data<int>({num}, context.GetPlace());
      int *num_data = rpn_rois_num->data<int>();
      memory::Copy(place, num_data, cpu_place, &tmp_num[0], sizeof(int) * num,
                   dev_ctx.stream());
      rpn_rois_num->Resize({num});
    }
    framework::LoD lod;
    lod.emplace_back(offset);
    rpn_rois->set_lod(lod);
    rpn_roi_probs->set_lod(lod);
    rpn_rois->Resize({num_proposals, 4});
    rpn_roi_probs->Resize({num_proposals, 1});
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(generate_proposals,
                        ops::CUDAGenerateProposalsKernel<
                            paddle::platform::CUDADeviceContext, float>);
