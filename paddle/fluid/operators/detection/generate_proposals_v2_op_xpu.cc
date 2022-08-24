/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef PADDLE_WITH_XPU

#include <paddle/fluid/memory/allocation/allocator.h>
#include <stdio.h>

#include <string>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/platform/device/device_wrapper.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

namespace {
template <typename T>
static void SortDescending(const platform::XPUDeviceContext &dev_ctx,
                           const Tensor &value,
                           Tensor *index_out,
                           int pre_nms_top_n) {
  auto *value_data = value.data<T>();
  auto place = dev_ctx.GetPlace();
  auto cpu_place = platform::CPUPlace();

  Tensor scores_slice_cpu;
  scores_slice_cpu.Resize({value.numel()});
  auto *scores_slice_cpu_data = scores_slice_cpu.mutable_data<T>(cpu_place);

  memory::Copy(cpu_place,
               scores_slice_cpu_data,
               place,
               value_data,
               sizeof(T) * value.numel());

  // Sort index
  Tensor index_t;
  int *index = index_t.mutable_data<int>({value.numel()}, cpu_place);
  for (int i = 0; i < value.numel(); ++i) {
    index[i] = i;
  }
  auto compare = [scores_slice_cpu_data](const int64_t &i, const int64_t &j) {
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

  int *idx_out =
      index_out->mutable_data<int>({index_t.numel()}, dev_ctx.GetPlace());
  memory::Copy(place, idx_out, cpu_place, index, sizeof(T) * index_t.numel());
}

template <typename T>
static std::pair<Tensor, Tensor> ProposalForOneImage(
    const platform::XPUDeviceContext &dev_ctx,
    const Tensor &im_shape,
    const Tensor &anchors,
    const Tensor &variances,
    const Tensor &bbox_deltas,  // [M, 4]
    const Tensor &scores,       // [N, 1]
    int pre_nms_top_n,
    int post_nms_top_n,
    float nms_thresh,
    float min_size,
    float eta,
    bool pixel_offset) {
  // 1. pre nms
  Tensor index_sort;
  SortDescending<T>(dev_ctx, scores, &index_sort, pre_nms_top_n);

  Tensor scores_sel, bbox_sel, anchor_sel, var_sel;
  scores_sel.mutable_data<T>({index_sort.numel(), 1}, dev_ctx.GetPlace());
  bbox_sel.mutable_data<T>({index_sort.numel(), 4}, dev_ctx.GetPlace());
  anchor_sel.mutable_data<T>({index_sort.numel(), 4}, dev_ctx.GetPlace());
  var_sel.mutable_data<T>({index_sort.numel(), 4}, dev_ctx.GetPlace());

  int r = xpu::gather<T>(dev_ctx.x_context(),
                         scores.data<T>(),
                         index_sort.data<int>(),
                         scores_sel.data<T>(),
                         {static_cast<int>(scores.numel()), 1},
                         index_sort.numel(),
                         0);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "gather");

  r = xpu::gather<T>(dev_ctx.x_context(),
                     bbox_deltas.data<T>(),
                     index_sort.data<int>(),
                     bbox_sel.data<T>(),
                     {static_cast<int>(bbox_deltas.numel()) / 4, 4},
                     index_sort.numel(),
                     0);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "gather");

  r = xpu::gather<T>(dev_ctx.x_context(),
                     anchors.data<T>(),
                     index_sort.data<int>(),
                     anchor_sel.data<T>(),
                     {static_cast<int>(anchors.numel()) / 4, 4},
                     index_sort.numel(),
                     0);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "gather");

  r = xpu::gather<T>(dev_ctx.x_context(),
                     variances.data<T>(),
                     index_sort.data<int>(),
                     var_sel.data<T>(),
                     {static_cast<int>(variances.numel()) / 4, 4},
                     index_sort.numel(),
                     0);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "gather");

  int num = scores.numel();
  int pre_nms_num = (pre_nms_top_n <= 0 || pre_nms_top_n > num) ? scores.numel()
                                                                : pre_nms_top_n;
  scores_sel.Resize({pre_nms_num, 1});
  index_sort.Resize({pre_nms_num, 1});

  // 2. box decode and clipping
  Tensor proposals;
  proposals.mutable_data<T>({pre_nms_num, 4}, dev_ctx.GetPlace());

  r = xpu::box_decoder<T>(dev_ctx.x_context(),
                          anchor_sel.data<T>(),
                          var_sel.data<T>(),
                          bbox_sel.data<T>(),
                          proposals.data<T>(),
                          pre_nms_num,
                          !pixel_offset,
                          true,
                          im_shape.data<T>());
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "box_decoder");

  // 3. filter
  Tensor keep_index, keep_num_t;
  keep_index.mutable_data<int>({pre_nms_num}, dev_ctx.GetPlace());
  keep_num_t.mutable_data<int>({1}, dev_ctx.GetPlace());
  min_size = std::max(min_size, 1.0f);
  r = xpu::remove_small_boxes<T>(dev_ctx.x_context(),
                                 proposals.data<T>(),
                                 im_shape.data<T>(),
                                 keep_index.data<int>(),
                                 keep_num_t.data<int>(),
                                 pre_nms_num,
                                 min_size,
                                 false,
                                 pixel_offset);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "remove_small_boxes");
  int keep_num;
  const auto xpu_place = dev_ctx.GetPlace();
  memory::Copy(platform::CPUPlace(),
               &keep_num,
               xpu_place,
               keep_num_t.data<int>(),
               sizeof(int));
  keep_index.Resize({keep_num});

  Tensor scores_filter, proposals_filter;
  // Handle the case when there is no keep index left
  if (keep_num == 0) {
    phi::funcs::SetConstant<platform::XPUDeviceContext, T> set_zero;
    proposals_filter.mutable_data<T>({1, 4}, dev_ctx.GetPlace());
    scores_filter.mutable_data<T>({1, 1}, dev_ctx.GetPlace());
    set_zero(dev_ctx, &proposals_filter, static_cast<T>(0));
    set_zero(dev_ctx, &scores_filter, static_cast<T>(0));
    return std::make_pair(proposals_filter, scores_filter);
  }
  proposals_filter.mutable_data<T>({keep_num, 4}, dev_ctx.GetPlace());
  scores_filter.mutable_data<T>({keep_num, 1}, dev_ctx.GetPlace());
  r = xpu::gather<T>(dev_ctx.x_context(),
                     proposals.data<T>(),
                     keep_index.data<int>(),
                     proposals_filter.data<T>(),
                     {pre_nms_num, 4},
                     keep_num,
                     0);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "gather");

  r = xpu::gather<T>(dev_ctx.x_context(),
                     scores_sel.data<T>(),
                     keep_index.data<int>(),
                     scores_filter.data<T>(),
                     {pre_nms_num, 1},
                     keep_num,
                     0);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "gather");

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

  Tensor scores_nms, proposals_nms;
  proposals_nms.mutable_data<T>({keep_index.numel(), 4}, dev_ctx.GetPlace());
  scores_nms.mutable_data<T>({keep_index.numel(), 1}, dev_ctx.GetPlace());
  r = xpu::gather<T>(dev_ctx.x_context(),
                     proposals_filter.data<T>(),
                     keep_index.data<int>(),
                     proposals_nms.data<T>(),
                     {keep_num, 4},
                     keep_index.numel(),
                     0);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "gather");
  r = xpu::gather<T>(dev_ctx.x_context(),
                     scores_filter.data<T>(),
                     keep_index.data<int>(),
                     scores_nms.data<T>(),
                     {keep_num, 1},
                     keep_index.numel(),
                     0);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "gather");
  if (dev_ctx.x_context()->xpu_stream) {
    dev_ctx.Wait();
  }
  return std::make_pair(proposals_nms, scores_nms);
}
}  // namespace

template <typename DeviceContext, typename T>
class XPUGenerateProposalsV2Kernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *scores = context.Input<Tensor>("Scores");
    auto *bbox_deltas = context.Input<Tensor>("BboxDeltas");
    auto *im_shape = context.Input<Tensor>("ImShape");
    auto anchors = GET_DATA_SAFELY(context.Input<Tensor>("Anchors"),
                                   "Input",
                                   "Anchors",
                                   "GenerateProposals");
    auto variances = GET_DATA_SAFELY(context.Input<Tensor>("Variances"),
                                     "Input",
                                     "Variances",
                                     "GenerateProposals");

    auto *rpn_rois = context.Output<LoDTensor>("RpnRois");
    auto *rpn_roi_probs = context.Output<LoDTensor>("RpnRoiProbs");

    int pre_nms_top_n = context.Attr<int>("pre_nms_topN");
    int post_nms_top_n = context.Attr<int>("post_nms_topN");
    float nms_thresh = context.Attr<float>("nms_thresh");
    float min_size = context.Attr<float>("min_size");
    float eta = context.Attr<float>("eta");
    bool pixel_offset = context.Attr<bool>("pixel_offset");
    PADDLE_ENFORCE_GE(eta,
                      1.,
                      platform::errors::InvalidArgument(
                          "Not support adaptive NMS. The attribute 'eta' "
                          "should not less than 1. But received eta=[%d]",
                          eta));

    auto &dev_ctx = context.template device_context<DeviceContext>();

    auto scores_dim = scores->dims();
    // the shape of bbox score
    int num = scores_dim[0];
    int c_score = scores_dim[1];
    int h_score = scores_dim[2];
    int w_score = scores_dim[3];

    auto bbox_dim = bbox_deltas->dims();
    int c_bbox = bbox_dim[1];
    int h_bbox = bbox_dim[2];
    int w_bbox = bbox_dim[3];

    Tensor bbox_deltas_swap, scores_swap;
    bbox_deltas_swap.mutable_data<T>({num, h_bbox, w_bbox, c_bbox},
                                     dev_ctx.GetPlace());
    scores_swap.mutable_data<T>({num, h_score, w_score, c_score},
                                dev_ctx.GetPlace());

    std::vector<int> axis = {0, 2, 3, 1};
    int r = xpu::transpose<T>(dev_ctx.x_context(),
                              bbox_deltas->data<T>(),
                              bbox_deltas_swap.data<T>(),
                              {num, c_bbox, h_bbox, w_bbox},
                              axis);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");
    r = xpu::transpose<T>(dev_ctx.x_context(),
                          scores->data<T>(),
                          scores_swap.data<T>(),
                          {num, c_score, h_score, w_score},
                          axis);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");

    anchors.Resize({anchors.numel() / 4, 4});
    variances.Resize({variances.numel() / 4, 4});

    // output
    rpn_rois->mutable_data<T>({bbox_deltas->numel() / 4, 4},
                              context.GetPlace());
    rpn_roi_probs->mutable_data<T>({scores->numel(), 1}, context.GetPlace());

    T *rpn_rois_data = rpn_rois->data<T>();
    T *rpn_roi_probs_data = rpn_roi_probs->data<T>();

    auto place = dev_ctx.GetPlace();
    auto cpu_place = platform::CPUPlace();

    int num_proposals = 0;
    std::vector<size_t> offset(1, 0);
    std::vector<int> tmp_num;

    for (int64_t i = 0; i < num; ++i) {
      Tensor im_shape_slice = im_shape->Slice(i, i + 1);
      Tensor bbox_deltas_slice = bbox_deltas_swap.Slice(i, i + 1);
      Tensor scores_slice = scores_swap.Slice(i, i + 1);

      bbox_deltas_slice.Resize({h_bbox * w_bbox * c_bbox / 4, 4});
      scores_slice.Resize({h_score * w_score * c_score, 1});

      std::pair<Tensor, Tensor> box_score_pair =
          ProposalForOneImage<T>(dev_ctx,
                                 im_shape_slice,
                                 anchors,
                                 variances,
                                 bbox_deltas_slice,
                                 scores_slice,
                                 pre_nms_top_n,
                                 post_nms_top_n,
                                 nms_thresh,
                                 min_size,
                                 eta,
                                 pixel_offset);

      Tensor &proposals = box_score_pair.first;
      Tensor &scores = box_score_pair.second;

      memory::Copy(place,
                   rpn_rois_data + num_proposals * 4,
                   place,
                   proposals.data<T>(),
                   sizeof(T) * proposals.numel());
      memory::Copy(place,
                   rpn_roi_probs_data + num_proposals,
                   place,
                   scores.data<T>(),
                   sizeof(T) * scores.numel());
      if (dev_ctx.x_context()->xpu_stream) {
        dev_ctx.Wait();
      }
      num_proposals += proposals.dims()[0];
      offset.emplace_back(num_proposals);
      tmp_num.push_back(proposals.dims()[0]);
    }
    if (context.HasOutput("RpnRoisNum")) {
      auto *rpn_rois_num = context.Output<Tensor>("RpnRoisNum");
      rpn_rois_num->mutable_data<int>({num}, context.GetPlace());
      int *num_data = rpn_rois_num->data<int>();
      memory::Copy(place, num_data, cpu_place, &tmp_num[0], sizeof(int) * num);
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
REGISTER_OP_XPU_KERNEL(
    generate_proposals_v2,
    ops::XPUGenerateProposalsV2Kernel<paddle::platform::XPUDeviceContext,
                                      float>);

#endif  // PADDLE_WITH_XPU
