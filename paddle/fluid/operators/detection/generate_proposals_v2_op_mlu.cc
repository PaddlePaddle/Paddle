/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;
using LoDTensor = framework::LoDTensor;

template <typename T>
class MLUGenerateProposalsV2Kernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *scores = context.Input<Tensor>("Scores");
    auto *bbox_deltas = context.Input<Tensor>("BboxDeltas");
    auto *im_shape = context.Input<Tensor>("ImShape");
    auto anchors = context.Input<Tensor>("Anchors");
    auto variances = context.Input<Tensor>("Variances");

    auto *rpn_rois = context.Output<LoDTensor>("RpnRois");
    auto *rpn_roi_probs = context.Output<LoDTensor>("RpnRoiProbs");
    auto *rpn_roi_num = context.Output<LoDTensor>("RpnRoisNum");

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

    auto scores_dim = scores->dims();
    int64_t num = scores_dim[0];
    int64_t c_score = scores_dim[1];
    int64_t h_score = scores_dim[2];
    int64_t w_score = scores_dim[3];

    auto &dev_ctx = context.template device_context<MLUDeviceContext>();

    rpn_rois->mutable_data<T>({num * post_nms_top_n, 4}, dev_ctx.GetPlace());
    rpn_roi_probs->mutable_data<T>({num * post_nms_top_n, 1},
                                   dev_ctx.GetPlace());
    Tensor rpn_roi_num_tmp;
    if (rpn_roi_num) {
      rpn_roi_num->mutable_data<int>({num}, dev_ctx.GetPlace());
      rpn_roi_num_tmp = *rpn_roi_num;
    } else {
      rpn_roi_num_tmp.mutable_data<int>({num}, dev_ctx.GetPlace());
    }

    Tensor scores_swap, bbox_deltas_swap;
    const std::vector<int> perm = {0, 2, 3, 1};
    TransposeFromMLUTensor<T>(
        context, perm, scores, &scores_swap, true /*need_reshape_or_alloc*/);
    TransposeFromMLUTensor<T>(context,
                              perm,
                              bbox_deltas,
                              &bbox_deltas_swap,
                              true /*need_reshape_or_alloc*/);
    MLUOpTensorDesc scores_desc(scores_swap);
    MLUOpTensorDesc bbox_deltas_desc(bbox_deltas_swap);
    MLUOpTensorDesc im_shape_desc(*im_shape);
    const int64_t anchor_var_shape_4D[4] = {h_score, w_score, c_score, 4};
    MLUOpTensorDesc anchors_desc(4, anchor_var_shape_4D, ToMluOpDataType<T>());
    MLUOpTensorDesc variances_desc(
        4, anchor_var_shape_4D, ToMluOpDataType<T>());
    MLUOpTensorDesc rpn_rois_desc(*rpn_rois);
    MLUOpTensorDesc rpn_rois_probs_desc(*rpn_roi_probs);
    MLUOpTensorDesc rpn_rois_num_desc(rpn_roi_num_tmp);

    Tensor rpn_rois_batch_size;
    rpn_rois_batch_size.mutable_data<int>({1}, dev_ctx.GetPlace());
    MLUOP::GenerateProposalsV2(context,
                               pre_nms_top_n,
                               post_nms_top_n,
                               nms_thresh,
                               min_size,
                               eta,
                               pixel_offset,
                               scores_desc.get(),
                               GetBasePtr(&scores_swap),
                               bbox_deltas_desc.get(),
                               GetBasePtr(&bbox_deltas_swap),
                               im_shape_desc.get(),
                               GetBasePtr(im_shape),
                               anchors_desc.get(),
                               GetBasePtr(anchors),
                               variances_desc.get(),
                               GetBasePtr(variances),
                               rpn_rois_desc.get(),
                               GetBasePtr(rpn_rois),
                               rpn_rois_probs_desc.get(),
                               GetBasePtr(rpn_roi_probs),
                               rpn_rois_num_desc.get(),
                               GetBasePtr(&rpn_roi_num_tmp),
                               GetBasePtr(&rpn_rois_batch_size));

    std::vector<int> rpn_rois_batch_size_cpu;
    framework::TensorToVector<int>(
        rpn_rois_batch_size, dev_ctx, &rpn_rois_batch_size_cpu);
    dev_ctx.Wait();

    int roi_num_final = rpn_rois_batch_size_cpu[0];
    rpn_rois->Resize({roi_num_final, 4});
    rpn_roi_probs->Resize({roi_num_final, 1});
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_MLU_KERNEL(generate_proposals_v2,
                       ops::MLUGenerateProposalsV2Kernel<float>);
