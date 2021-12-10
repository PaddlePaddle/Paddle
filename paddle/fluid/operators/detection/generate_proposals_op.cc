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

#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/operators/detection/bbox_util.h"
#include "paddle/fluid/operators/detection/nms_util.h"
#include "paddle/fluid/operators/gather.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

class GenerateProposalsOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Scores"), true,
        platform::errors::NotFound("Input(Scores) shouldn't be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("BboxDeltas"), true,
        platform::errors::NotFound("Input(BboxDeltas) shouldn't be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("ImInfo"), true,
        platform::errors::NotFound("Input(ImInfo) shouldn't be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Anchors"), true,
        platform::errors::NotFound("Input(Anchors) shouldn't be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Variances"), true,
        platform::errors::NotFound("Input(Variances) shouldn't be null."));

    ctx->SetOutputDim("RpnRois", {-1, 4});
    ctx->SetOutputDim("RpnRoiProbs", {-1, 1});
    if (!ctx->IsRuntime()) {
      ctx->SetLoDLevel("RpnRois", std::max(ctx->GetLoDLevel("Scores"), 1));
      ctx->SetLoDLevel("RpnRoiProbs", std::max(ctx->GetLoDLevel("Scores"), 1));
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Anchors"),
        ctx.device_context());
  }
};

template <typename T>
class GenerateProposalsKernel : public framework::OpKernel<T> {
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

    auto &dev_ctx =
        context.template device_context<platform::CPUDeviceContext>();

    auto &scores_dim = scores->dims();
    int64_t num = scores_dim[0];
    int64_t c_score = scores_dim[1];
    int64_t h_score = scores_dim[2];
    int64_t w_score = scores_dim[3];

    auto &bbox_dim = bbox_deltas->dims();
    int64_t c_bbox = bbox_dim[1];
    int64_t h_bbox = bbox_dim[2];
    int64_t w_bbox = bbox_dim[3];

    rpn_rois->mutable_data<T>({bbox_deltas->numel() / 4, 4},
                              context.GetPlace());
    rpn_roi_probs->mutable_data<T>({scores->numel(), 1}, context.GetPlace());

    Tensor bbox_deltas_swap, scores_swap;
    bbox_deltas_swap.mutable_data<T>({num, h_bbox, w_bbox, c_bbox},
                                     dev_ctx.GetPlace());
    scores_swap.mutable_data<T>({num, h_score, w_score, c_score},
                                dev_ctx.GetPlace());

    math::Transpose<platform::CPUDeviceContext, T, 4> trans;
    std::vector<int> axis = {0, 2, 3, 1};
    trans(dev_ctx, *bbox_deltas, &bbox_deltas_swap, axis);
    trans(dev_ctx, *scores, &scores_swap, axis);

    framework::LoD lod;
    lod.resize(1);
    auto &lod0 = lod[0];
    lod0.push_back(0);
    anchors.Resize({anchors.numel() / 4, 4});
    variances.Resize({variances.numel() / 4, 4});
    std::vector<int> tmp_num;

    int64_t num_proposals = 0;
    for (int64_t i = 0; i < num; ++i) {
      Tensor im_info_slice = im_info->Slice(i, i + 1);
      Tensor bbox_deltas_slice = bbox_deltas_swap.Slice(i, i + 1);
      Tensor scores_slice = scores_swap.Slice(i, i + 1);

      bbox_deltas_slice.Resize({h_bbox * w_bbox * c_bbox / 4, 4});
      scores_slice.Resize({h_score * w_score * c_score, 1});

      std::pair<Tensor, Tensor> tensor_pair =
          ProposalForOneImage(dev_ctx, im_info_slice, anchors, variances,
                              bbox_deltas_slice, scores_slice, pre_nms_top_n,
                              post_nms_top_n, nms_thresh, min_size, eta);
      Tensor &proposals = tensor_pair.first;
      Tensor &scores = tensor_pair.second;

      AppendProposals(rpn_rois, 4 * num_proposals, proposals);
      AppendProposals(rpn_roi_probs, num_proposals, scores);
      num_proposals += proposals.dims()[0];
      lod0.push_back(num_proposals);
      tmp_num.push_back(proposals.dims()[0]);
    }
    if (context.HasOutput("RpnRoisNum")) {
      auto *rpn_rois_num = context.Output<Tensor>("RpnRoisNum");
      rpn_rois_num->mutable_data<int>({num}, context.GetPlace());
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

  std::pair<Tensor, Tensor> ProposalForOneImage(
      const platform::CPUDeviceContext &ctx, const Tensor &im_info_slice,
      const Tensor &anchors, const Tensor &variances,
      const Tensor &bbox_deltas_slice,  // [M, 4]
      const Tensor &scores_slice,       // [N, 1]
      int pre_nms_top_n, int post_nms_top_n, float nms_thresh, float min_size,
      float eta) const {
    auto *scores_data = scores_slice.data<T>();

    // Sort index
    Tensor index_t;
    index_t.Resize({scores_slice.numel()});
    int *index = index_t.mutable_data<int>(ctx.GetPlace());
    for (int i = 0; i < scores_slice.numel(); ++i) {
      index[i] = i;
    }
    auto compare = [scores_data](const int64_t &i, const int64_t &j) {
      return scores_data[i] > scores_data[j];
    };

    if (pre_nms_top_n <= 0 || pre_nms_top_n >= scores_slice.numel()) {
      std::sort(index, index + scores_slice.numel(), compare);
    } else {
      std::nth_element(index, index + pre_nms_top_n,
                       index + scores_slice.numel(), compare);
      index_t.Resize({pre_nms_top_n});
    }

    Tensor scores_sel, bbox_sel, anchor_sel, var_sel;
    scores_sel.mutable_data<T>({index_t.numel(), 1}, ctx.GetPlace());
    bbox_sel.mutable_data<T>({index_t.numel(), 4}, ctx.GetPlace());
    anchor_sel.mutable_data<T>({index_t.numel(), 4}, ctx.GetPlace());
    var_sel.mutable_data<T>({index_t.numel(), 4}, ctx.GetPlace());

    CPUGather<T>(ctx, scores_slice, index_t, &scores_sel);
    CPUGather<T>(ctx, bbox_deltas_slice, index_t, &bbox_sel);
    CPUGather<T>(ctx, anchors, index_t, &anchor_sel);
    CPUGather<T>(ctx, variances, index_t, &var_sel);

    Tensor proposals;
    proposals.mutable_data<T>({index_t.numel(), 4}, ctx.GetPlace());
    BoxCoder<T>(ctx, &anchor_sel, &bbox_sel, &var_sel, &proposals);

    ClipTiledBoxes<T>(ctx, im_info_slice, proposals, &proposals, false);

    Tensor keep;
    FilterBoxes<T>(ctx, &proposals, min_size, im_info_slice, true, &keep);
    // Handle the case when there is no keep index left
    if (keep.numel() == 0) {
      math::SetConstant<platform::CPUDeviceContext, T> set_zero;
      bbox_sel.mutable_data<T>({1, 4}, ctx.GetPlace());
      set_zero(ctx, &bbox_sel, static_cast<T>(0));
      Tensor scores_filter;
      scores_filter.mutable_data<T>({1, 1}, ctx.GetPlace());
      set_zero(ctx, &scores_filter, static_cast<T>(0));
      return std::make_pair(bbox_sel, scores_filter);
    }

    Tensor scores_filter;
    bbox_sel.mutable_data<T>({keep.numel(), 4}, ctx.GetPlace());
    scores_filter.mutable_data<T>({keep.numel(), 1}, ctx.GetPlace());
    CPUGather<T>(ctx, proposals, keep, &bbox_sel);
    CPUGather<T>(ctx, scores_sel, keep, &scores_filter);
    if (nms_thresh <= 0) {
      return std::make_pair(bbox_sel, scores_filter);
    }

    Tensor keep_nms = NMS<T>(ctx, &bbox_sel, &scores_filter, nms_thresh, eta);

    if (post_nms_top_n > 0 && post_nms_top_n < keep_nms.numel()) {
      keep_nms.Resize({post_nms_top_n});
    }

    proposals.mutable_data<T>({keep_nms.numel(), 4}, ctx.GetPlace());
    scores_sel.mutable_data<T>({keep_nms.numel(), 1}, ctx.GetPlace());
    CPUGather<T>(ctx, bbox_sel, keep_nms, &proposals);
    CPUGather<T>(ctx, scores_filter, keep_nms, &scores_sel);

    return std::make_pair(proposals, scores_sel);
  }
};

class GenerateProposalsOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Scores",
             "(Tensor) The scores from conv is in shape (N, A, H, W), "
             "N is batch size, A is number of anchors, "
             "H and W are height and width of the feature map");
    AddInput("BboxDeltas",
             "(Tensor) Bounding box deltas from conv is in "
             "shape (N, 4*A, H, W).");
    AddInput("ImInfo",
             "(Tensor) Information for image reshape is in shape (N, 3), "
             "in format (height, width, scale)");
    AddInput("Anchors",
             "(Tensor) Bounding box anchors from anchor_generator_op "
             "is in shape (H, W, A, 4).");
    AddInput("Variances",
             "(Tensor) Bounding box variances with same shape as `Anchors`.");

    AddOutput("RpnRois",
              "(LoDTensor), Output proposals with shape (rois_num, 4).");
    AddOutput("RpnRoiProbs",
              "(LoDTensor) Scores of proposals with shape (rois_num, 1).");
    AddOutput("RpnRoisNum", "(Tensor), The number of Rpn RoIs in each image")
        .AsDispensable();
    AddAttr<int>("pre_nms_topN",
                 "Number of top scoring RPN proposals to keep before "
                 "applying NMS.");
    AddAttr<int>("post_nms_topN",
                 "Number of top scoring RPN proposals to keep after "
                 "applying NMS");
    AddAttr<float>("nms_thresh", "NMS threshold used on RPN proposals.");
    AddAttr<float>("min_size",
                   "Proposal height and width both need to be greater "
                   "than this min_size.");
    AddAttr<float>("eta", "The parameter for adaptive NMS.");
    AddComment(R"DOC(
This operator Generate bounding box proposals for Faster RCNN.
The propoasls are generated for a list of images based on image
score 'Scores', bounding box regression result 'BboxDeltas' as
well as predefined bounding box shapes 'anchors'. Greedy
non-maximum suppression is applied to generate the final bounding
boxes.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    generate_proposals, ops::GenerateProposalsOp, ops::GenerateProposalsOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(generate_proposals, ops::GenerateProposalsKernel<float>,
                       ops::GenerateProposalsKernel<double>);
REGISTER_OP_VERSION(generate_proposals)
    .AddCheckpoint(
        R"ROC(
              Incompatible upgrade of output [RpnRoisLod])ROC",
        paddle::framework::compatible::OpVersionDesc().DeleteOutput(
            "RpnRoisLod",
            "Delete RpnRoisLod due to incorrect output name and "
            "it is not used in object detection models yet."))
    .AddCheckpoint(
        R"ROC(
              Upgrade generate_proposals add a new output [RpnRoisNum])ROC",
        paddle::framework::compatible::OpVersionDesc().NewOutput(
            "RpnRoisNum",
            "The number of Rpn RoIs in each image. RpnRoisNum is "
            "dispensable."));
