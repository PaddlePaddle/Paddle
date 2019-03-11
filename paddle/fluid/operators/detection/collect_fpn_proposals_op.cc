/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
 Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
     http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.*/

#include "paddle/fluid/operators/detection/collect_fpn_proposals_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
class CollectFpnProposalsOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *context) const override {
    PADDLE_ENFORCE(context->HasInputs("MultiLayerRois"),
                   "Inputs(MultiLayerRois) shouldn't be null");
    PADDLE_ENFORCE(context->HasInputs("MultiLayerScores"),
                   "Inputs(MultiLayerScores) shouldn't be null");
    PADDLE_ENFORCE(context->HasOutput("FpnRois"),
                   "Outputs(MultiFpnRois) of DistributeOp should not be null");
    auto roi_dims = context->GetInputsDim("MultiLayerRois");
    auto score_dims = context->GetInputsDim("MultiLayerScores");
    auto post_nms_topN = context->Attrs().Get<int>("post_nms_topN");
    std::vector<int64_t> out_dims;
    for (auto &roi_dim : roi_dims) {
      PADDLE_ENFORCE_EQ(roi_dim[1], 4,
                        "Second dimension of Input(MultiLayerRois) must be 4");
    }
    for (auto &score_dim : score_dims) {
      PADDLE_ENFORCE_EQ(
          score_dim[1], 1,
          "Second dimension of Input(MultiLayerScores) must be 1");
    }
    context->SetOutputDim("FpnRois", {post_nms_topN, 4});
    if (!context->IsRuntime()) {  // Runtime LoD infershape will be computed
      // in Kernel.
      context->ShareLoD("MultiLayerRois", "FpnRois");
    }
    if (context->IsRuntime()) {
      std::vector<framework::InferShapeVarPtr> roi_inputs =
          context->GetInputVarPtrs("MultiLayerRois");
      std::vector<framework::InferShapeVarPtr> score_inputs =
          context->GetInputVarPtrs("MultiLayerScores");
      framework::LoD lod;
      for (size_t i = 0; i < roi_inputs.size(); ++i) {
        framework::Variable *roi_var =
            boost::get<framework::Variable *>(roi_inputs[i]);
        framework::Variable *score_var =
            boost::get<framework::Variable *>(score_inputs[i]);
        auto &roi_lod = roi_var->Get<LoDTensor>().lod();
        auto &score_lod = score_var->Get<LoDTensor>().lod();
        PADDLE_ENFORCE_EQ(roi_lod, score_lod,
                          "Inputs(MultiLayerRois) and Inputs(MultiLayerScores) "
                          "should have same lod.");
        if (lod.size() == 0) {
          lod = roi_lod;
        }
        PADDLE_ENFORCE_EQ(
            lod.size(), roi_lod.size(),
            "Each of Inputs(MultiLayerRois) should have same lod size.");
        PADDLE_ENFORCE_EQ(
            lod.size(), score_lod.size(),
            "Each of Inputs(MultiLayerScores) should have same lod size.");
      }
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto data_type =
        framework::GetDataTypeOfVar(ctx.MultiInputVar("MultiLayerRois")[0]);
    return framework::OpKernelType(data_type, ctx.GetPlace());
  }
};

class CollectFpnProposalsOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("MultiLayerRois",
             "(LoDTensor) Multiple roi LoDTensors from each levels in shape "
             "(N, 4), N is the number of rois")
        .AsDuplicable();
    AddInput("MultiLayerScores",
             "(LoDTensor) Multiple score LoDTensors from each levels in shape"
             " (N, 1), N is the number of rois.")
        .AsDuplicable();
    AddOutput("FpnRois", "(LoDTensor) All selected rois with highest scores");
    AddAttr<int>("post_nms_topN",
                 "Select post_nms_topN rois from"
                 " all images and all fpn layers");
    AddComment(R"DOC(
This operator collects all proposals from different images
 and different FPN levels. Then sort all of those proposals
by objectness confidence. Select the post_nms_topN rois in
 total. Finally, re-sort the rois in the order of batch Id. 
)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(collect_fpn_proposals, ops::CollectFpnProposalsOp,
                  ops::CollectFpnProposalsOpMaker,
                  paddle::framework::EmptyGradOpMaker);
REGISTER_OP_CPU_KERNEL(collect_fpn_proposals,
                       ops::CollectFpnProposalsOpKernel<float>,
                       ops::CollectFpnProposalsOpKernel<double>);
