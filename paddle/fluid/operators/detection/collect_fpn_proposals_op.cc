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

#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace operators {

class CollectFpnProposalsOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *context) const override {
    PADDLE_ENFORCE_EQ(
        context->HasInputs("MultiLevelRois"),
        true,
        platform::errors::NotFound("Inputs(MultiLevelRois) of "
                                   "CollectFpnProposalsOp is not found"));
    PADDLE_ENFORCE_EQ(
        context->HasInputs("MultiLevelScores"),
        true,
        platform::errors::NotFound("Inputs(MultiLevelScores) of "
                                   "CollectFpnProposalsOp is not found"));
    PADDLE_ENFORCE_EQ(
        context->HasOutput("FpnRois"),
        true,
        platform::errors::NotFound("Outputs(MultiFpnRois) of "
                                   "CollectFpnProposalsOp is not found"));
    auto roi_dims = context->GetInputsDim("MultiLevelRois");
    auto score_dims = context->GetInputsDim("MultiLevelScores");
    auto post_nms_topN = context->Attrs().Get<int>("post_nms_topN");
    std::vector<int64_t> out_dims;
    for (auto &roi_dim : roi_dims) {
      PADDLE_ENFORCE_EQ(
          roi_dim[1],
          4,
          platform::errors::InvalidArgument(
              "Second dimension of Input"
              "(MultiLevelRois) must be 4. But received dimension = %d",
              roi_dim[1]));
    }
    for (auto &score_dim : score_dims) {
      PADDLE_ENFORCE_EQ(
          score_dim[1],
          1,
          platform::errors::InvalidArgument(
              "Second dimension of Input"
              "(MultiLevelScores) must be 1. But received dimension = %d",
              score_dim[1]));
    }
    context->SetOutputDim("FpnRois", {post_nms_topN, 4});
    if (context->HasOutput("RoisNum")) {
      context->SetOutputDim("RoisNum", {-1});
    }
    if (!context->IsRuntime()) {  // Runtime LoD infershape will be computed
      // in Kernel.
      context->ShareLoD("MultiLevelRois", "FpnRois");
    }
    if (context->IsRuntime() && !context->HasInputs("MultiLevelRoIsNum")) {
      auto roi_inputs = context->GetInputVarPtrs("MultiLevelRois");
      auto score_inputs = context->GetInputVarPtrs("MultiLevelScores");
      for (size_t i = 0; i < roi_inputs.size(); ++i) {
        framework::Variable *roi_var =
            PADDLE_GET(framework::Variable *, roi_inputs[i]);
        framework::Variable *score_var =
            PADDLE_GET(framework::Variable *, score_inputs[i]);
        auto &roi_lod = roi_var->Get<phi::DenseTensor>().lod();
        auto &score_lod = score_var->Get<phi::DenseTensor>().lod();
        PADDLE_ENFORCE_EQ(
            roi_lod,
            score_lod,
            platform::errors::InvalidArgument(
                "Inputs(MultiLevelRois) and "
                "Inputs(MultiLevelScores) should have same lod."));
      }
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto data_type =
        OperatorWithKernel::IndicateVarDataType(ctx, "MultiLevelRois");
    return framework::OpKernelType(data_type, ctx.GetPlace());
  }
};

class CollectFpnProposalsOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("MultiLevelRois",
             "(phi::DenseTensor) Multiple roi phi::DenseTensors from each "
             "level in shape "
             "(N, 4), N is the number of RoIs")
        .AsDuplicable();
    AddInput("MultiLevelScores",
             "(phi::DenseTensor) Multiple score phi::DenseTensors from each "
             "level in shape"
             " (N, 1), N is the number of RoIs.")
        .AsDuplicable();
    AddInput(
        "MultiLevelRoIsNum",
        "(List of Tensor) The RoIs' number of each image on multiple levels."
        "The number on each level has the shape of (N), N is the number of "
        "images.")
        .AsDuplicable()
        .AsDispensable();
    AddOutput("FpnRois",
              "(phi::DenseTensor) All selected RoIs with highest scores");
    AddOutput("RoisNum", "(Tensor), Number of RoIs in each images.")
        .AsDispensable();
    AddAttr<int>("post_nms_topN",
                 "Select post_nms_topN RoIs from"
                 " all images and all fpn layers");
    AddComment(R"DOC(
This operator concats all proposals from different images
 and different FPN levels. Then sort all of those proposals
by objectness confidence. Select the post_nms_topN RoIs in
 total. Finally, re-sort the RoIs in the order of batch index.
)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    collect_fpn_proposals,
    ops::CollectFpnProposalsOp,
    ops::CollectFpnProposalsOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(collect_fpn_proposals,
                       ops::CollectFpnProposalsOpKernel<float>,
                       ops::CollectFpnProposalsOpKernel<double>);
REGISTER_OP_VERSION(collect_fpn_proposals)
    .AddCheckpoint(
        R"ROC(
              Upgrade collect_fpn_proposals add a new input
              [MultiLevelRoIsNum] and add a new output [RoisNum].)ROC",
        paddle::framework::compatible::OpVersionDesc()
            .NewInput("MultiLevelRoIsNum",
                      "The RoIs' number of each image on multiple levels."
                      "The number on each level has the shape of (N), "
                      "N is the number of images.")
            .NewOutput("RoisNum", "The number of RoIs in each image."));
