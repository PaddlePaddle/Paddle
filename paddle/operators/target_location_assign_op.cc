/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/operators/target_location_assign_op.h"

namespace paddle {
namespace operators {

class TargetLocationAssignOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInputs("Loc"),
                   "Input(Loc) of TargetLocationAssignOp should not be null");
    PADDLE_ENFORCE(
        ctx->HasInput("GTBoxes"),
        "Input(GTBoxes) of TargetLocationAssignOp should not be null");
    PADDLE_ENFORCE(
        ctx->HasInput("MatchIndices"),
        "Input(MatchIndices) of TargetLocationAssignOp should not be null");
    PADDLE_ENFORCE(
        ctx->HasInput("PriorBoxes"),
        "Input(PriorBoxes) of TargetLocationAssignOp should not be null");
    PADDLE_ENFORCE(
        ctx->HasInput("PriorVariances"),
        "Input(PriorVariances) of TargetLocationAssignOp should not be null");

    PADDLE_ENFORCE(
        ctx->HasOutput("LocGT"),
        "Output(LocGT) of TargetLocationAssignOp should not be null.");
    PADDLE_ENFORCE(
        ctx->HasOutput("LocPred"),
        "Output(LocPred) of TargetLocationAssignOp should not be null.");

    auto loc_dims = ctx->GetInputDim("Loc");
    auto gt_dims = ctx->GetInputDim("GTBoxes");
    auto mi_dims = ctx->GetInputDim("MatchIndices");
    auto pb_dims = ctx->GetInputDim("PriorBoxes");
    auto pv_dims = ctx->GetInputDim("PriorVariances");
    int prior_box_num = pb_dims[0];
    PADDLE_ENFORCE_EQ(loc_dims.size(), 3UL,
                      "The rank of Input(Loc) must be 3, the shape is "
                      "[batch_size, prior_box_num, 4].");
    PADDLE_ENFORCE_EQ(gt_dims.size(), 3UL,
                      "The rank of Input(GTBoxes) must be 3, the shape is "
                      "[N, prior_box_num, 4].");
    PADDLE_ENFORCE_EQ(mi_dims.size(), 2UL,
                      "The rank of Input(MatchIndices) must be 2, the shape is "
                      "[batch_size, prior_box_num].");
    PADDLE_ENFORCE_EQ(pb_dims.size(), 2UL,
                      "The rank of Input(PriorBoxes) must be 2, the shape is "
                      "[prior_box_num, 4].");
    PADDLE_ENFORCE_EQ(
        pv_dims.size(), 2UL,
        "The rank of Input(PriorVariances) must be 2, the shape is "
        "[prior_box_num, 4].");

    PADDLE_ENFORCE_EQ(loc_dims[0], mi_dims[0],
                      "The batch_size of Input(Loc) and "
                      "Input(MatchIndices) must be the same.");

    PADDLE_ENFORCE_EQ(loc_dims[1], prior_box_num,
                      "The prior_box_num of Input(Loc) and "
                      "Input(PriorBoxes) must be the same.");
    PADDLE_ENFORCE_EQ(mi_dims[1], prior_box_num,
                      "The prior_box_num of Input(MatchIndices) and "
                      "Input(PriorBoxes) must be the same.");
    PADDLE_ENFORCE_EQ(pv_dims[0], prior_box_num,
                      "The prior_box_num of Input(PriorVariances) and "
                      "Input(PriorBoxes) must be the same.");
    PADDLE_ENFORCE_EQ(pb_dims[1], 4UL,
                      "The shape of Input(PriorBoxes) is [prior_box_num, 4].");
    PADDLE_ENFORCE_EQ(
        pv_dims[1], 4UL,
        "The shape of Input(PriorVariances) is [prior_box_num, 4].");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<framework::Tensor>("Loc")->type()),
        ctx.device_context());
  }
};

class TargetLocationAssignOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  TargetLocationAssignOpMaker(OpProto* proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("Loc",
             "(Tensor, default Tensor<float>), The input localization "
             "predictions.");
    AddInput(
        "GTBoxes",
        "(LoDTensor, default LoDTensor<float>), The input ground-truth boxes.");
    AddInput("MatchIndices",
             "(LoDTensor, default LoDTensor<int>), The input matched indices, "
             "When it's equal to -1, it doesn't match any entity.");
    AddInput("PriorBoxes",
             "(Tensor, default Tensor<float>), The input prior boxes.");
    AddInput("PriorVariances",
             "(Tensor, default Tensor<float>), The input prior variances");
    AddOutput("LocGT",
              "(LoDTensor), The output ground-truth boxes encoded with "
              "corresponding prior box and MatchIndices filtering.");
    AddOutput("LocPred",
              "(LoDTensor), The output localization predictions after "
              "corresponding MatchIndices filtering.");
    AddAttr<bool>("encode_variance_in_target",
                  "(bool, default false), If true, encode the variance of "
                  "prior box in the LocGT and LocPred target.")
        .SetDefault(false);
    AddComment(R"DOC(
TargetLocationAssign operator

Encode ground-truth boxes and corresponding prior box and variance together
 when MatchIndices value is not -1, it produces the output LocGT.
 Filtering localization predictions using MatchIndices, and encode the variance
 of prior box when encode_variance_in_target is true. it produces the output LocPred.

    )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(target_location_assign,
                             ops::TargetLocationAssignOp,
                             ops::TargetLocationAssignOpMaker);
REGISTER_OP_CPU_KERNEL(
    target_location_assign,
    ops::TargetLocationAssignOpKernel<paddle::platform::CPUDeviceContext,
                                      float>,
    ops::TargetLocationAssignOpKernel<paddle::platform::CPUDeviceContext,
                                      double>);
