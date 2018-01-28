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

#include "paddle/operators/target_confidence_assign_op.h"

namespace paddle {
namespace operators {

class TargetConfidenceAssignOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(
        ctx->HasInput("Conf"),
        "Input(Conf) of TargetConfidenceAssignOp should not be null");
    PADDLE_ENFORCE(
        ctx->HasInput("GTLabels"),
        "Input(GTLabels) of TargetConfidenceAssignOp should not be null");
    PADDLE_ENFORCE(ctx->HasInput("MatchIndices"),
                   "Input(MatchIndices) of TargetConfidenceAssignOp should "
                   "not be null");
    PADDLE_ENFORCE(
        ctx->HasInput("NegIndices"),
        "Input(NegIndices) of TargetConfidenceAssignOp should not be null");

    PADDLE_ENFORCE(
        ctx->HasOutput("ConfGT"),
        "Output(ConfGT) of TargetConfidenceAssignOp should not be null.");
    PADDLE_ENFORCE(
        ctx->HasOutput("ConfPred"),
        "Output(ConfPred) of TargetConfidenceAssignOp should not be null.");

    auto conf_dims = ctx->GetInputDim("Conf");
    auto gt_dims = ctx->GetInputDim("GTLabels");
    auto mi_dims = ctx->GetInputDim("MatchIndices");
    auto neg_dims = ctx->GetInputDim("NegIndices");
    PADDLE_ENFORCE_EQ(conf_dims.size(), 3UL,
                      "The rank of Input(Conf) must be 3, the shape is "
                      "[batch_size, prior_box_num, class_num].");
    PADDLE_ENFORCE_EQ(gt_dims.size(), 2UL,
                      "The rank of Input(GTLabels) must be 2, the shape is "
                      "[N, 1].");
    PADDLE_ENFORCE_EQ(mi_dims.size(), 2UL,
                      "The rank of Input(MatchIndices) must be 2, the shape is "
                      "[batch_size, prior_box_num].");
    PADDLE_ENFORCE_EQ(neg_dims.size(), 2UL,
                      "The rank of Input(NegIndices) must be 2, the shape is "
                      "[N, 1].");

    PADDLE_ENFORCE_EQ(conf_dims[0], mi_dims[0],
                      "The batch_size of Input(Conf) and "
                      "Input(MatchIndices) must be the same.");

    PADDLE_ENFORCE_EQ(conf_dims[1], mi_dims[1],
                      "The prior_box_num of Input(Loc) and "
                      "Input(MatchIndices) must be the same.");
    PADDLE_ENFORCE_EQ(gt_dims[1], 1UL,
                      "The shape of Input(GTLabels) is [N, 1].");
    PADDLE_ENFORCE_EQ(neg_dims[1], 1UL,
                      "The shape of Input(NegIndices) is [Nneg, 1].");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<framework::Tensor>("Conf")->type()),
        ctx.device_context());
  }
};

class TargetConfidenceAssignOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  TargetConfidenceAssignOpMaker(OpProto* proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("Conf",
             "(Tensor, default Tensor<float>), The input confidence "
             "predictions.");
    AddInput(
        "GTLabels",
        "(LoDTensor, default LoDTensor<int>),  The input ground-truth labels.");
    AddInput("MatchIndices",
             "(LoDTensor, default LoDTensor<int>), The input matched indices, "
             "When it's equal to -1, it doesn't match any entity.");
    AddInput("NegIndices",
             "(LoDTensor, default LoDTensor<int>), The input negative example "
             "indics.");
    AddOutput("ConfGT",
              "(LoDTensor), The output ground-truth labels filtered by "
              "MatchIndices and append NegIndices examples.");
    AddOutput("ConfPred",
              "(LoDTensor), The output confidence predictions filtered by "
              "MatchIndices and append NegIndices examples.");
    AddAttr<int>("background_label_id",
                 "(int, default 0), Label id for background class.")
        .SetDefault(0);
    AddComment(R"DOC(
TargetConfidenceAssign operator

Filter ground-truth labels when the corresponding MatchIndices is not -1,
 and append negative examples with label background_label_id,
 it produces the output ConfGT.
 Filter confidence predictions when the corresponding MatchIndices is not -1,
 and append negative examples' confidence prediction.
 it produces the output ConfPred.

    )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(target_confidence_assign,
                             ops::TargetConfidenceAssignOp,
                             ops::TargetConfidenceAssignOpMaker);
REGISTER_OP_CPU_KERNEL(
    target_confidence_assign,
    ops::TargetConfidenceAssignOpKernel<paddle::platform::CPUDeviceContext,
                                        float>,
    ops::TargetConfidenceAssignOpKernel<paddle::platform::CPUDeviceContext,
                                        double>);
