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

#include "paddle/operators/multi_box_loss_op.h"

namespace paddle {
namespace operators {

class MultiBoxLossOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInputs("Loc"),
                   "Inputs(Loc) of MultiBoxLossOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInputs("Conf"),
                   "Inputs(Conf) of MultiBoxLossOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Label"),
                   "Input(Label) of MultiBoxLossOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Loss"),
                   "Output(Loss) of MultiBoxLossOp should not be null.");

    PADDLE_ENFORCE_EQ(ctx->Inputs("Loc").size(), ctx->Inputs("Conf").size(),
                      "The input number of Loc and Conf should be the same.");

    int input_num = ctx->Inputs("Loc").size();
    auto loc_dims = ctx->GetInputsDim("Loc");
    auto conf_dims = ctx->GetInputsDim("Conf");
    for (int i = 0; i < input_num; ++i) {
      PADDLE_ENFORCE_EQ(loc_dims[i].size(), 4UL,
                        "The format of input(loc %d) tensor is NCHW.", i);
      PADDLE_ENFORCE_EQ(conf_dims[i].size(), 4UL,
                        "The format of input(conf %d) tensor is NCHW.", i);
    }
    PADDLE_ENFORCE_EQ(ctx->GetInputDim("Label").size(), 2UL,
                      "The dim size of input(label) tensor is 2.");

    auto loss_dims = framework::make_ddim({1});
    ctx->SetOutputDim("Loss", loss_dims);
    auto couter_dims = framework::make_ddim({3});
    ctx->SetOutputDim("InterCounter", couter_dims);
  }

 protected:
  framework::OpKernelType GetKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<framework::Tensor>("Label")->type()),
        ctx.device_context());
  }
};

class MultiBoxLossGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    ctx->SetOutputsDim(framework::GradVarName("Loc"), ctx->GetInputsDim("Loc"));
    ctx->SetOutputsDim(framework::GradVarName("Conf"),
                       ctx->GetInputsDim("Conf"));
  }

 protected:
  framework::OpKernelType GetKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<framework::Tensor>("Label")->type()),
        ctx.device_context());
  }
};

class MultiBoxLossOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  MultiBoxLossOpMaker(framework::OpProto* proto,
                      framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("Loc", "The input predict locations.").AsDuplicable();
    AddInput("Conf", "The input priorbox confidence..").AsDuplicable();
    AddInput("PriorBox", "The input priorbox location.");
    AddInput("Label", "The input label.");
    AddOutput("Loss", "The output loss.");
    AddOutput("InterCounter", "Internal use counter.").AsIntermediate();
    AddOutput("AllMatchIndices", "All match indices, internal use only.")
        .AsIntermediate();
    AddOutput("AllNegIndices", "All negative indices, internal use only.")
        .AsIntermediate();
    AddOutput("LocGTData", "Locations ground truth data, internal use only.")
        .AsIntermediate();
    AddOutput("ConfGTData", "Confidence ground truth data, internal use only.")
        .AsIntermediate();
    AddOutput("LocDiff", "Locations difference data, internal use only.")
        .AsIntermediate();
    AddOutput("ConfProb", "Confidence possibility data, internal use only.")
        .AsIntermediate();
    AddAttr<int>("class_num", "The number of the classification.")
        .SetDefault(0);
    AddAttr<float>("overlap_threshold", "The threshold of the overlap.")
        .SetDefault(0.5);
    AddAttr<float>("neg_overlap", "The negative bbox overlap threshold.")
        .SetDefault(0.5);
    AddAttr<float>("neg_pos_ratio",
                   "The ratio of the negative bbox to the positive bbox.")
        .SetDefault(3);
    AddAttr<int>("background_label_id", "The background class index.")
        .SetDefault(-1);
    // AddOutput("MatchNum",
    //           "(Tensor), ")
    //     .AsIntermediate();
    AddComment(R"DOC(
MultiBoxLoss operator
Compute the location loss and the confidence loss for ssd.
Please get more information from the following papers:
https://arxiv.org/abs/1512.02325.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(multi_box_loss, ops::MultiBoxLossOp, ops::MultiBoxLossOpMaker,
            multi_box_loss_grad, ops::MultiBoxLossGradOp);
REGISTER_OP_CPU_KERNEL(
    multi_box_loss,
    ops::MultiBoxLossOpKernel<paddle::platform::CPUDeviceContext, float>);
REGISTER_OP_CPU_KERNEL(
    multi_box_loss_grad,
    ops::MultiBoxLossGradOpKernel<paddle::platform::CPUDeviceContext, float>);
