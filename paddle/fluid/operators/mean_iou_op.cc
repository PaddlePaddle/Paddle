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

#include "paddle/fluid/operators/mean_iou_op.h"

namespace paddle {
namespace operators {

class MeanIoUOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(
        ctx->HasInput("Predictions"), "Input", "Predictions", "MeanIoU");
    OP_INOUT_CHECK(ctx->HasInput("Labels"), "Input", "Labels", "MeanIoU");
    OP_INOUT_CHECK(
        ctx->HasOutput("OutMeanIou"), "Output", "OutMeanIou", "MeanIoU");
    OP_INOUT_CHECK(ctx->HasOutput("OutWrong"), "Output", "OutWrong", "MeanIoU");
    OP_INOUT_CHECK(
        ctx->HasOutput("OutCorrect"), "Output", "OutCorrect", "MeanIoU");

    int64_t num_classes =
        static_cast<int64_t>(ctx->Attrs().Get<int>("num_classes"));

    ctx->SetOutputDim("OutMeanIou", {1});
    ctx->SetOutputDim("OutWrong", {num_classes});
    ctx->SetOutputDim("OutCorrect", {num_classes});
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Predictions"),
        ctx.GetPlace());
  }
};

class MeanIoUOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Predictions",
             "(Tensor), A Tensor of prediction results for semantic labels"
             " with type int32 or int64. The rank should be greater than 1.");
    AddInput(
        "Labels",
        "(Tensor), A Tensor of ground truth labels with type int32 or int64."
        "Its shape should be the same as Input(Predictions).");
    AddInput("InWrongs",
             "(vector<Tensor>), A list of Tensor with shape "
             "[num_classes]. They are used to collect wrong number among "
             "batches. Empty list is also valid here.")
        .AsDuplicable()
        .AsDispensable();
    AddInput(
        "InCorrects",
        "(vector<Tensor>), A list of Tensor with shape "
        "[num_classes]. They are used to collect correct number among batches. "
        "Empty list is also valid here.")
        .AsDuplicable()
        .AsDispensable();
    AddInput("InMeanIou",
             "(vector<Tensor>), A list of Tensor that Output(mean_iou) should "
             "be added to. Empty list is also valid here.")
        .AsDuplicable()
        .AsDispensable();
    AddOutput("OutMeanIou",
              "(vector<Tensor>), A Tensor representing the"
              " mean intersection-over-union with shape [1].");
    AddOutput("OutWrong", "(Tensor), A Tensor with shape [num_classes]. ");
    AddOutput("OutCorrect", "(Tensor), A Tensor with shape [num_classes]. ");
    AddAttr<int>("num_classes", "(int), The possible number of labels.");

    AddComment(R"DOC(
mean-IOU Operator.
Mean Intersection-Over-Union is a common evaluation metric for
semantic image segmentation, which first computes the IOU for each
semantic class and then computes the average over classes.
IOU is defined as follows:
    IOU = true_positive / (true_positive + false_positive + false_negative).
It is based on pixel level area while "IOU Similarity Operator"
is based on area of rectangle.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    mean_iou,
    ops::MeanIoUOp,
    ops::MeanIoUOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(mean_iou,
                       ops::MeanIoUKernel<int>,
                       ops::MeanIoUKernel<int32_t>,
                       ops::MeanIoUKernel<int64_t>);
