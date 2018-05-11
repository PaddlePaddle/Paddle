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
    PADDLE_ENFORCE(ctx->HasInput("predictions"),
                   "Input (predictions) of MeanIoU op should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("labels"),
                   "Input (labels) of MeanIoU op should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("out_mean_iou"),
                   "Output (out_mean_iou) of MeanIoU op should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("out_wrong"),
                   "Output (out_wrong) of MeanIoU op should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("out_correct"),
                   "Output (out_wrong) of MeanIoU op should not be null.");

    int64_t num_classes =
        static_cast<int64_t>(ctx->Attrs().Get<int>("num_classes"));

    ctx->SetOutputDim("out_mean_iou", {1});
    ctx->SetOutputDim("out_wrong", {num_classes});
    ctx->SetOutputDim("out_correct", {num_classes});
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<Tensor>("predictions")->type()),
        ctx.GetPlace());
  }
};

class MeanIoUOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  MeanIoUOpMaker(OpProto* proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("predictions",
             "A Tensor of prediction results for semantic labels"
             " with type int32 or int64.");
    AddInput("labels",
             "A Tensor of ground truth labels with type int32 or int64."
             "Its shape should be the same as Input(predictions).");
    AddInput("in_wrongs",
             "A list of Tensor with shape "
             "[num_classes]. They are used to collect wrong number among "
             "batches. Empty list is also valid here.")
        .AsDuplicable()
        .AsDispensable();
    AddInput(
        "in_corrects",
        "A list of Tensor with shape "
        "[num_classes]. They are used to collect correct number among batches. "
        "Empty list is also valid here.")
        .AsDuplicable()
        .AsDispensable();
    AddInput("in_mean_iou",
             "A list of Tensor that Output(mean_iou) should "
             "be added to. Empty list is also valid here.")
        .AsDuplicable()
        .AsDispensable();
    AddOutput("out_mean_iou",
              "A Tensor representing the"
              " mean intersection-over-union.");
    AddOutput("out_wrong", "A Tensor with shape [num_classes]. ");
    AddOutput("out_correct", "A Tensor with shape [num_classes]. ");
    AddAttr<int>("num_classes", "The possible number of labels.");

    AddComment(R"DOC(
mean-IOU Operator.
Mean Intersection-Over-Union is a common evaluation metric for semantic image segmentation, which first computes the IOU for each semantic class and then computes the average over classes. IOU is defined as follows: IOU = true_positive / (true_positive + false_positive + false_negative). The predictions are accumulated in a confusion matrix and mean-IOU is then calculated from it.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(mean_iou, ops::MeanIoUOp, ops::MeanIoUOpMaker,
                  paddle::framework::EmptyGradOpMaker);
REGISTER_OP_CPU_KERNEL(mean_iou, ops::MeanIoUKernel<int>,
                       ops::MeanIoUKernel<int64_t>);
