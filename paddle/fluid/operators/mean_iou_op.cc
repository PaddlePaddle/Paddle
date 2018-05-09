/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

template <typename T>
struct GenConfusionMatrix<paddle::platform::CPUDeviceContext, T> {
  void operator()(const framework::ExecutionContext& ctx,
                  const int64_t num_classes, const int64_t count,
                  const T* predictions, const T* labels, float* out_cm) {
    int index;
    for (int i = 0; i < count; ++i) {
      index = predictions[i] * num_classes + labels[i];
      out_cm[index] += 1.0f;
    }
  }
};

template <typename T>
struct Replace<paddle::platform::CPUDeviceContext, T> {
  void operator()(const framework::ExecutionContext& ctx, const int64_t n,
                  T* data, T target, T value) {
    for (int i = 0; i < n; ++i) {
      if (data[i] == target) {
        data[i] = value;
      }
    }
  }
};

template <typename T>
struct Diagonal<paddle::platform::CPUDeviceContext, T> {
  void operator()(const framework::ExecutionContext& ctx, int64_t n, T* matrix,
                  T* diagonal) {
    int64_t stride = n + 1;
    int64_t index = 0;
    for (int64_t i = 0; i < n; ++i) {
      diagonal[i] = matrix[index];
      index += stride;
    }
  }
};

class MeanIoUOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("predictions"),
                   "Input (predictions) of accuracy op should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("labels"),
                   "Input (labels) of accuracy op should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("mean_iou"),
                   "Output (mean_iou) of AccuracyOp should not be null.");
    PADDLE_ENFORCE(
        ctx->HasOutput("out_confusion_matrix"),
        "Output (out_confusion_matrix) of AccuracyOp should not be null.");

    int64_t num_classes =
        static_cast<int64_t>(ctx->Attrs().Get<int>("num_classes"));

    ctx->SetOutputDim("mean_iou", {1});
    ctx->SetOutputDim("out_confusion_matrix", {num_classes, num_classes});
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
    AddInput("in_confusion_matrix",
             "A list of Tensor with shape "
             "[num_classes, num_classes]. They are used to collect confusion"
             "matrix values among batches. Empty list is also valid here.")
        .AsDuplicable()
        .AsDispensable();
    AddInput("in_mean_iou",
             "A list of Tensor that Output(mean_iou) should "
             "be added to. Empty list is also valid here.")
        .AsDuplicable()
        .AsDispensable();
    AddOutput("mean_iou",
              "A Tensor representing the"
              " mean intersection-over-union.");
    AddOutput(
        "out_confusion_matrix",
        "A Tensor with shape "
        "[num_classes, num_classes]. Input(in_confusion_matrix) will be"
        " updated by current batch data and output as out_confusion_matrix.");
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
REGISTER_OP_CPU_KERNEL(
    mean_iou, ops::MeanIoUKernel<paddle::platform::CPUDeviceContext, int>,
    ops::MeanIoUKernel<paddle::platform::CPUDeviceContext, int64_t>);
