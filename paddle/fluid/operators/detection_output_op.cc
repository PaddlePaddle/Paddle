/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
Indicesou may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/detection_output_op.h"
namespace paddle {
namespace operators {

class DetectionOutputOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  DetectionOutputOpMaker(OpProto* proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("Loc",
             "(Tensor) The input tensor of detection_output operator."
             "The input predict locations"
             "The format of input tensor is kNCHW. Where K is priorbox point "
             "numbers,"
             "N is How many boxes are there on each point, "
             "C is 4, H and W both are 1.");
    AddInput("Conf",
             "(Tensor) The input tensor of detection_output operator."
             "The input priorbox confidence."
             "The format of input tensor is kNCHW. Where K is priorbox point "
             "numbers,"
             "N is How many boxes are there on each point, "
             "C is the number of classes, H and W both are 1.");
    AddInput("PriorBox",
             "(Tensor) The input tensor of detection_output operator."
             "The format of input tensor is the position and variance "
             "of the boxes");
    AddOutput("Out",
              "(Tensor) The output tensor of detection_output operator.");
    AddAttr<int>("background_label_id", "(int), The background class index.");
    AddAttr<int>("num_classes", "(int), The number of the classification.");
    AddAttr<float>("nms_threshold",
                   "(float), The Non-maximum suppression threshold.");
    AddAttr<float>("confidence_threshold",
                   "(float), The classification confidence threshold.");
    AddAttr<int>("top_k", "(int), The bbox number kept of the layer’s output.");
    AddAttr<int>("nms_top_k",
                 "(int), The bbox number kept of the NMS’s output.");
    AddComment(R"DOC(
          detection output for SSD(single shot multibox detector)
          Apply the NMS to the output of network and compute the predict
          bounding box location. The output’s shape of this layer could
          be zero if there is no valid bounding box.
        )DOC");
  }
};

class DetectionOutputOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Loc"),
                   "Input(X) of DetectionOutputOp"
                   "should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Conf"),
                   "Input(X) of DetectionOutputOp"
                   "should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("PriorBox"),
                   "Input(X) of DetectionOutputOp"
                   "should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of DetectionOutputOp should not be null.");
    std::vector<int64_t> output_shape({1, 7});
    ctx->SetOutputDim("Out", framework::make_ddim(output_shape));
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(detection_output, ops::DetectionOutputOp,
                             ops::DetectionOutputOpMaker);
REGISTER_OP_CPU_KERNEL(
    detection_output,
    ops::DetectionOutputKernel<paddle::platform::CPUDeviceContext, float>,
    ops::DetectionOutputKernel<paddle::platform::CPUDeviceContext, double>);
