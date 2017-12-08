/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
Indicesou may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/operators/detection_output_op.h"
namespace paddle {
namespace operators {

class Detection_output_OpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  Detection_output_OpMaker(framework::OpProto* proto,
                           framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput(
        "Loc",
        "(Tensor) The input tensor of detection_output operator. "
        "The format of input tensor is NCHW. Where N is batch size, C is the "
        "number of channels, H and W is the height and width of feature.");
    AddInput(
        "Conf",
        "(Tensor) The input tensor of detection_output operator. "
        "The format of input tensor is NCHW. Where N is batch size, C is the "
        "number of channels, H and W is the height and width of feature.");
    AddInput(
        "PriorBox",
        "(Tensor) The input tensor of detection_output operator. "
        "The format of input tensor is NCHW. Where N is batch size, C is the "
        "number of channels, H and W is the height and width of feature.");
    AddOutput("Out",
              "(Tensor) The output tensor of detection_output operator."
              "N * M."
              "M = C * H * W");
    AddAttr<int>("background_label_id", "(int), multi level pooling");
    AddAttr<int>("num_classes", "(int), multi level pooling");
    AddAttr<float>("nms_threshold", "(int), multi level pooling");
    AddAttr<float>("confidence_threshold", "(int), multi level pooling");
    AddAttr<int>("top_k", "(int), multi level pooling");
    AddAttr<int>("nms_top_k", "(int), multi level pooling");
    AddComment(R"DOC(
        "Does spatial pyramid pooling on the input image by taking the max,
        etc. within regions so that the result vector of different sized
        images are of the same size
        Input shape: $(N, C_{in}, H_{in}, W_{in})$
        Output shape: $(H_{out}, W_{out})$
        Where
          $$
            H_{out} = N \\
            W_{out} = (((4^pyramid_height) - 1) / (4 - 1))$ * C_{in}
          $$
        )DOC");
  }
};

class Detection_output_Op : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of Detection_output_Op"
                   "should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of Detection_output_Op should not be null.");
    auto in_x_dims = ctx->GetInputDim("X");
    int pyramid_height = ctx->Attrs().Get<int>("pyramid_height");
    PADDLE_ENFORCE(in_x_dims.size() == 4,
                   "Detection_output_ing intput must be of 4-dimensional.");
    int outlen = ((std::pow(4, pyramid_height) - 1) / (4 - 1)) * in_x_dims[1];
    std::vector<int64_t> output_shape({in_x_dims[0], outlen});
    ctx->SetOutputDim("Out", framework::make_ddim(output_shape));
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(detection_output, ops::Detection_output_Op,
                             ops::Detection_output_OpMaker);
REGISTER_OP_CPU_KERNEL(
    detection_output,
    ops::Detection_output_Kernel<paddle::platform::CPUPlace, float>,
    ops::Detection_output_Kernel<paddle::platform::CPUPlace, double>);
