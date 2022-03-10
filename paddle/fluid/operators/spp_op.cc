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

#include "paddle/fluid/operators/spp_op.h"
#include <string>
#include <vector>
namespace paddle {
namespace operators {

class SppOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(
        "X",
        "(Tensor) The input tensor of spp operator. "
        "The format of input tensor is NCHW. Where N is batch size, C is the "
        "number of channels, H and W is the height and width of feature.");
    AddOutput("Out",
              "(Tensor) The output tensor of spp operator."
              "N * M."
              "M = C * H * W");
    AddAttr<int>("pyramid_height", "(int), multi level pooling");
    AddAttr<std::string>(
        "pooling_type",
        "(string), pooling type, can be \"max\" for max-pooling "
        "and \"avg\" for average-pooling.")
        .InEnum({"max", "avg"});
    AddComment(R"DOC(
        "With spatial pyramid pooling, the input image can
        be of any sizes. This not only allows arbitrary aspect
        ratios, but also allows arbitrary scales. We can resize
        the input image to any scale (e.g., min(w, h)=180, 224,
        ...) and apply the same deep network. When the
        input image is at different scales, the network (with
        the same filter sizes) will extract features at different
        scales. The scales play important roles in traditional
        methods.
        Input shape: $(N, C_{in}, H_{in}, W_{in})$
        Output shape: $(H_{out}, W_{out})$
        Where
          $$
            H_{out} = N \\
            W_{out} = (((4^pyramid_height) - 1) / (4 - 1))$ * C_{in}
          $$
        paper https://arxiv.org/pdf/1406.4729v4.pdf
        )DOC");
  }
};

class SppOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                      platform::errors::InvalidArgument(
                          "Input(X) of SppOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      platform::errors::InvalidArgument(
                          "Output(Out) of SppOp should not be null."));
    auto in_x_dims = ctx->GetInputDim("X");
    int pyramid_height = ctx->Attrs().Get<int>("pyramid_height");
    PADDLE_ENFORCE_EQ(in_x_dims.size(), 4,
                      platform::errors::InvalidArgument(
                          "Spping intput must be of 4-dimensional."));
    int outlen = ((std::pow(4, pyramid_height) - 1) / (4 - 1)) * in_x_dims[1];
    std::vector<int64_t> output_shape({in_x_dims[0], outlen});
    ctx->SetOutputDim("Out", phi::make_ddim(output_shape));
  }
};

class SppOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("X"), true,
        platform::errors::InvalidArgument("Input(X) must not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput(framework::GradVarName("X")), true,
        platform::errors::InvalidArgument("Input(X@GRAD) should not be null."));
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    spp, ops::SppOp, ops::SppOpMaker,
    paddle::framework::DefaultGradOpMaker<paddle::framework::OpDesc, true>,
    paddle::framework::DefaultGradOpMaker<paddle::imperative::OpBase, true>);
REGISTER_OPERATOR(spp_grad, ops::SppOpGrad);
REGISTER_OP_CPU_KERNEL(
    spp, ops::SppKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SppKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    spp_grad, ops::SppGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SppGradKernel<paddle::platform::CPUDeviceContext, double>);
