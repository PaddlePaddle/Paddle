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

#include "paddle/operators/spp_op.h"
namespace paddle {
namespace operators {

class SppOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SppOpMaker(framework::OpProto* proto, framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput(
        "X",
        "(Tensor) The input tensor of spp operator. "
        "The format of input tensor is NCHW. Where N is batch size, C is the "
        "number of channels, H and W is the height and width of feature.");
    AddOutput("Out",
              "(Tensor) The output tensor of spp operator."
              "N * M."
              "M = C * H * W");
    AddAttr<int>("pyramid_height", "int");
    AddComment(R"DOC(
        "Does spatial pyramid pooling on the input image by taking the max,
        etc. within regions so that the result vector of different sized
        images are of the same size
        Input shape: $(N, C_{in}, H_{in}, W_{in})$
        Output shape: $(H_{out}, W_{out})$
        Where
          $$
            H_{out} = N \\
            W_{out} = ((std::pow(4, pyramid_height) - 1) / (4 - 1)) * C_{in}
          $$
        )DOC");
  }
};

class SppOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of SppOp"
                   "should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of SppOp should not be null.");
    auto in_x_dims = ctx->GetInputDim("X");
    int pyramid_height = ctx->Attrs().Get<int>("pyramid_height");
    PADDLE_ENFORCE(in_x_dims.size() == 4,
                   "Spping intput must be of 4-dimensional.");
    int outlen = ((std::pow(4, pyramid_height) - 1) / (4 - 1)) * in_x_dims[1];
    std::vector<int64_t> output_shape({in_x_dims[0], outlen});
    ctx->SetOutputDim("Out", framework::make_ddim(output_shape));
  }
};

class SppOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) must not be null.");
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("X")),
                   "Input(X@GRAD) should not be null.");
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(spp, ops::SppOp, ops::SppOpMaker, spp_grad, ops::SppOpGrad);
REGISTER_OP_CPU_KERNEL(spp, ops::SppKernel<paddle::platform::CPUPlace, float>,
                       ops::SppKernel<paddle::platform::CPUPlace, double>);
REGISTER_OP_CPU_KERNEL(spp_grad,
                       ops::SppGradKernel<paddle::platform::CPUPlace, float>,
                       ops::SppGradKernel<paddle::platform::CPUPlace, double>);
