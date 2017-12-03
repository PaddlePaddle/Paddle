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
    AddAttr<int>("pyramid_height", ">= 1");
    AddComment(R"DOC(
        "Input shape: $(N, C_{in}, H_{in}, W_{in})$
        Output shape: $(H_{out}, W_{out})$
        Where
          $$
            H_{out} = (H_{in}−1) * strides[0] − 2 * paddings[0] + ksize[0] \\
            W_{out} = (W_{in}−1) * strides[1] − 2 * paddings[1] + ksize[1]
          $$
        )DOC");
  }
};

int OutputSize(int pyramid_level, int input_size) {
  int bins = std::pow(2, pyramid_level);
  int ksize = std::ceil(input_size / static_cast<double>(bins));
  int padding = (ksize * bins - input_size + 1) / 2;
  int output_size = (input_size - ksize + 2 * padding) / ksize + 1;
  // output_size = bins
  return output_size;
}

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
    int outlen = 0;
    for (int p = 0; p < pyramid_height; ++p) {
      int outh = OutputSize(p, in_x_dims[2]);
      int outw = OutputSize(p, in_x_dims[3]);
      int p_level_outlen = outh * outw * in_x_dims[1];
      outlen += p_level_outlen;
    }
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
