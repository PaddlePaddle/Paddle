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

#include "paddle/fluid/operators/unpool_op.h"
#include <string>
#include <vector>
namespace paddle {
namespace operators {

class Unpool2dOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(
        "X",
        "(Tensor) The input tensor of unpool operator. "
        "The format of input tensor is NCHW. Where N is batch size, C is the "
        "number of channels, H and W is the height and width of feature.");
    AddInput(
        "Indices",
        "(Tensor) The input tensor of the indices given out by MaxPool2d. "
        "The format of input tensor is NCHW. Where N is batch size, C is the "
        "number of channels, H and W is the height and width of feature.");
    AddOutput("Out",
              "(Tensor) The output tensor of unpool operator."
              "The format of output tensor is also NCHW."
              "Where N is batch size, C is "
              "the number of channels, H and W is the height and "
              "width of feature.");
    AddAttr<std::vector<int>>(
        "ksize",
        "(vector), the unpooling window size(height, width) "
        "of unpooling operator.");
    AddAttr<std::vector<int>>("strides",
                              "(vector, default:{1, 1}), "
                              "strides (height, width) of unpooling operator.")
        .SetDefault({1, 1});
    AddAttr<std::vector<int>>("paddings",
                              "(vector default:{0,0}), "
                              "paddings (height, width) of unpooling operator.")
        .SetDefault({0, 0});
    AddAttr<std::string>(
        "unpooling_type",
        "(string), unpooling type, can be \"max\" for max-unpooling ")
        .InEnum({"max"});
    AddComment(R"DOC(
Input shape is: $(N, C_{in}, H_{in}, W_{in})$, Output shape is:
$(N, C_{out}, H_{out}, W_{out})$, where
$$
H_{out} = (H_{in}-1) * strides[0] - 2 * paddings[0] + ksize[0] \\
W_{out} = (W_{in}-1) * strides[1] - 2 * paddings[1] + ksize[1]
$$
Paper: http://www.matthewzeiler.com/wp-content/uploads/2017/07/iccv2011.pdf
)DOC");
  }
};

int UnpoolOutputSize(int input_size, int ksize, int padding, int stride) {
  int output_size = (input_size - 1) * stride - 2 * padding + ksize;
  return output_size;
}

class UnpoolOp : public framework::OperatorWithKernel {
 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(ctx.Input<framework::Tensor>("X")->type(),
                                   ctx.device_context());
  }

 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of UnpoolOp"
                   "should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Indices"),
                   "Input(Indices) of UnpoolOp"
                   "should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of UnpoolOp should not be null.");
    auto in_x_dims = ctx->GetInputDim("X");
    auto in_y_dims = ctx->GetInputDim("Indices");
    std::string unpooling_type =
        ctx->Attrs().Get<std::string>("unpooling_type");
    std::vector<int> ksize = ctx->Attrs().Get<std::vector<int>>("ksize");
    std::vector<int> strides = ctx->Attrs().Get<std::vector<int>>("strides");
    std::vector<int> paddings = ctx->Attrs().Get<std::vector<int>>("paddings");
    PADDLE_ENFORCE(in_x_dims.size() == 4,
                   "Unpooling intput must be of 4-dimensional.");
    PADDLE_ENFORCE_EQ(in_x_dims, in_y_dims);

    std::vector<int64_t> output_shape({in_x_dims[0], in_x_dims[1]});
    for (size_t i = 0; i < ksize.size(); ++i) {
      if (!ctx->IsRuntime() && in_x_dims[i + 2] <= 0) {
        output_shape.push_back(-1);
      } else {
        output_shape.push_back(UnpoolOutputSize(in_x_dims[i + 2], ksize[i],
                                                paddings[i], strides[i]));
      }
    }
    ctx->SetOutputDim("Out", framework::make_ddim(output_shape));
  }
};

class UnpoolOpGrad : public framework::OperatorWithKernel {
 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(ctx.Input<framework::Tensor>("X")->type(),
                                   ctx.device_context());
  }

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
REGISTER_OPERATOR(unpool, ops::UnpoolOp, ops::Unpool2dOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(unpool_grad, ops::UnpoolOpGrad);
REGISTER_OP_CPU_KERNEL(
    unpool, ops::UnpoolKernel<paddle::platform::CPUDeviceContext, float>,
    ops::UnpoolKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    unpool_grad,
    ops::UnpoolGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::UnpoolGradKernel<paddle::platform::CPUDeviceContext, double>);
