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

#include "paddle/fluid/operators/fc_op.h"
#include <vector>
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/fc_compute.h"

namespace paddle {
namespace operators {

void FCOp::InferShape(framework::InferShapeContext* ctx) const {
  PADDLE_ENFORCE(ctx->HasInput("Input"),
                 "X(Input) of Fully Connected should not be null.");
  PADDLE_ENFORCE(ctx->HasOutput("Out"),
                 "Out(Output) of Fully Connected should not be null.");
  PADDLE_ENFORCE(ctx->HasInput("W"),
                 "W(Input) of Fully Connected should not be null.");
  // NCHW
  auto in_dims = ctx->GetInputDim("Input");
  // IO, I=C*H*W
  auto w_dims = ctx->GetInputDim("W");
  std::vector<int64_t> output_shape({in_dims[0], w_dims[1]});

  if (ctx->HasInput("Bias")) {
    auto bias_dims = ctx->GetInputDim("Bias");
    if (bias_dims.size() == 2) {
      PADDLE_ENFORCE_EQ(bias_dims[0], 1, "The shape of Bias must be [1, dim].");
      PADDLE_ENFORCE_EQ(bias_dims[1], w_dims[1],
                        "The shape of Bias must be [1, dim].");
    } else if (bias_dims.size() == 1) {
      PADDLE_ENFORCE_EQ(bias_dims[0], w_dims[1],
                        "The shape of Bias must be [1, dim].");
    }
  }
  PADDLE_ENFORCE(in_dims.size() == 2 || in_dims.size() == 4,
                 "Fully Connected input should be 2-D or 4-D tensor.");
  PADDLE_ENFORCE_EQ(w_dims.size(), 2UL,
                    "Fully Connected input should be 2-D tensor.");
  PADDLE_ENFORCE_EQ(framework::product(in_dims) / in_dims[0], w_dims[0],
                    "Fully Connected input and weigth size do not match.");

  ctx->SetOutputDim("Out", framework::make_ddim(output_shape));
}

framework::OpKernelType FCOp::GetExpectedKernelType(
    const framework::ExecutionContext& ctx) const {
  framework::LibraryType library = framework::LibraryType::kPlain;
  framework::DataLayout layout = framework::DataLayout::kAnyLayout;
  if (ctx.Attr<bool>("use_mkldnn")) {
    library = framework::LibraryType::kMKLDNN;
    layout = framework::DataLayout::kMKLDNN;
  }
  return framework::OpKernelType(
      framework::ToDataType(ctx.Input<Tensor>("Input")->type()), ctx.GetPlace(),
      layout, library);
}

void FCOpGrad::InferShape(framework::InferShapeContext* ctx) const {
  auto in_dims = ctx->GetInputDim("Input");
  auto w_dims = ctx->GetInputDim("W");

  if (ctx->HasOutput(framework::GradVarName("Input"))) {
    ctx->SetOutputDim(framework::GradVarName("Input"), in_dims);
  }
  if (ctx->HasOutput(framework::GradVarName("W"))) {
    ctx->SetOutputDim(framework::GradVarName("W"), w_dims);
  }

  if (ctx->HasInput("Bias")) {
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("Bias")),
                   "Should have bias grad");
    auto bias_dims = ctx->GetInputDim("Bias");
    ctx->SetOutputDim(framework::GradVarName("Bias"), bias_dims);
  }
}

framework::OpKernelType FCOpGrad::GetExpectedKernelType(
    const framework::ExecutionContext& ctx) const {
  framework::LibraryType library = framework::LibraryType::kPlain;
  framework::DataLayout layout = framework::DataLayout::kAnyLayout;
  if (ctx.Attr<bool>("use_mkldnn")) {
    library = framework::LibraryType::kMKLDNN;
    layout = framework::DataLayout::kMKLDNN;
  }
  return framework::OpKernelType(
      framework::ToDataType(ctx.Input<Tensor>("Input")->type()), ctx.GetPlace(),
      layout, library);
}

void FCOpMaker::Make() {
  AddInput("Input",
           "(Tensor), The input tensor of fully connected operator with format "
           "(NCHW). ");
  AddInput("W", "(Tensor), The weight fc op with shape (I, O).");
  AddInput("Bias", "(Tensor, optional) Bias vector with shape (1 x O")
      .AsDispensable();
  AddOutput("Out", "(Tensor) The output tensor of fully connected operator. ");
  AddAttr<bool>("use_mkldnn",
                "(bool, default false) Only used in mkldnn kernel")
      .SetDefault(false);
  AddComment(R"DOC(
  Fully Connected Operator.

  The fully connected operation calculates the output based on the input, weights and bias.
  The size of each dimension of the parameters checked in the infer-shape.
)DOC");
}

template <typename T>
class FCOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");
    auto input = ctx.Input<Tensor>("Input");
    auto w = ctx.Input<Tensor>("W");
    auto bias = ctx.Input<Tensor>("Bias");
    auto output = ctx.Output<Tensor>("Out");
    auto in_dims = input->dims();
    auto w_dims = w->dims();

    const T* input_data = input->data<T>();
    const T* w_data = w->data<T>();
    T* output_data = output->mutable_data<T>(ctx.GetPlace());
    auto blas = math::GetBlas<platform::CPUDeviceContext, T>(ctx);
    math::FCCompute<platform::CPUDeviceContext, T>(
        blas, in_dims[0], w_dims[1], w_dims[0], input_data, w_data, output_data,
        bias ? bias->data<T>() : NULL);

    // TODO(TJ): fuse act
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(fc, ops::FCOp, ops::FCOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(fc_grad, ops::FCOpGrad);
REGISTER_OP_CPU_KERNEL(fc, ops::FCOpKernel<float>, ops::FCOpKernel<double>);
