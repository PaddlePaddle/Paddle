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

namespace paddle {
namespace operators {

class FCOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Input"),
                   "X(Input) of Fully Connected should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Out(Output) of Fully Connected should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("W"),
                   "W(Input) of Fully Connected should not be null.");

    auto in_dims = ctx->GetInputDim("Input");
    auto w_dims = ctx->GetInputDim("W");

    if (ctx->HasInput("Bias")) {
      auto bias_dims = ctx->GetInputDim("Bias");
      if (bias_dims.size() == 2) {
        PADDLE_ENFORCE_EQ(bias_dims[0], 1,
                          "The shape of Bias must be [1, dim].");
        PADDLE_ENFORCE_EQ(bias_dims[1], w_dims[1],
                          "The shape of Bias must be [1, dim].");
      } else if (bias_dims.size() == 1) {
        PADDLE_ENFORCE_EQ(bias_dims[0], w_dims[1],
                          "The shape of Bias must be [1, dim].");
      }
    }

    if (ctx->Attrs().Get<bool>("use_mkldnn")) {
      PADDLE_ENFORCE(in_dims.size() == 2 || in_dims.size() == 4,
                     "Fully Connected input should be 2-D or 4-D tensor.");
    }
    PADDLE_ENFORCE_EQ(w_dims.size(), 2,
                      "Fully Connected input should be 2-D tensor.");
    int in_num_col_dims = ctx->Attrs().Get<int>("in_num_col_dims");
    PADDLE_ENFORCE_GT(
        in_dims.size(), in_num_col_dims,
        "The input tensor Input's rank of FCOp should be larger than "
        "in_num_col_dims.");

    std::vector<int64_t> output_dims;
    FCOutputSize(in_dims, w_dims, output_dims, in_num_col_dims);

    ctx->SetOutputDim("Out", framework::make_ddim(output_dims));
    ctx->ShareLoD("Input", "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    framework::LibraryType library = framework::LibraryType::kPlain;
    framework::DataLayout layout = framework::DataLayout::kAnyLayout;
    if (ctx.Attr<bool>("use_mkldnn")) {
      library = framework::LibraryType::kMKLDNN;
      layout = framework::DataLayout::kMKLDNN;
    }
    return framework::OpKernelType(ctx.Input<Tensor>("Input")->type(),
                                   ctx.GetPlace(), layout, library);
  }
};

class FCOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
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

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    framework::LibraryType library = framework::LibraryType::kPlain;
    framework::DataLayout layout = framework::DataLayout::kAnyLayout;
    if (ctx.Attr<bool>("use_mkldnn")) {
      library = framework::LibraryType::kMKLDNN;
      layout = framework::DataLayout::kMKLDNN;
    }
    return framework::OpKernelType(ctx.Input<Tensor>("Input")->type(),
                                   ctx.GetPlace(), layout, library);
  }
};

class FCOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input",
             "(Tensor), The input tensor of fully connected operator.");
    AddInput("W", "(Tensor), The weight fc op with shape (I, O).");
    AddInput("Bias", "(Tensor, optional) Bias vector with shape (1 x O")
        .AsDispensable();
    AddAttr<int>("in_num_col_dims",
                 "(int, default 1), The fc op can take tensors with more than "
                 "two dimensions as its inputs.")
        .SetDefault(1)
        .EqualGreaterThan(1);
    AddOutput("Out",
              "(Tensor) The output tensor of fully connected operator. ");
    AddAttr<bool>("use_mkldnn",
                  "(bool, default false) Only used in mkldnn kernel")
        .SetDefault(false);
    AddAttr<bool>(framework::kAllKernelsMustComputeRuntimeShape,
                  "Skip calling InferShape() function in the runtime.")
        .SetDefault(true);
    AddComment(R"DOC(
Fully Connected Operator.

The fully connected operation calculates the output based on the input, weights and bias.
The size of each dimension of the parameters checked in the infer-shape.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(fc, ops::FCOp, ops::FCOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(fc_grad, ops::FCOpGrad);
REGISTER_OP_CPU_KERNEL(
    fc, ops::FCOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::FCOpKernel<paddle::platform::CPUDeviceContext, double>);
