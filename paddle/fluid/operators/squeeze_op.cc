/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/squeeze_op.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

class SqueezeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "Squeeze");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "Squeeze");

    const auto &x_dims = ctx->GetInputDim("X");
    // Check input tensor dims (<6) Eigen limit.
    PADDLE_ENFORCE_LE(x_dims.size(),
                      6,
                      platform::errors::InvalidArgument(
                          "The dimensions of Input(X) "
                          "should be in the range of [1, 6] (Eigen limit)."
                          "But received X's dimensions = %d, X's shape=[%s].",
                          x_dims.size(),
                          x_dims));

    const auto &axes = ctx->Attrs().Get<std::vector<int>>("axes");
    auto out_dims = GetOutputShape(axes, x_dims, false);
    ctx->SetOutputDim("Out", out_dims);
    if (x_dims[0] == out_dims[0]) {
      // Only pass LoD when the first dimension of output and Input(X)
      // are the same.
      ctx->ShareLoD("X", "Out");
    }
  }

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type =
        framework::OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return phi::KernelKey(input_data_type, ctx.GetPlace());
  }
};

class SqueezeGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *context) const override {
    context->SetOutputDim(framework::GradVarName("X"),
                          context->GetInputDim("X"));
    context->ShareLoD("X", framework::GradVarName("X"));
  }

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type = framework::OperatorWithKernel::IndicateVarDataType(
        ctx, framework::GradVarName("Out"));
    return phi::KernelKey(input_data_type, ctx.GetPlace());
  }
};

class SqueezeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor). The input tensor of squeeze operator.");
    AddOutput("Out", "(Tensor). The output tensor of squeeze operator.");
    AddAttr<std::vector<int>>("axes",
                              "(std::vector<int>). List of integers,"
                              " indicating the dimensions to squeeze.")
        .SetDefault({})
        .SupportTensor();
    AddAttr<bool>("use_mkldnn",
                  "(bool, default false) Only used in mkldnn kernel")
        .SetDefault(false)
        .AsExtra();
    AddAttr<std::string>(
        "mkldnn_data_type",
        "(string, default \"float32\"). Data type of mkldnn kernel")
        .SetDefault("float32")
        .InEnum({"float32", "bfloat16"})
        .AsExtra();
    AddComment(R"DOC(
        Squeeze Operator.

        Remove single-dimensional entries from the shape of a tensor.
        Takes a parameter axes with a list of axes to squeeze.
        If axes is not provided, all the single dimensions will be removed from the shape.
        If an axis is selected with shape entry not equal to one, an error is raised.

        Examples:
        Case 1:
          Given
            X.shape = (1, 3, 1, 5)
          and
            axes = [0]
          we get:
            Out.shape = (3, 1, 5)

        Case 2:
          Given
            X.shape = (1, 3, 1, 5)
          and
            axes = []
          we get:
            Out.shape = (3, 5)
    )DOC");
  }
};

template <typename T>
class SqueezeGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("squeeze_grad");
    grad_op->SetInput("X", this->Input("X"));
    grad_op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    grad_op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    grad_op->SetAttrMap(this->Attrs());
  }
};

template <typename T>
class SqueezeDoubleGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("squeeze");
    grad_op->SetInput("X", this->OutputGrad(framework::GradVarName("X")));
    grad_op->SetOutput("Out", this->InputGrad(framework::GradVarName("Out")));
    grad_op->SetAttrMap(this->Attrs());
  }
};

DECLARE_INPLACE_OP_INFERER(SqueezeInplaceInferer, {"X", "Out"});
DECLARE_INPLACE_OP_INFERER(SqueezeGradInplaceInferer,
                           {framework::GradVarName("Out"),
                            framework::GradVarName("X")});
DECLARE_NO_NEED_BUFFER_VARS_INFERER(SqueezeGradNoNeedBufferVarsInferer, "X");
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(squeezeaa,
                  ops::SqueezeOp,
                  ops::SqueezeOpMaker,
                  ops::SqueezeGradOpMaker<paddle::framework::OpDesc>,
                  ops::SqueezeGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(squeeze_gradaa,
                  ops::SqueezeGradOp,
                  ops::SqueezeDoubleGradOpMaker<paddle::framework::OpDesc>,
                  ops::SqueezeDoubleGradOpMaker<paddle::imperative::OpBase>,
                  ops::SqueezeGradNoNeedBufferVarsInferer);

REGISTER_OP_CPU_KERNEL(
    squeezeaa,
    ops::SqueezeKernel<phi::CPUContext, float>,
    ops::SqueezeKernel<phi::CPUContext, double>,
    ops::SqueezeKernel<phi::CPUContext, bool>,
    ops::SqueezeKernel<phi::CPUContext, int>,
    ops::SqueezeKernel<phi::CPUContext, uint8_t>,
    ops::SqueezeKernel<phi::CPUContext, int8_t>,
    ops::SqueezeKernel<phi::CPUContext, int64_t>,
    ops::SqueezeKernel<phi::CPUContext, paddle::platform::complex<float>>,
    ops::SqueezeKernel<phi::CPUContext, paddle::platform::complex<double>>,
    ops::SqueezeKernel<phi::CPUContext, paddle::platform::bfloat16>);
REGISTER_OP_CPU_KERNEL(
    squeeze_gradaa,
    ops::SqueezeGradKernel<phi::CPUContext, float>,
    ops::SqueezeGradKernel<phi::CPUContext, double>,
    ops::SqueezeGradKernel<phi::CPUContext, bool>,
    ops::SqueezeGradKernel<phi::CPUContext, int>,
    ops::SqueezeGradKernel<phi::CPUContext, uint8_t>,
    ops::SqueezeGradKernel<phi::CPUContext, int8_t>,
    ops::SqueezeGradKernel<phi::CPUContext, int64_t>,
    ops::SqueezeGradKernel<phi::CPUContext, paddle::platform::complex<float>>,
    ops::SqueezeGradKernel<phi::CPUContext, paddle::platform::complex<double>>,
    ops::SqueezeGradKernel<phi::CPUContext, paddle::platform::bfloat16>);
