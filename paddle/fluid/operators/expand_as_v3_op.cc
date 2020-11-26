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

#include "paddle/fluid/operators/expand_as_v3_op.h"
#include <memory>
#include <vector>

namespace paddle {
namespace operators {

using framework::Tensor;

class ExpandAsV3Op : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "ExpandAsV3");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "ExpandAsV3");
    auto x_dims = ctx->GetInputDim("X");
    auto target_shape = ctx->Attrs().Get<std::vector<int>>("target_shape");
    PADDLE_ENFORCE_GE(
        target_shape.size(), static_cast<size_t>(x_dims.size()),
        platform::errors::InvalidArgument(
            "The rank of target_shape must be greater than or equal "
            "to the rank of Input(X). But received Input(X): input "
            "rank %u; received target_shape: rank %u.",
            x_dims.size(), target_shape.size()));
    PADDLE_ENFORCE_LE(target_shape.size(), MAX_RANK_SUPPORTED,
                      platform::errors::InvalidArgument(
                          "The rank of target_shape must be less than or equal "
                          "to %d. But received: rank %u.",
                          MAX_RANK_SUPPORTED, target_shape.size()));
    ctx->SetOutputDim("Out", framework::make_ddim(target_shape));
  }
};

class ExpandAsV3OpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor, default Tensor<float>). A tensor with rank in [1, 6]."
             "X is the input to be expanded.");
    AddOutput("Out",
              "(Tensor, default Tensor<float>). A tensor with rank in [1, 6]."
              "The rank of Output(Out) have the same with Input(X). "
              "After expanding, size of each dimension of Output(Out) is equal "
              "to size of the corresponding dimension of Input(X) multiplying "
              "the corresponding value given by Attr(expand_times).");
    AddAttr<std::vector<int>>("target_shape",
                              "Expand shape for each dimension.")
        .SetDefault({});
    AddComment(R"DOC(
Expand the input to the given shape.
)DOC");
  }
};

class ExpandAsV3GradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "ExpandAsV3Grad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   framework::GradVarName("Out"), "ExpandAsV3Grad");

    auto x_dims = ctx->GetInputDim("X");
    auto x_grad_name = framework::GradVarName("X");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
    }
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.device_context());
  }
};

template <typename T>
class ExpandAsV3GradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("expand_as_v3_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(ExpandAsV3GradNoNeedBufVarsInferer, "X");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(expand_as_v3, ops::ExpandAsV3Op, ops::ExpandAsV3OpMaker,
                  ops::ExpandAsV3GradOpMaker<paddle::framework::OpDesc>,
                  ops::ExpandAsV3GradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(expand_as_v3_grad, ops::ExpandAsV3GradOp,
                  ops::ExpandAsV3GradNoNeedBufVarsInferer);
REGISTER_OP_CPU_KERNEL(
    expand_as_v3,
    ops::ExpandAsV3Kernel<paddle::platform::CPUDeviceContext, float>,
    ops::ExpandAsV3Kernel<paddle::platform::CPUDeviceContext, double>,
    ops::ExpandAsV3Kernel<paddle::platform::CPUDeviceContext, int>,
    ops::ExpandAsV3Kernel<paddle::platform::CPUDeviceContext, int64_t>,
    ops::ExpandAsV3Kernel<paddle::platform::CPUDeviceContext, bool>);
REGISTER_OP_CPU_KERNEL(
    expand_as_v3_grad,
    ops::ExpandAsV3GradKernel<paddle::platform::CPUDeviceContext, int>,
    ops::ExpandAsV3GradKernel<paddle::platform::CPUDeviceContext, int64_t>,
    ops::ExpandAsV3GradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ExpandAsV3GradKernel<paddle::platform::CPUDeviceContext, double>);
