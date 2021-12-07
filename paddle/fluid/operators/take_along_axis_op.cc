/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/take_along_axis_op.h"
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace operators {

class TakeAlongAxisOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("Input"), true,
                      platform::errors::InvalidArgument(
                          "Input(Input) of GatherOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasInput("Index"), true,
                      platform::errors::InvalidArgument(
                          "Input(Index) of GatherOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Result"), true,
                      platform::errors::InvalidArgument(
                          "Output(Result) of GatherOp should not be null."));

    auto index_dim = ctx->GetInputDim("Index");

    ctx->SetOutputDim("Result", index_dim);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Input"),
        ctx.device_context());
  }
  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const framework::Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override {
    if (var_name == "Dim") {
      return expected_kernel_type;
    }
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   tensor.place(), tensor.layout());
  }
};

class TakeAlongAxisOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input", "The input tensor of TakeAlongAxis op");
    AddInput("Index", "The index tensor of TakeAlongAxis op");
    AddOutput("Result", "The result tensor of TakeAlongAxis op");
    AddAttr<int>("Dim",
                 "The Tensor which contains the dim that we do TakeAlongAxis "
                 "operation.");
    AddComment(R"DOC(
        Take_along_axis Operator.)
    )DOC");
  }
};

class TakeAlongAxisGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    ctx->SetOutputDim(framework::GradVarName("Input"),
                      ctx->GetInputDim("Input"));
    ctx->ShareLoD("Input", /*-->*/ framework::GradVarName("Input"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Result")),
                                   ctx.device_context());
  }
  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const framework::Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override {
    // if (var_name == "Dim") {
    //   return expected_kernel_type;
    // }
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   tensor.place(), tensor.layout());
  }
};

template <typename T>
class TakeAlongAxisGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("take_along_axis_grad");
    op->SetInput("Index", this->Input("Index"));
    op->SetInput("Input", this->Input("Input"));

    op->SetInput(framework::GradVarName("Result"), this->OutputGrad("Result"));
    op->SetOutput(framework::GradVarName("Input"), this->InputGrad("Input"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    take_along_axis, ops::TakeAlongAxisOp, ops::TakeAlongAxisOpMaker,
    ops::TakeAlongAxisGradOpMaker<paddle::framework::OpDesc>,
    ops::TakeAlongAxisGradOpMaker<paddle::imperative::OpBase>);  //,
// ops::TakeAlongAxisGradOpMaker<paddle::framework::OpDesc>,
// ops::TakeAlongAxisGradOpMaker<paddle::imperative::OpBase>);

// REGISTER_OPERATOR(take_along_axisgra, ops::TakeAlongAxisOpGradOp);

REGISTER_OPERATOR(take_along_axis_grad, ops::TakeAlongAxisGradOp)
REGISTER_OP_CPU_KERNEL(take_along_axis, ops::TakeAlongAxisOpKernel<float>,
                       ops::TakeAlongAxisOpKernel<double>,
                       ops::TakeAlongAxisOpKernel<int>,
                       ops::TakeAlongAxisOpKernel<uint8_t>,
                       ops::TakeAlongAxisOpKernel<int64_t>);
