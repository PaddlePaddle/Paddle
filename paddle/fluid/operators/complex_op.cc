/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/complex_op.h"

#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"

namespace paddle {
namespace operators {

class ComplexOpMaker : public framework::OpProtoAndCheckerMaker {
 protected:
  void Make() override {
    AddInput("X", "(Tensor), real part of complex_op");
    AddInput("Y", "(Tensor), image part of complex_op");
    AddOutput("Out", "(Tensor), output of complex_op");
    AddComment(R"DOC(
Complex Operator.

Return a complex tensor given the real and image tensors.

)DOC");
  }
};

template <typename T>
class ComplexGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("complex_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Y", this->Input("Y"));
    // op->SetInput("Out", this->Output("Out"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));
    op->SetAttrMap(this->Attrs());
  }
};

class ComplexOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "complex");
    OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "complex");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "complex");

    if (ctx->GetInputDim("X") == ctx->GetInputDim("Y")) {
      ctx->ShareDim("X", /*->*/ "Out");
      // NOTE(chenfeiyu): lod & broadcasting is intrinsically contradictory
      // so tensors with lod are not supported here
    } else {
      auto x_dims = ctx->GetInputDim("X");
      auto y_dims = ctx->GetInputDim("Y");
      int max_dim = std::max(x_dims.size(), y_dims.size());

      // start align axis
      int axis = std::abs(x_dims.size() - y_dims.size());
      std::vector<int> x_dims_array(max_dim);
      std::vector<int> y_dims_array(max_dim);
      std::vector<int> out_dims_array(max_dim);
      GetBroadcastDimsArrays(x_dims, y_dims, x_dims_array.data(),
                             y_dims_array.data(), out_dims_array.data(),
                             max_dim, axis);
      ctx->SetOutputDim("Out", framework::make_ddim(out_dims_array));
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(data_type, ctx.GetPlace());
  }
};

class ComplexGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "complex_grad");
    OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "kron_complex_gradgrad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   framework::GradVarName("Out"), "complex_grad");

    auto x_grad_name = framework::GradVarName("X");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->ShareDim("X", /*->*/ x_grad_name);
    }

    auto y_grad_name = framework::GradVarName("Y");
    if (ctx->HasOutput(y_grad_name)) {
      ctx->ShareDim("Y", /*->*/ y_grad_name);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto out_grad_name = framework::GradVarName("Out");
    auto computation_dtype = framework::ToRealType(
        OperatorWithKernel::IndicateVarDataType(ctx, out_grad_name));
    return framework::OpKernelType(computation_dtype, ctx.GetPlace());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(complex, ops::ComplexOp, ops::ComplexOpMaker,
                  ops::ComplexGradOpMaker<paddle::framework::OpDesc>,
                  ops::ComplexGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(complex_grad, ops::ComplexGradOp);

REGISTER_OP_CPU_KERNEL(
    complex, ops::ComplexKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ComplexKernel<paddle::platform::CPUDeviceContext, double>);

REGISTER_OP_CPU_KERNEL(
    complex_grad,
    ops::ComplexGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ComplexGradKernel<paddle::platform::CPUDeviceContext, double>);
