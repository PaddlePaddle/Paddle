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

#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/ternary.h"

namespace paddle {
namespace operators {

class PutAlongAxisOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

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
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   tensor.place(), tensor.layout());
  }
};

class PutAlongAxisOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input", "The input tensor of PutAlongAxisOp");
    AddInput("Index", "The index tensor of PutAlongAxisOp");
    AddInput("Value", "The value tensor of PutAlongAxisOp");
    AddOutput("Result", "The result tensor of PutAlongAxisOp");
    AddAttr<int>("Axis", "The axis that we do PutAlongAxis operation");
    AddAttr<std::string>("Reduce", "The reduce operation for scatter")
        .SetDefault("assign");
    AddComment(R"DOC(
        PutAlongAxis Operator.)
    )DOC");
  }
};

class PutAlongAxisGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    ctx->SetOutputDim(framework::GradVarName("Input"),
                      ctx->GetInputDim("Input"));
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
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   tensor.place(), tensor.layout());
  }
};

template <typename T>
class PutAlongAxisGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("put_along_axis_grad");
    op->SetInput("Index", this->Input("Index"));
    op->SetInput("Input", this->Input("Input"));

    op->SetInput(framework::GradVarName("Result"), this->OutputGrad("Result"));
    op->SetOutput(framework::GradVarName("Input"), this->InputGrad("Input"));
    op->SetOutput(framework::GradVarName("Value"), this->InputGrad("Value"));
    op->SetAttrMap(this->Attrs());
  }
};

DECLARE_INPLACE_OP_INFERER(PutAlongAxisInplaceInferer, {"Input", "Result"});

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(put_along_axis, PutAlongAxisInferShapeFunctor,
                            PD_INFER_META(phi::PutAlongAxisInferMeta));
REGISTER_OPERATOR(put_along_axis, ops::PutAlongAxisOp, ops::PutAlongAxisOpMaker,
                  ops::PutAlongAxisGradOpMaker<paddle::framework::OpDesc>,
                  ops::PutAlongAxisGradOpMaker<paddle::imperative::OpBase>,
                  paddle::operators::PutAlongAxisInplaceInferer,
                  PutAlongAxisInferShapeFunctor);

REGISTER_OPERATOR(put_along_axis_grad, ops::PutAlongAxisGradOp);
