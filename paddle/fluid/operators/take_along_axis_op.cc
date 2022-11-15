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
#include "paddle/phi/infermeta/binary.h"

namespace paddle {
namespace operators {

class TakeAlongAxisOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Input"),
        ctx.device_context());
  }
};

class TakeAlongAxisOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input", "The input tensor of TakeAlongAxisOp");
    AddInput("Index", "The index tensor of TakeAlongAxisOp");
    AddOutput("Result", "The result tensor of TakeAlongAxisOp");
    AddAttr<int>("Axis",
                 "The Tensor which contains the axis that we do TakeAlongAxis "
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
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Result")),
                                   ctx.device_context());
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
DECLARE_INFER_SHAPE_FUNCTOR(take_along_axis,
                            TakeAlongAxisInferShapeFunctor,
                            PD_INFER_META(phi::TakeAlongAxisInferMeta));
REGISTER_OPERATOR(take_along_axis,
                  ops::TakeAlongAxisOp,
                  ops::TakeAlongAxisOpMaker,
                  ops::TakeAlongAxisGradOpMaker<paddle::framework::OpDesc>,
                  ops::TakeAlongAxisGradOpMaker<paddle::imperative::OpBase>,
                  TakeAlongAxisInferShapeFunctor);

REGISTER_OPERATOR(take_along_axis_grad, ops::TakeAlongAxisGradOp);
