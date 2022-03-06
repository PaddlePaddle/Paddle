/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

class RealOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

class RealOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor of real op.");
    AddOutput("Out", "(Tensor), The output tensor of real op.");
    AddComment(R"DOC( 
Real Operator. 

This operator is used to get a new tensor containing real values 
from a tensor with complex data type.

)DOC");
  }
};

class RealGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   "Out@Grad", "RealGrad");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("X")), "Output",
                   "X@Grad", "RealGrad");

    auto dout_dims = ctx->GetInputDim(framework::GradVarName("Out"));
    ctx->SetOutputDim(framework::GradVarName("X"), dout_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto dtype = OperatorWithKernel::IndicateVarDataType(
        ctx, framework::GradVarName("Out"));
    auto complex_dtype = framework::ToComplexType(dtype);
    return framework::OpKernelType(complex_dtype, ctx.GetPlace());
  }
};

template <typename T>
class RealGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("real_grad");
    grad_op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    grad_op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }
};

DECLARE_INPLACE_OP_INFERER(RealOpInplaceInferer, {"X", "Out"});
DECLARE_INPLACE_OP_INFERER(RealGradOpInplaceInferer,
                           {framework::GradVarName("Out"),
                            framework::GradVarName("X")});

}  // namespace operators
}  // namespace paddle

DECLARE_INFER_SHAPE_FUNCTOR(real, RealInferShapeFunctor,
                            PD_INFER_META(phi::RealAndImagInferMeta));

namespace ops = paddle::operators;

REGISTER_OPERATOR(real, ops::RealOp, ops::RealOpMaker,
                  ops::RealGradOpMaker<::paddle::framework::OpDesc>,
                  ops::RealGradOpMaker<::paddle::imperative::OpBase>,
                  RealInferShapeFunctor);
REGISTER_OPERATOR(real_grad, ops::RealGradOp);
