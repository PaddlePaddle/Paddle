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

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/backward.h"
#include "paddle/phi/infermeta/binary.h"

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

DECLARE_INFER_SHAPE_FUNCTOR(complex,
                            ComplexInferShapeFunctor,
                            PD_INFER_META(phi::ComplexInferMeta));

REGISTER_OPERATOR(complex,
                  ops::ComplexOp,
                  ops::ComplexOpMaker,
                  ops::ComplexGradOpMaker<paddle::framework::OpDesc>,
                  ops::ComplexGradOpMaker<paddle::imperative::OpBase>,
                  ComplexInferShapeFunctor);

DECLARE_INFER_SHAPE_FUNCTOR(complex_grad,
                            ComplexGradInferShapeFunctor,
                            PD_INFER_META(phi::ComplexGradInferMeta));

REGISTER_OPERATOR(complex_grad,
                  ops::ComplexGradOp,
                  ComplexGradInferShapeFunctor);
