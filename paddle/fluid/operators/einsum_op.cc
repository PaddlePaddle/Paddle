// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/kernels/impl/einsum_impl.h"

namespace paddle {
namespace operators {
class EinsumOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

class EinsumOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Operands", "(Tensor), The input tensor of svd op.")
        .AsDuplicable();
    AddOutput("Out", "(Tensor), The output VH tensor of svd op.");
    AddAttr<std::string>("equation",
                         "(string) A einsum equation. such as `ij,jk->ik`"
                         "There must have `->` and the number of operands in "
                         "equation must equals the `Operands` length.");
    AddComment(R"DOC(
Einsum Operator.

This operator is used to perform einsum operation for given operands and equation.
)DOC");
  }
};

class EinsumGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    auto in_x = "Operands";
    auto out_x_g_n = framework::GradVarName(in_x);
    ctx->SetOutputsDim(out_x_g_n, ctx->GetInputsDim(in_x));
    ctx->ShareAllLoD(in_x, out_x_g_n);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto dtype = OperatorWithKernel::IndicateVarDataType(
        ctx, framework::GradVarName("Out"));
    return framework::OpKernelType(dtype, ctx.GetPlace());
  }
};

template <typename T>
class EinsumGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("einsum_grad");
    retv->SetInput("Operands", this->Input("Operands"));
    retv->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    retv->SetAttrMap(this->Attrs());
    retv->SetOutput(framework::GradVarName("Operands"),
                    this->InputGrad("Operands", false));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(einsum, EinsumInferShapeFunctor,
                            PD_INFER_META(phi::EinsumInferShape));

REGISTER_OPERATOR(einsum, ops::EinsumOp, ops::EinsumOpMaker,
                  EinsumInferShapeFunctor,
                  ops::EinsumGradMaker<paddle::framework::OpDesc>,
                  ops::EinsumGradMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(einsum_grad, ops::EinsumGradOp);
