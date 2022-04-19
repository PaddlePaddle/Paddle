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
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/kernels/impl/einsum_impl.h"
#include "paddle/fluid/framework/infershape_utils.h"

namespace paddle {
namespace operators {
class EinsumOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

class EinsumOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Operands", "(Tensor), The input tensor of svd op.").AsDuplicable();
    AddOutput("Out", "(Tensor), The output VH tensor of svd op.");
    AddAttr<std::string>("equation",
                  "(string) A einsum equation. such as `ij,jk->ik`"
                  "There must have `->` and the number of operands in equation must equals the `Operands` length.");
    AddComment(R"DOC(
Einsum Operator.

This operator is used to perform einsum operation for given operands and equation.
)DOC");
  }
};

//class SvdGradOp : public framework::OperatorWithKernel {
 //public:
  //using framework::OperatorWithKernel::OperatorWithKernel;
  //void InferShape(framework::InferShapeContext* ctx) const override {
    //OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("U")), "Input",
                   //"U@Grad", "SvdGrad");
    //OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("VH")), "Input",
                   //"VH@Grad", "SvdGrad");
    //OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("S")), "Input",
                   //"S@Grad", "SvdGrad");
    //OP_INOUT_CHECK(ctx->HasInput("U"), "Input", "U", "SvdGrad");
    //OP_INOUT_CHECK(ctx->HasInput("S"), "Input", "S", "SvdGrad");
    //OP_INOUT_CHECK(ctx->HasInput("VH"), "Input", "VH", "SvdGrad");
    //OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("X")), "Output",
                   //"X@Grad", "SvdGrad");

    //auto d_x = ctx->GetInputDim(("X"));
    //ctx->SetOutputDim(framework::GradVarName("X"), d_x);
  //}

 //protected:
  //framework::OpKernelType GetExpectedKernelType(
      //const framework::ExecutionContext& ctx) const override {
    //auto dtype = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    //return framework::OpKernelType(dtype, ctx.GetPlace());
  //}
//};

//template <typename T>
//class SvdGradMaker : public framework::SingleGradOpMaker<T> {
 //public:
  //using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  //void Apply(GradOpPtr<T> retv) const override {
    //retv->SetType("svd_grad");
    //retv->SetInput(framework::GradVarName("U"), this->OutputGrad("U"));
    //retv->SetInput(framework::GradVarName("VH"), this->OutputGrad("VH"));
    //retv->SetInput(framework::GradVarName("S"), this->OutputGrad("S"));
    //retv->SetInput("U", this->Output("U"));
    //retv->SetInput("VH", this->Output("VH"));
    //retv->SetInput("S", this->Output("S"));
    //retv->SetInput("X", this->Input("X"));
    //retv->SetAttrMap(this->Attrs());
    //retv->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  //}
//};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(einsum, EinsumInferShapeFunctor,
                            PD_INFER_META(phi::EinsumInferShape));

REGISTER_OPERATOR(einsum, ops::EinsumOp, ops::EinsumOpMaker, EinsumInferShapeFunctor);
                  //ops::SvdGradMaker<paddle::framework::OpDesc>,
                  //ops::SvdGradMaker<paddle::imperative::OpBase>);

//REGISTER_OPERATOR(svd_grad, ops::SvdGradOp);

//REGISTER_OP_CPU_KERNEL(
    //svd_grad, ops::SvdGradKernel<paddle::platform::CPUDeviceContext, float>,
    //ops::SvdGradKernel<paddle::platform::CPUDeviceContext, double>);
