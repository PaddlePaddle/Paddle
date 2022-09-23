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

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"

#include "paddle/phi/infermeta/backward.h"
#include "paddle/phi/infermeta/binary.h"

namespace paddle {
namespace operators {

class LU_UnpackOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddComment(R"DOC(Unpack L U and P to single matrix tensor,
                unpack L and U matrix from LU, unpack permutation matrix Pmat from Pivtos .
                )DOC");
    AddInput("X", "(Tensor) The input LU tensor, shape of (*,m,n)");
    AddInput("Pivots",
             "(Tensor) The input Pivots tensor, shape of (*,min(m,n))");
    AddOutput(
        "Pmat",
        "(Tensor) The output permutation matrix tensor, shape of (*, m, m)");
    AddOutput("L", "(Tensor) The output lower triangular matrix tensor");
    AddOutput("U", "(Tensor) The output upper triangular matrix tensor");
    AddAttr<bool>("unpack_ludata", "Whether to unpack L and U")
        .SetDefault(true);
    AddAttr<bool>("unpack_pivots", "Whether to unpack permutation matrix")
        .SetDefault(true);
  }
};

class LU_UnpackOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class LU_UnpackOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    auto var_type = ctx->GetInputType("X", 0);
    auto data_type = ctx->GetInputDataType("X", 0);

    ctx->SetOutputType("L", var_type, framework::ALL_ELEMENTS);
    ctx->SetOutputDataType("L", data_type, framework::ALL_ELEMENTS);

    ctx->SetOutputType("U", var_type, framework::ALL_ELEMENTS);
    ctx->SetOutputDataType("U", data_type, framework::ALL_ELEMENTS);

    ctx->SetOutputType("Pmat", var_type, framework::ALL_ELEMENTS);
    ctx->SetOutputDataType("Pmat", data_type, framework::ALL_ELEMENTS);
  }
};

template <typename T>
class LU_UnpackOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("lu_unpack_grad");
    retv->SetInput("X", this->Input("X"));
    retv->SetInput("Pivots", this->Input("Pivots"));
    retv->SetInput("L", this->Output("L"));
    retv->SetInput("U", this->Output("U"));
    retv->SetInput("Pmat", this->Output("Pmat"));

    retv->SetInput(framework::GradVarName("L"), this->OutputGrad("L"));
    retv->SetInput(framework::GradVarName("U"), this->OutputGrad("U"));
    retv->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    retv->SetAttrMap(this->Attrs());
  }
};

class LU_UnpackGradOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    auto var_type = ctx->GetInputType("X", 0);
    auto data_type = ctx->GetInputDataType("X", 0);

    ctx->SetOutputType(
        framework::GradVarName("X"), var_type, framework::ALL_ELEMENTS);
    ctx->SetOutputDataType(
        framework::GradVarName("X"), data_type, framework::ALL_ELEMENTS);
  }
};

class LU_UnpackGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto dtype = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(dtype, ctx.GetPlace());
  }
};

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;
namespace plat = paddle::platform;

DECLARE_INFER_SHAPE_FUNCTOR(lu_unpack,
                            LUUnpackInferMetaFunctor,
                            PD_INFER_META(phi::LUUnpackInferMeta));
DECLARE_INFER_SHAPE_FUNCTOR(lu_unpack_grad,
                            LUUnpackGradInferMetaFunctor,
                            PD_INFER_META(phi::LUUnpackGradInferMeta));

REGISTER_OPERATOR(lu_unpack,
                  ops::LU_UnpackOp,
                  ops::LU_UnpackOpMaker,
                  ops::LU_UnpackOpVarTypeInference,
                  ops::LU_UnpackOpGradMaker<paddle::framework::OpDesc>,
                  ops::LU_UnpackOpGradMaker<paddle::imperative::OpBase>,
                  LUUnpackInferMetaFunctor);
REGISTER_OPERATOR(lu_unpack_grad,
                  ops::LU_UnpackGradOp,
                  ops::LU_UnpackGradOpVarTypeInference,
                  LUUnpackGradInferMetaFunctor);
