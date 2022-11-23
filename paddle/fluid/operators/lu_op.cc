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
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

class LUOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddComment(R"DOC(LU decomposition,
                Computes the LU factorization of a matrix or batches of matrices A.
                )DOC");
    AddInput("X", "(Tensor) The input tensor, shape of (*,m,n)");
    AddOutput("Out", "(Tensor) The output tensor, shape same to X");
    AddOutput("Pivots",
              "Stores all the intermediate transpositions of rows. shape of "
              "(*,min(m,n))");
    AddOutput("Infos",
              "(Tensor) This is a tensor of size (*) where non-zero values "
              "indicate whether factorization for the matrix has succeeded");
    AddAttr<bool>("pivots", "Whether pivoting is done").SetDefault(true);
  }
};

class LUOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class LUOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    auto var_type = ctx->GetInputType("X", 0);
    auto data_type = ctx->GetInputDataType("X", 0);

    ctx->SetOutputType("Out", var_type, framework::ALL_ELEMENTS);
    ctx->SetOutputDataType("Out", data_type, framework::ALL_ELEMENTS);

    ctx->SetOutputType("Pivots", var_type, framework::ALL_ELEMENTS);
    ctx->SetOutputDataType(
        "Pivots", framework::proto::VarType::INT32, framework::ALL_ELEMENTS);

    ctx->SetOutputType("Infos", var_type, framework::ALL_ELEMENTS);
    ctx->SetOutputDataType(
        "Infos", framework::proto::VarType::INT32, framework::ALL_ELEMENTS);
  }
};

template <typename T>
class LUOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("lu_grad");
    retv->SetInput("X", this->Input("X"));
    retv->SetInput("Out", this->Output("Out"));
    retv->SetInput("Pivots", this->Output("Pivots"));
    retv->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    retv->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    retv->SetAttrMap(this->Attrs());
  }
};

class LUGradOpVarTypeInference : public framework::VarTypeInference {
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

class LUGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto dtype = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(dtype, ctx.GetPlace());
  }
};

DECLARE_INPLACE_OP_INFERER(LUOpInplaceInferer, {"X", "Out"});
DECLARE_INPLACE_OP_INFERER(LUGradOpInplaceInferer,
                           {framework::GradVarName("Out"),
                            framework::GradVarName("X")});

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;
namespace plat = paddle::platform;

DECLARE_INFER_SHAPE_FUNCTOR(lu,
                            LUInferMetaFunctor,
                            PD_INFER_META(phi::LUInferMeta));
DECLARE_INFER_SHAPE_FUNCTOR(lu_grad,
                            LUGradInferMetaFunctor,
                            PD_INFER_META(phi::LUGradInferMeta));

REGISTER_OPERATOR(lu,
                  ops::LUOp,
                  ops::LUOpMaker,
                  ops::LUOpVarTypeInference,
                  ops::LUOpGradMaker<paddle::framework::OpDesc>,
                  ops::LUOpGradMaker<paddle::imperative::OpBase>,
                  LUInferMetaFunctor);
REGISTER_OPERATOR(lu_grad,
                  ops::LUGradOp,
                  ops::LUGradOpVarTypeInference,
                  LUGradInferMetaFunctor);
