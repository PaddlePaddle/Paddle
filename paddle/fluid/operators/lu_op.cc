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

#include "paddle/fluid/operators/lu_op.h"

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

  void InferShape(framework::InferShapeContext *context) const override {
    OP_INOUT_CHECK(context->HasInput("X"), "Input", "X", "LU");
    OP_INOUT_CHECK(context->HasOutput("Out"), "Output", "Out", "LU");
    bool pivots = context->Attrs().Get<bool>("pivots");
    auto x_dims = context->GetInputDim("X");
    int x_rank = x_dims.size();
    PADDLE_ENFORCE_GE(x_rank,
                      2,
                      platform::errors::InvalidArgument(
                          "the rank of input must greater than 2"));
    context->SetOutputDim("Out", x_dims);
    int m = x_dims[x_rank - 1];
    int n = x_dims[x_rank - 2];
    int min_mn = std::min(m, n);
    auto dims_vec = phi::vectorize(x_dims);
    OP_INOUT_CHECK(context->HasOutput("Infos"), "Output", "Infos", "LU");
    if (x_rank == 2) {
      auto Infos_dim = std::vector<int>(1);
      context->SetOutputDim("Infos", phi::make_ddim(Infos_dim));
    } else {
      auto Infos_dim =
          std::vector<int>(dims_vec.begin(), dims_vec.begin() + x_rank - 2);
      context->SetOutputDim("Infos", phi::make_ddim(Infos_dim));
    }
    if (pivots) {
      OP_INOUT_CHECK(context->HasOutput("Pivots"), "Output", "Pivots", "LU");
      auto Pivots_dim =
          std::vector<int>(dims_vec.begin(), dims_vec.begin() + x_rank - 1);
      Pivots_dim[x_rank - 2] = min_mn;
      context->SetOutputDim("Pivots", phi::make_ddim(Pivots_dim));
    }
  }

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

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "lu");
    OP_INOUT_CHECK(ctx->HasInput("Out"), "Input", "Out", "lu");
    OP_INOUT_CHECK(ctx->HasInput("Pivots"), "Input", "Pivots", "lu");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")),
                   "Input",
                   "Out@GRAD",
                   "lu");

    auto x_dims = ctx->GetInputDim("X");
    auto x_grad_name = framework::GradVarName("X");

    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
    }
  }

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

REGISTER_OPERATOR(lu,
                  ops::LUOp,
                  ops::LUOpMaker,
                  ops::LUOpVarTypeInference,
                  ops::LUOpGradMaker<paddle::framework::OpDesc>,
                  ops::LUOpGradMaker<paddle::imperative::OpBase>,
                  ops::LUOpInplaceInferer);
REGISTER_OPERATOR(lu_grad,
                  ops::LUGradOp,
                  ops::LUGradOpVarTypeInference,
                  ops::LUGradOpInplaceInferer);
