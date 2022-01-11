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

#include "paddle/fluid/operators/lu_unpack_op.h"

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

  void InferShape(framework::InferShapeContext *context) const override {
    OP_INOUT_CHECK(context->HasInput("X"), "Input", "X", "LU_Unpack");
    OP_INOUT_CHECK(context->HasInput("Pivots"), "Input", "Pivots", "LU_Unpack");
    OP_INOUT_CHECK(context->HasOutput("L"), "Output", "L", "LU_Unpack");
    OP_INOUT_CHECK(context->HasOutput("U"), "Output", "U", "LU_Unpack");
    OP_INOUT_CHECK(context->HasOutput("Pmat"), "Output", "Pmat", "LU_Unpack");
    bool unpack_ludata = context->Attrs().Get<bool>("unpack_ludata");
    bool unpack_pivots = context->Attrs().Get<bool>("unpack_pivots");

    auto x_dims = context->GetInputDim("X");
    int x_rank = x_dims.size();
    PADDLE_ENFORCE_GE(x_rank, 2, platform::errors::InvalidArgument(
                                     "the rank of input must greater than 2"));

    // context->SetOutputDim("Out", x_dims);
    int m = x_dims[x_rank - 1];
    int n = x_dims[x_rank - 2];
    int min_mn = std::min(m, n);
    if (unpack_ludata) {
      auto ldims = x_dims;
      auto udims = x_dims;
      if (m >= n) {
        udims[x_rank - 2] = min_mn;
      } else {
        ldims[x_rank - 1] = min_mn;
      }
      context->SetOutputDim("U", udims);
      context->SetOutputDim("L", ldims);
    }
    if (unpack_pivots) {
      auto pdims = x_dims;
      pdims[x_rank - 1] = m;
      context->SetOutputDim("Pmat", pdims);
    }
  }

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

    ctx->SetOutputType(framework::GradVarName("X"), var_type,
                       framework::ALL_ELEMENTS);
    ctx->SetOutputDataType(framework::GradVarName("X"), data_type,
                           framework::ALL_ELEMENTS);
  }
};

class LU_UnpackGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "lu_unpack");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("L")), "Input",
                   "L@GRAD", "lu_unpack");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("U")), "Input",
                   "U@GRAD", "lu_unpack");

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

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OPERATOR(lu_unpack, ops::LU_UnpackOp, ops::LU_UnpackOpMaker,
                  ops::LU_UnpackOpVarTypeInference,
                  ops::LU_UnpackOpGradMaker<paddle::framework::OpDesc>,
                  ops::LU_UnpackOpGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(lu_unpack_grad, ops::LU_UnpackGradOp,
                  ops::LU_UnpackGradOpVarTypeInference);

REGISTER_OP_CPU_KERNEL(lu_unpack,
                       ops::LU_UnpackKernel<plat::CPUDeviceContext, float>,
                       ops::LU_UnpackKernel<plat::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    lu_unpack_grad, ops::LU_UnpackGradKernel<plat::CPUDeviceContext, float>,
    ops::LU_UnpackGradKernel<plat::CPUDeviceContext, double>);
