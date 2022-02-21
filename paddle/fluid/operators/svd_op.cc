// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/svd_op.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/phi/core/ddim.h"
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace operators {

using DDim = framework::DDim;
static DDim UDDim(const DDim& x_dim, int k) {
  // get x_dim and return the ddim of U
  auto x_vec = vectorize(x_dim);
  x_vec[x_vec.size() - 1] = k;
  return phi::make_ddim(x_vec);
}
static DDim VHDDim(const DDim& x_dim, int k) {
  // get x_dim and return the ddim of U
  auto x_vec = vectorize(x_dim);
  x_vec[x_vec.size() - 2] = k;
  return phi::make_ddim(x_vec);
}
static DDim SDDim(const DDim& x_dim, int k) {
  // get x_dim and return the ddim of U
  auto x_vec = vectorize(x_dim);
  x_vec[x_vec.size() - 2] = k;
  x_vec.erase(x_vec.end() - 1);  // rank - 1
  return phi::make_ddim(x_vec);
}

class SvdOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "svd");
    OP_INOUT_CHECK(ctx->HasOutput("U"), "Output", "U", "svd");
    OP_INOUT_CHECK(ctx->HasOutput("VH"), "Output", "VH", "svd");
    OP_INOUT_CHECK(ctx->HasOutput("S"), "Output", "S", "svd");

    auto in_dims = ctx->GetInputDim("X");
    int x_rank = in_dims.size();
    PADDLE_ENFORCE_GE(in_dims.size(), 2,
                      platform::errors::InvalidArgument(
                          "the rank of input must greater than 2"));
    int m = in_dims[x_rank - 2];
    int n = in_dims[x_rank - 1];
    int k = std::min(m, n);
    const bool full_uv = ctx->Attrs().Get<bool>("full_matrices");
    ctx->SetOutputDim("U", !full_uv ? UDDim(in_dims, k) : UDDim(in_dims, m));
    ctx->SetOutputDim("VH", !full_uv ? VHDDim(in_dims, k) : VHDDim(in_dims, n));
    ctx->SetOutputDim("S", SDDim(in_dims, k));

    ctx->ShareLoD("X", /*->*/ "U");
    ctx->ShareLoD("X", /*->*/ "VH");
    ctx->ShareLoD("X", /*->*/ "S");
  }
};

class SvdOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor of svd op.");
    AddOutput("U", "(Tensor), The output U tensor of svd op.");
    AddOutput("S", "(Tensor), The output S tensor of svd op.");
    AddOutput("VH", "(Tensor), The output VH tensor of svd op.");
    AddAttr<bool>("full_matrices",
                  "(bool, default false) Only Compute the thin U and V"
                  "when set as True, the gradient have some random "
                  "attribute.")
        .SetDefault(false);
    AddComment(R"DOC(
Svd Operator.

This operator is used to perform SVD operation for batched matrics $X$.
$$U, S, VH = svd(X)$$

)DOC");
  }
};

class SvdGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("U")), "Input",
                   "U@Grad", "SvdGrad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("VH")), "Input",
                   "VH@Grad", "SvdGrad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("S")), "Input",
                   "S@Grad", "SvdGrad");
    OP_INOUT_CHECK(ctx->HasInput("U"), "Input", "U", "SvdGrad");
    OP_INOUT_CHECK(ctx->HasInput("S"), "Input", "S", "SvdGrad");
    OP_INOUT_CHECK(ctx->HasInput("VH"), "Input", "VH", "SvdGrad");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("X")), "Output",
                   "X@Grad", "SvdGrad");

    auto d_x = ctx->GetInputDim(("X"));
    ctx->SetOutputDim(framework::GradVarName("X"), d_x);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto dtype = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(dtype, ctx.GetPlace());
  }
};

template <typename T>
class SvdGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("svd_grad");
    retv->SetInput(framework::GradVarName("U"), this->OutputGrad("U"));
    retv->SetInput(framework::GradVarName("VH"), this->OutputGrad("VH"));
    retv->SetInput(framework::GradVarName("S"), this->OutputGrad("S"));
    retv->SetInput("U", this->Output("U"));
    retv->SetInput("VH", this->Output("VH"));
    retv->SetInput("S", this->Output("S"));
    retv->SetInput("X", this->Input("X"));
    retv->SetAttrMap(this->Attrs());
    retv->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(svd, ops::SvdOp, ops::SvdOpMaker,
                  ops::SvdGradMaker<paddle::framework::OpDesc>,
                  ops::SvdGradMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(svd_grad, ops::SvdGradOp);

REGISTER_OP_CPU_KERNEL(svd, ops::SvdCPUKernel<float>,
                       ops::SvdCPUKernel<double>);

REGISTER_OP_CPU_KERNEL(
    svd_grad, ops::SvdGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SvdGradKernel<paddle::platform::CPUDeviceContext, double>);
