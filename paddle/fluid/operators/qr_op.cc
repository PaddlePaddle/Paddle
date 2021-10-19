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

#include "paddle/fluid/operators/qr_op.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/framework/ddim.h"
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace operators {
using DDim = framework::DDim;

class QrOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "qr");
    OP_INOUT_CHECK(ctx->HasOutput("Q"), "Output", "Q", "qr");
    OP_INOUT_CHECK(ctx->HasOutput("R"), "Output", "R", "qr");

    auto x_dims = ctx->GetInputDim("X");
    int x_rank = x_dims.size();
    PADDLE_ENFORCE_GE(x_dims.size(), 2,
                      platform::errors::InvalidArgument(
                          "the rank of input must greater than 2"));
    bool compute_q;
    bool reduced_mode;
    int m = x_dims[x_rank - 2];
    int n = x_dims[x_rank - 1];
    int min_mn = std::min(m, n);
    std::string mode = ctx->Attrs().Get<std::string>("mode");
    std::tie(compute_q, reduced_mode) = _parse_qr_mode(mode);

    if (compute_q) {
      int k = reduced_mode ? min_mn : m;
      auto q_dims_vec = framework::vectorize(x_dims);
      q_dims_vec[q_dims_vec.size() - 1] = k;
      ctx->SetOutputDim("Q", framework::make_ddim(q_dims_vec));
    } else {
      ctx->SetOutputDim("Q", framework::make_ddim({0}));
    }

    int k = reduced_mode ? min_mn : m;
    auto r_dims_vec = framework::vectorize(x_dims);
    r_dims_vec[r_dims_vec.size() - 2] = k;
    r_dims_vec[r_dims_vec.size() - 1] = n;
    ctx->SetOutputDim("R", framework::make_ddim(r_dims_vec));

    ctx->ShareLoD("X", /*->*/ "Q");
    ctx->ShareLoD("X", /*->*/ "R");
  }
};

class QrOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor of qr op.");
    AddOutput("Q", "(Tensor), The output Q tensor of qr op.");
    AddOutput("R", "(Tensor), The output R tensor of qr op.");
    AddAttr<std::string>(
        "mode",
        "(string, default \"reduced\"). "
        "If mode is \"reduced\", Qr op will return reduced Q and R matrices. "
        "If mode is \"complete\", Qr op will return complete Q and R matrices. "
        "If mode is \"r\", Qr op will only return reduced R matrix.")
        .SetDefault("reduced");
    AddComment(R"DOC(
Qr Operator.

This operator is used to perform QR operation for batched matrics $X$.
$$Q, R = qr(X)$$

)DOC");
  }
};

class QrGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Q")), "Input",
                   "Q@Grad", "QrGrad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("R")), "Input",
                   "R@Grad", "QrGrad");
    OP_INOUT_CHECK(ctx->HasInput("Q"), "Input", "Q", "QrGrad");
    OP_INOUT_CHECK(ctx->HasInput("R"), "Input", "R", "QrGrad");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("X")), "Output",
                   "X@Grad", "QrGrad");

    auto x_dims = ctx->GetInputDim(("X"));
    ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto dtype = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(dtype, ctx.GetPlace());
  }
};

template <typename T>
class QrGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("qr_grad");
    retv->SetInput(framework::GradVarName("Q"), this->OutputGrad("Q"));
    retv->SetInput(framework::GradVarName("R"), this->OutputGrad("R"));
    retv->SetInput("Q", this->Output("Q"));
    retv->SetInput("R", this->Output("R"));
    retv->SetInput("X", this->Input("X"));
    retv->SetAttrMap(this->Attrs());
    retv->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(qr, ops::QrOp, ops::QrOpMaker,
                  ops::QrGradMaker<paddle::framework::OpDesc>,
                  ops::QrGradMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(qr_grad, ops::QrGradOp);

REGISTER_OP_CPU_KERNEL(qr, ops::QrCPUKernel<float>, ops::QrCPUKernel<double>);

REGISTER_OP_CPU_KERNEL(
    qr_grad, ops::QrGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::QrGradKernel<paddle::platform::CPUDeviceContext, double>);
