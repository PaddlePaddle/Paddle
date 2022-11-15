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

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {
using DDim = framework::DDim;

class QrOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
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
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Q")),
                   "Input",
                   "Q@Grad",
                   "QrGrad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("R")),
                   "Input",
                   "R@Grad",
                   "QrGrad");
    OP_INOUT_CHECK(ctx->HasInput("Q"), "Input", "Q", "QrGrad");
    OP_INOUT_CHECK(ctx->HasInput("R"), "Input", "R", "QrGrad");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("X")),
                   "Output",
                   "X@Grad",
                   "QrGrad");

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
DECLARE_INFER_SHAPE_FUNCTOR(qr,
                            QrInferShapeFunctor,
                            PD_INFER_META(phi::QrInferMeta));

REGISTER_OPERATOR(qr,
                  ops::QrOp,
                  ops::QrOpMaker,
                  ops::QrGradMaker<paddle::framework::OpDesc>,
                  ops::QrGradMaker<paddle::imperative::OpBase>,
                  QrInferShapeFunctor);

REGISTER_OPERATOR(qr_grad, ops::QrGradOp);
