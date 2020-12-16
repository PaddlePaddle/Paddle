// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/conj_op.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace operators {

class ConjOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "conj");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "conj");

    auto in_dims = ctx->GetInputDim("X");

    ctx->SetOutputDim("Out", in_dims);
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

class ConjOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor of conj op.");
    AddOutput("Out", "(Tensor), The output tensor of conj op.");
    AddComment(R"DOC(
Conj Operator.

This operator is used to perform elementwise conjugate for input $X$.

)DOC");
  }
};

template <typename T>
class ConjGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("conj");
    retv->SetInput("X", this->OutputGrad("Out"));
    retv->SetAttrMap(this->Attrs());
    retv->SetOutput("Out", this->InputGrad("X"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(conj, ops::ConjOp, ops::ConjOpMaker,
                  ops::ConjGradMaker<paddle::framework::OpDesc>,
                  ops::ConjGradMaker<paddle::imperative::OpBase>);

REGISTER_OP_CPU_KERNEL(
    conj, ops::ConjKernel<paddle::platform::CPUDeviceContext,
                          paddle::platform::complex64>,
    ops::ConjKernel<paddle::platform::CPUDeviceContext,
                    paddle::platform::complex128>,
    ops::ConjKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ConjKernel<paddle::platform::CPUDeviceContext, double>,
    ops::ConjKernel<paddle::platform::CPUDeviceContext, int>,
    ops::ConjKernel<paddle::platform::CPUDeviceContext, int64_t>);
