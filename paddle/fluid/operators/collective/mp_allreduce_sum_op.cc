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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/collective/c_allreduce_op.h"

namespace paddle::framework {
class OpDesc;
}  // namespace paddle::framework
namespace paddle::imperative {
class OpBase;
}  // namespace paddle::imperative

namespace paddle::operators {

class MpAllReduceSumOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext *ctx) const override {
    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
  }
};

class MpAllReduceSumOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), tensor to be allreduced in model parallel.");
    AddOutput("Out", "(Tensor) the allreduced result in model parallel.");
    AddAttr<int>("ring_id", "(int default 0) communication ring id.")
        .SetDefault(0);

    AddComment(string::Sprintf(R"DOC(
MpAllReduceSum Operator

Call collective AllReduceSum in model parallel. If input and output are
the same variable, in-place allreduce will be used.
Reference: https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/usage/operations.html#allreduce
)DOC"));
  }
};

template <typename T>
class MpAllReduceSumOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("c_identity");
    retv->SetInput("X", this->OutputGrad("Out"));
    retv->SetOutput("Out", this->InputGrad("X"));
    retv->SetAttrMap(this->Attrs());
  }
};

DECLARE_INPLACE_OP_INFERER(MpAllReduceSumInplaceInferer, {"X", "Out"});

DEFINE_C_ALLREDUCE_CPU_KERNEL(MpAllReduceSum, kRedSum);

}  // namespace paddle::operators

namespace ops = paddle::operators;

REGISTER_OPERATOR(mp_allreduce_sum,
                  ops::MpAllReduceSumOp,
                  ops::MpAllReduceSumOpGradMaker<paddle::framework::OpDesc>,
                  ops::MpAllReduceSumOpGradMaker<paddle::imperative::OpBase>,
                  ops::MpAllReduceSumOpMaker,
                  ops::MpAllReduceSumInplaceInferer);
