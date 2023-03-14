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

namespace paddle {
namespace framework {
class OpDesc;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
}  // namespace paddle

namespace paddle {
namespace operators {

class MpAllReduceSumOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext *ctx) const override {
    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
  }
};

class MpAllReduceSumOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "(Tensor), tensor to be allreduced in model parallel.");
    AddOutput("Out", "(Tensor) the allreduced result in model parallel.");
    AddAttr<int>("ring_id", "(int default 0) communication ring id.")
        .SetDefault(0);
#if defined(PADDLE_WITH_ASCEND_CL)
    AddAttr<std::string>("tag", "(string default tag) tag for all reduce.")
        .SetDefault("tag");
#endif
    AddAttr<bool>(
        "use_calc_stream",
        "(bool default false) eject CUDA operations to calculation stream.")
        .SetDefault(false);
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

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OPERATOR(mp_allreduce_sum,
                  ops::MpAllReduceSumOp,
                  ops::MpAllReduceSumOpGradMaker<paddle::framework::OpDesc>,
                  ops::MpAllReduceSumOpGradMaker<paddle::imperative::OpBase>,
                  ops::MpAllReduceSumOpMaker,
                  ops::MpAllReduceSumInplaceInferer);

REGISTER_OP_CPU_KERNEL(mp_allreduce_sum,
                       ops::CAllReduceOpCPUKernel<ops::kRedSum, float>,
                       ops::CAllReduceOpCPUKernel<ops::kRedSum, double>,
                       ops::CAllReduceOpCPUKernel<ops::kRedSum, int>,
                       ops::CAllReduceOpCPUKernel<ops::kRedSum, int64_t>,
                       ops::CAllReduceOpCPUKernel<ops::kRedSum, plat::float16>)
