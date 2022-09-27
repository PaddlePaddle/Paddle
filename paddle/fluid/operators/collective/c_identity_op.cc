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

#include "paddle/fluid/operators/collective/c_identity_op.h"

namespace paddle {
namespace operators {

class CIdentityOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "c_identity");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "c_identity");
    int ring_id = ctx->Attrs().Get<int>("ring_id");
    PADDLE_ENFORCE_GE(
        ring_id,
        0,
        platform::errors::InvalidArgument(
            "The ring_id (%d) for c_identity must be non-negative.", ring_id));
    framework::DDim dim = ctx->GetInputDim("X");
    ctx->SetOutputDim("Out", dim);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class CIdentityOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "(Tensor) identity tensor.");
    AddOutput("Out", "(Tensor) identity tensor.");
    AddAttr<int>("ring_id", "(int default 0) nccl communication ring id.")
        .SetDefault(0);
    AddAttr<bool>(
        "use_calc_stream",
        "(bool default true) eject CUDA operations to calculation stream.")
        .SetDefault(true);
    AddAttr<bool>("use_model_parallel",
                  "(bool default true) use this op with model parallel.")
        .SetDefault(true);
    AddComment(R"DOC(
Identity Operator which returns a copy of itself.
)DOC");
  }
};

template <typename T>
class CIdentityOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("c_allreduce_sum");
    retv->SetInput("X", this->OutputGrad("Out"));
    retv->SetOutput("Out", this->InputGrad("X"));
    retv->SetAttrMap(this->Attrs());
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OPERATOR(c_identity,
                  ops::CIdentityOp,
                  ops::CIdentityOpGradMaker<paddle::framework::OpDesc>,
                  ops::CIdentityOpGradMaker<paddle::imperative::OpBase>,
                  ops::CIdentityOpMaker);

REGISTER_OP_CPU_KERNEL(c_identity,
                       ops::CIdentityOpCPUKernel<float>,
                       ops::CIdentityOpCPUKernel<double>,
                       ops::CIdentityOpCPUKernel<int>,
                       ops::CIdentityOpCPUKernel<int64_t>,
                       ops::CIdentityOpCPUKernel<plat::float16>);
