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

#include "paddle/fluid/operators/collective/alltoall_op.h"

namespace paddle {
namespace operators {

class AllToAllOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "AllToAll");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "AllToAll");
    int ring_id = ctx->Attrs().Get<int>("ring_id");
    PADDLE_ENFORCE_GE(
        ring_id,
        0,
        platform::errors::InvalidArgument(
            "The ring_id (%d) for alltoall op must be non-negative.", ring_id));
    framework::DDim dim = ctx->GetInputDim("X");
    if (dim[0] < 0) dim[0] = -1;
    ctx->SetOutputDim("Out", dim);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class AllToAllOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "(Tensor) tensor send.");
    AddOutput("Out", "(Tensor) the result of alltoall.");
    AddAttr<int>("ring_id", "(int default 0) nccl communication ring id.")
        .SetDefault(0);
    AddAttr<bool>(
        "use_calc_stream",
        "(bool default false) eject CUDA operations to calculation stream.")
        .SetDefault(false);
    AddComment(R"DOC(
AllToAll Operator
Scatter tensors from all participators to all participators.
)DOC");
  }
};

template <typename T>
class AllToAllOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("alltoall");
    retv->SetInput("X", this->OutputGrad("Out"));
    retv->SetOutput("Out", this->InputGrad("X"));
    retv->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OPERATOR(alltoall,
                  ops::AllToAllOp,
                  ops::AllToAllOpMaker,
                  ops::AllToAllOpGradMaker<paddle::framework::OpDesc>,
                  ops::AllToAllOpGradMaker<paddle::imperative::OpBase>)

REGISTER_OP_CPU_KERNEL(alltoall,
                       ops::AllToAllOpCPUKernel<float>,
                       ops::AllToAllOpCPUKernel<double>,
                       ops::AllToAllOpCPUKernel<int>,
                       ops::AllToAllOpCPUKernel<int64_t>,
                       ops::AllToAllOpCPUKernel<plat::float16>);
