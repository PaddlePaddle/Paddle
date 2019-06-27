/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <future>  // NOLINT
#include <ostream>

#include "paddle/fluid/operators/collective/c_allreduce_op.h"

namespace paddle {
namespace operators {

class CAllReduceOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(ctx.Input<framework::Tensor>("X")->type(),
                                   ctx.GetPlace());
  }
};

class CAllReduceOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "(Tensor), tensor to be allreduced.");
    AddOutput("Out", "(Tensor) the allreduced result.");
    AddAttr<int>("reduce_type", "(int default 0) determin the reduce type.")
        .SetDefault(0);
    AddAttr<int>("ring_id", "(int default 0) communication ring id.")
        .SetDefault(0);
    AddAttr<bool>(
        "use_calc_stream",
        "(bool default false) eject CUDA operations to calculation stream.")
        .SetDefault(false);
    AddComment(R"DOC(
***CAllReduce Operator***

Call NCCL collective AllReduce internally. Note that this op must be used when one
thread is managing one GPU device.

For speed reasons, reduce_type should be an integer:

0: sum
1: prod
2: max
3: min 
If input and output are the same variable, in-place allreduce will be used.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_WITHOUT_GRADIENT(c_allreduce, ops::CAllReduceOp,
                             ops::CAllReduceOpMaker);

REGISTER_OP_CPU_KERNEL(
    c_allreduce, ops::CAllReduceOpKernel<plat::CPUDeviceContext, float>,
    ops::CAllReduceOpKernel<plat::CPUDeviceContext, double>,
    ops::CAllReduceOpKernel<plat::CPUDeviceContext, int>,
    ops::CAllReduceOpKernel<plat::CPUDeviceContext, int64_t>,
    ops::CAllReduceOpKernel<plat::CPUDeviceContext, plat::float16>);
