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

#include "paddle/fluid/operators/collective/c_barrier_op.h"
#include <memory>
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

class CBarrierOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "C_Barrier");
  }
};

class CBarrierOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "(Tensor) Input data used in NPUKernel.");
    AddAttr<int>("ring_id", "(int default 0) communication ring id.")
        .SetDefault(0);
    AddAttr<bool>(
        "use_calc_stream",
        "(bool default false) eject NPU operations to calculation stream.")
        .SetDefault(false);
    AddComment(R"DOC(
CBarrier Operator - Barrier among all pariticapitors.)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_WITHOUT_GRADIENT(c_barrier, paddle::operators::CBarrierOp,
                             paddle::operators::CBarrierOpMaker);

REGISTER_OP_CPU_KERNEL(c_barrier, ops::CBarrierOpCPUKernel<float>,
                       ops::CBarrierOpCPUKernel<double>);
