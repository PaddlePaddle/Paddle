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

#include "paddle/fluid/operators/collective/barrier_op.h"

#include <memory>

namespace paddle {
namespace operators {

class BarrierOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext *ctx) const override {}
};

class BarrierOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "(Tensor) Dependency of the variable need to sync");
    AddInput("Out", "(Tensor) Dependency of the variable need to sync");
    AddAttr<int>("ring_id", "(int default 0) communication ring id.")
        .SetDefault(0);
    AddComment(R"DOC(
Barrier Operator.
Barrier among all pariticapitors.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_WITHOUT_GRADIENT(barrier, ops::BarrierOp, ops::BarrierOpMaker);

REGISTER_OP_CPU_KERNEL(barrier, ops::BarrierOpCPUKernel<float>,
                       ops::BarrierOpCPUKernel<double>,
                       ops::BarrierOpCPUKernel<int>,
                       ops::BarrierOpCPUKernel<int64_t>,
                       ops::BarrierOpCPUKernel<plat::float16>);
