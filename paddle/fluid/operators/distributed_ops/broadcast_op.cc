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

#include <algorithm>
#include <ostream>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

class BroadcastOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of BroadcastOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Output) of ConvOp should not be null.");
  }
};

class BroadcastOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "(Tensor), tensor to be broadcast.");
    AddOutput("Out", "(Tensor) the result of broadcast.");
    AddAttr<bool>(
        "sync_mode",
        "(bool) whether to synchronize the CUDA stream after nccl call.")
        .SetDefault(false);
    AddAttr<int>("root", "(int).").SetDefault(0).EqualGreaterThan(0);
    AddComment(R"DOC(
***Broadcast Operator***

Call NCCL Broadcast internally. Note that this op must be used when one
thread is managing one GPU device.
)DOC");
  }
};

template <typename T>
class BroadcastOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_THROW("Broadcast op can run on gpu place only for now.");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_WITHOUT_GRADIENT(broadcast, ops::BroadcastOp,
                             ops::BroadcastOpMaker);

REGISTER_OP_CPU_KERNEL(broadcast, ops::BroadcastOpKernel<float>,
                       ops::BroadcastOpKernel<double>,
                       ops::BroadcastOpKernel<int>,
                       ops::BroadcastOpKernel<int64_t>,
                       ops::BroadcastOpKernel<plat::float16>);
