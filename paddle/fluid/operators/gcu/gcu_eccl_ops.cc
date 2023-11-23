//   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef PADDLE_WITH_GCU

#include <algorithm>
#include <ostream>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

template <typename T>
class EcclBroadcastOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {}
};

class EcclBroadcastOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    ctx->SetOutputsDim("OutputList", ctx->GetInputsDim("InputList"));
  }
};

class EcclBroadcastOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("InputList", "(Tensor List), tensors to be broadcast.");
    AddOutput("OutputList", "(Tensor List) the results of broadcast.");
    AddAttr<bool>(
        "sync_mode",
        "(bool) whether to synchronize the eccl stream after eccl call.")
        .SetDefault(true);
    AddAttr<int>("root_rank", "(int).").SetDefault(0).EqualGreaterThan(0);
    AddComment(R"DOC(
***Broadcast Operator***

Call ECCL Broadcast internally. Note that this op must be used when one
thread is managing one GCU device.
)DOC");
  }
};

template <typename T>
class EcclAllReduceOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {}
};

class EcclAllReduceOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    ctx->SetOutputsDim("OutputList", ctx->GetInputsDim("InputList"));
  }

 protected:
  //   framework::OpKernelType GetExpectedKernelType(
  //       const framework::ExecutionContext& ctx) const override {
  //     return framework::OpKernelType(
  //         OperatorWithKernel::IndicateVarDataType(ctx, "InputList"),
  //         ctx.GetPlace());
  //  }
};

class EcclAllReduceOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("InputList", "(Tensor List), tensors to be allreduced.");
    AddOutput("OutputList", "(Tensor List) the results of allreduced.");
    AddAttr<int>("reduce_type", "(int) determin the reduce type.")
        .SetDefault(0);
    AddAttr<bool>(
        "sync_mode",
        "(bool) whether to synchronize the eccl stream after eccl call.")
        .SetDefault(true);
    AddComment(R"DOC(
***AllReduce Operator***

Call ECCL AllReduce internally. Note that this op must be used when one
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

REGISTER_OP_WITHOUT_GRADIENT(eccl_broadcast,
                             ops::EcclBroadcastOp,
                             ops::EcclBroadcastOpMaker);

REGISTER_OP_CPU_KERNEL(eccl_broadcast,
                       ops::EcclBroadcastOpKernel<float>,
                       ops::EcclBroadcastOpKernel<double>,
                       ops::EcclBroadcastOpKernel<int>,
                       ops::EcclBroadcastOpKernel<int64_t>,
                       ops::EcclBroadcastOpKernel<plat::float16>);

REGISTER_OP_WITHOUT_GRADIENT(eccl_allreduce,
                             ops::EcclAllReduceOp,
                             ops::EcclAllReduceOpMaker);

REGISTER_OP_CPU_KERNEL(eccl_allreduce,
                       ops::EcclAllReduceOpKernel<float>,
                       ops::EcclAllReduceOpKernel<double>,
                       ops::EcclAllReduceOpKernel<int>,
                       ops::EcclAllReduceOpKernel<int64_t>,
                       ops::EcclAllReduceOpKernel<plat::float16>);

#endif  // PADDLE_WITH_GCU
