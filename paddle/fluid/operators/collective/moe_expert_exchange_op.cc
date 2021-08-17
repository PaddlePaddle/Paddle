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

#include "paddle/fluid/operators/collective/moe_expert_exchange_op.h"

namespace paddle {
namespace operators {

class MoeExpertExchangeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("local_expert_count"), "Input",
                   "local_expert_count", "MoeExpertExchange");
    // OP_INOUT_CHECK(ctx->HasInput("local_expert_count"), "Input",
    // "local_expert_count", "SelectScatter");
    // OP_INOUT_CHECK(ctx->HasInput("global_expert_count"), "Input",
    // "global_expert_count", "SelectScatter");
    // OP_INOUT_CHECK(ctx->HasInput("global_expert_count"), "Input",
    // "global_expert_count", "MoeExpertExchange");
    // OP_INOUT_CHECK(ctx->HasInput("in_feat"), "Input", "in_feat",
    // "SelectScatter");
    // OP_INOUT_CHECK(ctx->HasInput("n_expert"), "Input", "n_expert",
    // "SelectScatter");
    // OP_INOUT_CHECK(ctx->HasInput("world_size"), "Input", "world_size",
    // "SelectScatter");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "MoeExpertExchange");
    int ring_id = ctx->Attrs().Get<int>("ring_id");
    PADDLE_ENFORCE_GE(
        ring_id, 0,
        platform::errors::InvalidArgument(
            "The ring_id (%d) for alltoall op must be non-negative.", ring_id));
    framework::DDim dim = ctx->GetInputDim("local_expert_count");
    if (dim[0] < 0) dim[0] = -1;
    ctx->SetOutputDim("Out", dim);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "local_expert_count"),
        ctx.GetPlace());
  }
};

class MoeExpertExchangeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("local_expert_count", "(Tensor) tensor send.");
    // AddInput("global_expert_count", "(Tensor) tensor send.");
    // AddAttr<std::vector<int>>("local_expert_count", "The shape of the
    // output");
    // AddAttr<std::vector<int>>("global_expert_count", "The shape of the
    // output");
    // AddInput("local_expert_count", "(Tensor) tensor send.");
    // AddInput("global_expert_count", "(Tensor) tensor send.");
    // AddInput("input_buf", "(Tensor) tensor send.");
    // AddInput("in_feat", "(Tensor) tensor send.");
    // AddInput("n_expert", "(Tensor) tensor send.");
    // AddInput("world_size", "(Tensor) tensor send.");
    // AddAttr<int>("in_feat", "(int default 0) nccl communication ring id.")
    //     .SetDefault(0);
    AddAttr<int>("n_expert", "(int default 0) nccl communication ring id.")
        .SetDefault(0);
    AddAttr<int>("world_size", "(int default 0) nccl communication ring id.")
        .SetDefault(0);
    AddOutput("Out", "(Tensor) the result of alltoall.");

    AddAttr<int>("ring_id", "(int default 0) nccl communication ring id.")
        .SetDefault(0);
    AddAttr<bool>(
        "use_calc_stream",
        "(bool default false) eject CUDA operations to calculation stream.")
        .SetDefault(false);
    AddComment(R"DOC(
MoeExpertExchange Operator
Scatter tensors from all participators to all participators.
)DOC");
  }
};

// DECLARE_INPLACE_OP_INFERER(AllToAllInplaceInferer, {"X", "Out"});

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_WITHOUT_GRADIENT(moe_expert_exchange, ops::MoeExpertExchangeOp,
                             ops::MoeExpertExchangeOpMaker);
// REGISTER_OPERATOR(moe_expert_exchange, ops::MoeExpertExchangeOp,
// ops::MoeExpertExchangeOpMaker,
//                   ops::MoeExpertExchangeOpGradMaker<paddle::framework::OpDesc>,
//                   ops::MoeExpertExchangeOpGradMaker<paddle::imperative::OpBase>)

REGISTER_OP_CPU_KERNEL(moe_expert_exchange,
                       ops::MoeExpertExchangeOpCPUKernel<float>,
                       ops::MoeExpertExchangeOpCPUKernel<double>,
                       ops::MoeExpertExchangeOpCPUKernel<int>,
                       ops::MoeExpertExchangeOpCPUKernel<int64_t>,
                       ops::MoeExpertExchangeOpCPUKernel<plat::float16>);
