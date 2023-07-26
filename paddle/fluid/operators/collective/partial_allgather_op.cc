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

#include "paddle/fluid/operators/collective/partial_allgather_op.h"

namespace paddle {
namespace operators {

class PartialAllGatherOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "PartialAllGather");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Input", "Out", "PartialAllGather");
    int nranks = ctx->Attrs().Get<int>("nranks");
    int rank = ctx->Attrs().Get<int>("rank");

    PADDLE_ENFORCE_GE(nranks,
                      2,
                      platform::errors::InvalidArgument(
                          "The value of nranks should be >=2."));
    PADDLE_ENFORCE_EQ(
        (rank >= 0 && rank < nranks),
        true,
        platform::errors::InvalidArgument(
            "The rank (%d) for partial_allgather op must >=0 and <nranks (%d)",
            rank,
            nranks));

    framework::DDim dim = ctx->GetInputDim("X");
    ctx->SetOutputDim("Out", dim);
  }
};

class PartialAllGatherOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) tensor to be partial allgather");
    AddOutput("Out", "(Tensor) the allgather result");
    AddAttr<int>("ring_id", "(int default 0) communication ring id.")
        .SetDefault(0);

    AddAttr<bool>(
        "use_calc_stream",
        "(bool default false) eject CUDA operations to calculation stream.")
        .SetDefault(false);
    AddAttr<int>("nranks",
                 "Total trainer count of the distributed training job");
    AddAttr<int>("rank", "Rand of the distributed training job");
    AddComment(R"DOC(
PartialAllGather Operator.
Divide the Input into nranks copies and only use the rank part.
Each rank receives the aggregation of data from all ranks in the order of the ranks.


reference: https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/usage/operations.html#allgather
)DOC");
  }
};

DECLARE_INPLACE_OP_INFERER(PartialAllGatherOpInplaceInferer, {"X", "Out"});

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OPERATOR(
    partial_allgather,
    ops::PartialAllGatherOp,
    ops::PartialAllGatherOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    ops::PartialAllGatherOpInplaceInferer)

PD_REGISTER_STRUCT_KERNEL(partial_allgather,
                          CPU,
                          ALL_LAYOUT,
                          ops::PartialAllGatherOpCPUKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          plat::float16) {}
