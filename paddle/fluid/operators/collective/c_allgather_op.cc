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

#include "paddle/fluid/operators/collective/c_allgather_op.h"

#include <memory>

namespace paddle::operators {

class CAllGatherOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"),
                      true,
                      common::errors::PreconditionNotMet(
                          "Input 'X' of AllGather must be provided."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"),
                      true,
                      common::errors::PreconditionNotMet(
                          "Output 'Out' of AllGather must be provided."));
    int nranks = ctx->Attrs().Get<int>("nranks");
    PADDLE_ENFORCE_GE(
        nranks,
        2,
        common::errors::InvalidArgument("The value of nranks should be >=2."));
    phi::DDim dim = ctx->GetInputDim("X");
    // 0D use stack/unstack while others use concat/split
    if (dim.size() == 0) {
      dim = common::make_ddim({nranks});
    } else {
      dim[0] = dim[0] * nranks;
      if (dim[0] < 0) dim[0] = -1;
    }
    ctx->SetOutputDim("Out", dim);
  }
};

class CAllGatherOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) tensor to be allgather");
    AddOutput("Out", "(Tensor) the allgather result");
    AddAttr<int>("ring_id", "(int default 0) communication ring id.")
        .SetDefault(0);
    AddAttr<bool>(
        "use_calc_stream",
        "(bool default false) eject CUDA operations to calculation stream.")
        .SetDefault(false);
    AddAttr<int>("nranks",
                 "Total trainer count of the distributed training job");
    AddComment(R"DOC(
CAllGather Operator
each rank receives the aggregation of data from all ranks in the order of the ranks

reference: https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/usage/operations.html#allgather
)DOC");
  }
};

}  // namespace paddle::operators

namespace ops = paddle::operators;

REGISTER_OP_WITHOUT_GRADIENT(c_allgather,
                             ops::CAllGatherOp,
                             ops::CAllGatherOpMaker);
