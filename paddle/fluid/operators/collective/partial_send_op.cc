/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle::operators {

class PartialSendOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "PartialSend");
    int peer = ctx->Attrs().Get<int>("peer");
    int ring_id = ctx->Attrs().Get<int>("ring_id");
    int num = ctx->Attrs().Get<int>("num");
    int id = ctx->Attrs().Get<int>("id");

    PADDLE_ENFORCE_GE(
        peer,
        0,
        common::errors::InvalidArgument(
            "The peer (%d) for partial_send op must be non-negative.", peer));
    PADDLE_ENFORCE_GE(
        ring_id,
        0,
        common::errors::InvalidArgument(
            "The ring_id (%d) for partial_send op must be non-negative.",
            ring_id));
    PADDLE_ENFORCE_GE(num,
                      1,
                      common::errors::InvalidArgument(
                          "The num (%d) for partial_send op must >=1", num));
    PADDLE_ENFORCE_EQ(
        (id >= 0 && id < num),
        true,
        common::errors::InvalidArgument(
            "The id (%d) for partial_send op must >=0 and <num (%d)", id, num));
  }

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return phi::KernelKey(OperatorWithKernel::IndicateVarDataType(ctx, "X"),
                          ctx.GetPlace());
  }
};

class PartialSendMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) tensor to be sent.");
    AddAttr<int>("ring_id", "(int default 0) nccl communication ring id.")
        .SetDefault(0);
    AddAttr<int>("peer", "(int default 0) rank id for receiver.").SetDefault(0);

    AddAttr<int>("num", "(int default 1) The number of Input to be cut.")
        .SetDefault(1);
    AddAttr<int>("id",
                 "(int default 0) ID of the part to be sent after Input cut.")
        .SetDefault(0);
    AddComment(R"DOC(
PartialSend Operator.
Divide the Input into num copies and only send the id part.

Reference: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html#sendrecv
)DOC");
  }
};

}  // namespace paddle::operators

namespace ops = paddle::operators;

REGISTER_OP_WITHOUT_GRADIENT(partial_send,
                             ops::PartialSendOp,
                             ops::PartialSendMaker);
