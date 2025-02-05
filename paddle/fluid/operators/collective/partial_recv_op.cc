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

class PartialRecvOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "PartialRecv");
    int peer = ctx->Attrs().Get<int>("peer");
    int ring_id = ctx->Attrs().Get<int>("ring_id");
    int num = ctx->Attrs().Get<int>("num");
    int id = ctx->Attrs().Get<int>("id");
    auto out_shape = ctx->Attrs().Get<std::vector<int>>("out_shape");

    PADDLE_ENFORCE_GE(
        peer,
        0,
        common::errors::InvalidArgument(
            "The peer (%d) for partial_recv op must be non-negative.", peer));
    PADDLE_ENFORCE_GE(
        ring_id,
        0,
        common::errors::InvalidArgument(
            "The ring_id (%d) for partial_recv op must be non-negative.",
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
    PADDLE_ENFORCE_GE(out_shape.size(),
                      1,
                      common::errors::InvalidArgument(
                          "The size of the output shape must be greater than 0 "
                          "but the value given is %d.",
                          out_shape.size()));

    for (size_t i = 0; i < out_shape.size(); ++i) {
      PADDLE_ENFORCE_GE(out_shape[i],
                        1,
                        common::errors::InvalidArgument(
                            "The shape attribute for partial_recv must be set "
                            "explicitly, but the %dth element is %d which "
                            "is less than 1.",
                            i,
                            out_shape[i]));
    }
    auto out_dims = common::make_ddim(out_shape);
    int64_t numel = common::product(out_dims);
    PADDLE_ENFORCE_EQ(
        (numel % num),
        0,
        common::errors::InvalidArgument(
            "The output numel (%d) must be divisible by num(%d)", numel, num));

    ctx->SetOutputDim("Out", common::make_ddim(out_shape));
  }

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    int dtype = ctx.Attr<int>("dtype");
    framework::proto::VarType::Type type =
        framework::proto::VarType::Type(dtype);
    return phi::KernelKey(type, ctx.GetPlace());
  }
};

class PartialRecvOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddOutput("Out", "(Tensor) tensor to receive.");
    AddAttr<int>("ring_id", "(int default 0) nccl communication ring id.")
        .SetDefault(0);
    AddAttr<int>("peer", "(int default 0) rank id for sender.").SetDefault(0);
    AddAttr<int>("dtype", "(int default 5('float32')) data type of tensor.")
        .SetDefault(5);

    AddAttr<std::vector<int>>("out_shape", "shape of the output tensor.")
        .SetDefault(std::vector<int>());
    AddAttr<int>("num", "(int default 1) The number of Output to be cut.")
        .SetDefault(1);
    AddAttr<int>("id",
                 "(int default 0) ID of the part to be recv after Output cut.")
        .SetDefault(0);
    AddComment(R"DOC(
Recv Operator.
Divide the Output into num copies and only recv the id part.

Reference: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html#sendrecv
)DOC");
  }
};

}  // namespace paddle::operators

namespace ops = paddle::operators;

REGISTER_OP_WITHOUT_GRADIENT(partial_recv,
                             ops::PartialRecvOp,
                             ops::PartialRecvOpMaker);
