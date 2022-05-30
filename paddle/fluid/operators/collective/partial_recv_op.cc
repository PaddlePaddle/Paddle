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

#include "paddle/fluid/operators/collective/partial_recv_op.h"
#include <string>

namespace paddle {
namespace operators {

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
        peer, 0,
        platform::errors::InvalidArgument(
            "The peer (%d) for partial_recv op must be non-negative.", peer));
    PADDLE_ENFORCE_GE(
        ring_id, 0,
        platform::errors::InvalidArgument(
            "The ring_id (%d) for partial_recv op must be non-negative.",
            ring_id));
    PADDLE_ENFORCE_GE(num, 1,
                      platform::errors::InvalidArgument(
                          "The num (%d) for partial_send op must >=1", num));
    PADDLE_ENFORCE_EQ(
        (id >= 0 && id < num), true,
        platform::errors::InvalidArgument(
            "The id (%d) for partial_send op must >=0 and <num (%d)", id, num));
    PADDLE_ENFORCE_GE(out_shape.size(), 1,
                      platform::errors::InvalidArgument(
                          "The size of the output shape must be greater than 0 "
                          "but the value given is %d.",
                          out_shape.size()));

    for (size_t i = 0; i < out_shape.size(); ++i) {
      PADDLE_ENFORCE_GE(out_shape[i], 1,
                        platform::errors::InvalidArgument(
                            "The shape attribute for partial_recv must be set "
                            "explicitly, but the %dth element is %d which "
                            "is less than 1.",
                            i, out_shape[i]));
    }
    auto out_dims = phi::make_ddim(out_shape);
    int numel = phi::product(out_dims);
    PADDLE_ENFORCE_EQ(
        (numel % num), 0,
        platform::errors::InvalidArgument(
            "The output numel (%d) must be divisible by num(%d)", numel, num));

    ctx->SetOutputDim("Out", phi::make_ddim(out_shape));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    int dtype = ctx.Attr<int>("dtype");
    framework::proto::VarType::Type type =
        framework::proto::VarType::Type(dtype);
    return framework::OpKernelType(type, ctx.GetPlace());
  }
};

class PartialRecvOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddOutput("Out", "(Tensor) tensor to receive.");
    AddAttr<int>("ring_id", "(int default 0) nccl communication ring id.")
        .SetDefault(0);
    AddAttr<int>("peer", "(int default 0) rank id for sender.").SetDefault(0);
    AddAttr<int>("dtype", "(int default 5('float32')) data type of tensor.")
        .SetDefault(5);
#if defined(PADDLE_WITH_ASCEND_CL)
    AddAttr<std::string>("tag", "(string default tag) tag for broadcasting.")
        .SetDefault("tag");
    AddAttr<int>("srTag", "(string default tag) tag for broadcasting.")
        .SetDefault(0);
#endif
    AddAttr<std::vector<int>>("out_shape", "shape of the output tensor.")
        .SetDefault(std::vector<int>());
    AddAttr<bool>(
        "use_calc_stream",
        "(bool default false) eject CUDA operations to calculation stream.")
        .SetDefault(false);
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

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_WITHOUT_GRADIENT(partial_recv, ops::PartialRecvOp,
                             ops::PartialRecvOpMaker);

REGISTER_OP_CPU_KERNEL(partial_recv, ops::PartialRecvOpCPUKernel<float>,
                       ops::PartialRecvOpCPUKernel<double>,
                       ops::PartialRecvOpCPUKernel<int>,
                       ops::PartialRecvOpCPUKernel<int64_t>,
                       ops::PartialRecvOpCPUKernel<plat::float16>);
