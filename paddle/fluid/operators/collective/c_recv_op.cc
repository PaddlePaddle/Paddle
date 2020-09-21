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

#include "paddle/fluid/operators/collective/c_recv_op.h"
#include <string>

namespace paddle {
namespace operators {

class CRecvOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "CRecv");
    int peer = ctx->Attrs().Get<int>("peer");
    int ring_id = ctx->Attrs().Get<int>("ring_id");
    PADDLE_ENFORCE_GE(
        peer, 0,
        platform::errors::InvalidArgument(
            "The peer (%d) for c_send_op must be non-negative.", peer));
    PADDLE_ENFORCE_GE(
        ring_id, 0,
        platform::errors::InvalidArgument(
            "The ring_id (%d) for c_send_op must be non-negative.", ring_id));
    auto out_shape = ctx->Attrs().Get<std::vector<int>>("out_shape");
    PADDLE_ENFORCE_GE(out_shape.size(), 1,
                      platform::errors::InvalidArgument(
                          "The size of the output shape must be greater than 0 "
                          "but the value given is %d.",
                          out_shape.size()));
    ctx->SetOutputDim("Out", framework::make_ddim(out_shape));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    VLOG(0) << "wow1";
    int dtype = ctx.Attr<int>("dtype");
    framework::proto::VarType::Type type;
    if (dtype == framework::proto::VarType::FP32) {
      type = framework::proto::VarType::FP32;
    } else if (dtype == framework::proto::VarType::FP64) {
      type = framework::proto::VarType::FP64;
    } else if (dtype == framework::proto::VarType::FP16) {
      type = framework::proto::VarType::FP16;
    } else if (dtype == framework::proto::VarType::INT32) {
      type = framework::proto::VarType::INT32;
    } else if (dtype == framework::proto::VarType::INT64) {
      type = framework::proto::VarType::INT64;
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Unknown data type %s for c_recv op.", dtype));
    }
    VLOG(0) << "wow2";
    return framework::OpKernelType(type, ctx.GetPlace());
    // OperatorWithKernel::IndicateVarDataType(ctx, "Out"), ctx.GetPlace());
  }
};

class CRecvOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddOutput("Out", "(Tensor) tensor to receive.");
    AddAttr<int>("ring_id", "(int default 0) nccl communication ring id.")
        .SetDefault(0);
    AddAttr<int>("peer", "(int default 0) rank id for sender.").SetDefault(0);
    AddAttr<int>("dtype",
                 "(std::string default 5(float32)) data type of tensor.")
        .SetDefault(5);
    AddAttr<std::vector<int>>("out_shape", "shape of the output tensor.")
        .SetDefault(std::vector<int>());
    AddAttr<bool>(
        "use_calc_stream",
        "(bool default false) eject CUDA operations to calculation stream.")
        .SetDefault(false);
    AddComment(R"DOC(
CRecv Operator

Reference: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html#sendrecv
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_WITHOUT_GRADIENT(c_recv, ops::CRecvOp, ops::CRecvOpMaker);

REGISTER_OP_CPU_KERNEL(c_recv, ops::CRecvOpCPUKernel<float>,
                       ops::CRecvOpCPUKernel<double>,
                       ops::CRecvOpCPUKernel<int>,
                       ops::CRecvOpCPUKernel<int64_t>,
                       ops::CRecvOpCPUKernel<plat::float16>);
