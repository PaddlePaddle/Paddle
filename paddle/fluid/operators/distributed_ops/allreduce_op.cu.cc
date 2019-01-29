/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <future>  // NOLINT
#include <ostream>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/nccl_helper.h"

namespace paddle {
namespace operators {

class AllReduceOp : public framework::OperatorBase {
  using OperatorBase::OperatorBase;

  void RunImpl(const framework::Scope& scope,
               const platform::Place& place) const override {
    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    auto* ctx = pool.Get(place);
    auto in_names = Inputs("X");
    auto out_names = Outputs("Out");
    PADDLE_ENFORCE_EQ(in_names.size(), 1);
    PADDLE_ENFORCE_EQ(out_names.size(), 1);

    auto* in = scope.FindVar(in_names[0]);
    auto* out = scope.FindVar(out_names[0]);

    PADDLE_ENFORCE(in->IsType<framework::Tensor>() ||
                   in->IsType<framework::LoDTensor>());

    int dtype = -1;
    auto in_tensor = in->Get<framework::Tensor>();
    dtype = platform::ToNCCLDataType(in_tensor.type());

    int64_t numel = in_tensor.numel();
    auto* sendbuff = in_tensor.data<void>();
    auto* out_tensor = out->GetMutable<framework::Tensor>();
    auto* recvbuff =
        static_cast<void*>(out_tensor->mutable_data<uint8_t>(place));

    auto cuda_ctx = static_cast<platform::CUDADeviceContext*>(ctx);
    auto* comm = cuda_ctx->nccl_comm();
    auto stream = cuda_ctx->stream();

    PADDLE_ENFORCE(platform::dynload::ncclAllReduce(
        sendbuff, recvbuff, numel, static_cast<ncclDataType_t>(dtype), ncclSum,
        comm, stream));
  }
};

class AllReduceOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "(Tensor), tensor to be allreduced.");
    AddOutput("Out", "(Tensor) the result of allreduced");
    AddComment(R"DOC(
***AllReduce Operator***

Call NCCL AllReduce internally. Note that this op must be used when one
thread is managing one GPU device.

If input and output are the same variable, in-place allreduce will be used.
)DOC");
  }
};

class AllReduceOpShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext* ctx) const override {}
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(allreduce, ops::AllReduceOp,
                  paddle::framework::EmptyGradOpMaker, ops::AllReduceOpMaker,
                  ops::AllReduceOpShapeInference);
