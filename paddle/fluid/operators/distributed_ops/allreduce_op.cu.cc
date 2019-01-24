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
  // inherit op base constructor.
  using OperatorBase::OperatorBase;

  void RunImpl(const framework::Scope& scope,
               const platform::Place& place) const override {
    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    auto& ctx = *pool.Get(place);
    auto* in = Input("X");
    auto* out = Output("Out");
    PADDLE_ENFORCE(in->IsType<>(framework::Tensor) ||
                   in->IsType<>(framework::LoDTensor));

    auto in_tensor = in->Get<Tensor>();
    int64_t numel = in_tensor.numel();
    auto* sendbuff = in_tensor.data<void>(place);
    auto* out_tensor = out->GetMutable<Tensor>();
    auto* recvbuff = out_tensor->mutable_data<void>(place);

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

REGISTER_OPERATOR(send, ops::AllReduceOp, paddle::framework::EmptyGradOpMaker,
                  ops::AllReduceOpMaker, ops::AllReduceOpShapeInference);
