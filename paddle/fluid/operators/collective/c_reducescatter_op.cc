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
#include "paddle/fluid/operators/collective/c_reducescatter_op.h"

namespace paddle{
namespace operators{

class CReduceScatterOp:public framework::OperatorWithKernel{
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext *ctx) const override{
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                                 "Output(Out) of SyncFCGather op should not be null.");
    int nranks = ctx->Attrs().Get<int>("nranks");
    auto in_dim = ctx->GetInputDim("X");
    in_dim[0] = in_dim[0]/nranks;
    ctx->SetOutputDim("Out", in_dim);
  }
};

class CReduceScatterOpMaker: public framework::OpProtoAndCheckerMaker{
 public:
  void Make(){
   AddInput("X","(Tensor) tensor to be allgather");
   AddOutput("Out","(Tensor) the allgather result");
   AddAttr<int>("ring_id", "(int) communication ring id.").SetDefault(0);
   AddAttr<int>("nranks","Total trainer count of the distributed training job").SetDefault(1);
   AddComment(R"DOC(
***CAllGather Operator***

Call NCCL collective  AllGather internally. Note that this op must be used when one
thread is managing one GPU device.
)DOC");
  }
};

} //namespace operators
} //namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_WITHOUT_GRADIENT(c_reducescatter, ops::CReduceScatterOp,
                             ops::CReduceScatterOpMaker);
REGISTER_OP_CPU_KERNEL(
    c_reducescatter, ops::CReduceScatterOpKernel<plat::CPUDeviceContext, float>,
    ops::CReduceScatterOpKernel<plat::CPUDeviceContext, double>,
    ops::CReduceScatterOpKernel<plat::CPUDeviceContext, int>,
    ops::CReduceScatterOpKernel<plat::CPUDeviceContext, int64_t>,
    ops::CReduceScatterOpKernel<plat::CPUDeviceContext, plat::float16>);

