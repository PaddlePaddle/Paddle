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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/nccl_helper.h"

namespace paddle {
namespace operators {

template <typename T>
class SyncFCAllGatherKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto x_tensor = ctx.Input<framework::LoDTensor>("X");
    auto out_tensor = ctx.Output<framework::LoDTensor>("Out");

    auto &place = boost::get<platform::CUDAPlace>(ctx.GetPlace());
    int gpu_id = place.GetDeviceId();
    auto &nccl_map = platform::NCCLContextMap::Instance();
    int nccl_nranks = nccl_map.nranks_;
    auto &nccl_ctx = nccl_map.at(gpu_id);

    auto out_dims = x_tensor->dims();
    out_dims[0] *= static_cast<int64_t>(nccl_nranks);
    out_tensor->mutable_data<T>(out_dims, place);

    auto stream = nccl_ctx.stream();
    auto comm = nccl_ctx.comm_;
    int64_t send_count = x_tensor->numel();

    const T *send_buff = x_tensor->data<T>();
    T *recv_buff = out_tensor->data<T>();
    int dtype = platform::ToNCCLDataType(x_tensor->type());

    PADDLE_ENFORCE(platform::dynload::ncclAllGather(
        send_buff, recv_buff, send_count, static_cast<ncclDataType_t>(dtype),
        comm, stream));
    nccl_ctx.ctx_->Wait();
  }
};

template <typename T>
class SyncFCAllGatherGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto x_tensor =
        ctx.Input<framework::LoDTensor>(framework::GradVarName("Out"));
    auto out_tensor =
        ctx.Output<framework::LoDTensor>(framework::GradVarName("X"));

    auto &place = boost::get<platform::CUDAPlace>(ctx.GetPlace());
    int gpu_id = place.GetDeviceId();
    auto &nccl_map = platform::NCCLContextMap::Instance();
    auto &nccl_ctx = nccl_map.at(gpu_id);
    int nccl_nranks = nccl_map.nranks_;
    int nccl_rank = nccl_ctx.rank_;

    auto stream = nccl_ctx.stream();
    auto comm = nccl_ctx.comm_;

    auto out_dims = x_tensor->dims();
    out_dims[0] = out_dims[0] / static_cast<int64_t>(nccl_nranks);
    out_tensor->mutable_data<T>(out_dims, place);
    int64_t send_count = out_tensor->numel();

    const T *send_buff = x_tensor->data<T>() + nccl_rank * send_count;
    T *recv_buff = out_tensor->data<T>();
    int dtype = platform::ToNCCLDataType(x_tensor->type());
    PADDLE_ENFORCE(platform::dynload::ncclReduce(
        send_buff, recv_buff, send_count, static_cast<ncclDataType_t>(dtype),
        ncclSum, nccl_rank, comm, stream));
    nccl_ctx.ctx_->Wait();
  }
};

class SyncFCAllGatherOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(LoDTensor) Input tensor of sync fc allgather operator.");
    AddOutput("Out",
              "(LoDTensor) Output tensor of sync fc allgather operator.");
    AddAttr<int>("trainers",
                 "Total trainer count of the distributed training job.")
        .SetDefault(1);
    AddComment(R"DOC(
All gather operator of sync fc operator
)DOC");
  }
};

class SyncFCAllGatherOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null.");

    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of SumOp should not be null.");
    auto in_dim = ctx->GetInputDim("X");
    auto dev_cnt = platform::GetCUDADeviceCount();
    int trainers = ctx->Attrs().Get<int>("trainers");
    in_dim[0] = in_dim[0] * trainers * dev_cnt;
    ctx->SetOutputDim("Out", in_dim);
  }
};

class SyncFCAllGatherOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Grad(Out)) should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("X")),
                   "Output(Grad(X)) should not be null.");
    auto in_dim = ctx->GetInputDim(framework::GradVarName("Out"));
    int trainers = ctx->Attrs().Get<int>("trainers");
    auto dev_cnt = platform::GetCUDADeviceCount();

    in_dim[0] = in_dim[0] / trainers / dev_cnt;
    ctx->SetOutputDim(framework::GradVarName("X"), in_dim);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(sync_fc_allgather, ops::SyncFCAllGatherOp,
                  ops::SyncFCAllGatherOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(sync_fc_allgather_grad, ops::SyncFCAllGatherOpGrad);

REGISTER_OP_CUDA_KERNEL(sync_fc_allgather, ops::SyncFCAllGatherKernel<float>);
REGISTER_OP_CUDA_KERNEL(sync_fc_allgather_grad,
                        ops::SyncFCAllGatherGradKernel<float>);
