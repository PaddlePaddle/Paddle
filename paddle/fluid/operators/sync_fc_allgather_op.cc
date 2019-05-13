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

#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
#include "paddle/fluid/platform/nccl_helper.h"
#endif

namespace paddle {
namespace operators {

template <typename T>
class SyncFCAllGatherKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
#ifdef PADDLE_WITH_CUDA
    auto x_tensor = ctx.Input<framework::LoDTensor>("X");
    auto out_tensor = ctx.Output<framework::LoDTensor>("Out");

    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto &place = boost::get<platform::CUDAPlace>(ctx.GetPlace());
    auto stream = dev_ctx.stream();
    int nccl_nranks = dev_ctx.nranks();
    auto *comm = dev_ctx.nccl_comm();

    auto out_dims = x_tensor->dims();
    out_dims[0] *= static_cast<int64_t>(nccl_nranks);
    out_tensor->mutable_data<T>(out_dims, place);

    int64_t send_numel = x_tensor->numel();

    const T *send_buff = x_tensor->data<T>();
    T *recv_buff = out_tensor->data<T>();
    int dtype = platform::ToNCCLDataType(x_tensor->type());

    PADDLE_ENFORCE(platform::dynload::ncclAllGather(
        send_buff, recv_buff, send_numel, static_cast<ncclDataType_t>(dtype),
        comm, stream));
    PADDLE_ENFORCE(cudaStreamSynchronize(stream));
#else
    PADDLE_THROW("Paddle should compile with GPU.");
#endif
  }
};

template <typename T>
class SyncFCAllGatherGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
#ifdef PADDLE_WITH_CUDA
    auto x_tensor =
        ctx.Input<framework::LoDTensor>(framework::GradVarName("Out"));
    auto out_tensor =
        ctx.Output<framework::LoDTensor>(framework::GradVarName("X"));

    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto &place = boost::get<platform::CUDAPlace>(ctx.GetPlace());
    auto stream = dev_ctx.stream();
    int nccl_nranks = dev_ctx.nranks();
    auto *comm = dev_ctx.nccl_comm();

    auto out_dims = x_tensor->dims();
    out_dims[0] = out_dims[0] / static_cast<int64_t>(nccl_nranks);
    out_tensor->mutable_data<T>(out_dims, place);
    int64_t send_numel = out_tensor->numel();

    T *recv_buff = out_tensor->data<T>();
    for (int nccl_idx = 0; nccl_idx < nccl_nranks; ++nccl_idx) {
      const T *send_buff = x_tensor->data<T>() + nccl_idx * send_numel;
      int dtype = platform::ToNCCLDataType(x_tensor->type());
      PADDLE_ENFORCE(platform::dynload::ncclReduce(
          send_buff, recv_buff, send_numel, static_cast<ncclDataType_t>(dtype),
          ncclSum, nccl_idx, comm, stream));
    }
    PADDLE_ENFORCE(cudaStreamSynchronize(stream));
#else
    PADDLE_THROW("Paddle should compile with GPU.");
#endif
  }
};

class SyncFCAllGatherOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(LoDTensor) Input tensor of sync fc allgather operator.");
    AddOutput("Out",
              "(LoDTensor) Output tensor of sync fc allgather operator.");
    AddAttr<int>("nranks",
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
    int nranks = ctx->Attrs().Get<int>("nranks");
    in_dim[0] = in_dim[0] * nranks;
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
    int nranks = ctx->Attrs().Get<int>("nranks");
    in_dim[0] = in_dim[0] / nranks;
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
