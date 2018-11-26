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
#include "paddle/fluid/operators/math/concat_and_split.h"
#include "paddle/fluid/platform/nccl_helper.h"

namespace paddle {
namespace operators {

template <typename T>
class SyncFCGatherKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto in_tensor = ctx.Input<framework::LoDTensor>("X");
    auto out_tensor = ctx.Output<framework::LoDTensor>("Out");
    auto place = boost::get<platform::CUDAPlace>(ctx.GetPlace());
    int dev_id = place.device;
    auto &nccl_map = platform::NCCLContextMap::Instance();
    auto &nccl_ctx = nccl_map.at(dev_id);
    int nccl_nranks = nccl_map.nranks_;
    int nccl_rank = nccl_ctx.rank_;
    auto stream = nccl_ctx.stream();
    auto comm = nccl_ctx.comm_;

    int dtype = platform::ToNCCLDataType(in_tensor->type());
    framework::Tensor tmp_tensor;
    tmp_tensor.Resize(in_tensor->dims());
    tmp_tensor.mutable_data(ctx.GetPlace(), in_tensor->type());

    // resize output tensor by concat with axis=1
    auto in_dim = in_tensor->dims();
    in_dim[0] = in_dim[0] / nccl_nranks;
    in_dim[1] = in_dim[1] * nccl_nranks;
    out_tensor->Resize(in_dim);
    out_tensor->mutable_data(ctx.GetPlace(), in_tensor->type());

    PADDLE_ENFORCE_EQ(
        in_tensor->numel() % nccl_nranks, 0,
        "The numel of in tensor should be integer multiple of nccl nranks.");
    int64_t shard_numel = in_tensor->numel() / nccl_nranks;

    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto &compute_ctx = static_cast<const platform::CUDADeviceContext &>(
        *pool.Get(ctx.GetPlace()));

    for (int i = 0; i < nccl_nranks; ++i) {
      const float *send_buff = in_tensor->data<float>() + shard_numel * i;
      float *recv_buff = tmp_tensor.data<float>();
      PADDLE_ENFORCE(platform::dynload::ncclAllGather(
          send_buff, recv_buff, shard_numel, static_cast<ncclDataType_t>(dtype),
          comm, stream));
      nccl_ctx.ctx_->Wait();
      if (i == nccl_rank) {
        std::vector<framework::Tensor> inputs;
        for (int shard_idx = 0; shard_idx < nccl_nranks; ++shard_idx) {
          int begin_idx = shard_idx * (tmp_tensor.dims()[0] / nccl_nranks);
          int end_idx = (shard_idx + 1) * (tmp_tensor.dims()[0] / nccl_nranks);
          // Tensor Slice doesn't copy data, just easy to reuse the
          // concat_and_split kernel
          inputs.push_back(tmp_tensor.Slice(begin_idx, end_idx));
        }
        paddle::operators::math::ConcatFunctor<platform::CUDADeviceContext, T>
            concat_functor;
        concat_functor(compute_ctx, inputs, 1, out_tensor);
      }
    }
  }
};

template <typename T>
class SyncFCGatherGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto in_tensor =
        ctx.Input<framework::LoDTensor>(framework::GradVarName("Out"));
    auto out_tensor =
        ctx.Output<framework::LoDTensor>(framework::GradVarName("X"));
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    platform::NCCLContextMap &nccl_map = platform::NCCLContextMap::Instance();
    auto place = boost::get<platform::CUDAPlace>(ctx.GetPlace());
    int dev_id = place.device;

    auto &nccl_ctx = nccl_map.at(dev_id);
    auto stream = nccl_ctx.stream();
    auto comm = nccl_ctx.comm_;
    int nccl_nranks = nccl_map.nranks_;
    int nccl_rank = nccl_ctx.rank_;
    int dtype = platform::ToNCCLDataType(in_tensor->type());

    auto &compute_ctx = static_cast<const platform::CUDADeviceContext &>(
        *pool.Get(ctx.GetPlace()));

    auto out_dims = in_tensor->dims();
    out_dims[0] = out_dims[0] * nccl_nranks;
    out_dims[1] = out_dims[1] / nccl_nranks;
    out_tensor->Resize(out_dims);
    out_tensor->mutable_data<T>(ctx.GetPlace());

    std::vector<const framework::Tensor *> outs_const_ref;
    std::vector<framework::Tensor *> outs_ref;
    auto sliced_dims = in_tensor->dims();
    sliced_dims[1] = sliced_dims[1] / nccl_nranks;

    framework::Tensor split_tmp;
    split_tmp.mutable_data<T>(out_dims, place);
    std::vector<framework::Tensor> split_outs;
    split_outs.reserve(nccl_nranks);
    for (int i = 0; i < nccl_nranks; ++i) {
      int begin_idx = i * (split_tmp.dims()[0] / nccl_nranks);
      int end_idx = (i + 1) * (split_tmp.dims()[0] / nccl_nranks);
      split_outs.push_back(split_tmp.Slice(begin_idx, end_idx));
    }
    for (int i = 0; i < nccl_nranks; ++i) {
      outs_const_ref.push_back(&split_outs[i]);
      outs_ref.push_back(&split_outs[i]);
    }

    framework::Tensor tmp;
    tmp.mutable_data<T>(out_dims, place);
    paddle::operators::math::SplitFunctor<platform::CUDADeviceContext, T>
        split_functor;
    split_functor(compute_ctx, *in_tensor, outs_const_ref, 1, &outs_ref);
    for (int i = 0; i < nccl_nranks; ++i) {
      auto *send_tensor = outs_ref[i];
      const void *send_buff = send_tensor->data<void>();
      size_t send_count = static_cast<size_t>(send_tensor->numel());
      void *recv_buff = tmp.data<void>();
      PADDLE_ENFORCE(platform::dynload::ncclAllGather(
          send_buff, recv_buff, send_count, static_cast<ncclDataType_t>(dtype),
          comm, stream));
      nccl_ctx.ctx_->Wait();
      if (i == nccl_rank) {
        memory::Copy(place, out_tensor->mutable_data<T>(place), place,
                     tmp.data<T>(), tmp.numel() * sizeof(T),
                     compute_ctx.stream());
        compute_ctx.Wait();
      }
    }
  }
};

class SyncFCGatherOpMaker : public framework::OpProtoAndCheckerMaker {
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

class SyncFCGatherOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of SyncFCGather op should not be null.");
    int trainers = ctx->Attrs().Get<int>("trainers");
    auto in_dim = ctx->GetInputDim("X");
    in_dim[0] = in_dim[0] / trainers / platform::GetCUDADeviceCount();
    in_dim[1] = in_dim[1] * trainers * platform::GetCUDADeviceCount();
    ctx->SetOutputDim("Out", in_dim);
  }
};

class SyncFCGatherOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *ctx) const override {
    auto in_out_g = framework::GradVarName("Out");
    auto out_x_g = framework::GradVarName("X");
    int trainers = ctx->Attrs().Get<int>("trainers");
    PADDLE_ENFORCE(ctx->HasInput(in_out_g),
                   "Input(GRAD@Out) should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput(out_x_g),
                   "output(GRAD@X) should not be null.");
    auto in_dim = ctx->GetInputDim(in_out_g);
    in_dim[0] = in_dim[0] * trainers * platform::GetCUDADeviceCount();
    in_dim[1] = in_dim[1] / trainers / platform::GetCUDADeviceCount();
    ctx->SetOutputDim(out_x_g, in_dim);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(sync_fc_gather, ops::SyncFCGatherOp, ops::SyncFCGatherOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<false>);
REGISTER_OPERATOR(sync_fc_gather_grad, ops::SyncFCGatherOpGrad);

REGISTER_OP_CUDA_KERNEL(sync_fc_gather, ops::SyncFCGatherKernel<float>);
REGISTER_OP_CUDA_KERNEL(sync_fc_gather_grad,
                        ops::SyncFCGatherGradKernel<float>);
