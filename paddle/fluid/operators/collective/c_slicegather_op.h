/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <algorithm>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/operators/math/concat_and_split.h"

#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/nccl_helper.h"
#endif

namespace paddle {
namespace operators {
template <typename DeviceContext, typename T>
class CSliceGatherOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto place = ctx.GetPlace();
    PADDLE_ENFORCE(is_gpu_place(place),
                   "CSliceGather op only supports GPUKernel for now.");
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
    auto in_tensor = ctx.Input<framework::Tensor>("X");
    auto out_tensor = ctx.Output<framework::Tensor>("Out");
    int rid = ctx.Attr<int>("ring_id");
    auto comm = platform::NCCLCommContext::Instance().Get(rid);
    int nccl_nranks = comm->nranks();
    int local_rank = comm->rank();
    int dtype = platform::ToNCCLDataType(in_tensor->type());
    framework::Tensor tmp_tensor;
    tmp_tensor.Resize(in_tensor->dims());
    tmp_tensor.mutable_data(place, in_tensor->type());

    auto in_dim = in_tensor->dims();
    in_dim[0] = in_dim[0] / nccl_nranks;
    in_dim[1] = in_dim[1] * nccl_nranks;
    out_tensor->Resize(in_dim);
    out_tensor->mutable_data(place, in_tensor->type());
    PADDLE_ENFORCE_EQ(in_tensor->numel() % nccl_nranks, 0,
                      "The numel(%d) of in tensor should be integer multiple "
                      "of nccl nranks(%d).",
                      in_tensor->numel(), nccl_nranks);
    int64_t shard_numel = in_tensor->numel() / nccl_nranks;

    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto &compute_ctx = static_cast<const platform::CUDADeviceContext &>(
        pool->Get(ctx.GetPlace()));

    for (int i = 0; i < nccl_nranks; ++i) {
      const T *send_buff = in_tensor->data<T>() + shard_numel * i;
      T *recv_buff = tmp_tensor.data<T>();
      PADDLE_ENFORCE(platform::dynload::ncclAllGather(
          send_buff, recv_buff, shard_numel, static_cast<ncclDataType_t>(dtype),
          comm->comm(), comm->stream()));
      PADDLE_ENFORCE(cudaStreamSynchronize(comm->stream()));
      if (i == local_rank) {
        std::vector<framework::Tensor> inputs;
        for (int shard_idx = 0; shard_idx < nccl_nranks; ++shard_idx) {
          int begin_idx = shard_idx * (tmp_tensor.dims()[0] / nccl_nranks);
          int end_idx = (shard_idx + 1) * (tmp_tensor.dims()[0] / nccl_nranks);
          // Tensor Slice doesn't copy data, just reuse the
          // concat_and_split kernel
          inputs.push_back(tmp_tensor.Slice(begin_idx, end_idx));
        }
        paddle::operators::math::ConcatFunctor<platform::CUDADeviceContext, T>
            concat_functor;
        concat_functor(compute_ctx, inputs, 1, out_tensor);
        compute_ctx.Wait();
      }
    }
#else
    PADDLE_THROW(
        "PaddlePaddle should compile with GPU to use CSliceGather op.");
#endif
  }
};

template <typename T>
class CSliceGatherGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
    auto in_tensor =
        ctx.Input<framework::LoDTensor>(framework::GradVarName("Out"));
    auto out_tensor =
        ctx.Output<framework::LoDTensor>(framework::GradVarName("X"));
    int rid = ctx.Attr<int>("ring_id");
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto place = boost::get<platform::CUDAPlace>(ctx.GetPlace());
    auto comm = platform::NCCLCommContext::Instance().Get(rid);
    int nccl_nranks = comm->nranks();
    int local_rank = comm->rank();

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
          comm->comm(), comm->stream()));
      PADDLE_ENFORCE(cudaStreamSynchronize(comm->stream()));
      if (i == local_rank) {
        memory::Copy(place, out_tensor->mutable_data<T>(place), place,
                     tmp.data<T>(), tmp.numel() * sizeof(T),
                     compute_ctx.stream());
        compute_ctx.Wait();
      }
    }
#else
    PADDLE_THROW("Paddle should compile with GPU.");
#endif
  }
};

}  // namespace operators
}  // namespace paddle
