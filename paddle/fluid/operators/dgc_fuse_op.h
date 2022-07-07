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

#pragma once
#include <vector>

#include "dgc/dgc.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/operators/dgc_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/distributed/collective/ProcessGroup.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#endif

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class DGCFuseOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto grad_out = ctx.Output<framework::Tensor>("Grad_out");
    auto place = ctx.GetPlace();
    auto& dev_ctx = ctx.template device_context<DeviceContext>();

    // nranks
    auto nranks_tensor = ctx.Input<framework::Tensor>("nranks");
    const int nranks = static_cast<const int>(*nranks_tensor->data<float>());
    PADDLE_ENFORCE_GT(nranks, 1,
                      platform::errors::PreconditionNotMet(
                          "DGC is not useful when num_trainers <= 1. Please "
                          "use multi card or multi machine GPU"));
    // stream
    const int ring_id = ctx.Attr<int>("ring_id");
    auto comm = platform::NCCLCommContext::Instance().Get(ring_id, place);
    gpuStream_t stream = comm->stream();
    gpuStream_t compute_stream = dev_ctx.stream();
    ncclDataType_t dtype = platform::ToNCCLDataType(
        framework::TransToProtoVarType(grad_out->dtype()));

    bool is_use_dgc = ctx.Attr<bool>("is_use_dgc");
    if (!is_use_dgc) {
#ifdef PADDLE_WITH_HIP
      PADDLE_ENFORCE_GPU_SUCCESS(
          hipEventRecord(comm->compute_event(), dev_ctx.stream()));
      PADDLE_ENFORCE_GPU_SUCCESS(
          hipStreamWaitEvent(stream, comm->compute_event(), 0));
#else
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaEventRecord(comm->compute_event(), dev_ctx.stream()));
      PADDLE_ENFORCE_GPU_SUCCESS(
          cudaStreamWaitEvent(stream, comm->compute_event(), 0));
#endif

      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllReduce(
          grad_out->data<T>(), grad_out->data<T>(), grad_out->numel(), dtype,
          ncclSum, comm->comm(), stream));

      return;
    }

    // reuse dgc op
    DGCOpFunction<DeviceContext, T>(ctx);

    auto encode_grad_out = ctx.Output<framework::Tensor>("EncodeGrad");
    auto gather_buff = ctx.Output<framework::Tensor>("GatherBuff");
    T* encode_grad_out_data = encode_grad_out->data<T>();

    auto k_out = ctx.Output<framework::Tensor>("k");
    int64_t k = static_cast<int64_t>(*k_out->data<T>());

#ifdef PADDLE_WITH_HIP
    PADDLE_ENFORCE_GPU_SUCCESS(
        hipEventRecord(comm->compute_event(), dev_ctx.stream()));
    PADDLE_ENFORCE_GPU_SUCCESS(
        hipStreamWaitEvent(stream, comm->compute_event(), 0));
#else
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaEventRecord(comm->compute_event(), dev_ctx.stream()));
    PADDLE_ENFORCE_GPU_SUCCESS(
        cudaStreamWaitEvent(stream, comm->compute_event(), 0));
#endif

    // do dgc comm
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
        encode_grad_out_data, gather_buff->data<T>(), 2 * k, dtype,
        comm->comm(), stream));

    PADDLE_ENFORCE_EQ(
        paddle::communication::dgc::sparseReduce(
            static_cast<void*>(gather_buff->data()), k, grad_out->data<T>(),
            grad_out->numel(), nranks, stream),
        true, platform::errors::Unavailable("Calling sparseReduce() failed."));
  }
};
}  // namespace operators
}  // namespace paddle
