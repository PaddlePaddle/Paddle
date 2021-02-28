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

#include "paddle/fluid/operators/collective/send_v2_op.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/nccl_helper.h"
#endif

namespace paddle {
namespace operators {

template <typename T>
class SendOpV2CUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if (defined(PADDLE_WITH_RCCL) || defined(PADDLE_WITH_NCCL)) && \
    NCCL_VERSION_CODE >= 2703
    auto x = ctx.Input<framework::LoDTensor>("X");
    int numel = x->numel();

    int rid = ctx.Attr<int>("ring_id");
    PADDLE_ENFORCE_GE(
        rid, 0,
        platform::errors::InvalidArgument(
            "The ring_id (%d) for send_v2 op must be non-negative.", rid));

    int peer = ctx.Attr<int>("peer");
    PADDLE_ENFORCE_GE(
        peer, 0,
        platform::errors::InvalidArgument(
            "The peer (%d) for send_v2 op must be non-negative.", peer));
    gpuStream_t stream = nullptr;
    auto place = ctx.GetPlace();
    auto comm = platform::NCCLCommContext::Instance().Get(rid, place);
    if (ctx.Attr<bool>("use_calc_stream")) {
      auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
      stream = static_cast<platform::CUDADeviceContext*>(dev_ctx)->stream();
    } else {
      stream = comm->stream();
    }
    PADDLE_ENFORCE_LT(
        peer, comm->nranks(),
        platform::errors::InvalidArgument("The value of peer (%d) you set must "
                                          "be less than comm->nranks (%d).",
                                          peer, comm->nranks()));
    ncclDataType_t dtype = platform::ToNCCLDataType(x->type());
    // Send number of elements to the receiver, as the receiver may have
    // no information of the Tensor size.
    int* numel_ptr = nullptr;
#ifdef PADDLE_WITH_RCCL
    PADDLE_ENFORCE_CUDA_SUCCESS(hipMalloc(&numel_ptr, sizeof(int)));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        hipMemcpy(numel_ptr, &numel, sizeof(int), hipMemcpyHostToDevice));
#else
    PADDLE_ENFORCE_CUDA_SUCCESS(cudaMalloc(&numel_ptr, sizeof(int)));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        cudaMemcpy(numel_ptr, &numel, sizeof(int), cudaMemcpyHostToDevice));
#endif

    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclSend(
        numel_ptr, 1, ncclInt, peer, comm->comm(), stream));
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclSend(
        x->data<T>(), numel, dtype, peer, comm->comm(), stream));
    VLOG(3) << "rank " << comm->rank() << " send "
            << framework::product(x->dims()) << " to " << peer;
#else
    PADDLE_THROW(platform::errors::Unavailable(
        "PaddlePaddle should be compiled with NCCL "
        "and NCCL version >= 2.7.3 is needed."));
#endif
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(send_v2, ops::SendOpV2CUDAKernel<float>,
                        ops::SendOpV2CUDAKernel<double>,
                        ops::SendOpV2CUDAKernel<int>,
                        ops::SendOpV2CUDAKernel<int64_t>,
                        ops::SendOpV2CUDAKernel<plat::float16>);
