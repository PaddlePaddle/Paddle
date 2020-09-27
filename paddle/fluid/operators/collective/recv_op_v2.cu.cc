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

#include "paddle/fluid/operators/collective/send_op_v2.h"

#if defined(PADDLE_WITH_NCCL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/nccl_helper.h"
#endif

namespace paddle {
namespace operators {

template <typename T>
class RecvOpV2CUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
#if defined(PADDLE_WITH_NCCL)
    auto out = ctx.Output<framework::LoDTensor>("Out");
    int data_type = ctx.Attr<int>("dtype");
    framework::proto::VarType::Type type =
        framework::proto::VarType::Type(data_type);
    ncclDataType_t dtype = platform::ToNCCLDataType(type);

    auto out_dims = out->dims();
    // Recv the number of element first
    int numel = 0;
    int *numel_ptr = nullptr;
    PADDLE_ENFORCE_CUDA_SUCCESS(cudaMalloc(&numel_ptr, sizeof(int)));

    int rid = ctx.Attr<int>("ring_id");
    auto place = ctx.GetPlace();
    auto comm = platform::NCCLCommContext::Instance().Get(rid, place);
    int peer = ctx.Attr<int>("peer");
    PADDLE_ENFORCE_LT(
        peer, comm->nranks(),
        platform::errors::InvalidArgument("The value of peer (%d) you set must "
                                          "be less than comm->nranks (%d).",
                                          peer, comm->nranks()));

    cudaStream_t stream = nullptr;
    if (ctx.Attr<bool>("use_calc_stream")) {
      auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
      stream = static_cast<platform::CUDADeviceContext *>(dev_ctx)->stream();
    } else {
      stream = comm->stream();
    }

    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclGroupStart());
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::ncclRecv(static_cast<void *>(numel_ptr), 1, ncclInt,
                                    peer, comm->comm(), stream));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        cudaMemcpy(&numel, numel_ptr, sizeof(int), cudaMemcpyDeviceToHost));

    int rest_numel = 1;
    for (size_t i = 1; i < out_dims.size(); ++i) {
      rest_numel = rest_numel * out_dims[i];
    }
    out_dims[0] = numel / rest_numel;
    out->mutable_data<T>(out_dims, place);

    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclRecv(
        out->data<T>(), numel, dtype, peer, comm->comm(), stream));
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclGroupEnd());
    VLOG(3) << "rank " << comm->rank() << " recv "
            << framework::product(out->dims()) << " from " << peer;
#else
    PADDLE_THROW(
        platform::errors::Unavailable("PaddlePaddle should compile with GPU."));
#endif
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(recv_v2, ops::RecvOpV2CUDAKernel<float>,
                        ops::RecvOpV2CUDAKernel<double>,
                        ops::RecvOpV2CUDAKernel<int>,
                        ops::RecvOpV2CUDAKernel<int64_t>,
                        ops::RecvOpV2CUDAKernel<plat::float16>);
