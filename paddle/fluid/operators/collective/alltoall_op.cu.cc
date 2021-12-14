/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/collective/alltoall_op.h"

#if defined(PADDLE_WITH_NCCL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#endif

namespace paddle {
namespace operators {

template <typename T>
class AllToAllOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_NCCL)
#if NCCL_VERSION_CODE >= 2703
    auto x = ctx.Input<framework::LoDTensor>("X");
    auto out = ctx.Output<framework::LoDTensor>("Out");
    int send_numel = x->numel();
    ncclDataType_t dtype = platform::ToNCCLDataType(x->type());

    int ring_id = ctx.Attr<int>("ring_id");
    PADDLE_ENFORCE_GE(
        ring_id, 0,
        platform::errors::InvalidArgument(
            "The ring_id (%d) for alltoall op must be non-negative.", ring_id));
    auto place = ctx.GetPlace();
    auto comm = platform::NCCLCommContext::Instance().Get(ring_id, place);
    int nranks = comm->nranks();

    cudaStream_t stream = nullptr;
    if (ctx.Attr<bool>("use_calc_stream")) {
      auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
      stream = static_cast<platform::CUDADeviceContext*>(dev_ctx)->stream();
    } else {
      stream = comm->stream();
    }

    framework::DDim x_dims = x->dims();
    framework::DDim out_dims(x_dims);
    PADDLE_ENFORCE_EQ(
        x_dims[0] % nranks, 0,
        platform::errors::InvalidArgument(
            "The first dimension size (%d) of the input tensor must be "
            "divisible by the number of ranks (%d).",
            x_dims[0], nranks));
    auto send_buf = x->data<T>();
    auto recv_buf = out->mutable_data<T>(out_dims, place);
    size_t offset = 0;
    send_numel /= nranks;
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
    for (auto i = 0; i < nranks; ++i) {
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclSend(
          send_buf + offset, send_numel, dtype, i, comm->comm(), stream));
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclRecv(
          recv_buf + offset, send_numel, dtype, i, comm->comm(), stream));
      offset += send_numel;
    }
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
#else
    PADDLE_THROW(
        platform::errors::Unavailable("NCCL version >= 2.7.3 is needed."));
#endif
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

REGISTER_OP_CUDA_KERNEL(alltoall, ops::AllToAllOpCUDAKernel<float>,
                        ops::AllToAllOpCUDAKernel<double>,
                        ops::AllToAllOpCUDAKernel<int>,
                        ops::AllToAllOpCUDAKernel<int64_t>,
                        ops::AllToAllOpCUDAKernel<plat::float16>);
