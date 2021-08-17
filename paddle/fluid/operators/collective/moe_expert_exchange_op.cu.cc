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

#include "paddle/fluid/operators/collective/moe_expert_exchange_op.h"

#if defined(PADDLE_WITH_NCCL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/nccl_helper.h"
#endif

namespace paddle {
namespace operators {
template <typename T>
class MoeExpertExchangeOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_NCCL)
#if NCCL_VERSION_CODE >= 2703
    auto local_expert_count =
        ctx.Input<framework::LoDTensor>("local_expert_count");
    auto n_expert = ctx.Attr<int>("n_expert");
    auto world_size = ctx.Attr<int>("world_size");
    auto out = ctx.Output<framework::LoDTensor>("Out");

    ncclDataType_t dtype = platform::ToNCCLDataType(local_expert_count->type());
    int ring_id = ctx.Attr<int>("ring_id");
    PADDLE_ENFORCE_GE(
        ring_id, 0,
        platform::errors::InvalidArgument(
            "The ring_id (%d) for selectscatter op must be non-negative.",
            ring_id));
    auto place = ctx.GetPlace();
    auto comm = platform::NCCLCommContext::Instance().Get(ring_id, place);

    cudaStream_t stream = nullptr;
    if (ctx.Attr<bool>("use_calc_stream")) {
      auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
      stream = static_cast<platform::CUDADeviceContext*>(dev_ctx)->stream();
    } else {
      stream = comm->stream();
    }
    framework::DDim global_expert_count_dims = local_expert_count->dims();
    framework::DDim out_dims(global_expert_count_dims);
    auto send_buf = local_expert_count->data<T>();
    auto recv_buf = out->mutable_data<T>(out_dims, place);
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclGroupStart());
    for (auto i = 0; i < world_size; i++) {
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclSend(
          send_buf + n_expert * i, n_expert, dtype, i, comm->comm(), stream));
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclRecv(
          recv_buf + n_expert * i, n_expert, dtype, i, comm->comm(), stream));
    }
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclGroupEnd());

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

REGISTER_OP_CUDA_KERNEL(moe_expert_exchange,
                        ops::MoeExpertExchangeOpCUDAKernel<float>,
                        ops::MoeExpertExchangeOpCUDAKernel<double>,
                        ops::MoeExpertExchangeOpCUDAKernel<int>,
                        ops::MoeExpertExchangeOpCUDAKernel<int64_t>,
                        ops::MoeExpertExchangeOpCUDAKernel<plat::float16>);
