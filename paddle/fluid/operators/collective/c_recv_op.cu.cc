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

#include "paddle/fluid/operators/collective/c_send_op.h"

#if defined(PADDLE_WITH_NCCL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/nccl_helper.h"
#endif

namespace paddle {
namespace operators {

template <typename T>
class CSendOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_NCCL)
    auto out = ctx.Output<framework::LoDTensor>("Out");
    int numel = out->numel();
    ncclDataType_t dtype = platform::ToNCCLDataType(out->type());

    int rid = ctx.Attr<int>("ring_id");
    auto place = ctx.GetPlace();
    auto comm = platform::NCCLCommContext::Instance().Get(rid, place);

    cudaStream_t stream = nullptr;
    if (ctx.Attr<bool>("use_calc_stream")) {
      auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
      stream = static_cast<platform::CUDADeviceContext*>(dev_ctx)->stream();
    } else {
      stream = comm->stream();
    }

    int peer = ctx.Attr<int>("peer");
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclSend(
        out->mutable_data<T>(place), numel, dtype, peer, comm->comm(), stream));
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

REGISTER_OP_CUDA_KERNEL(c_recv, ops::CRecvOpCUDAKernel<float>,
                        ops::CRecvOpCUDAKernel<double>,
                        ops::CRecvOpCUDAKernel<int>,
                        ops::CRecvOpCUDAKernel<int64_t>,
                        ops::CRecvOpCUDAKernel<plat::float16>);
