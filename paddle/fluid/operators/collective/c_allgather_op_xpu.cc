/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/collective/c_allgather_op.h"

#ifdef PADDLE_WITH_XPU_BKCL
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/xpu/bkcl_helper.h"
#endif

namespace paddle {
namespace operators {

template <typename T>
class CAllGatherOpXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_XPU_BKCL)
    auto in = ctx.Input<phi::DenseTensor>("X");
    auto out = ctx.Output<phi::DenseTensor>("Out");
    BKCLDataType dtype =
        platform::ToBKCLDataType(framework::TransToProtoVarType(in->dtype()));

    int nranks = ctx.Attr<int>("nranks");
    int rid = ctx.Attr<int>("ring_id");
    auto place = ctx.GetPlace();

    auto comm = platform::BKCLCommContext::Instance().Get(rid, place);
    PADDLE_ENFORCE_EQ(
        nranks,
        comm->nranks(),
        platform::errors::InvalidArgument(
            "nranks: %s should equal to %s", nranks, comm->nranks()));

    framework::DDim out_dims = in->dims();
    out_dims[0] *= nranks;

    size_t numel = in->numel();
    const void* sendbuff = in->data<T>();
    void* recvbuff = out->mutable_data<T>(out_dims, place);

    XPUStream stream = nullptr;
    if (ctx.Attr<bool>("use_calc_stream")) {
      auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
      stream = static_cast<platform::XPUDeviceContext*>(dev_ctx)
                   ->x_context()
                   ->xpu_stream;
    } else {
      stream = comm->stream();
    }

    // BKCLResult_t bkcl_all_gather(const BKCLContext_t ctx, const void*
    // sendbuf, size_t sendcnt, void* recvbuf, BKCLDataType datatype, XPUStream
    // stream);
    PADDLE_ENFORCE_EQ(
        bkcl_all_gather(comm->comm(), sendbuff, numel, recvbuff, dtype, stream),
        BKCL_SUCCESS,
        platform::errors::PreconditionNotMet("BKCL all gather failed"));
#else
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "PaddlePaddle should be compiled with XPU and bkcl."));
#endif
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_XPU_KERNEL(c_allgather,
                       ops::CAllGatherOpXPUKernel<float>,
                       ops::CAllGatherOpXPUKernel<double>,
                       ops::CAllGatherOpXPUKernel<int>,
                       ops::CAllGatherOpXPUKernel<int64_t>,
                       ops::CAllGatherOpXPUKernel<plat::float16>);
