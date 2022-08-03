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

#include "dgc/dgc.h"
#include "paddle/fluid/memory/allocation/allocator_facade.h"
#include "paddle/fluid/operators/dgc_comm_op.h"
#include "paddle/phi/core/dense_tensor.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#endif

namespace paddle {
namespace operators {
template <typename DeviceContext, typename T>
class DGCCommOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto x = ctx.Input<phi::DenseTensor>("X");
    auto out = ctx.Output<phi::DenseTensor>("Out");
    auto gather_out = ctx.Output<phi::DenseTensor>("Gather_Out");

    auto place = ctx.GetPlace();
    const int ring_id = ctx.Attr<int>("ring_id");
    const int nranks = ctx.Attr<int>("nranks");
    const int k = ctx.Attr<int>("k_var");
    const int out_numel = static_cast<int>(out->numel());

    auto comm = platform::NCCLCommContext::Instance().Get(ring_id, place);
    auto& dev_ctx =
        ctx.template device_context<phi::GPUContext>();
    gpuStream_t stream = comm->stream();
    ncclDataType_t dtype =
        platform::ToNCCLDataType(framework::TransToProtoVarType(x->dtype()));

    const T* x_buff = x->data<T>();
    T* gather_buff = gather_out->data<T>();

    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
        x_buff, gather_buff, static_cast<int64_t>(2 * k),
        static_cast<ncclDataType_t>(dtype), comm->comm(), stream));

    PADDLE_ENFORCE_EQ(
        paddle::communication::dgc::sparseReduce(
            static_cast<void*>(gather_buff), k, out->data<T>(), out_numel,
            nranks, stream),
        true, platform::errors::Unavailable("Calling sparseReduce() failed."));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(
    dgc_comm,
    ops::DGCCommOpCUDAKernel<phi::GPUContext, float>);
