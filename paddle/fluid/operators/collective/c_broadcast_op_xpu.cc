/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/collective/c_broadcast_op.h"

#ifdef PADDLE_WITH_XPU_BKCL
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/xpu/bkcl_helper.h"
#endif

namespace paddle {
namespace operators {

template <typename T>
class CBroadcastOpXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_XPU_BKCL)
    auto x = ctx.Input<phi::DenseTensor>("X");
    auto out = ctx.Output<phi::DenseTensor>("Out");
    size_t numel = x->numel();

    BKCLDataType dtype =
        platform::ToBKCLDataType(framework::TransToProtoVarType(x->dtype()));
    int ring_id = ctx.Attr<int>("ring_id");
    auto place = ctx.GetPlace();
    auto comm =
        paddle::platform::BKCLCommContext::Instance().Get(ring_id, place);

    XPUStream stream = nullptr;
    auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
    if (ctx.Attr<bool>("use_calc_stream")) {
      stream = static_cast<platform::XPUDeviceContext*>(dev_ctx)
                   ->x_context()
                   ->xpu_stream;
    } else {
      stream = comm->stream();
    }

    int root = ctx.Attr<int>("root");
    VLOG(3) << "begin bkcl broadcast, parameter is: "
            << "root " << root << ", comm: " << comm->comm()
            << ", stream: " << stream;
    void* send_recv_buffer = nullptr;
    if (root == comm->rank()) {
      // API: BKCLResult_t bkcl_broadcast(const BKCLContext_t ctx,
      //                                  const void* sendbuf,
      //                                  void* recvbuf,
      //                                  size_t count, BKCLDataType datatype,
      //                                  int root,
      //                                  XPUStream stream);
      send_recv_buffer = reinterpret_cast<void*>(const_cast<T*>(x->data<T>()));
      auto ret = bkcl_broadcast(comm->comm(),
                                send_recv_buffer,
                                send_recv_buffer,
                                numel,
                                dtype,
                                root,
                                stream);
      PADDLE_ENFORCE_EQ(ret,
                        BKCL_SUCCESS,
                        platform::errors::PreconditionNotMet(
                            "XPU BKCL c_broadcast execute failed"));
      if (out != x) {
        framework::TensorCopy(
            *static_cast<const phi::DenseTensor*>(x),
            place,
            *platform::DeviceContextPool::Instance().Get(place),
            static_cast<phi::DenseTensor*>(out));
      }
    } else {
      auto& dev_ctx = ctx.template device_context<platform::XPUDeviceContext>();
      dev_ctx.template Alloc<T>(out);
      send_recv_buffer = out->data<T>();
      auto ret = bkcl_broadcast(comm->comm(),
                                send_recv_buffer,
                                send_recv_buffer,
                                numel,
                                dtype,
                                root,
                                stream);
      PADDLE_ENFORCE_EQ(ret,
                        BKCL_SUCCESS,
                        platform::errors::PreconditionNotMet(
                            "XPU BKCL c_broadcast execute failed"));
    }

    VLOG(3) << "rank " << comm->rank() << " invoke Bcast. received "
            << phi::product(out->dims());
    out->Resize(x->dims());
    out->set_lod(x->lod());
#else
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "PaddlePaddle should be compiled with XPU and BKCL."));
#endif
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_XPU_KERNEL(c_broadcast,
                       ops::CBroadcastOpXPUKernel<float>,
                       ops::CBroadcastOpXPUKernel<plat::float16>);
