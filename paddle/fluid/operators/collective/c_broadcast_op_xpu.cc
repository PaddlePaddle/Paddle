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
#include "paddle/phi/core/distributed/comm_context_manager.h"

#ifdef PADDLE_WITH_XPU_BKCL
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/xpu/bkcl_helper.h"
#include "paddle/phi/core/distributed/bkcl_comm_context.h"
#include "paddle/phi/core/flags.h"
PHI_DECLARE_bool(dynamic_static_unified_comm);
#endif
#include "paddle/fluid/distributed/collective/process_group.h"

namespace paddle {
namespace operators {

template <typename T, typename DeviceContext>
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
    int root = ctx.Attr<int>("root");

    platform::BKCLComm* comm = nullptr;
    phi::distributed::BKCLCommContext* comm_ctx = nullptr;
    XPUStream stream = nullptr;
    const auto& comm_context_manager =
        phi::distributed::CommContextManager::GetInstance();
    if (FLAGS_dynamic_static_unified_comm) {
      PADDLE_ENFORCE_EQ(comm_context_manager.Has(std::to_string(ring_id)),
                        true,
                        platform::errors::InvalidArgument(
                            "You choose to use new communication library by "
                            "setting environment "
                            "variable FLAGS_dynamic_static_unified_comm True. "
                            "But ring_id(%d) is "
                            "not found in comm_context_manager.",
                            std::to_string(ring_id)));
      comm_ctx = static_cast<phi::distributed::BKCLCommContext*>(
          comm_context_manager.Get(std::to_string(ring_id)));
      PADDLE_ENFORCE_NE(comm_ctx,
                        nullptr,
                        platform::errors::Unavailable(
                            "BKCLCommContext is nullptr, collective op should "
                            "has ring_id attr."));
      stream = comm_ctx->GetStream();
      VLOG(3) << "new comm_context_manager has rid " << ring_id;
    } else {  // old comm_context
      comm = paddle::platform::BKCLCommContext::Instance().Get(ring_id, place);
      stream = comm->stream();
      VLOG(3) << "old BKCLCommContext has rid " << ring_id;
    }
    if (ctx.Attr<bool>("use_calc_stream")) {
      auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
      stream = static_cast<platform::XPUDeviceContext*>(dev_ctx)
                   ->x_context()
                   ->xpu_stream;
    }
    if (comm_ctx) {
      comm_ctx->Broadcast(out, *x, root, stream);
    } else {
      void* send_recv_buffer = nullptr;
      if (root == comm->rank()) {
        send_recv_buffer =
            reinterpret_cast<void*>(const_cast<T*>(x->data<T>()));
        PADDLE_ENFORCE_XPU_SUCCESS(bkcl_broadcast(comm->comm(),
                                                  send_recv_buffer,
                                                  send_recv_buffer,
                                                  numel,
                                                  dtype,
                                                  root,
                                                  stream));
        VLOG(3) << "rank " << comm->rank() << " invoke Bcast. sent "
                << x->numel();
        if (out != x) {
          framework::TensorCopy(
              *static_cast<const phi::DenseTensor*>(x),
              place,
              *platform::DeviceContextPool::Instance().Get(place),
              static_cast<phi::DenseTensor*>(out));
        }
      } else {
        auto& dev_ctx =
            ctx.template device_context<platform::XPUDeviceContext>();
        dev_ctx.template Alloc<T>(out);
        send_recv_buffer = out->data<T>();
        PADDLE_ENFORCE_XPU_SUCCESS(bkcl_broadcast(comm->comm(),
                                                  send_recv_buffer,
                                                  send_recv_buffer,
                                                  numel,
                                                  dtype,
                                                  root,
                                                  stream));
        VLOG(3) << "rank " << comm->rank() << " invoke Bcast. received "
                << phi::product(out->dims());
      }
    }
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

PD_REGISTER_STRUCT_KERNEL(c_broadcast,
                          XPU,
                          ALL_LAYOUT,
                          ops::CBroadcastOpXPUKernel,
                          float,
                          double,
                          plat::float16,
                          int,
                          int64_t) {}
