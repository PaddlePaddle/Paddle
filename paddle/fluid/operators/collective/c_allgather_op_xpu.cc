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
    auto map = distributed::ProcessGroupMapFromGid::getInstance();
    if (map->has(rid)) {
      // Use ProcessGroup
      distributed::ProcessGroup* pg = map->get(rid);
      std::vector<phi::DenseTensor> in_tensor;
      std::vector<phi::DenseTensor> out_tensor;
      in_tensor.push_back(*in);
      out_tensor.push_back(*out);
      auto task = pg->AllGather(in_tensor, out_tensor);
      task->Wait();
      return;
    }
    auto place = ctx.GetPlace();

    size_t numel = in->numel();
    const void* sendbuff = in->data<T>();
    void* recvbuff = out->mutable_data<T>(place);

    XPUStream stream = nullptr;
    platform::BKCLComm* comm = nullptr;
    phi::distributed::BKCLCommContext* comm_ctx = nullptr;
    const auto& comm_context_manager =
        phi::distributed::CommContextManager::GetInstance();
    if (FLAGS_dynamic_static_unified_comm) {
      PADDLE_ENFORCE_EQ(comm_context_manager.Has(std::to_string(rid)),
                        true,
                        platform::errors::InvalidArgument(
                            "You choose to use new communication library by "
                            "setting environment "
                            "variable FLAGS_dynamic_static_unified_comm True. "
                            "But ring_id(%d) is "
                            "not found in comm_context_manager.",
                            std::to_string(rid)));
      comm_ctx = static_cast<phi::distributed::BKCLCommContext*>(
          comm_context_manager.Get(std::to_string(rid)));
      PADDLE_ENFORCE_NE(comm_ctx,
                        nullptr,
                        platform::errors::Unavailable(
                            "BKCLCommContext is nullptr, collective op should "
                            "has ring_id attr."));
      stream = comm_ctx->GetStream();
      VLOG(3) << "new comm_context_manager has rid " << rid;
    } else {  // old comm_context
      comm = platform::BKCLCommContext::Instance().Get(rid, place);
      PADDLE_ENFORCE_EQ(
          nranks,
          comm->nranks(),
          platform::errors::InvalidArgument(
              "nranks: %s should equal to %s", nranks, comm->nranks()));
      stream = comm->stream();
      VLOG(3) << "old BKCLCommContext has rid " << rid;
    }

    if (ctx.Attr<bool>("use_calc_stream")) {
      auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
      stream = static_cast<platform::XPUDeviceContext*>(dev_ctx)
                   ->x_context()
                   ->xpu_stream;
    }

    if (comm_ctx) {
      comm_ctx->AllGather(out, *in, stream);
    } else {
      PADDLE_ENFORCE_XPU_SUCCESS(bkcl_all_gather(
          comm->comm(), sendbuff, numel, recvbuff, dtype, stream));
    }
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

PD_REGISTER_STRUCT_KERNEL(c_allgather,
                          XPU,
                          ALL_LAYOUT,
                          ops::CAllGatherOpXPUKernel,
                          float,
                          double,
                          plat::float16,
                          int,
                          int64_t,
                          uint8_t,
                          bool) {}
