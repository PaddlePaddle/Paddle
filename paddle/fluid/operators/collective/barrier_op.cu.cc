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

#include "paddle/fluid/operators/collective/barrier_op.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#include "paddle/phi/core/distributed/nccl_comm_context.h"
#include "paddle/phi/core/flags.h"
PHI_DECLARE_bool(dynamic_static_unified_comm);
#endif

namespace paddle {
namespace operators {

template <typename T, typename DeviceContext>
class BarrierOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    auto in = ctx.Input<phi::DenseTensor>("X");
    auto out = ctx.Output<phi::DenseTensor>("Out");

    auto place = ctx.GetPlace();
    ncclDataType_t dtype =
        platform::ToNCCLDataType(framework::TransToProtoVarType(in->dtype()));
    int64_t numel = in->numel();
    const void* sendbuff = in->data();
    void* recvbuff = out->mutable_data<T>(place);

    int rid = ctx.Attr<int>("ring_id");
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
      auto comm_ctx = static_cast<phi::distributed::NCCLCommContext*>(
          comm_context_manager.Get(std::to_string(rid)));
      PADDLE_ENFORCE_NE(comm_ctx,
                        nullptr,
                        platform::errors::Unavailable(
                            "NCCLCommContext is nullptr, collective op should "
                            "has ring_id attr."));
      auto stream = comm_ctx->GetStream();
      ncclRedOp_t nccl_red_type = ncclSum;
      comm_ctx->AllReduce(out, *in, nccl_red_type, stream);
      platform::GpuStreamSync(stream);
      VLOG(3) << "new NCCLCommContext has rid " << rid;
    } else {
      auto comm = platform::NCCLCommContext::Instance().Get(rid, place);
      // should ExecutionContext for calc stream.
      auto stream = ctx.cuda_device_context().stream();
      ncclRedOp_t nccl_red_type = ncclSum;
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllReduce(sendbuff,
                                                                  recvbuff,
                                                                  numel,
                                                                  dtype,
                                                                  nccl_red_type,
                                                                  comm->comm(),
                                                                  stream));
      platform::GpuStreamSync(stream);
      VLOG(3) << "old NCCLCommContext has rid " << rid;
    }
#else
    PADDLE_THROW(platform::errors::Unavailable(
        "PaddlePaddle should compile with NCCL."));
#endif
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

PD_REGISTER_STRUCT_KERNEL(
    barrier, GPU, ALL_LAYOUT, ops::BarrierOpCUDAKernel, int) {}
