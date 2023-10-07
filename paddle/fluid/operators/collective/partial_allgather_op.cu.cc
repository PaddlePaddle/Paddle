/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/collective/partial_allgather_op.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/distributed/collective/process_group.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#include "paddle/phi/core/distributed/nccl_comm_context.h"
#include "paddle/phi/core/flags.h"
PHI_DECLARE_bool(dynamic_static_unified_comm);
#endif

#include "paddle/fluid/distributed/collective/utils.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"

namespace paddle {
namespace operators {

template <typename T, typename DeviceContext>
class PartialAllGatherOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    auto in = ctx.Input<phi::DenseTensor>("X");
    auto out = ctx.Output<phi::DenseTensor>("Out");
    int64_t numel = in->numel();
    ncclDataType_t dtype =
        platform::ToNCCLDataType(framework::TransToProtoVarType(in->dtype()));

    int nranks = ctx.Attr<int>("nranks");
    int rank = ctx.Attr<int>("rank");
    int rid = ctx.Attr<int>("ring_id");
    auto place = ctx.GetPlace();
    gpuStream_t stream = nullptr;

    platform::NCCLComm* comm = nullptr;
    phi::distributed::NCCLCommContext* comm_ctx = nullptr;

    const auto& comm_context_manager =
        phi::distributed::CommContextManager::GetInstance();

    int real_nranks = 0;
    int real_rank = 0;
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
      comm_ctx = static_cast<phi::distributed::NCCLCommContext*>(
          comm_context_manager.Get(std::to_string(rid)));
      PADDLE_ENFORCE_NE(comm_ctx,
                        nullptr,
                        platform::errors::Unavailable(
                            "NCCLCommContext is nullptr, collective op should "
                            "has ring_id attr."));

      stream = comm_ctx->GetStream();
      real_nranks = comm_ctx->GetSize();
      real_rank = comm_ctx->GetRank();
      VLOG(3) << "new comm_context_manager has ring_id " << rid;
    } else {  // old comm_context
      comm = platform::NCCLCommContext::Instance().Get(rid, place);

      stream = comm->stream();
      real_nranks = comm->nranks();
      real_rank = comm->rank();
      VLOG(3) << "old NCCLCommContext has ring_id " << rid;
    }

    PADDLE_ENFORCE_EQ(
        nranks,
        real_nranks,
        platform::errors::InvalidArgument(
            "nranks: %s should equal to %s", nranks, real_nranks));
    PADDLE_ENFORCE_EQ(rank,
                      real_rank,
                      platform::errors::InvalidArgument(
                          "rank: %s should equal to %s", rank, real_rank));

    PADDLE_ENFORCE_EQ(
        (numel % nranks),
        0,
        platform::errors::InvalidArgument(
            "The input numel (%d) must be divisible by nranks(%d)",
            numel,
            nranks));

    framework::DDim dims = in->dims();
    out->mutable_data<T>(dims, place);

    int64_t send_numel = numel / nranks;
    int offset = send_numel * rank;

    auto map = distributed::ProcessGroupMapFromGid::getInstance();
    if (map->has(rid)) {
      // Use ProcessGroup
      distributed::ProcessGroup* pg = map->get(rid);
      auto task = pg->AllGather(out, *in, offset, send_numel, /*sync_op*/ true);
      task->Wait();
    } else {
      if (ctx.Attr<bool>("use_calc_stream")) {
        // should ExecutionContext for calc stream.
        stream = ctx.cuda_device_context().stream();
      }

      if (comm_ctx) {
        auto send_buf = distributed::GetPartialTensor(*in, offset, send_numel);

        comm_ctx->AllGather(out, send_buf, stream);
      } else {
        const T* send_buff = in->data<T>() + offset;
        T* recv_buff = out->data<T>();
        PADDLE_ENFORCE_GPU_SUCCESS(
            platform::dynload::ncclAllGather(send_buff,
                                             recv_buff,
                                             send_numel,
                                             static_cast<ncclDataType_t>(dtype),
                                             comm->comm(),
                                             stream));
      }
    }
#else
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU."));
#endif
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

PD_REGISTER_STRUCT_KERNEL(partial_allgather,
                          GPU,
                          ALL_LAYOUT,
                          ops::PartialAllGatherOpCUDAKernel,
                          float,
                          double,
#if NCCL_VERSION_CODE >= 21000 && CUDA_VERSION >= 11000
                          plat::bfloat16,
#endif
                          int,
                          int64_t,
                          plat::float16) {
}
