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

#include "paddle/fluid/operators/collective/partial_recv_op.h"

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
class PartialRecvOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if (defined(PADDLE_WITH_RCCL) || defined(PADDLE_WITH_NCCL)) && \
    NCCL_VERSION_CODE >= 2703
    auto out = ctx.Output<phi::DenseTensor>("Out");
    auto out_dims = out->dims();
    auto numel = out->numel();

    int rid = ctx.Attr<int>("ring_id");
    int peer = ctx.Attr<int>("peer");
    int data_type = ctx.Attr<int>("dtype");
    int num = ctx.Attr<int>("num");
    int id = ctx.Attr<int>("id");
    framework::proto::VarType::Type type =
        framework::proto::VarType::Type(data_type);

    PADDLE_ENFORCE_GE(
        rid,
        0,
        platform::errors::InvalidArgument(
            "The ring_id (%d) for partial_recv op must be non-negative.", rid));
    PADDLE_ENFORCE_GE(
        peer,
        0,
        platform::errors::InvalidArgument(
            "The peer (%d) for partial_recv op must be non-negative.", peer));
    PADDLE_ENFORCE_GE(num,
                      1,
                      platform::errors::InvalidArgument(
                          "The num (%d) for partial_recv op must >=1", num));
    PADDLE_ENFORCE_EQ(
        (id >= 0 && id < num),
        true,
        platform::errors::InvalidArgument(
            "The id (%d) for partial_recv op must >=0 and <num (%d)", id, num));
    PADDLE_ENFORCE_EQ(
        (numel % num),
        0,
        platform::errors::InvalidArgument(
            "The input numel (%d) must be divisible by num(%d)", numel, num));

    auto place = ctx.GetPlace();
    out->mutable_data<T>(out_dims, place);
    int64_t recv_numel = numel / num;
    int64_t offset = recv_numel * id;

    auto map = distributed::ProcessGroupMapFromGid::getInstance();
    if (map->has(rid)) {
      // Use ProcessGroup
      distributed::ProcessGroup* pg = map->get(rid);
      auto task = pg->Recv(out, peer, offset, recv_numel, /*sync_op*/ true);
      task->Wait();
    } else {
      gpuStream_t stream = nullptr;
      platform::NCCLComm* comm = nullptr;
      phi::distributed::NCCLCommContext* comm_ctx = nullptr;

      int nranks = 0;
      int rank = 0;

      const auto& comm_context_manager =
          phi::distributed::CommContextManager::GetInstance();

      if (FLAGS_dynamic_static_unified_comm) {
        // Use New Communication Library
        PADDLE_ENFORCE_EQ(
            comm_context_manager.Has(std::to_string(rid)),
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
        PADDLE_ENFORCE_NE(
            comm_ctx,
            nullptr,
            platform::errors::Unavailable(
                "NCCLCommContext is nullptr, collective op should "
                "has ring_id attr."));

        stream = comm_ctx->GetStream();
        nranks = comm_ctx->GetSize();
        rank = comm_ctx->GetRank();

        VLOG(3) << "new comm_context_manager has ring_id " << rid;
      } else {
        comm = platform::NCCLCommContext::Instance().Get(rid, place);

        stream = comm->stream();
        nranks = comm->nranks();
        rank = comm->rank();

        VLOG(3) << "old NCCLCommContext has ring_id" << rid;
      }

      if (ctx.Attr<bool>("use_calc_stream")) {
        // should ExecutionContext for calc stream.
        stream = ctx.cuda_device_context().stream();
      }

      PADDLE_ENFORCE_LT(peer,
                        nranks,
                        platform::errors::InvalidArgument(
                            "The value of peer (%d) you set must "
                            "be less than nranks (%d).",
                            peer,
                            nranks));

      ncclDataType_t dtype = platform::ToNCCLDataType(type);

      if (comm_ctx) {
        auto recv_buf = distributed::GetPartialTensor(*out, offset, recv_numel);

        comm_ctx->Recv(&recv_buf, recv_numel, peer, stream);
      } else {
        PADDLE_ENFORCE_GPU_SUCCESS(
            platform::dynload::ncclRecv(out->data<T>() + offset,
                                        recv_numel,
                                        dtype,
                                        peer,
                                        comm->comm(),
                                        stream));
      }
      VLOG(3) << "rank " << rank << " recv " << recv_numel << " from offset["
              << offset << "] from " << peer;
    }
#else
    PADDLE_THROW(platform::errors::Unavailable(
        "PaddlePaddle should be compiled with NCCL and "
        "NCCL version >= 2.7.3 is needed."));
#endif
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

PD_REGISTER_STRUCT_KERNEL(partial_recv,
                          GPU,
                          ALL_LAYOUT,
                          ops::PartialRecvOpCUDAKernel,
                          float,
                          double,
#if NCCL_VERSION_CODE >= 21000 && CUDA_VERSION >= 11000
                          plat::bfloat16,
#endif
                          int,
                          int64_t,
                          plat::float16) {
}
