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

#include "paddle/fluid/operators/collective/alltoall_op.h"
#include "paddle/fluid/distributed/collective/utils.h"
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
class AllToAllOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#if NCCL_VERSION_CODE >= 2703
    auto x = ctx.Input<phi::DenseTensor>("X");
    auto out = ctx.Output<phi::DenseTensor>("Out");
    int send_numel = x->numel();
    ncclDataType_t dtype =
        platform::ToNCCLDataType(framework::TransToProtoVarType(x->dtype()));

    int ring_id = ctx.Attr<int>("ring_id");
    PADDLE_ENFORCE_GE(
        ring_id,
        0,
        platform::errors::InvalidArgument(
            "The ring_id (%d) for alltoall op must be non-negative.", ring_id));
    auto place = ctx.GetPlace();

    gpuStream_t stream = nullptr;
    platform::NCCLComm* comm = nullptr;
    phi::distributed::NCCLCommContext* comm_ctx = nullptr;
    int nranks = 0;

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
      comm_ctx = static_cast<phi::distributed::NCCLCommContext*>(
          comm_context_manager.Get(std::to_string(ring_id)));
      PADDLE_ENFORCE_NE(comm_ctx,
                        nullptr,
                        platform::errors::Unavailable(
                            "NCCLCommContext is nullptr, collective op should "
                            "has ring_id attr."));
      stream = comm_ctx->GetStream();
      nranks = comm_ctx->GetSize();
      VLOG(3) << "new comm_context_manager has rid " << ring_id;
    } else {
      comm = platform::NCCLCommContext::Instance().Get(ring_id, place);
      stream = comm->stream();
      nranks = comm->nranks();
      VLOG(3) << "old NCCLCommContext has rid " << ring_id;
    }

    if (ctx.Attr<bool>("use_calc_stream")) {
      // should ExecutionContext for calc stream.
      stream = ctx.cuda_device_context().stream();
    }

    framework::DDim x_dims = x->dims();
    framework::DDim out_dims(x_dims);
    PADDLE_ENFORCE_EQ(
        x_dims[0] % nranks,
        0,
        platform::errors::InvalidArgument(
            "The first dimension size (%d) of the input tensor must be "
            "divisible by the number of ranks (%d).",
            x_dims[0],
            nranks));
    auto send_buf = x->data<T>();
    auto recv_buf = out->mutable_data<T>(out_dims, place);
    size_t offset = 0;
    send_numel /= nranks;
    if (comm_ctx) {
      comm_ctx->GroupStart();
      for (auto i = 0; i < nranks; ++i) {
        auto send_buf = distributed::GetPartialTensor(*x, offset, send_numel);
        comm_ctx->Send(send_buf, send_numel, i, stream);
        auto recv_buf = distributed::GetPartialTensor(*out, offset, send_numel);
        comm_ctx->Recv(&recv_buf, send_numel, i, stream);
        offset += send_numel;
      }
      comm_ctx->GroupEnd();
      VLOG(3) << "new comm_context_manager has rid " << ring_id;
    } else {
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
      for (auto i = 0; i < nranks; ++i) {
        PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclSend(
            send_buf + offset, send_numel, dtype, i, comm->comm(), stream));
        PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclRecv(
            recv_buf + offset, send_numel, dtype, i, comm->comm(), stream));
        offset += send_numel;
      }
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
      VLOG(3) << "old NCCLCommContext has rid " << ring_id;
    }
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

PD_REGISTER_STRUCT_KERNEL(alltoall,
                          GPU,
                          ALL_LAYOUT,
                          ops::AllToAllOpCUDAKernel,
                          float,
                          double,
#if NCCL_VERSION_CODE >= 21000 && CUDA_VERSION >= 11000
                          plat::bfloat16,
#endif
                          int,
                          int64_t,
                          plat::float16) {
}
