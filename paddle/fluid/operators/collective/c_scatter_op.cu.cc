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

#include "paddle/fluid/operators/collective/c_scatter_op.h"
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
class CScatterOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    auto x = ctx.Input<phi::DenseTensor>("X");
    auto out = ctx.Output<phi::DenseTensor>("Out");
    int numel = x->numel();
    ncclDataType_t dtype =
        platform::ToNCCLDataType(framework::TransToProtoVarType(x->dtype()));

    int nranks = ctx.Attr<int>("nranks");
    int root_id = ctx.Attr<int>("root");
    int ring_id = ctx.Attr<int>("ring_id");
    auto place = ctx.GetPlace();
    gpuStream_t stream = nullptr;
    platform::NCCLComm* comm = nullptr;
    phi::distributed::NCCLCommContext* comm_ctx = nullptr;
    PADDLE_ENFORCE_GE(
        root_id,
        0,
        platform::errors::InvalidArgument(
            "The root_id (%d) for c_scatter_op must be non-negative.",
            root_id));
    PADDLE_ENFORCE_GE(
        ring_id,
        0,
        platform::errors::InvalidArgument(
            "The ring_id (%d) for c_scatter_op must be non-negative.",
            ring_id));

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
      PADDLE_ENFORCE_EQ(nranks,
                        comm_ctx->GetSize(),
                        platform::errors::InvalidArgument(
                            "The number of ranks (%d) you set of must "
                            "be equal to comm_ctx->GetSize() (%d).",
                            nranks,
                            comm_ctx->GetSize()));

      stream = comm_ctx->GetStream();
      VLOG(3) << "new comm_context_manager has ring_id " << ring_id;
    } else {  // old comm_context
      comm = platform::NCCLCommContext::Instance().Get(ring_id, place);
      PADDLE_ENFORCE_EQ(nranks,
                        comm->nranks(),
                        platform::errors::InvalidArgument(
                            "The number of ranks (%d) you set of must "
                            "be equal to comm->nranks (%d).",
                            nranks,
                            comm->nranks()));

      stream = comm->stream();
      VLOG(3) << "old NCCLCommContext has ring_id " << ring_id;
    }
    if (ctx.Attr<bool>("use_calc_stream")) {
      // should ExecutionContext for calc stream.
      stream = ctx.cuda_device_context().stream();
    }

    framework::DDim x_dims = x->dims();
    framework::DDim out_dims(x_dims);
    phi::DenseTensor temp;
    auto out_ptr = temp.mutable_data<T>(out_dims, place);

    if (FLAGS_dynamic_static_unified_comm) {
      if (root_id == comm_ctx->GetRank()) {
        comm_ctx->Broadcast(
            const_cast<phi::DenseTensor*>(x), *x, root_id, stream);
        framework::TensorCopy(
            *static_cast<const phi::DenseTensor*>(x),
            place,
            *platform::DeviceContextPool::Instance().Get(place),
            static_cast<phi::DenseTensor*>(&temp));
      } else {
        comm_ctx->Broadcast(&temp, temp, root_id, stream);
      }
    } else {
      if (root_id == comm->rank()) {
        PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclBcast(
            reinterpret_cast<void*>(const_cast<T*>(x->data<T>())),
            numel,
            dtype,
            root_id,
            comm->comm(),
            stream));

        framework::TensorCopy(
            *static_cast<const phi::DenseTensor*>(x),
            place,
            *platform::DeviceContextPool::Instance().Get(place),
            static_cast<phi::DenseTensor*>(&temp));
      } else {
        PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclBcast(
            out_ptr, numel, dtype, root_id, comm->comm(), stream));
      }
    }

    out_dims[0] = out_dims[0] / nranks;
    auto start_index = FLAGS_dynamic_static_unified_comm
                           ? out_dims[0] * comm_ctx->GetRank()
                           : out_dims[0] * comm->rank();
    auto end_index = start_index + out_dims[0];
    temp = temp.Slice(start_index, end_index);
    temp.Resize(out_dims);
    out->mutable_data<T>(out_dims, place);
    framework::TensorCopySync(*static_cast<const phi::DenseTensor*>(&temp),
                              place,
                              static_cast<phi::DenseTensor*>(out));
    out->Resize(out_dims);
#else
    PADDLE_ENFORCE_EQ(
        true,
        false,
        platform::errors::Unavailable("PaddlePaddle should compile with GPU."));
#endif
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

PD_REGISTER_STRUCT_KERNEL(c_scatter,
                          GPU,
                          ALL_LAYOUT,
                          ops::CScatterOpCUDAKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          plat::float16) {}
