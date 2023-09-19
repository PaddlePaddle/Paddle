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

#include "paddle/fluid/operators/collective/global_gather_op.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#endif
#include "paddle/fluid/distributed/collective/utils.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/distributed/nccl_comm_context.h"
#include "paddle/phi/core/flags.h"
PHI_DECLARE_bool(dynamic_static_unified_comm);

namespace paddle {
namespace operators {

template <typename T>
struct GlobalGatherFunctor<phi::GPUContext, T> {
  void operator()(const framework::ExecutionContext& ctx) {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#if NCCL_VERSION_CODE >= 2703
    auto x = ctx.Input<phi::DenseTensor>("X");
    auto local_count = ctx.Input<phi::DenseTensor>("local_count");
    auto global_count = ctx.Input<phi::DenseTensor>("global_count");
    auto local_count_type =
        framework::TransToProtoVarType(local_count->dtype());
    auto global_count_type =
        framework::TransToProtoVarType(global_count->dtype());
    if (local_count_type != framework::proto::VarType::INT64) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Please use int64 type in local_count."));
    }
    if (global_count_type != framework::proto::VarType::INT64) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Please use int64 type in global_count."));
    }
    auto out = ctx.Output<phi::DenseTensor>("Out");
    const int64_t* cpu_local_count_data;
    const int64_t* cpu_global_count_data;
    auto local_count_len = 0;

    phi::DenseTensor cpu_local_count;
    if (platform::is_cpu_place(local_count->place())) {
      cpu_local_count_data = local_count->data<int64_t>();
      local_count_len = local_count->numel();
    } else {
      framework::TensorCopySync(
          *local_count, platform::CPUPlace(), &cpu_local_count);
      cpu_local_count_data = cpu_local_count.data<int64_t>();
      local_count_len = cpu_local_count.numel();
    }

    phi::DenseTensor cpu_global_count;
    if (platform::is_cpu_place(global_count->place())) {
      cpu_global_count_data = global_count->data<int64_t>();
    } else {
      framework::TensorCopySync(
          *global_count, platform::CPUPlace(), &cpu_global_count);
      cpu_global_count_data = cpu_global_count.data<int64_t>();
    }

    ncclDataType_t dtype =
        platform::ToNCCLDataType(framework::TransToProtoVarType(x->dtype()));

    int ring_id = ctx.Attr<int>("ring_id");
    PADDLE_ENFORCE_GE(
        ring_id,
        0,
        platform::errors::InvalidArgument(
            "The ring_id (%d) for global gather op must be non-negative.",
            ring_id));
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
                            "variable FLAGS_dynamic_static_unified_comm "
                            "True. But ring_id(%d) is "
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
    } else {
      comm = platform::NCCLCommContext::Instance().Get(ring_id, place);
      stream = comm->stream();
      nranks = comm->nranks();
    }
    if (ctx.Attr<bool>("use_calc_stream")) {
      // should ExecutionContext for calc stream.
      stream = ctx.cuda_device_context().stream();
    }

    auto in_feat = x->dims()[1];
    auto n_expert = local_count->dims()[0] / nranks;

    auto fwd_count = 0;

    for (auto i = 0; i < local_count_len; ++i) {
      fwd_count += cpu_local_count_data[i];
    }
    framework::DDim out_dims = phi::make_ddim({fwd_count, in_feat});
    int64_t* expert_ptr = new int64_t[n_expert * nranks];
    expert_ptr[0] = 0;
    auto tot_experts = n_expert * nranks;
    for (auto i = 1; i < tot_experts; ++i) {
      expert_ptr[i] = expert_ptr[i - 1] + cpu_local_count_data[i - 1];
    }
    auto send_ptr = 0;
    out->mutable_data<T>(out_dims, place);

    if (comm_ctx) {
      for (auto i = 0; i < n_expert; ++i) {
        comm_ctx->GroupStart();
        for (auto j = 0; j < nranks; ++j) {
          int idx = i + j * n_expert;
          if (cpu_global_count_data[idx]) {
            auto send_buf = distributed::GetPartialTensor(
                *x, send_ptr * in_feat, cpu_global_count_data[idx] * in_feat);
            comm_ctx->Send(
                send_buf, cpu_global_count_data[idx] * in_feat, j, stream);
            send_ptr += cpu_global_count_data[idx];
          }
          if (cpu_local_count_data[idx]) {
            auto recv_buf = distributed::GetPartialTensor(
                *out,
                expert_ptr[idx] * in_feat,
                cpu_local_count_data[idx] * in_feat);
            comm_ctx->Recv(
                &recv_buf, cpu_local_count_data[idx] * in_feat, j, stream);
          }
        }
        comm_ctx->GroupEnd();
      }
    } else {
      auto send_buf = x->data<T>();
      auto recv_buf = out->data<T>();
      for (auto i = 0; i < n_expert; ++i) {
        PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
        for (auto j = 0; j < nranks; ++j) {
          int idx = i + j * n_expert;
          if (cpu_global_count_data[idx]) {
            PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclSend(
                send_buf + send_ptr * in_feat,
                cpu_global_count_data[idx] * in_feat,
                dtype,
                j,
                comm->comm(),
                stream));
            send_ptr += cpu_global_count_data[idx];
          }
          if (cpu_local_count_data[idx]) {
            PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclRecv(
                recv_buf + expert_ptr[idx] * in_feat,
                cpu_local_count_data[idx] * in_feat,
                dtype,
                j,
                comm->comm(),
                stream));
          }
        }
        PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
      }
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

template <typename T>
struct GlobalGatherProcessGroupFunctor<phi::GPUContext, T> {
  void operator()(const framework::ExecutionContext& ctx) {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#if NCCL_VERSION_CODE >= 2703
    auto x = ctx.Input<phi::DenseTensor>("X");
    auto local_count = ctx.Input<phi::DenseTensor>("local_count");
    auto global_count = ctx.Input<phi::DenseTensor>("global_count");
    auto local_count_type =
        framework::TransToProtoVarType(local_count->dtype());
    auto global_count_type =
        framework::TransToProtoVarType(global_count->dtype());
    if (local_count_type != framework::proto::VarType::INT64) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Please use int64 type in local_count."));
    }
    if (global_count_type != framework::proto::VarType::INT64) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Please use int64 type in global_count."));
    }
    auto out = ctx.Output<phi::DenseTensor>("Out");
    const int64_t* cpu_local_count_data;
    const int64_t* cpu_global_count_data;
    auto local_count_len = 0;

    phi::DenseTensor cpu_local_count;
    if (platform::is_cpu_place(local_count->place())) {
      cpu_local_count_data = local_count->data<int64_t>();
      local_count_len = local_count->numel();
    } else {
      framework::TensorCopySync(
          *local_count, platform::CPUPlace(), &cpu_local_count);
      cpu_local_count_data = cpu_local_count.data<int64_t>();
      local_count_len = cpu_local_count.numel();
    }

    phi::DenseTensor cpu_global_count;
    if (platform::is_cpu_place(global_count->place())) {
      cpu_global_count_data = global_count->data<int64_t>();
    } else {
      framework::TensorCopySync(
          *global_count, platform::CPUPlace(), &cpu_global_count);
      cpu_global_count_data = cpu_global_count.data<int64_t>();
    }

    int ring_id = ctx.Attr<int>("ring_id");
    PADDLE_ENFORCE_GE(
        ring_id,
        0,
        platform::errors::InvalidArgument(
            "The ring_id (%d) for global gather op must be non-negative.",
            ring_id));
    auto place = ctx.GetPlace();

    auto map = distributed::ProcessGroupMapFromGid::getInstance();
    distributed::ProcessGroup* pg = map->get(ring_id);

    int nranks = pg->GetSize();
    auto in_feat = x->dims()[1];
    auto n_expert = local_count->dims()[0] / nranks;

    auto fwd_count = 0;

    for (auto i = 0; i < local_count_len; ++i) {
      fwd_count += cpu_local_count_data[i];
    }
    framework::DDim out_dims = phi::make_ddim({fwd_count, in_feat});
    int64_t* expert_ptr = new int64_t[n_expert * nranks];
    expert_ptr[0] = 0;
    auto tot_experts = n_expert * nranks;
    for (auto i = 1; i < tot_experts; ++i) {
      expert_ptr[i] = expert_ptr[i - 1] + cpu_local_count_data[i - 1];
    }
    auto send_ptr = 0;
    out->mutable_data<T>(out_dims, place);

    for (auto i = 0; i < n_expert; ++i) {
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
      for (auto j = 0; j < nranks; ++j) {
        int idx = i + j * n_expert;
        if (cpu_global_count_data[idx]) {
          phi::DenseTensor tmp = *x;
          pg->Send(tmp,
                   j,
                   send_ptr * in_feat,
                   cpu_global_count_data[idx] * in_feat,
                   /*sync_op*/ true);
          send_ptr += cpu_global_count_data[idx];
        }
        if (cpu_local_count_data[idx]) {
          pg->Recv(out,
                   j,
                   expert_ptr[idx] * in_feat,
                   cpu_local_count_data[idx] * in_feat,
                   /*sync_op*/ true);
        }
      }
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
    }

#ifdef PADDLE_WITH_CUDA
    PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
#else
    PADDLE_ENFORCE_GPU_SUCCESS(hipDeviceSynchronize());
#endif

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

template <typename T, typename DeivceContext>
class GlobalGatherOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const int rid = ctx.Attr<int>("ring_id");
    auto map = distributed::ProcessGroupMapFromGid::getInstance();
    if (map->has(rid)) {
      GlobalGatherProcessGroupFunctor<phi::GPUContext, T> functor_;
      functor_(ctx);
    } else {
      GlobalGatherFunctor<phi::GPUContext, T> functor_;
      functor_(ctx);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

PD_REGISTER_STRUCT_KERNEL(global_gather,
                          GPU,
                          ALL_LAYOUT,
                          ops::GlobalGatherOpCUDAKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          plat::float16) {}
