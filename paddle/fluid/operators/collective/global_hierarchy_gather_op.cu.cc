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

#include "paddle/fluid/operators/collective/global_hierarchy_gather_op.h"

#if defined(PADDLE_WITH_NCCL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#endif

namespace paddle {
namespace operators {
using framework::Tensor;
template <typename DeviceContext, typename T>
class GlobalHierarchyGatherOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_NCCL)
#if NCCL_VERSION_CODE >= 2703
    auto x = ctx.Input<framework::LoDTensor>("X");
    auto local_count = ctx.Input<framework::LoDTensor>("local_count");
    auto mp_global_count = ctx.Input<framework::LoDTensor>("mp_global_count");
    auto mp_fused_global_count = ctx.Input<framework::LoDTensor>("mp_fused_global_count");
    auto dp_global_count = ctx.Input<framework::LoDTensor>("dp_global_count");

    auto local_count_type = framework::TransToProtoVarType(local_count->type());
    if (local_count_type != framework::proto::VarType::INT64) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Please use int64 type in local_count."));
    }
    auto out = ctx.Output<framework::LoDTensor>("Out");
    const int64_t* cpu_local_count_data;
    const int64_t* cpu_mp_global_count_data;
    const int64_t* cpu_mp_fused_global_count_data;
    const int64_t* cpu_dp_global_count_data;

    framework::Tensor cpu_local_count, cpu_mp_global_count, cpu_mp_fused_global_count, cpu_dp_global_count;
    if (platform::is_cpu_place(local_count->place())) {
      cpu_local_count_data = local_count->data<int64_t>();
    } else {
      framework::TensorCopySync(*local_count, platform::CPUPlace(),
                                &cpu_local_count);
      cpu_local_count_data = cpu_local_count.data<int64_t>();
    }

    if (platform::is_cpu_place(mp_global_count->place())) {
      cpu_mp_global_count_data = mp_global_count->data<int64_t>();
    } else {
      framework::TensorCopySync(*mp_global_count, platform::CPUPlace(),
                                &cpu_mp_global_count);
      cpu_mp_global_count_data = cpu_mp_global_count.data<int64_t>();
    }

    if (platform::is_cpu_place(mp_fused_global_count->place())) {
      cpu_mp_fused_global_count_data = mp_fused_global_count->data<int64_t>();
    } else {
      framework::TensorCopySync(*mp_fused_global_count, platform::CPUPlace(),
                                &cpu_mp_fused_global_count);
      cpu_mp_fused_global_count_data = cpu_mp_fused_global_count.data<int64_t>();
    }

    if (platform::is_cpu_place(dp_global_count->place())) {
      cpu_dp_global_count_data = dp_global_count->data<int64_t>();
    } else {
      framework::TensorCopySync(*dp_global_count, platform::CPUPlace(),
                                &cpu_dp_global_count);
      cpu_dp_global_count_data = cpu_dp_global_count.data<int64_t>();
    }

    ncclDataType_t dtype = platform::ToNCCLDataType(framework::TransToProtoVarType(x->type()));

    int inside_ring_id = ctx.Attr<int>("inside_ring_id");
    PADDLE_ENFORCE_GE(
        inside_ring_id, 0,
        platform::errors::InvalidArgument("The inside_ring_id (%d) for global "
                                          "gather op must be non-negative.",
                                          inside_ring_id));

    int outside_ring_id = ctx.Attr<int>("outside_ring_id");
    PADDLE_ENFORCE_GE(
        outside_ring_id, 0,
        platform::errors::InvalidArgument("The outside_ring_id (%d) for global "
                                          "gather op must be non-negative.",
                                          outside_ring_id));

    auto place = ctx.GetPlace();
    auto inside_comm =
        platform::NCCLCommContext::Instance().Get(inside_ring_id, place);
    auto outside_comm =
        platform::NCCLCommContext::Instance().Get(outside_ring_id, place);

    cudaStream_t stream = nullptr;
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();

    if (ctx.Attr<bool>("use_calc_stream")) {
      stream = dev_ctx.stream();
    } else {
      stream = inside_comm->stream();
    }
    int inside_nranks = inside_comm->nranks();
    int outside_nranks = outside_comm->nranks();
    auto in_feat = x->dims()[1];
    auto n_expert = local_count->dims()[0] / inside_nranks / outside_nranks;
    //auto inside_all_experts = n_expert * inside_nranks;

    // Step1: outside all_to_all
    int dp_fwd_count = 0;
    for (auto i = 0; i < cpu_mp_fused_global_count.numel(); ++i) {
      dp_fwd_count += cpu_mp_fused_global_count_data[i];
    }
    Tensor mp_global_res;
    mp_global_res = ctx.AllocateTmpTensor<T, DeviceContext>(
        {dp_fwd_count, in_feat}, dev_ctx);
    auto outside_send_ptr = 0;
    auto outside_recv_ptr = 0;
    auto outside_send_buf = x->data<T>();
    auto outside_recv_buf = mp_global_res.mutable_data<T>(place);

    int recv_src = 0;
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
    for (auto i = 0; i < n_expert; ++i) {
      for (auto j = 0; j < outside_nranks; ++j) {
        int idx = j + i * outside_nranks;
        if (cpu_dp_global_count_data[idx]) {
          PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclSend(
              outside_send_buf + outside_send_ptr * in_feat,
              cpu_dp_global_count_data[idx] * in_feat, dtype, j,
              outside_comm->comm(), stream));
          outside_send_ptr += cpu_dp_global_count_data[idx];
        }
        if (cpu_mp_fused_global_count_data[idx]) {
          PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclRecv(
              outside_recv_buf + outside_recv_ptr * in_feat,
              cpu_mp_fused_global_count_data[idx] * in_feat, dtype,
              recv_src / n_expert, outside_comm->comm(), stream));
          outside_recv_ptr += cpu_mp_fused_global_count_data[idx];
        }
        recv_src++;
      }
    }
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());

    // Step2: inside all_to_all
    int mp_fwd_count = 0;
    for (auto i = 0; i < cpu_local_count.numel(); ++i) {
      mp_fwd_count += cpu_local_count_data[i];
    }

    auto inside_send_ptr = 0;
    auto inside_recv_ptr = 0;
    auto inside_send_buf = mp_global_res.data<T>();
    auto inside_recv_buf = out->mutable_data<T>({mp_fwd_count, in_feat}, place);

    recv_src = 0;
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
    for (auto j = 0; j < outside_nranks * n_expert; ++j) {
      for (auto k = 0; k < inside_nranks; ++k) {
        int idx = k + j * inside_nranks;
        if (cpu_mp_global_count_data[idx]) {
          PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclSend(
              inside_send_buf + inside_send_ptr * in_feat,
              cpu_mp_global_count_data[idx] * in_feat, dtype, k,
              inside_comm->comm(), stream));
          inside_send_ptr += cpu_mp_global_count_data[idx];
        }
        if (cpu_local_count_data[idx]) {
          PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclRecv(
              inside_recv_buf + inside_recv_ptr * in_feat,
              cpu_local_count_data[idx] * in_feat, dtype,
              recv_src / n_expert % inside_nranks, inside_comm->comm(),
              stream));
          inside_recv_ptr += cpu_local_count_data[idx];
        }
        recv_src++;
      }
    }
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());

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
using GPUCtx = paddle::platform::CUDADeviceContext;

REGISTER_OP_CUDA_KERNEL(
    global_hierarchy_gather,
    ops::GlobalHierarchyGatherOpCUDAKernel<GPUCtx, float>,
    ops::GlobalHierarchyGatherOpCUDAKernel<GPUCtx, double>,
    ops::GlobalHierarchyGatherOpCUDAKernel<GPUCtx, int>,
    ops::GlobalHierarchyGatherOpCUDAKernel<GPUCtx, int64_t>,
    ops::GlobalHierarchyGatherOpCUDAKernel<GPUCtx, plat::float16>);
