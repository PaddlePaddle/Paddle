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

#if defined(PADDLE_WITH_NCCL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#endif
#include "paddle/fluid/framework/convert_utils.h"

namespace paddle {
namespace operators {
template <typename T>
class GlobalGatherOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_NCCL)
#if NCCL_VERSION_CODE >= 2703
    auto x = ctx.Input<framework::LoDTensor>("X");
    auto local_count = ctx.Input<framework::LoDTensor>("local_count");
    auto global_count = ctx.Input<framework::LoDTensor>("global_count");
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
    auto out = ctx.Output<framework::LoDTensor>("Out");
    const int64_t* cpu_local_count_data;
    const int64_t* cpu_global_count_data;
    auto local_count_len = 0;

    framework::Tensor cpu_local_count;
    if (platform::is_cpu_place(local_count->place())) {
      cpu_local_count_data = local_count->data<int64_t>();
      local_count_len = local_count->numel();
    } else {
      framework::TensorCopySync(*local_count, platform::CPUPlace(),
                                &cpu_local_count);
      cpu_local_count_data = cpu_local_count.data<int64_t>();
      local_count_len = cpu_local_count.numel();
    }

    framework::Tensor cpu_global_count;
    if (platform::is_cpu_place(global_count->place())) {
      cpu_global_count_data = global_count->data<int64_t>();
    } else {
      framework::TensorCopySync(*global_count, platform::CPUPlace(),
                                &cpu_global_count);
      cpu_global_count_data = cpu_global_count.data<int64_t>();
    }

    ncclDataType_t dtype =
        platform::ToNCCLDataType(framework::TransToProtoVarType(x->dtype()));

    int ring_id = ctx.Attr<int>("ring_id");
    PADDLE_ENFORCE_GE(
        ring_id, 0,
        platform::errors::InvalidArgument(
            "The ring_id (%d) for global gather op must be non-negative.",
            ring_id));
    auto place = ctx.GetPlace();
    auto comm = platform::NCCLCommContext::Instance().Get(ring_id, place);
    cudaStream_t stream = nullptr;
    if (ctx.Attr<bool>("use_calc_stream")) {
      auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
      stream = static_cast<platform::CUDADeviceContext*>(dev_ctx)->stream();
    } else {
      stream = comm->stream();
    }
    int nranks = comm->nranks();
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
    auto send_buf = x->data<T>();
    auto recv_buf = out->mutable_data<T>(out_dims, place);

    for (auto i = 0; i < n_expert; ++i) {
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
      for (auto j = 0; j < nranks; ++j) {
        int idx = i + j * n_expert;
        if (cpu_global_count_data[idx]) {
          PADDLE_ENFORCE_GPU_SUCCESS(
              platform::dynload::ncclSend(send_buf + send_ptr * in_feat,
                                          cpu_global_count_data[idx] * in_feat,
                                          dtype, j, comm->comm(), stream));
          send_ptr += cpu_global_count_data[idx];
        }
        if (cpu_local_count_data[idx]) {
          PADDLE_ENFORCE_GPU_SUCCESS(
              platform::dynload::ncclRecv(recv_buf + expert_ptr[idx] * in_feat,
                                          cpu_local_count_data[idx] * in_feat,
                                          dtype, j, comm->comm(), stream));
        }
      }
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
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

REGISTER_OP_CUDA_KERNEL(global_gather, ops::GlobalGatherOpCUDAKernel<float>,
                        ops::GlobalGatherOpCUDAKernel<double>,
                        ops::GlobalGatherOpCUDAKernel<int>,
                        ops::GlobalGatherOpCUDAKernel<int64_t>,
                        ops::GlobalGatherOpCUDAKernel<plat::float16>);
