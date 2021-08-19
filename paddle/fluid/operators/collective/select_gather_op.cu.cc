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

#include "paddle/fluid/operators/collective/select_gather_op.h"

#if defined(PADDLE_WITH_NCCL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/nccl_helper.h"
#endif

namespace paddle {
namespace operators {
template <typename T>
class SelectGatherOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_NCCL)
#if NCCL_VERSION_CODE >= 2703
    auto output_buf = ctx.Input<framework::LoDTensor>("output_buf");
    auto local_expert_count =
        ctx.Input<framework::LoDTensor>("local_expert_count");
    auto global_expert_count =
        ctx.Input<framework::LoDTensor>("global_expert_count");
    // auto local_expert_count =
    // ctx.Attr<std::vector<int>>("local_expert_count");
    // auto global_expert_count =
    // ctx.Attr<std::vector<int>>("global_expert_count");
    auto out_feat = static_cast<int>(ctx.Attr<int>("out_feat"));
    auto n_expert = static_cast<int>(ctx.Attr<int>("n_expert"));
    auto world_size = static_cast<int>(ctx.Attr<int>("world_size"));
    auto out = ctx.Output<framework::LoDTensor>("Out");

    framework::Tensor cpu_local_expert_count;
    framework::TensorCopy(*local_expert_count, platform::CPUPlace(),
                          &cpu_local_expert_count);
    framework::Tensor cpu_global_expert_count;
    framework::TensorCopy(*global_expert_count, platform::CPUPlace(),
                          &cpu_global_expert_count);
    int* cpu_local_expert_count_data = cpu_local_expert_count.data<int>();
    int* cpu_global_expert_count_data = cpu_global_expert_count.data<int>();
    // int64_t data_numel = local_input_buf->numel();
    // T* cpu_local_input_buf_data = cpu_local_input_buf.data<T>();
    // for (auto i = 0; i < data_numel; i++)
    //     VLOG(1) << cpu_local_input_buf_data[i];
    // VLOG(1) << "local_input_buf";

    ncclDataType_t dtype = platform::ToNCCLDataType(output_buf->type());

    int ring_id = ctx.Attr<int>("ring_id");
    PADDLE_ENFORCE_GE(
        ring_id, 0,
        platform::errors::InvalidArgument(
            "The ring_id (%d) for select gather op must be non-negative.",
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

    auto fwd_expert_count = 0;
    auto global_expert_count_len = cpu_local_expert_count.numel();
    for (auto i = 0; i < global_expert_count_len; ++i)
      fwd_expert_count += cpu_local_expert_count_data[i];
    framework::DDim out_dims =
        framework::make_ddim({fwd_expert_count, out_feat});
    int* expert_ptr = new int[n_expert * world_size];
    expert_ptr[0] = 0;
    auto tot_experts = n_expert * world_size;
    for (auto i = 1; i < tot_experts; ++i) {
      expert_ptr[i] = expert_ptr[i - 1] + cpu_local_expert_count_data[i - 1];
    }
    auto send_ptr = 0;
    auto send_buf = output_buf->data<T>();
    auto recv_buf = out->mutable_data<T>(out_dims, place);

    for (auto i = 0; i < n_expert; ++i) {
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclGroupStart());
      for (auto j = 0; j < world_size; ++j) {
        int idx = i + j * n_expert;
        if (cpu_global_expert_count_data[idx]) {
          PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclSend(
              send_buf + send_ptr * out_feat,
              cpu_global_expert_count_data[idx] * out_feat, dtype, j,
              comm->comm(), stream));
          send_ptr += cpu_global_expert_count_data[idx];
        }
        if (cpu_local_expert_count_data[idx]) {
          PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclRecv(
              recv_buf + expert_ptr[idx] * out_feat,
              cpu_local_expert_count_data[idx] * out_feat, dtype, j,
              comm->comm(), stream));
        }
      }
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclGroupEnd());
      PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamSynchronize(stream));
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

REGISTER_OP_CUDA_KERNEL(select_gather, ops::SelectGatherOpCUDAKernel<float>,
                        ops::SelectGatherOpCUDAKernel<double>,
                        ops::SelectGatherOpCUDAKernel<int>,
                        ops::SelectGatherOpCUDAKernel<int64_t>,
                        ops::SelectGatherOpCUDAKernel<plat::float16>);
