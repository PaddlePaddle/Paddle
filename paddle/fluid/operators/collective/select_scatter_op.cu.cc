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

#include "paddle/fluid/operators/collective/select_scatter_op.h"

#if defined(PADDLE_WITH_NCCL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/nccl_helper.h"
#endif

namespace paddle {
namespace operators {

template <typename T>
class SelectScatterOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_NCCL)
#if NCCL_VERSION_CODE >= 2703
    auto local_input_buf = ctx.Input<framework::LoDTensor>("local_input_buf");
    auto local_expert_count =
        ctx.Input<framework::LoDTensor>("local_expert_count");
    auto global_expert_count =
        ctx.Input<framework::LoDTensor>("global_expert_count");
    auto input_buf = ctx.Input<framework::LoDTensor>("input_buf");
    auto in_feat = ctx.Input<framework::LoDTensor>("in_feat");
    auto n_expert = ctx.Input<framework::LoDTensor>("n_expert");
    auto world_size = ctx.Input<framework::LoDTensor>("world_size");
    auto out = ctx.Output<framework::LoDTensor>("Out");
    // const T* local_input_buf_d = local_input_buf->data<T>();
    const int64* local_expert_count_d = local_expert_count->data<int64>();
    const int64* global_expert_count_d = global_expert_count->data<int64>();
    const int in_feat_d = in_feat->data<int>()[0];
    const int n_expert_d = n_expert->data<int>()[0];
    const int world_size_d = world_size->data<int>()[0];

    ncclDataType_t dtype = platform::ToNCCLDataType(input_buf->type());
    int ring_id = ctx.Attr<int>("ring_id");
    PADDLE_ENFORCE_GE(
        ring_id, 0,
        platform::errors::InvalidArgument(
            "The ring_id (%d) for selectscatter op must be non-negative.",
            ring_id));
    auto place = ctx.GetPlace();
    auto comm = platform::NCCLCommContext::Instance().Get(ring_id, place);
    // int nranks = comm->nranks();

    cudaStream_t stream = nullptr;
    if (ctx.Attr<bool>("use_calc_stream")) {
      auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
      stream = static_cast<platform::CUDADeviceContext*>(dev_ctx)->stream();
    } else {
      stream = comm->stream();
    }
    framework::DDim input_buf_dims = input_buf->dims();
    framework::DDim out_dims(input_buf_dims);

    int64* expert_ptr = new int64[n_expert_d * world_size_d];
    expert_ptr[0] = 0;
    for (auto i = 1; i < n_expert_d * world_size_d; ++i) {
      expert_ptr[i] = expert_ptr[i - 1] + local_expert_count_d[i - 1];
    }
    int recv_ptr = 0;
    auto send_buf = local_input_buf->data<T>();
    auto recv_buf = out->mutable_data<T>(out_dims, place);

    for (auto i = 0; i < n_expert_d; ++i) {
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclGroupStart());
      for (auto j = 0; j < world_size_d; ++j) {
        int idx = i + j * n_expert_d;
        if (local_expert_count_d[idx]) {
          PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclSend(
              send_buf + expert_ptr[idx] * in_feat_d,
              local_expert_count_d[idx] * in_feat_d * sizeof(T), ncclChar, j,
              comm->comm(), stream));
        }
        if (global_expert_count_d[idx]) {
          PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclRecv(
              recv_buf + recv_ptr * in_feat_d,
              global_expert_count_d[idx] * in_feat_d * sizeof(T), dtype, j,
              comm->comm(), stream));
          recv_ptr += global_expert_count_d[idx];
        }
      }
    }
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclGroupEnd());
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

REGISTER_OP_CUDA_KERNEL(selectscatter, ops::SelectScatterOpCUDAKernel<float>,
                        ops::SelectScatterOpCUDAKernel<double>,
                        ops::SelectScatterOpCUDAKernel<int>,
                        ops::SelectScatterOpCUDAKernel<int64_t>,
                        ops::SelectScatterOpCUDAKernel<plat::float16>);
