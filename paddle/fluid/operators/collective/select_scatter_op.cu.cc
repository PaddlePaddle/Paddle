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
    auto local_expert_count = ctx.Attr<std::vector<int>>("local_expert_count");
    auto global_expert_count =
        ctx.Attr<std::vector<int>>("global_expert_count");
    // auto local_expert_count =
    // ctx.Input<framework::LoDTensor>("local_expert_count");
    // auto global_expert_count =
    // ctx.Input<framework::LoDTensor>("global_expert_count");
    auto input_buf = ctx.Input<framework::LoDTensor>("input_buf");
    auto in_feat = ctx.Attr<int>("in_feat");
    auto n_expert = ctx.Attr<int>("n_expert");
    auto world_size = ctx.Attr<int>("world_size");
    // auto in_feat = ctx.Input<framework::LoDTensor>("in_feat");
    // auto n_expert = ctx.Input<framework::LoDTensor>("n_expert");
    // auto world_size = ctx.Input<framework::LoDTensor>("world_size");
    auto out = ctx.Output<framework::LoDTensor>("Out");
    VLOG(1) << "local_input_buf";
    // int64_t in_data_numel = local_input_buf->numel();
    // Tensor cpu_in_data = new T[in_data_numel];
    // cudaMemcpy(cpu_in_data, local_input_buf, in_data_numel * sizeof(T),
    //         cudaMemcpyDeviceToHost);
    // for (auto i = 0; i < in_data_numel; i ++)
    //     VLOG(1) << cpu_in_data[i];

    framework::Tensor cpu_local_input_buf;
    framework::TensorCopy(*local_input_buf, platform::CPUPlace(),
                          &cpu_local_input_buf);
    int64_t data_numel = local_input_buf->numel();
    T* cpu_local_input_buf_data = cpu_local_input_buf.data<T>();
    for (auto i = 0; i < data_numel; i++)
      VLOG(1) << cpu_local_input_buf_data[i];
    VLOG(1) << "local_input_buf";

    // const T* local_input_buf_d = local_input_buf->data<T>();
    // VLOG(1) << "defination";

    // const int* local_expert_count_d = local_expert_count->data<int>();
    // VLOG(1) << "local_expert_count_d";
    // VLOG(1) << local_expert_count_d[0];

    // const int* global_expert_count_d = global_expert_count->data<int>();
    // VLOG(1) << "global_expert_count_d";
    // VLOG(1) << global_expert_count;
    // VLOG(1) << "in_feat_d";
    // VLOG(1) << in_feat->data<int32_t>()[0];
    // VLOG(1) << "n_expert_d";
    // VLOG(1) << n_expert[0];
    // VLOG(1) << "world_size";
    // VLOG(1) << world_size[0];
    // const int* in_feat_d = in_feat->data<int>();
    // const int* n_expert_d = n_expert->data<int>();
    // const int* world_size_d = world_size->data<int>();
    // VLOG(1) << "world_size";
    // VLOG(1) << in_feat_d[0];
    VLOG(1) << "local_input_buf type: ";
    VLOG(1) << local_input_buf->type();
    ncclDataType_t dtype = platform::ToNCCLDataType(local_input_buf->type());
    VLOG(1) << "nccl type: ";
    VLOG(1) << dtype;
    int ring_id = ctx.Attr<int>("ring_id");
    PADDLE_ENFORCE_GE(
        ring_id, 0,
        platform::errors::InvalidArgument(
            "The ring_id (%d) for selectscatter op must be non-negative.",
            ring_id));
    auto place = ctx.GetPlace();
    auto comm = platform::NCCLCommContext::Instance().Get(ring_id, place);
    // int nranks = comm->nranks();
    VLOG(1) << "defination3";
    cudaStream_t stream = nullptr;
    if (ctx.Attr<bool>("use_calc_stream")) {
      auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
      stream = static_cast<platform::CUDADeviceContext*>(dev_ctx)->stream();
    } else {
      stream = comm->stream();
    }
    framework::DDim input_buf_dims = input_buf->dims();
    framework::DDim out_dims(input_buf_dims);
    // VLOG(1) << "local_expert_count";
    // for (auto i = 0; i < (int)local_expert_count.size(); ++i)
    //   VLOG(1) << local_expert_count[i] << " ";
    // VLOG(1) << "local_expert_count";
    // VLOG(1) << "global_expert_count";
    // for (auto i = 0; i < (int)global_expert_count.size(); ++i)
    //   VLOG(1) << global_expert_count[i] << " ";
    // VLOG(1) << "global_expert_count";
    // VLOG(1) << "expert_ptr";
    int* expert_ptr = new int[n_expert * world_size];
    expert_ptr[0] = 0;
    for (auto i = 1; i < n_expert * world_size; ++i) {
      expert_ptr[i] = expert_ptr[i - 1] + local_expert_count[i - 1];
      VLOG(1) << expert_ptr[i];
    }
    VLOG(1) << "expert_ptr";
    size_t recv_ptr = 0;
    size_t recv_print_ptr = 0;
    auto send_buf = local_input_buf->data<T>();
    auto recv_buf = out->mutable_data<T>(out_dims, place);
    // VLOG(1) << "local_expert_count" << local_expert_count;
    // VLOG(1) << "global_expert_count" << global_expert_count;
    for (auto i = 0; i < n_expert; ++i) {
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclGroupStart());
      for (auto j = 0; j < world_size; ++j) {
        // VLOG(1) << "type j";
        // VLOG(1) << typeid(j).name();
        // VLOG(1) << "type j";
        int idx = i + j * n_expert;
        if (local_expert_count[idx]) {
          // VLOG(1) << "send ahah: " << idx;
          // // int64_t in_data_numel = local_input_buf->numel();
          // T* cpu_in_data = new T[local_expert_count[idx] * in_feat];
          // cudaMemcpy(cpu_in_data, send_buf + expert_ptr[idx] * in_feat,
          // local_expert_count[idx] * in_feat * sizeof(T),
          //         cudaMemcpyDeviceToHost);
          // for (auto i = 0; i < local_expert_count[idx] * in_feat; i ++)
          //     VLOG(1) << cpu_in_data[i];
          // VLOG(1) << "send ahah: " << idx;
          // 只用j是因为只有这么多张卡
          PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclSend(
              send_buf + expert_ptr[idx] * in_feat,
              local_expert_count[idx] * in_feat * sizeof(T), dtype, j,
              comm->comm(), stream));
        }
        if (global_expert_count[idx]) {
          // VLOG(1) << "recv ahah";
          PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclRecv(
              recv_buf + recv_ptr * in_feat,
              global_expert_count[idx] * in_feat * sizeof(T), dtype, j,
              comm->comm(), stream));

          // VLOG(1) << "recv ahah: " << idx;
          // // int64_t in_data_numel = local_input_buf->numel();
          // // 在这内部打印是不行的，因为还没有执行ncclGroupEnd()
          // T* cpu_in_data = new T[global_expert_count[idx] * in_feat];
          // cudaMemcpy(cpu_in_data, recv_buf + recv_ptr * in_feat,
          // global_expert_count[idx] * in_feat * sizeof(T),
          //         cudaMemcpyDeviceToHost);
          // for (auto i = 0; i < global_expert_count[idx] * in_feat; i ++)
          //     VLOG(1) << cpu_in_data[i];
          // VLOG(1) << "recv ahah: " << idx;
          recv_ptr += global_expert_count[idx];
          // VLOG(1) << "out";
          // VLOG(1) << "recv_ptr: " << recv_ptr;
          // framework::Tensor cpu_local_input_buf;
          // framework::TensorCopy(*out, platform::CPUPlace(),
          // &cpu_local_input_buf);
          // int64_t data_numel = out->numel();
          // T* cpu_local_input_buf_data = cpu_local_input_buf.data<T>();
          // for (auto i = 0; i < data_numel; i++)
          //     VLOG(1) << cpu_local_input_buf_data[i];
          // VLOG(1) << "out";
        }
      }
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclGroupEnd());
      // 打印
      for (auto j = 0; j < world_size; ++j) {
        int idx = i + j * n_expert;
        if (local_expert_count[idx]) {
          VLOG(1) << "send ahah: " << idx;
          // int64_t in_data_numel = local_input_buf->numel();
          T* cpu_in_data = new T[local_expert_count[idx] * in_feat];
          cudaMemcpy(cpu_in_data, send_buf + expert_ptr[idx] * in_feat,
                     local_expert_count[idx] * in_feat * sizeof(T),
                     cudaMemcpyDeviceToHost);
          for (auto i = 0; i < local_expert_count[idx] * in_feat; i++)
            VLOG(1) << cpu_in_data[i];
          VLOG(1) << "send ahah: " << idx;
        }
        if (global_expert_count[idx]) {
          // VLOG(1) << "recv ahah";
          // PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclRecv(
          //     recv_buf + recv_print_ptr * in_feat, global_expert_count[idx] *
          //     in_feat * sizeof(T), dtype, j, comm->comm(), stream));

          VLOG(1) << "recv ahah: " << idx;
          // int64_t in_data_numel = local_input_buf->numel();
          // 在这内部打印是不行的，因为还没有执行ncclGroupEnd()
          T* cpu_in_data = new T[global_expert_count[idx] * in_feat];
          cudaMemcpy(cpu_in_data, recv_buf + recv_print_ptr * in_feat,
                     global_expert_count[idx] * in_feat * sizeof(T),
                     cudaMemcpyDeviceToHost);
          for (auto i = 0; i < global_expert_count[idx] * in_feat; i++)
            VLOG(1) << cpu_in_data[i];
          VLOG(1) << "recv ahah: " << idx;
          recv_print_ptr += global_expert_count[idx];
        }
      }
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

REGISTER_OP_CUDA_KERNEL(select_scatter, ops::SelectScatterOpCUDAKernel<float>,
                        ops::SelectScatterOpCUDAKernel<double>,
                        ops::SelectScatterOpCUDAKernel<int>,
                        ops::SelectScatterOpCUDAKernel<int64_t>,
                        ops::SelectScatterOpCUDAKernel<plat::float16>);
