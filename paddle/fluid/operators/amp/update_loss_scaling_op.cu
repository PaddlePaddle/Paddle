/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/amp/update_loss_scaling_op.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void GpuUpdateLossScaling(
    const bool* found_inf_data, const T* pre_loss_scaling_data,
    const int* good_in_data, const int* bad_in_data,
    const int incr_every_n_steps, const int decr_every_n_nan_or_inf,
    const float incr_ratio, const float decr_ratio,
    T* updated_loss_scaling_data, int* good_out_data, int* bad_out_data) {
  Update<T>(found_inf_data, pre_loss_scaling_data, good_in_data, bad_in_data,
            incr_every_n_steps, decr_every_n_nan_or_inf, incr_ratio, decr_ratio,
            updated_loss_scaling_data, good_out_data, bad_out_data);
}

template <typename T>
__global__ void FusedFillIf(T** outs, const size_t xs_size,
                            const int64_t* starts, const T value,
                            const bool* has_inf) {
  if (!(*has_inf)) return;

  const int tid = threadIdx.x + blockIdx.x * blockDim.x;

  // copy starts array from global memory to shared memory
  extern __shared__ int64_t s_starts[];
  for (int i = threadIdx.x; i <= xs_size; i += blockDim.x) {
    s_starts[i] = starts[i];
  }
  __syncthreads();

  const int64_t total_num = s_starts[xs_size];
  int out_index = 0;

  for (int64_t id = tid; id < total_num; id += blockDim.x * gridDim.x) {
    // get the "out" index of "id"
    // For example:
    // id = 15, starts = [0, 10, 10, 20, 30]
    // because 10 <= id < 20 ==>
    // the id element locate in the 3rd tensor (notice the 2nd tensor size is 0)
    int next_out_index = out_index;
    while (id >= s_starts[next_out_index]) next_out_index++;
    out_index = next_out_index - 1;

    // get data pointer and index
    T* out_data = outs[out_index];
    int64_t idx = id - s_starts[out_index];

    // set value
    out_data[idx] = value;
  }
}

template <typename T>
class UpdateLossScalingFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const platform::CUDADeviceContext& dev_ctx,
                  const bool* found_inf_data, const T* pre_loss_scaling_data,
                  const int* good_in_data, const int* bad_in_data,
                  const int incr_every_n_steps,
                  const int decr_every_n_nan_or_inf, const float incr_ratio,
                  const float decr_ratio, T* updated_loss_scaling_data,
                  int* good_out_data, int* bad_out_data) const {
    GpuUpdateLossScaling<T><<<1, 1, 0, dev_ctx.stream()>>>(
        found_inf_data, pre_loss_scaling_data, good_in_data, bad_in_data,
        incr_every_n_steps, decr_every_n_nan_or_inf, incr_ratio, decr_ratio,
        updated_loss_scaling_data, good_out_data, bad_out_data);
  }
};

template <typename T>
class LazyZeros<platform::CUDADeviceContext, T> {
 public:
  void operator()(const platform::CUDADeviceContext& dev_ctx,
                  const bool* found_inf_data,
                  const std::vector<const framework::Tensor*>& xs,
                  const std::vector<framework::Tensor*>& outs) const {
    size_t xs_size = xs.size();
    if (xs_size == 0) return;

    const auto& cpu_place = platform::CPUPlace();
    // alloc each tensor's start index and copy to device
    auto h_in_starts_mem =
        memory::Alloc(cpu_place, (xs_size + 1) * sizeof(int64_t));
    int64_t* h_starts = reinterpret_cast<int64_t*>(h_in_starts_mem->ptr());

    auto d_in_starts_mem =
        memory::Alloc(dev_ctx, (xs_size + 1) * sizeof(int64_t));
    int64_t* d_starts = reinterpret_cast<int64_t*>(d_in_starts_mem->ptr());

    // the start index value of each tensor is
    // the sum of previous tensor's size. For example:
    // outs = [10, 0, 10, 10] ==> starts = [0, 10, 10, 20, 30]
    h_starts[0] = 0;
    for (int i = 0; i < xs_size; i++) {
      h_starts[i + 1] = h_starts[i] + outs[i]->numel();
    }
    memory::Copy(dev_ctx.GetPlace(), d_starts, cpu_place, h_starts,
                 (xs_size + 1) * sizeof(int64_t), dev_ctx.stream());

    // copy each tensor of "outs" data address array to device
    auto h_out_addrs_mem = memory::Alloc(cpu_place, xs_size * sizeof(T*));
    T** h_out_addrs = reinterpret_cast<T**>(h_out_addrs_mem->ptr());

    auto d_out_addrs_mem = memory::Alloc(dev_ctx, xs_size * sizeof(T*));
    T** d_out_addrs = reinterpret_cast<T**>(d_out_addrs_mem->ptr());

    for (size_t i = 0; i < xs_size; ++i) {
      h_out_addrs[i] = outs[i]->mutable_data<T>(dev_ctx.GetPlace());
    }
    memory::Copy(dev_ctx.GetPlace(), d_out_addrs, cpu_place, h_out_addrs,
                 xs_size * sizeof(T*), dev_ctx.stream());

    // launch cuda kernel
    int64_t total_num = h_starts[xs_size];
    int64_t threads_per_block = std::min(static_cast<int64_t>(1024), total_num);
    int64_t elements_per_block =
        threads_per_block * 50;  // each thread deal with 50 data
    int64_t blocks_per_grid =
        (total_num + elements_per_block - 1) / elements_per_block;
    FusedFillIf<T><<<blocks_per_grid, threads_per_block,
                     (xs_size + 1) * sizeof(int64_t), dev_ctx.stream()>>>(
        d_out_addrs, xs_size, d_starts, static_cast<T>(0), found_inf_data);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
using GPU = paddle::platform::CUDADeviceContext;

REGISTER_OP_CUDA_KERNEL(update_loss_scaling,
                        ops::UpdateLossScalingKernel<GPU, float>,
                        ops::UpdateLossScalingKernel<GPU, double>,
                        ops::UpdateLossScalingKernel<GPU, plat::float16>);
