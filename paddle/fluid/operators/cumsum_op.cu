/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/cum_op.h"
#include "paddle/fluid/platform/gpu_launch_param_config.h"

using Tensor = paddle::framework::Tensor;
using LoDTensor = paddle::framework::LoDTensor;
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFILICT_FREE_OFFSET(n) \
  ((n >> LOG_NUM_BANKS) + (n) >> (2 * LOG_NUM_BANKS))
#define CUDA_ERROR(err, msg)                                            \
  {                                                                     \
    if (err != cudaSuccess) {                                           \
      printf("%s: %s in %s at line %d\n", msg, cudaGetErrorString(err), \
             __FILE__, __LINE__);                                       \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }
namespace paddle {
namespace operators {

template <typename T>

// inclusive scan
__global__ void OuterScan(const T* in, T* out, unsigned inner_dim_size,
                          unsigned outer_dim_size, unsigned scan_dim_size,
                          bool exclusive) {
  int id = blockIdx.y * blockDim.x + threadIdx.x;

  for (unsigned outer_index = blockIdx.x; outer_index < outer_dim_size;
       outer_index += gridDim.x) {
    for (unsigned inner_index = blockIdx.y * blockDim.x + threadIdx.x;
         inner_index < inner_dim_size; inner_index += gridDim.y * blockDim.x) {
      int src_index =
          outer_index * scan_dim_size * inner_dim_size + inner_index;
      int dst_index =
          outer_index * scan_dim_size * inner_dim_size + inner_index;
      T acc = 0;
      for (unsigned scan_index = 0; scan_index < scan_dim_size; ++scan_index) {
        acc = in[src_index] + acc;
        out[dst_index] = acc;
        src_index += inner_dim_size;
        dst_index += inner_dim_size;
      }
    }
  }
}

// exclusive scan
template <typename T>
__global__ void BlellochScan(const T* in, T* out, unsigned inner_dim_size,
                             unsigned outer_dim_size, unsigned scan_dim_size,
                             int size) {
  // https://stackoverflow.com/questions/27570552/templated-cuda-kernel-with-dynamic-shared-memory
  extern __shared__ __align__(sizeof(T)) unsigned char raw_tmp[];
  T* share_tmp = reinterpret_cast<T*>(raw_tmp);

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int thread_idx = threadIdx.x;
  int offset = 1;
  int element_size = inner_dim_size;
  int active_thread_size = inner_dim_size / 2;
  if (idx > active_thread_size) return;

  for (; idx < active_thread_size; idx += blockDim.x * gridDim.x) {
    for (size_t i = 0; i < outer_dim_size; i++) {
      // load data to share memory
      share_tmp[2 * thread_idx] = in[(2 * idx) + inner_dim_size * i];
      if ((2 * thread_idx + 1) < element_size) {
        share_tmp[2 * thread_idx + 1] = in[(2 * idx + 1) + inner_dim_size * i];
      }
      __syncthreads();

      // parallel reduction(up-sweep)
      for (int s = element_size >> 1; s > 0; s >>= 1) {
        if (thread_idx < s && idx < active_thread_size) {
          int ai = offset * (2 * thread_idx + 1) - 1;
          int bi = offset * (2 * thread_idx + 2) - 1;
          share_tmp[bi] += share_tmp[ai];
        }
        offset *= 2;
        __syncthreads();
      }
      // set the last element to be zero
      if (thread_idx == 0) share_tmp[element_size - 1] = 0;
      __syncthreads();

      // Down-sweep
      for (int s = 1; s < element_size; s <<= 1) {
        offset >>= 1;
        if (thread_idx < s && idx < active_thread_size) {
          int ai = offset * (2 * thread_idx + 1) - 1;
          int bi = offset * (2 * thread_idx + 2) - 1;
          T tmp = share_tmp[ai];
          share_tmp[ai] = share_tmp[bi];
          share_tmp[bi] += tmp;
        }
        __syncthreads();
      }

      // write back to memory
      if (thread_idx < active_thread_size) {
        out[(2 * idx) + inner_dim_size * i] = share_tmp[2 * thread_idx];
        out[(2 * idx + 1) + inner_dim_size * i] = share_tmp[2 * thread_idx + 1];
      }
    }
  }
}

template <typename DeviceContext, typename T>
class CumCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in = context.Input<framework::Tensor>("X");
    auto* out = context.Output<framework::Tensor>("Out");

    int axis = context.Attr<int>("axis");
    bool exclusive = context.Attr<bool>("exclusive");
    bool reverse = context.Attr<bool>("reverse");
    auto in_dims = in->dims();
    auto size = in->numel();

    if (axis == -1) {
      axis = in_dims.size() - 1;
    }
    PADDLE_ENFORCE_LT(
        axis, in_dims.size(),
        platform::errors::InvalidArgument("axis(%d) should be less than the "
                                          "dimension(%d) of the input tensor.",
                                          axis, in_dims.size()));

    unsigned scan_dim_size = in_dims[axis];
    unsigned outer_dim_size = 1;
    unsigned inner_dim_size = 1;
    // treat all dim index < axis as outer_dim_size
    for (size_t i = 0; i < axis; i++) {
      outer_dim_size *= in_dims[i];
    }
    // treat all dim index > axis as innner_dim_size
    for (size_t i = axis + 1; i < in_dims.size(); i++) {
      inner_dim_size *= in_dims[i];
    }

    T* out_data = out->mutable_data<T>(context.GetPlace());
    const T* in_data = in->data<T>();

    auto& dev_ctx = context.template device_context<DeviceContext>();
    bool optimize_condition = false;
    if (optimize_condition) {
      int mem_per_block = size * sizeof(T);
      dim3 block(1024);
      dim3 grid((size + block.x - 1) / block.x);

    } else {
      dim3 block(std::min(512u, inner_dim_size));
      dim3 grid(outer_dim_size, (inner_dim_size + block.x - 1) / block.x);
      OuterScan<T><<<block, grid, 0, dev_ctx.stream()>>>(
          in_data, out_data, inner_dim_size, outer_dim_size, scan_dim_size,
          exclusive);
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    cumsum, ops::CumCUDAKernel<paddle::platform::CUDADeviceContext, float>,
    ops::CumCUDAKernel<paddle::platform::CUDADeviceContext, double>,
    ops::CumCUDAKernel<paddle::platform::CUDADeviceContext, int>,
    ops::CumCUDAKernel<paddle::platform::CUDADeviceContext, int64_t>);
