// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>
#include <hiprand_kernel.h>
#else
#include <cuda_runtime.h>
#include <curand_kernel.h>
#endif

#include "cub/cub.cuh"
#include "math.h"
#include "paddle/fluid/memory/memory.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/hostdevice.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/graph_weighted_sample_neighbors_kernel.h"
#define SAMPLE_SIZE_THRESHOLD 1024

namespace phi {

template <typename T, bool NeedNeighbor = false>
__global__ void GetSampleCountAndNeighborCountKernel(const T* col_ptr,
                                                     const T* input_nodes,
                                                     int* actual_size,
                                                     int* neighbor_count,
                                                     int sample_size,
                                                     int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n) return;
  T nid = input_nodes[i];
  int neighbor_size = (int)(col_ptr[nid + 1] - col_ptr[nid]);
  // sample_size < 0 means sample all.
  int k = neighbor_size;
  if (sample_size >= 0) {
    k = min(neighbor_size, sample_size);
  }
  actual_size[i] = k;
  if (NeedNeighbor) {
    neighbor_count[i] = (neighbor_size <= sample_size) ? 0 : neighbor_size;
  }
}

template <typename T, unsigned int BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE) __global__
    void WeightSampleLargeKernel(T* sample_output,
                                 const int* sample_offset,
                                 const int* target_neighbor_offset,
                                 float* weight_keys_buf,
                                 const T* input_nodes,
                                 int input_node_count,
                                 const T* in_rows,
                                 const T* col_ptr,
                                 const float* edge_weight,
                                 int max_sample_count,
                                 unsigned long long random_seed) {
  int input_idx = blockIdx.x;
  if (input_idx >= input_node_count) return;
  int gidx = threadIdx.x + blockIdx.x * BLOCK_SIZE;
  IdType nid = input_nodes[input_idx];
  int neighbor_count = (int)(col_ptr[nid + 1] - col_ptr[nid]);
}

template <typename T, typename Context>
void GraphWeightedSampleNeighborsKernel(
    const Context& dev_ctx,
    const DenseTensor& row,
    const DenseTensor& col_ptr,
    const DenseTensor& edge_weights,
    const DenseTensor& x,
    const paddle::optional<DenseTensor>& eids,
    int sample_size,
    bool return_eids,
    DenseTensor* out,
    DenseTensor* out_count,
    DenseTensor* out_eids) {
  auto* row_data = row.data<T>();
  auto* col_ptr_data = col_ptr.data<T>();
  auto* weights_data = edge_weights.data<float>();
  auto* x_data = x.data<T>();
  int bs = x.dims()[0];
  int64_t len_col_ptr = col_ptr.dims()[0];

  const bool need_neighbor_count = sample_size > SAMPLE_SIZE_THRESHOLD;

  out_count->Resize({bs});
  int* out_count_data = dev_ctx.template Alloc<int>(out_count);
  int* neighbor_count_ptr = nullptr;

  int grid_size = (bs + 127) / 128;
  if (need_neighbor_count) {
    auto neighbor_count = paddle::memory::Alloc(
        dev_ctx.GetPlace(),
        (bs + 1) * sizeof(int),
        phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
    neighbor_count_ptr = reinterpret_cast<int*>(neighbor_count->ptr());
    GetSampleCountAndNeighborCountKernel<T, true>
        <<<grid_size, 128, 0, dev_ctx.stream()>>>(col_ptr_data,
                                                  x_data,
                                                  out_count_data,
                                                  neighbor_count_ptr,
                                                  sample_size,
                                                  bs);
  } else {
    GetSampleCountAndNeighborCountKernel<T, false>
        <<<grid_size, 128, 0, dev_ctx.stream()>>>(
            col_ptr_data, x_data, out_count_data, nullptr, sample_size, bs);
  }

#ifdef PADDLE_WITH_CUDA
  paddle::memory::ThrustAllocator<cudaStream_t> allocator(dev_ctx.GetPlace(),
                                                          dev_ctx.stream());
  const auto& exec_policy = thrust::cuda::par(allocator).on(dev_ctx.stream());
#else
  const auto& exec_policy = thrust::hip::par.on(dev_ctx.stream());
#endif

  thrust::exclusive_scan(
      exec_policy, sample_count, sample_count + bs + 1, sample_offset);
  int total_sample_size;

#ifdef PADDLE_WITH_CUDA
  cudaMemcpyAsync(&total_sample_size,
                  sample_offset + bs,
                  sizeof(int),
                  cudaMemcpyDeviceToHost,
                  dev_ctx.stream());
#else
  hipMemcpyAsync(&total_sample_size,
                 sample_offset + bs,
                 sizeof(int),
                 hipMemcpyDeviceToHost,
                 dev_ctx.stream());
#endif

  out->Resize({static_cast<int>(total_sample_size)});
  T* out_data = dev_ctx.template Alloc<T>(out);

  // large sample size
  if (sample_size > sample_count_threshold) {
    thrust::exclusive_scan(exec_policy,
                           neighbor_counts,
                           neighbor_counts + bs + 1,
                           neighbor_counts);
    int* neighbor_offset = neighbor_counts;
    int target_neighbor_counts;
#ifdef PADDLE_WITH_CUDA
    cudaMemcpyAsync(&target_neighbor_counts,
                    neighbor_offset + bs,
                    sizeof(int),
                    cudaMemcpyDeviceToHost,
                    dev_ctx.stream());
    cudaStreamSynchronize(dev_ctx.stream());
#else
    hipMemcpyAsync(&target_neighbor_counts,
                   neighbor_offset + bs,
                   sizeof(int),
                   hipMemcpyDeviceToHost,
                   dev_ctx.stream());
    hipStreamSynchronize(dev_ctx.stream());
#endif

    auto tmh_weights = paddle::memory::Alloc(
        dev_ctx.GetPlace(),
        target_neighbor_counts * sizeof(float),
        phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
    float* target_weights_keys_buf_ptr =
        reinterpret_cast<float*>(tmh_weights->ptr());
    constexpr int BLOCK_SIZE = 256;
    WeightSampleLargeKernel<><<<>>>();
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(graph_weighted_sample_neighbors,
                   GPU,
                   ALL_LAYOUT,
                   phi::GraphWeightedSampleNeighborsKernel,
                   int,
                   int64_t) {}
