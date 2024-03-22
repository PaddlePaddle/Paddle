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

#ifdef PADDLE_WITH_CUDA
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "cub/cub.cuh"
#endif

#include "math.h"  // NOLINT
#include "paddle/common/hostdevice.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/block_radix_topk.cuh"
#include "paddle/phi/kernels/funcs/random.cuh"
#include "paddle/phi/kernels/weighted_sample_neighbors_kernel.h"
#define SAMPLE_SIZE_THRESHOLD 1024

namespace phi {

#ifdef PADDLE_WITH_CUDA
__device__ __forceinline__ float GenKeyFromWeight(
    const float weight,
    RandomNumGen& rng) {  // NOLINT
  rng.NextValue();
  float u = -rng.RandomUniformFloat(1.0f, 0.5f);
  long long random_num2 = 0;  // NOLINT
  int seed_count = -1;
  do {
    random_num2 = rng.Random64();
    seed_count++;
  } while (!random_num2);
  int one_bit = __clzll(random_num2) + seed_count * 64;
  u *= exp2f(-one_bit);
  float logk = (log1pf(u) / logf(2.0)) * (1 / weight);
  return logk;
}
#endif

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
  int neighbor_size = static_cast<int>(col_ptr[nid + 1] - col_ptr[nid]);
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

#ifdef PADDLE_WITH_CUDA
template <typename T, unsigned int BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE) __global__
    void WeightedSampleLargeKernel(T* sample_output,
                                   const int* sample_offset,
                                   const int* target_neighbor_offset,
                                   float* weight_keys_buf,
                                   const T* input_nodes,
                                   int input_node_count,
                                   const T* in_rows,
                                   const T* col_ptr,
                                   const float* edge_weight,
                                   const T* eids,
                                   int max_sample_count,
                                   unsigned long long random_seed,  // NOLINT
                                   T* out_eids,
                                   bool return_eids) {
  int i = blockIdx.x;
  if (i >= input_node_count) return;
  int gidx = threadIdx.x + blockIdx.x * BLOCK_SIZE;
  T nid = input_nodes[i];
  T start = col_ptr[nid + 1];
  T end = col_ptr[nid];
  int neighbor_count = static_cast<int>(end - start);

  float* weight_keys_local_buff = weight_keys_buf + target_neighbor_offset[i];
  int offset = sample_offset[i];
  if (neighbor_count <= max_sample_count) {
    for (int j = threadIdx.x; j < neighbor_count; j += BLOCK_SIZE) {
      sample_output[offset + j] = in_rows[start + j];
      if (return_eids) {
        out_eids[offset + j] = eids[start + j];
      }
    }
  } else {
    RandomNumGen rng(gidx, random_seed);
    for (int j = threadIdx.x; j < neighbor_count; j += BLOCK_SIZE) {
      float thread_weight = edge_weight[start + j];
      weight_keys_local_buff[j] =
          static_cast<float>(GenKeyFromWeight(thread_weight, rng));
    }
    __syncthreads();

    float topk_val;
    bool topk_is_unique;

    using BlockRadixSelectT =
        paddle::framework::BlockRadixTopKGlobalMemory<float, BLOCK_SIZE, true>;
    __shared__ typename BlockRadixSelectT::TempStorage share_storage;

    BlockRadixSelectT{share_storage}.radixTopKGetThreshold(
        weight_keys_local_buff,
        max_sample_count,
        neighbor_count,
        topk_val,
        topk_is_unique);
    __shared__ int cnt;

    if (threadIdx.x == 0) {
      cnt = 0;
    }
    __syncthreads();

    // We use atomicAdd 1 operations instead of binaryScan to calculate the
    // write index, since we do not need to keep the relative positions of
    // element.

    if (topk_is_unique) {
      for (int j = threadIdx.x; j < neighbor_count; j += BLOCK_SIZE) {
        float key = weight_keys_local_buff[j];
        bool has_topk = (key >= topk_val);

        if (has_topk) {
          int write_index = atomicAdd(&cnt, 1);
          sample_output[offset + write_index] = in_rows[start + j];
          if (return_eids) {
            out_eids[offset + write_index] = eids[start + j];
          }
        }
      }
    } else {
      for (int j = threadIdx.x; j < neighbor_count; j += BLOCK_SIZE) {
        float key = weight_keys_local_buff[j];
        bool has_topk = (key > topk_val);

        if (has_topk) {
          int write_index = atomicAdd(&cnt, 1);
          sample_output[offset + write_index] = in_rows[start + j];
          if (return_eids) {
            out_eids[offset + write_index] = eids[start + j];
          }
        }
      }
      __syncthreads();

      for (int j = threadIdx.x; j < neighbor_count; j += BLOCK_SIZE) {
        float key = weight_keys_local_buff[j];
        bool has_topk = (key == topk_val);
        if (has_topk) {
          int write_index = atomicAdd(&cnt, 1);
          if (write_index >= max_sample_count) {
            break;
          }
          sample_output[offset + write_index] = in_rows[start + j];
          if (return_eids) {
            out_eids[offset + write_index] = eids[start + j];
          }
        }
      }
    }
  }
}
#endif

template <typename T>
__global__ void SampleAllKernel(T* sample_output,
                                const int* sample_offset,
                                const T* input_nodes,
                                int input_node_count,
                                const T* in_rows,
                                const T* col_ptr,
                                const T* eids,
                                T* out_eids,
                                bool return_eids) {
  int i = blockIdx.x;
  if (i >= input_node_count) return;
  T nid = input_nodes[i];
  T start = col_ptr[nid + 1];
  T end = col_ptr[nid];
  int neighbor_count = static_cast<int>(end - start);
  if (neighbor_count <= 0) return;
  int offset = sample_offset[i];
  for (int j = threadIdx.x; j < neighbor_count; j += blockDim.x) {
    sample_output[offset + j] = in_rows[start + j];
    if (return_eids) {
      out_eids[offset + j] = eids[start + j];
    }
  }
}

// A-RES algorithm
#ifdef PADDLE_WITH_CUDA
template <typename T, unsigned int ITEMS_PER_THREAD, unsigned int BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE) __global__
    void WeightedSampleKernel(T* sample_output,
                              const int* sample_offset,
                              const T* input_nodes,
                              int input_node_count,
                              const T* in_rows,
                              const T* col_ptr,
                              const float* edge_weight,
                              const T* eids,
                              int max_sample_count,
                              unsigned long long random_seed,  // NOLINT
                              T* out_eids,
                              bool return_eids) {
  int i = blockIdx.x;
  if (i >= input_node_count) return;
  int gidx = threadIdx.x + blockIdx.x * BLOCK_SIZE;
  T nid = input_nodes[i];
  T start = col_ptr[nid];
  T end = col_ptr[nid + 1];
  int neighbor_count = static_cast<int>(end - start);
  int offset = sample_offset[i];

  if (neighbor_count <= max_sample_count) {
    for (int j = threadIdx.x; j < neighbor_count; j += BLOCK_SIZE) {
      sample_output[offset + j] = in_rows[start + j];
      if (return_eids) {
        out_eids[offset + j] = eids[start + j];
      }
    }
  } else {
    RandomNumGen rng(gidx, random_seed);
    float weight_keys[ITEMS_PER_THREAD];
    int neighbor_idxs[ITEMS_PER_THREAD];
    using BlockRadixTopKT = paddle::framework::
        BlockRadixTopKRegister<float, BLOCK_SIZE, ITEMS_PER_THREAD, true, int>;
    __shared__ typename BlockRadixTopKT::TempStorage sort_tmp_storage;

    const int tx = threadIdx.x;
#pragma unroll
    for (int j = 0; j < ITEMS_PER_THREAD; j++) {
      int idx = BLOCK_SIZE * j + tx;
      if (idx < neighbor_count) {
        float thread_weight = edge_weight[start + idx];
        weight_keys[j] = GenKeyFromWeight(thread_weight, rng);
        neighbor_idxs[j] = idx;
      }
    }
    const int valid_count = (neighbor_count < (BLOCK_SIZE * ITEMS_PER_THREAD))
                                ? neighbor_count
                                : (BLOCK_SIZE * ITEMS_PER_THREAD);
    BlockRadixTopKT{sort_tmp_storage}.radixTopKToStriped(
        weight_keys, neighbor_idxs, max_sample_count, valid_count);
    __syncthreads();
    const int stride = BLOCK_SIZE * ITEMS_PER_THREAD - max_sample_count;

    for (int idx_offset = ITEMS_PER_THREAD * BLOCK_SIZE;
         idx_offset < neighbor_count;
         idx_offset += stride) {
#pragma unroll
      for (int j = 0; j < ITEMS_PER_THREAD; j++) {
        int local_idx = BLOCK_SIZE * j + tx - max_sample_count;
        int target_idx = idx_offset + local_idx;
        if (local_idx >= 0 && target_idx < neighbor_count) {
          float thread_weight = edge_weight[start + target_idx];
          weight_keys[j] = GenKeyFromWeight(thread_weight, rng);
          neighbor_idxs[j] = target_idx;
        }
      }
      const int iter_valid_count =
          ((neighbor_count - idx_offset) >= stride)
              ? (BLOCK_SIZE * ITEMS_PER_THREAD)
              : (max_sample_count + neighbor_count - idx_offset);
      BlockRadixTopKT{sort_tmp_storage}.radixTopKToStriped(
          weight_keys, neighbor_idxs, max_sample_count, iter_valid_count);
      __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < ITEMS_PER_THREAD; j++) {
      int idx = j * BLOCK_SIZE + tx;
      if (idx < max_sample_count) {
        sample_output[offset + idx] = in_rows[start + neighbor_idxs[j]];
        if (return_eids) {
          out_eids[offset + idx] = eids[start + neighbor_idxs[j]];
        }
      }
    }
  }
}
#endif

template <typename T, typename Context>
void WeightedSampleNeighborsKernel(const Context& dev_ctx,
                                   const DenseTensor& row,
                                   const DenseTensor& col_ptr,
                                   const DenseTensor& edge_weight,
                                   const DenseTensor& x,
                                   const paddle::optional<DenseTensor>& eids,
                                   int sample_size,
                                   bool return_eids,
                                   DenseTensor* out,
                                   DenseTensor* out_count,
                                   DenseTensor* out_eids) {
  auto* row_data = row.data<T>();
  auto* col_ptr_data = col_ptr.data<T>();
  auto* weights_data = edge_weight.data<float>();
  auto* x_data = x.data<T>();
  auto* eids_data =
      (eids.get_ptr() == nullptr ? nullptr : eids.get_ptr()->data<T>());
  int bs = x.dims()[0];

  thread_local std::random_device rd;
  thread_local std::mt19937 gen(rd());
  thread_local std::uniform_int_distribution<unsigned long long>  // NOLINT
      distrib;
  unsigned long long random_seed = distrib(gen);  // NOLINT
  const bool need_neighbor_count = sample_size > SAMPLE_SIZE_THRESHOLD;

  out_count->Resize({bs});
  int* out_count_data =
      dev_ctx.template Alloc<int>(out_count);  // finally copy sample_count
  int* neighbor_count_ptr = nullptr;
  std::shared_ptr<phi::Allocation> neighbor_count;
  auto sample_count = phi::memory_utils::Alloc(
      dev_ctx.GetPlace(),
      (bs + 1) * sizeof(int),
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  int* sample_count_ptr = reinterpret_cast<int*>(sample_count->ptr());

  int grid_size = (bs + 127) / 128;
  if (need_neighbor_count) {
    neighbor_count = phi::memory_utils::AllocShared(
        dev_ctx.GetPlace(),
        (bs + 1) * sizeof(int),
        phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
    neighbor_count_ptr = reinterpret_cast<int*>(neighbor_count->ptr());
    GetSampleCountAndNeighborCountKernel<T, true>
        <<<grid_size, 128, 0, dev_ctx.stream()>>>(col_ptr_data,
                                                  x_data,
                                                  sample_count_ptr,
                                                  neighbor_count_ptr,
                                                  sample_size,
                                                  bs);
  } else {
    GetSampleCountAndNeighborCountKernel<T, false>
        <<<grid_size, 128, 0, dev_ctx.stream()>>>(
            col_ptr_data, x_data, sample_count_ptr, nullptr, sample_size, bs);
  }

  auto sample_offset = phi::memory_utils::Alloc(
      dev_ctx.GetPlace(),
      (bs + 1) * sizeof(int),
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  int* sample_offset_ptr = reinterpret_cast<int*>(sample_offset->ptr());

#ifdef PADDLE_WITH_CUDA
  const auto& exec_policy = thrust::cuda::par.on(dev_ctx.stream());
#else
  const auto& exec_policy = thrust::hip::par.on(dev_ctx.stream());
#endif
  thrust::exclusive_scan(exec_policy,
                         sample_count_ptr,
                         sample_count_ptr + bs + 1,
                         sample_offset_ptr);
  int total_sample_size = 0;
#ifdef PADDLE_WITH_CUDA
  cudaMemcpyAsync(&total_sample_size,
                  sample_offset_ptr + bs,
                  sizeof(int),
                  cudaMemcpyDeviceToHost,
                  dev_ctx.stream());
  cudaMemcpyAsync(out_count_data,
                  sample_count_ptr,
                  sizeof(int) * bs,
                  cudaMemcpyDeviceToDevice,
                  dev_ctx.stream());
  cudaStreamSynchronize(dev_ctx.stream());
#else
  hipMemcpyAsync(&total_sample_size,
                 sample_offset_ptr + bs,
                 sizeof(int),
                 hipMemcpyDeviceToHost,
                 dev_ctx.stream());
  hipMemcpyAsync(out_count_data,
                 sample_count_ptr,
                 sizeof(int) * bs,
                 hipMemcpyDeviceToDevice,
                 dev_ctx.stream());
  hipStreamSynchronize(dev_ctx.stream());
#endif

  out->Resize({static_cast<int>(total_sample_size)});
  T* out_data = dev_ctx.template Alloc<T>(out);
  T* out_eids_data = nullptr;
  if (return_eids) {
    out_eids->Resize({static_cast<int>(total_sample_size)});
    out_eids_data = dev_ctx.template Alloc<T>(out_eids);
  }

  // large sample size
#ifdef PADDLE_WITH_CUDA
  if (sample_size > SAMPLE_SIZE_THRESHOLD) {
    thrust::exclusive_scan(exec_policy,
                           neighbor_count_ptr,
                           neighbor_count_ptr + bs + 1,
                           neighbor_count_ptr);
    int* neighbor_offset = neighbor_count_ptr;
    int target_neighbor_counts;
    cudaMemcpyAsync(&target_neighbor_counts,
                    neighbor_offset + bs,
                    sizeof(int),
                    cudaMemcpyDeviceToHost,
                    dev_ctx.stream());
    cudaStreamSynchronize(dev_ctx.stream());

    auto tmh_weights = phi::memory_utils::Alloc(
        dev_ctx.GetPlace(),
        target_neighbor_counts * sizeof(float),
        phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
    float* target_weights_keys_buf_ptr =
        reinterpret_cast<float*>(tmh_weights->ptr());
    constexpr int BLOCK_SIZE = 256;
    WeightedSampleLargeKernel<T, BLOCK_SIZE>
        <<<bs, BLOCK_SIZE, 0, dev_ctx.stream()>>>(out_data,
                                                  sample_offset_ptr,
                                                  neighbor_offset,
                                                  target_weights_keys_buf_ptr,
                                                  x_data,
                                                  bs,
                                                  row_data,
                                                  col_ptr_data,
                                                  weights_data,
                                                  eids_data,
                                                  sample_size,
                                                  random_seed,
                                                  out_eids_data,
                                                  return_eids);
    cudaStreamSynchronize(dev_ctx.stream());
  } else if (sample_size <= 0) {
    SampleAllKernel<T><<<bs, 64, 0, dev_ctx.stream()>>>(out_data,
                                                        sample_offset_ptr,
                                                        x_data,
                                                        bs,
                                                        row_data,
                                                        col_ptr_data,
                                                        eids_data,
                                                        out_eids_data,
                                                        return_eids);
    cudaStreamSynchronize(dev_ctx.stream());
  } else {  // sample_size < sample_count_threshold
    using WeightedSampleFuncType = void (*)(T*,
                                            const int*,
                                            const T*,
                                            int,
                                            const T*,
                                            const T*,
                                            const float*,
                                            const T*,
                                            int,
                                            unsigned long long,  // NOLINT
                                            T*,
                                            bool);
    static const WeightedSampleFuncType func_array[7] = {
        WeightedSampleKernel<T, 4, 128>,
        WeightedSampleKernel<T, 6, 128>,
        WeightedSampleKernel<T, 4, 256>,
        WeightedSampleKernel<T, 5, 256>,
        WeightedSampleKernel<T, 6, 256>,
        WeightedSampleKernel<T, 8, 256>,
        WeightedSampleKernel<T, 8, 512>,
    };
    const int block_sizes[7] = {128, 128, 256, 256, 256, 256, 512};
    auto choose_func_idx = [](int sample_size) {
      if (sample_size <= 128) {
        return 0;
      }
      if (sample_size <= 384) {
        return (sample_size - 129) / 64 + 4;
      }
      if (sample_size <= 512) {
        return 5;
      } else {
        return 6;
      }
    };
    int func_idx = choose_func_idx(sample_size);
    int block_size = block_sizes[func_idx];
    func_array[func_idx]<<<bs, block_size, 0, dev_ctx.stream()>>>(
        out_data,
        sample_offset_ptr,
        x_data,
        bs,
        row_data,
        col_ptr_data,
        weights_data,
        eids_data,
        sample_size,
        random_seed,
        out_eids_data,
        return_eids);
    cudaStreamSynchronize(dev_ctx.stream());
  }
#endif
}

}  // namespace phi

PD_REGISTER_KERNEL(weighted_sample_neighbors,
                   GPU,
                   ALL_LAYOUT,
                   phi::WeightedSampleNeighborsKernel,
                   int,
                   int64_t) {}
