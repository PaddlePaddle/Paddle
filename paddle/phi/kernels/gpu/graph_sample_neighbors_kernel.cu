// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/graph_sample_neighbors_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/hostdevice.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T>
struct DegreeFunctor {
  const T* col_ptr;
  HOSTDEVICE explicit inline DegreeFunctor(const T* x) { this->col_ptr = x; }
  HOSTDEVICE inline int operator()(T i) const {
    return col_ptr[i + 1] - col_ptr[i];
  }
};

struct MaxFunctor {
  int cap;
  HOSTDEVICE explicit inline MaxFunctor(int cap) { this->cap = cap; }
  HOSTDEVICE inline int operator()(int x) const {
    if (x > cap) {
      return cap;
    }
    return x;
  }
};

template <typename T, int WARP_SIZE, int BLOCK_WARPS, int TILE_SIZE>
__global__ void SampleKernel(const uint64_t rand_seed,
                             int k,
                             const int64_t num_nodes,
                             const T* nodes,
                             const T* row,
                             const T* col_ptr,
                             const T* eids,
                             T* output,
                             T* output_eids,
                             int* output_ptr,
                             bool return_eids) {
  assert(blockDim.x == WARP_SIZE);
  assert(blockDim.y == BLOCK_WARPS);

  int64_t out_row = blockIdx.x * TILE_SIZE + threadIdx.y;
  const int64_t last_row =
      min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_nodes);
#ifdef PADDLE_WITH_HIP
  hiprandState rng;
  hiprand_init(rand_seed * gridDim.x + blockIdx.x,
               threadIdx.y * WARP_SIZE + threadIdx.x,
               0,
               &rng);
#else
  curandState rng;
  curand_init(rand_seed * gridDim.x + blockIdx.x,
              threadIdx.y * WARP_SIZE + threadIdx.x,
              0,
              &rng);
#endif

  while (out_row < last_row) {
    T node = nodes[out_row];
    T in_row_start = col_ptr[node];
    int deg = col_ptr[node + 1] - in_row_start;
    int out_row_start = output_ptr[out_row];

    if (deg <= k) {
      for (int idx = threadIdx.x; idx < deg; idx += WARP_SIZE) {
        output[out_row_start + idx] = row[in_row_start + idx];
        if (return_eids) {
          output_eids[out_row_start + idx] = eids[in_row_start + idx];
        }
      }
    } else {
      for (int idx = threadIdx.x; idx < k; idx += WARP_SIZE) {
        output[out_row_start + idx] = idx;
      }
#ifdef PADDLE_WITH_CUDA
      __syncwarp();
#endif

      for (int idx = k + threadIdx.x; idx < deg; idx += WARP_SIZE) {
#ifdef PADDLE_WITH_HIP
        const int num = hiprand(&rng) % (idx + 1);
#else
        const int num = curand(&rng) % (idx + 1);
#endif
        if (num < k) {
          atomicMax(reinterpret_cast<unsigned int*>(  // NOLINT
                        output + out_row_start + num),
                    static_cast<unsigned int>(idx));  // NOLINT
        }
      }
#ifdef PADDLE_WITH_CUDA
      __syncwarp();
#endif

      for (int idx = threadIdx.x; idx < k; idx += WARP_SIZE) {
        T perm_idx = output[out_row_start + idx] + in_row_start;
        output[out_row_start + idx] = row[perm_idx];
        if (return_eids) {
          output_eids[out_row_start + idx] = eids[perm_idx];
        }
      }
    }

    out_row += BLOCK_WARPS;
  }
}

template <typename T, typename Context>
int GetTotalSampleNum(const thrust::device_ptr<const T> input,
                      const T* col_ptr,
                      thrust::device_ptr<int> output_count,
                      int sample_size,
                      int bs) {
  thrust::transform(input, input + bs, output_count, DegreeFunctor<T>(col_ptr));
  if (sample_size >= 0) {
    thrust::transform(
        output_count, output_count + bs, output_count, MaxFunctor(sample_size));
  }
  int total_sample_num = thrust::reduce(output_count, output_count + bs);
  return total_sample_num;
}

template <typename T, typename Context>
void SampleNeighbors(const Context& dev_ctx,
                     const T* row,
                     const T* col_ptr,
                     const T* eids,
                     const thrust::device_ptr<const T> input,
                     thrust::device_ptr<T> output,
                     thrust::device_ptr<int> output_count,
                     thrust::device_ptr<T> output_eids,
                     int sample_size,
                     int bs,
                     int total_sample_num,
                     bool return_eids) {
  thrust::device_vector<int> output_ptr;
  output_ptr.resize(bs);
  thrust::exclusive_scan(
      output_count, output_count + bs, output_ptr.begin(), 0);

  constexpr int WARP_SIZE = 32;
  constexpr int BLOCK_WARPS = 128 / WARP_SIZE;
  constexpr int TILE_SIZE = BLOCK_WARPS * 16;
  const dim3 block(WARP_SIZE, BLOCK_WARPS);
  const dim3 grid((bs + TILE_SIZE - 1) / TILE_SIZE);
  SampleKernel<T,
               WARP_SIZE,
               BLOCK_WARPS,
               TILE_SIZE><<<grid, block, 0, dev_ctx.stream()>>>(
      0,
      sample_size,
      bs,
      thrust::raw_pointer_cast(input),
      row,
      col_ptr,
      eids,
      thrust::raw_pointer_cast(output),
      thrust::raw_pointer_cast(output_eids),
      thrust::raw_pointer_cast(output_ptr.data()),
      return_eids);
}

template <typename T, int WARP_SIZE, int BLOCK_WARPS, int TILE_SIZE>
__global__ void FisherYatesSampleKernel(const uint64_t rand_seed,
                                        int k,
                                        const int64_t num_rows,
                                        const T* in_rows,
                                        T* src,
                                        const T* dst_count) {
  assert(blockDim.x == WARP_SIZE);
  assert(blockDim.y == BLOCK_WARPS);

  int64_t out_row = blockIdx.x * TILE_SIZE + threadIdx.y;
  const int64_t last_row =
      min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_rows);
#ifdef PADDLE_WITH_HIP
  hiprandState rng;
  hiprand_init(
      rand_seed * gridDim.x + blockIdx.x, threadIdx.y + threadIdx.x, 0, &rng);
#else
  curandState rng;
  curand_init(
      rand_seed * gridDim.x + blockIdx.x, threadIdx.y + threadIdx.x, 0, &rng);
#endif

  while (out_row < last_row) {
    const T row = in_rows[out_row];
    const T in_row_start = dst_count[row];
    const int deg = dst_count[row + 1] - in_row_start;
    int split;
    if (k < deg) {
      if (deg < 2 * k) {
        split = k;
      } else {
        split = deg - k;
      }
      for (int idx = split + threadIdx.x; idx <= deg - 1; idx += WARP_SIZE) {
#ifdef PADDLE_WITH_HIP
        const int num = hiprand(&rng) % (idx + 1);
#else
        const int num = curand(&rng) % (idx + 1);
#endif
        src[in_row_start + idx] = static_cast<T>(
            atomicExch(reinterpret_cast<unsigned long long int*>(  // NOLINT
                           src + in_row_start + num),
                       static_cast<unsigned long long int>(  //  NOLINT
                           src[in_row_start + idx])));
      }
#ifdef PADDLE_WITH_CUDA
      __syncwarp();
#endif
    }
    out_row += BLOCK_WARPS;
  }
}

template <typename T, int WARP_SIZE, int BLOCK_WARPS, int TILE_SIZE>
__global__ void GatherEdge(int k,
                           int64_t num_rows,
                           const T* in_rows,
                           const T* src,
                           const T* dst_count,
                           const T* eids,
                           T* outputs,
                           T* output_eids,
                           int* output_ptr,
                           T* perm_data,
                           bool return_eids) {
  assert(blockDim.x == WARP_SIZE);
  assert(blockDim.y == BLOCK_WARPS);

  int64_t out_row = blockIdx.x * TILE_SIZE + threadIdx.y;
  const int64_t last_row =
      min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_rows);

  while (out_row < last_row) {
    const T row = in_rows[out_row];
    const T in_row_start = dst_count[row];
    const int deg = dst_count[row + 1] - in_row_start;
    const T out_row_start = output_ptr[out_row];

    if (deg <= k) {
      for (int idx = threadIdx.x; idx < deg; idx += WARP_SIZE) {
        outputs[out_row_start + idx] = src[in_row_start + idx];
        if (return_eids) {
          output_eids[out_row_start + idx] = eids[in_row_start + idx];
        }
      }
    } else {
      int split = k;
      int begin, end;
      if (deg < 2 * k) {
        begin = 0;
        end = k;
      } else {
        begin = deg - k;
        end = deg;
      }

      for (int idx = begin + threadIdx.x; idx < end; idx += WARP_SIZE) {
        outputs[out_row_start + idx - begin] =
            src[perm_data[in_row_start + idx]];
        if (return_eids) {
          output_eids[out_row_start + idx - begin] =
              eids[perm_data[in_row_start + idx]];
        }
      }
    }
    out_row += BLOCK_WARPS;
  }
}

template <typename T, typename Context>
void FisherYatesSampleNeighbors(const Context& dev_ctx,
                                const T* row,
                                const T* col_ptr,
                                const T* eids,
                                T* perm_data,
                                const thrust::device_ptr<const T> input,
                                thrust::device_ptr<T> output,
                                thrust::device_ptr<int> output_count,
                                thrust::device_ptr<T> output_eids,
                                int sample_size,
                                int bs,
                                int total_sample_num,
                                bool return_eids) {
  thrust::device_vector<int> output_ptr;
  output_ptr.resize(bs);
  thrust::exclusive_scan(
      output_count, output_count + bs, output_ptr.begin(), 0);

  constexpr int WARP_SIZE = 32;
  constexpr int BLOCK_WARPS = 128 / WARP_SIZE;
  constexpr int TILE_SIZE = BLOCK_WARPS * 16;
  const dim3 block(WARP_SIZE, BLOCK_WARPS);
  const dim3 grid((bs + TILE_SIZE - 1) / TILE_SIZE);

  FisherYatesSampleKernel<T,
                          WARP_SIZE,
                          BLOCK_WARPS,
                          TILE_SIZE><<<grid, block, 0, dev_ctx.stream()>>>(
      0, sample_size, bs, thrust::raw_pointer_cast(input), perm_data, col_ptr);

  GatherEdge<T,
             WARP_SIZE,
             BLOCK_WARPS,
             TILE_SIZE><<<grid, block, 0, dev_ctx.stream()>>>(
      sample_size,
      bs,
      thrust::raw_pointer_cast(input),
      row,
      col_ptr,
      eids,
      thrust::raw_pointer_cast(output),
      thrust::raw_pointer_cast(output_eids),
      thrust::raw_pointer_cast(output_ptr.data()),
      perm_data,
      return_eids);
}

template <typename T, typename Context>
void GraphSampleNeighborsKernel(
    const Context& dev_ctx,
    const DenseTensor& row,
    const DenseTensor& col_ptr,
    const DenseTensor& x,
    paddle::optional<const DenseTensor&> eids,
    paddle::optional<const DenseTensor&> perm_buffer,
    int sample_size,
    bool return_eids,
    bool flag_perm_buffer,
    DenseTensor* out,
    DenseTensor* out_count,
    DenseTensor* out_eids) {
  auto* row_data = row.data<T>();
  auto* col_ptr_data = col_ptr.data<T>();
  auto* x_data = x.data<T>();
  int bs = x.dims()[0];

  const thrust::device_ptr<const T> input(x_data);

  out_count->Resize({bs});
  int* out_count_data = dev_ctx.template Alloc<int>(out_count);
  thrust::device_ptr<int> output_count(out_count_data);

  int total_sample_size = GetTotalSampleNum<T, Context>(
      input, col_ptr_data, output_count, sample_size, bs);

  out->Resize({static_cast<int>(total_sample_size)});
  T* out_data = dev_ctx.template Alloc<T>(out);
  thrust::device_ptr<T> output(out_data);

  if (return_eids) {
    auto* eids_data = eids.get_ptr()->data<T>();
    out_eids->Resize({static_cast<int>(total_sample_size)});
    T* out_eids_data = dev_ctx.template Alloc<T>(out_eids);
    thrust::device_ptr<T> output_eids(out_eids_data);
    if (!flag_perm_buffer) {
      SampleNeighbors<T, Context>(dev_ctx,
                                  row_data,
                                  col_ptr_data,
                                  eids_data,
                                  input,
                                  output,
                                  output_count,
                                  output_eids,
                                  sample_size,
                                  bs,
                                  total_sample_size,
                                  return_eids);
    } else {
      DenseTensor perm_buffer_out(perm_buffer->type());
      const auto* p_perm_buffer = perm_buffer.get_ptr();
      perm_buffer_out.ShareDataWith(*p_perm_buffer);
      T* perm_buffer_out_data = perm_buffer_out.template data<T>();
      FisherYatesSampleNeighbors<T, Context>(dev_ctx,
                                             row_data,
                                             col_ptr_data,
                                             eids_data,
                                             perm_buffer_out_data,
                                             input,
                                             output,
                                             output_count,
                                             output_eids,
                                             sample_size,
                                             bs,
                                             total_sample_size,
                                             return_eids);
    }
  } else {
    // How to set null value for output_eids(thrust::device_ptr<T>)?
    // We use `output` to fill the position of unused output_eids.
    if (!flag_perm_buffer) {
      SampleNeighbors<T, Context>(dev_ctx,
                                  row_data,
                                  col_ptr_data,
                                  nullptr,
                                  input,
                                  output,
                                  output_count,
                                  output,
                                  sample_size,
                                  bs,
                                  total_sample_size,
                                  return_eids);
    } else {
      DenseTensor perm_buffer_out(perm_buffer->type());
      const auto* p_perm_buffer = perm_buffer.get_ptr();
      perm_buffer_out.ShareDataWith(*p_perm_buffer);
      T* perm_buffer_out_data = perm_buffer_out.template data<T>();
      FisherYatesSampleNeighbors<T, Context>(dev_ctx,
                                             row_data,
                                             col_ptr_data,
                                             nullptr,
                                             perm_buffer_out_data,
                                             input,
                                             output,
                                             output_count,
                                             output,
                                             sample_size,
                                             bs,
                                             total_sample_size,
                                             return_eids);
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(graph_sample_neighbors,
                   GPU,
                   ALL_LAYOUT,
                   phi::GraphSampleNeighborsKernel,
                   int,
                   int64_t) {}
