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

#include "paddle/phi/kernels/gpu/graph_reindex_funcs.h"
#include "paddle/phi/kernels/graph_reindex_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

constexpr int WARP_SIZE = 32;

template <typename T, typename Context>
void FillHashTable(const Context& dev_ctx,
                   const T* input,
                   int num_input,
                   int64_t len_hashtable,
                   thrust::device_vector<T>* unique_items,
                   T* keys,
                   int* values,
                   int* key_index) {
#ifdef PADDLE_WITH_HIP
  int block = 256;
#else
  int block = 1024;
#endif
  int max_grid_dimx = dev_ctx.GetCUDAMaxGridDimSize()[0];
  int grid_tmp = (num_input + block - 1) / block;
  int grid = grid_tmp < max_grid_dimx ? grid_tmp : max_grid_dimx;
  // Insert data into keys and values.
  BuildHashTable<T><<<grid, block, 0, dev_ctx.stream()>>>(
      input, num_input, len_hashtable, keys, key_index);

  // Get item index count.
  thrust::device_vector<int> item_count(num_input + 1, 0);
  GetItemIndexCount<T><<<grid, block, 0, dev_ctx.stream()>>>(
      input,
      thrust::raw_pointer_cast(item_count.data()),
      num_input,
      len_hashtable,
      keys,
      key_index);

  thrust::exclusive_scan(
      item_count.begin(), item_count.end(), item_count.begin());
  size_t total_unique_items = item_count[num_input];
  unique_items->resize(total_unique_items);

  // Get unique items
  FillUniqueItems<T><<<grid, block, 0, dev_ctx.stream()>>>(
      input,
      num_input,
      len_hashtable,
      thrust::raw_pointer_cast(unique_items->data()),
      thrust::raw_pointer_cast(item_count.data()),
      keys,
      values,
      key_index);
}

template <typename T, typename Context>
void Reindex(const Context& dev_ctx,
             const T* inputs,
             thrust::device_vector<T>* src_outputs,
             thrust::device_vector<T>* out_nodes,
             int num_inputs) {
  out_nodes->resize(num_inputs + src_outputs->size());
  thrust::copy(inputs, inputs + num_inputs, out_nodes->begin());
  thrust::copy(src_outputs->begin(),
               src_outputs->end(),
               out_nodes->begin() + num_inputs);
  thrust::device_vector<T> unique_nodes;
  unique_nodes.clear();

  // Fill hash table
  int64_t num = out_nodes->size();
  int64_t log_num = 1 << static_cast<size_t>(1 + std::log2(num >> 1));
  int64_t table_size = log_num << 1;
  T* keys;
  int *values, *key_index;
  cudaMalloc(&keys, table_size * sizeof(T));
  cudaMalloc(&values, table_size * sizeof(int));
  cudaMalloc(&key_index, table_size * sizeof(int));
  cudaMemset(keys, -1, table_size * sizeof(T));
  cudaMemset(values, -1, table_size * sizeof(int));
  cudaMemset(key_index, -1, table_size * sizeof(int));
  FillHashTable<T, Context>(dev_ctx,
                            thrust::raw_pointer_cast(out_nodes->data()),
                            out_nodes->size(),
                            table_size,
                            &unique_nodes,
                            keys,
                            values,
                            key_index);
  out_nodes->resize(unique_nodes.size());
  thrust::copy(unique_nodes.begin(), unique_nodes.end(), out_nodes->begin());

// Fill outputs with reindex result.
// Fill outputs with reindex result.
#ifdef PADDLE_WITH_HIP
  int block = 256;
#else
  int block = 1024;
#endif
  int max_grid_dimx = dev_ctx.GetCUDAMaxGridDimSize()[0];
  int grid_tmp = (src_outputs->size() + block - 1) / block;
  int grid = grid_tmp < max_grid_dimx ? grid_tmp : max_grid_dimx;
  ReindexSrcOutput<T><<<grid, block, 0, dev_ctx.stream()>>>(
      thrust::raw_pointer_cast(src_outputs->data()),
      src_outputs->size(),
      table_size,
      keys,
      values);
  cudaFree(keys);
  cudaFree(values);
  cudaFree(key_index);
}

template <typename T, int BLOCK_WARPS, int TILE_SIZE>
__global__ void GetDstEdgeCUDAKernel(const int64_t num_rows,
                                     const int* in_rows,
                                     const int* dst_counts,
                                     const int* dst_ptr,
                                     T* dst_outputs) {
  assert(blockDim.x == WARP_SIZE);
  assert(blockDim.y == BLOCK_WARPS);

  int64_t out_row = blockIdx.x * TILE_SIZE + threadIdx.y;
  const int64_t last_row =
      min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_rows);

  while (out_row < last_row) {
    const int row = in_rows[out_row];
    const int dst_sample_size = dst_counts[out_row];
    const int out_row_start = dst_ptr[out_row];
    for (int idx = threadIdx.x; idx < dst_sample_size; idx += WARP_SIZE) {
      dst_outputs[out_row_start + idx] = row;
    }
#ifdef PADDLE_WITH_CUDA
    __syncwarp();
#endif

    out_row += BLOCK_WARPS;
  }
}

template <typename T, typename Context>
void GraphReindexKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& neighbors,
                        const DenseTensor& count,
                        DenseTensor* reindex_src,
                        DenseTensor* reindex_dst,
                        DenseTensor* out_nodes) {
  const T* x_data = x.data<T>();
  const T* neighbors_data = neighbors.data<T>();
  const int* count_data = count.data<int>();
  const int bs = x.dims()[0];
  const int num_edges = neighbors.dims()[0];

  thrust::device_vector<T> src_outputs(num_edges);
  thrust::device_vector<T> dst_outputs(num_edges);
  thrust::device_vector<T> unique_nodes;
  thrust::copy(neighbors_data, neighbors_data + num_edges, src_outputs.begin());
  Reindex<T, Context>(dev_ctx, x_data, &src_outputs, &unique_nodes, bs);

  // Get reindex dst edge.
  thrust::device_vector<int> unique_dst_reindex(bs);
  thrust::sequence(unique_dst_reindex.begin(), unique_dst_reindex.end());
  thrust::device_vector<int> dst_ptr(bs);
  thrust::exclusive_scan(count_data, count_data + bs, dst_ptr.begin());
  constexpr int BLOCK_WARPS = 128 / WARP_SIZE;
  constexpr int TILE_SIZE = BLOCK_WARPS * 16;
  const dim3 block(WARP_SIZE, BLOCK_WARPS);
  const dim3 grid((bs + TILE_SIZE - 1) / TILE_SIZE);
  GetDstEdgeCUDAKernel<T,
                       BLOCK_WARPS,
                       TILE_SIZE><<<grid, block, 0, dev_ctx.stream()>>>(
      bs,
      thrust::raw_pointer_cast(unique_dst_reindex.data()),
      count_data,
      thrust::raw_pointer_cast(dst_ptr.data()),
      thrust::raw_pointer_cast(dst_outputs.data()));

  reindex_src->Resize({num_edges});
  T* reindex_src_data = dev_ctx.template Alloc<T>(reindex_src);
  thrust::copy(src_outputs.begin(), src_outputs.end(), reindex_src_data);
  reindex_dst->Resize({num_edges});
  T* reindex_dst_data = dev_ctx.template Alloc<T>(reindex_dst);
  thrust::copy(dst_outputs.begin(), dst_outputs.end(), reindex_dst_data);
  out_nodes->Resize({static_cast<int>(unique_nodes.size())});
  T* out_nodes_data = dev_ctx.template Alloc<T>(out_nodes);
  thrust::copy(unique_nodes.begin(), unique_nodes.end(), out_nodes_data);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    graph_reindex, GPU, ALL_LAYOUT, phi::GraphReindexKernel, int, int64_t) {}
