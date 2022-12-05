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

#include "paddle/phi/kernels/graph_reindex_kernel.h"

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/gpu/graph_reindex_funcs.h"

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
void FillBufferHashTable(const Context& dev_ctx,
                         const T* input,
                         int num_input,
                         thrust::device_vector<T>* unique_items,
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
  // Insert data.
  BuildHashTable<T>
      <<<grid, block, 0, dev_ctx.stream()>>>(input, num_input, key_index);

  // Get item index count.
  thrust::device_vector<int> item_count(num_input + 1, 0);
  GetItemIndexCount<T><<<grid, block, 0, dev_ctx.stream()>>>(
      input, thrust::raw_pointer_cast(item_count.data()), num_input, key_index);

  thrust::exclusive_scan(
      item_count.begin(), item_count.end(), item_count.begin());
  size_t total_unique_items = item_count[num_input];
  unique_items->resize(total_unique_items);

  // Get unique items
  FillUniqueItems<T><<<grid, block, 0, dev_ctx.stream()>>>(
      input,
      num_input,
      thrust::raw_pointer_cast(unique_items->data()),
      thrust::raw_pointer_cast(item_count.data()),
      values,
      key_index);
}

template <typename T, typename Context>
void ResetBufferHashTable(const Context& dev_ctx,
                          const T* input,
                          int num_input,
                          thrust::device_vector<T>* unique_items,
                          int* values,
                          int* key_index) {
#ifdef PADDLE_WITH_HIP
  int block = 256;
#else
  int block = 1024;
#endif
  int max_grid_dimx = dev_ctx.GetCUDAMaxGridDimSize()[0];
  int grid_tmp = (unique_items->size() + block - 1) / block;
  int grid = grid_tmp < max_grid_dimx ? grid_tmp : max_grid_dimx;
  ResetHashTable<T><<<grid, block, 0, dev_ctx.stream()>>>(
      thrust::raw_pointer_cast(unique_items->data()),
      unique_items->size(),
      key_index,
      values);
}

template <typename T, typename Context>
void Reindex(const Context& dev_ctx,
             const T* inputs,
             thrust::device_ptr<T> src_outputs,
             thrust::device_vector<T>* out_nodes,
             int num_inputs,
             int num_edges) {
  out_nodes->resize(num_inputs + num_edges);
  thrust::copy(inputs, inputs + num_inputs, out_nodes->begin());
  thrust::copy(
      src_outputs, src_outputs + num_edges, out_nodes->begin() + num_inputs);
  thrust::device_vector<T> unique_nodes;
  unique_nodes.clear();

  // Fill hash table
  int64_t num = out_nodes->size();
  int64_t log_num = 1 << static_cast<size_t>(1 + std::log2(num >> 1));
  int64_t table_size = log_num << 1;
  T* keys;
  int *values, *key_index;

#ifdef PADDLE_WITH_HIP
  hipMalloc(&keys, table_size * sizeof(T));
  hipMalloc(&values, table_size * sizeof(int));
  hipMalloc(&key_index, table_size * sizeof(int));
  hipMemset(keys, -1, table_size * sizeof(T));
  hipMemset(values, -1, table_size * sizeof(int));
  hipMemset(key_index, -1, table_size * sizeof(int));
#else
  cudaMalloc(&keys, table_size * sizeof(T));
  cudaMalloc(&values, table_size * sizeof(int));
  cudaMalloc(&key_index, table_size * sizeof(int));
  cudaMemset(keys, -1, table_size * sizeof(T));
  cudaMemset(values, -1, table_size * sizeof(int));
  cudaMemset(key_index, -1, table_size * sizeof(int));
#endif

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
#ifdef PADDLE_WITH_HIP
  int block = 256;
#else
  int block = 1024;
#endif
  int max_grid_dimx = dev_ctx.GetCUDAMaxGridDimSize()[0];
  int grid_tmp = (num_edges + block - 1) / block;
  int grid = grid_tmp < max_grid_dimx ? grid_tmp : max_grid_dimx;
  ReindexSrcOutput<T><<<grid, block, 0, dev_ctx.stream()>>>(
      thrust::raw_pointer_cast(src_outputs),
      num_edges,
      table_size,
      keys,
      values);
#ifdef PADDLE_WITH_HIP
  hipFree(keys);
  hipFree(values);
  hipFree(key_index);
#else
  cudaFree(keys);
  cudaFree(values);
  cudaFree(key_index);
#endif
}

template <typename T, typename Context>
void BufferReindex(const Context& dev_ctx,
                   const T* inputs,
                   thrust::device_ptr<T> src_outputs,
                   thrust::device_vector<T>* out_nodes,
                   int num_inputs,
                   int* hashtable_value,
                   int* hashtable_index,
                   int num_edges) {
  out_nodes->resize(num_inputs + num_edges);
  thrust::copy(inputs, inputs + num_inputs, out_nodes->begin());
  thrust::copy(
      src_outputs, src_outputs + num_edges, out_nodes->begin() + num_inputs);
  thrust::device_vector<T> unique_nodes;
  unique_nodes.clear();

  // Fill hash table
  FillBufferHashTable<T, Context>(dev_ctx,
                                  thrust::raw_pointer_cast(out_nodes->data()),
                                  out_nodes->size(),
                                  &unique_nodes,
                                  hashtable_value,
                                  hashtable_index);
  out_nodes->resize(unique_nodes.size());
  thrust::copy(unique_nodes.begin(), unique_nodes.end(), out_nodes->begin());

// Fill outputs with reindex result.
#ifdef PADDLE_WITH_HIP
  int block = 256;
#else
  int block = 1024;
#endif
  int max_grid_dimx = dev_ctx.GetCUDAMaxGridDimSize()[0];
  int grid_tmp = (num_edges + block - 1) / block;
  int grid = grid_tmp < max_grid_dimx ? grid_tmp : max_grid_dimx;
  ReindexSrcOutput<T><<<grid, block, 0, dev_ctx.stream()>>>(
      thrust::raw_pointer_cast(src_outputs), num_edges, hashtable_value);

  ResetBufferHashTable<T, Context>(dev_ctx,
                                   thrust::raw_pointer_cast(out_nodes->data()),
                                   out_nodes->size(),
                                   &unique_nodes,
                                   hashtable_value,
                                   hashtable_index);
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
    out_row += BLOCK_WARPS;
  }
}

template <typename T, typename Context>
void GraphReindexKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& neighbors,
                        const DenseTensor& count,
                        const paddle::optional<DenseTensor>& hashtable_value,
                        const paddle::optional<DenseTensor>& hashtable_index,
                        bool flag_buffer_hashtable,
                        DenseTensor* reindex_src,
                        DenseTensor* reindex_dst,
                        DenseTensor* out_nodes) {
  const T* x_data = x.data<T>();
  const T* neighbors_data = neighbors.data<T>();
  const int* count_data = count.data<int>();
  const int bs = x.dims()[0];
  const int num_edges = neighbors.dims()[0];
  reindex_src->Resize({num_edges});

  T* reindex_src_data = dev_ctx.template Alloc<T>(reindex_src);
  thrust::device_ptr<T> src_outputs(reindex_src_data);

  thrust::device_vector<T> unique_nodes;
  thrust::copy(neighbors_data, neighbors_data + num_edges, src_outputs);

  if (flag_buffer_hashtable) {
    // Here we directly use buffer tensor to act as a hash table.
    DenseTensor hashtable_value_out(hashtable_value->type());
    const auto* ph_value = hashtable_value.get_ptr();
    hashtable_value_out.ShareDataWith(*ph_value);
    DenseTensor hashtable_index_out(hashtable_index->type());
    const auto* ph_index = hashtable_index.get_ptr();
    hashtable_index_out.ShareDataWith(*ph_index);
    int* hashtable_value_data =
        dev_ctx.template Alloc<int>(&hashtable_value_out);
    int* hashtable_index_data =
        dev_ctx.template Alloc<int>(&hashtable_index_out);
    BufferReindex<T, Context>(dev_ctx,
                              x_data,
                              src_outputs,
                              &unique_nodes,
                              bs,
                              hashtable_value_data,
                              hashtable_index_data,
                              num_edges);
  } else {
    Reindex<T, Context>(
        dev_ctx, x_data, src_outputs, &unique_nodes, bs, num_edges);
  }

  // Get reindex dst edge.
  // Add support for multi-type edges reindex.
  int num_ac_count = count.dims()[0];
  int num_edge_types = num_ac_count / bs;
  thrust::device_vector<int> unique_dst_reindex(bs);
  thrust::sequence(unique_dst_reindex.begin(), unique_dst_reindex.end());
  constexpr int BLOCK_WARPS = 128 / WARP_SIZE;
  constexpr int TILE_SIZE = BLOCK_WARPS * 16;
  const dim3 block(WARP_SIZE, BLOCK_WARPS);
  const dim3 grid((bs + TILE_SIZE - 1) / TILE_SIZE);
  reindex_dst->Resize({num_edges});
  T* reindex_dst_data = dev_ctx.template Alloc<T>(reindex_dst);
  int begin = 0;
  for (int i = 0; i < num_edge_types; i++) {
    thrust::device_vector<int> dst_ptr(bs);
    thrust::exclusive_scan(
        count_data + i * bs, count_data + (i + 1) * bs, dst_ptr.begin());

    GetDstEdgeCUDAKernel<T, BLOCK_WARPS, TILE_SIZE>
        <<<grid, block, 0, dev_ctx.stream()>>>(
            bs,
            thrust::raw_pointer_cast(unique_dst_reindex.data()),
            count_data + i * bs,
            thrust::raw_pointer_cast(dst_ptr.data()),
            reindex_dst_data + begin);

    int count_i =
        thrust::reduce(thrust::device_pointer_cast(count_data) + i * bs,
                       thrust::device_pointer_cast(count_data) + (i + 1) * bs);
    begin += count_i;
  }

  out_nodes->Resize({static_cast<int>(unique_nodes.size())});
  T* out_nodes_data = dev_ctx.template Alloc<T>(out_nodes);
  thrust::copy(unique_nodes.begin(), unique_nodes.end(), out_nodes_data);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    graph_reindex, GPU, ALL_LAYOUT, phi::GraphReindexKernel, int, int64_t) {}
