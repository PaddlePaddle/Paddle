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

#ifdef __NVCC__
#include <cub/cub.cuh>
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/gpu/graph_reindex_funcs.h"

namespace phi {

constexpr int WARP_SIZE = 32;
const int CUDA_NUM_THREADS = 512;
inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

template <typename T>
__global__ void InitializeHashTable(T* tensor, int len) {
  CUDA_KERNEL_LOOP(idx, len) { tensor[idx] = -1; }
}

template <typename T, typename Context>
std::shared_ptr<phi::Allocation> FillHashTable(const Context& dev_ctx,
                                               const T* input,
                                               int num_input,
                                               int64_t len_hashtable,
                                               T* keys,
                                               int* values,
                                               int* key_index,
                                               int* final_nodes_len) {
  const auto place = dev_ctx.GetPlace();

  int block = 1024;
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

  int total_unique_items = item_count[num_input];
  auto unique_items =
      phi::memory_utils::AllocShared(place, total_unique_items * sizeof(T));
  T* unique_items_data = reinterpret_cast<T*>(unique_items->ptr());
  *final_nodes_len = total_unique_items;

  // Get unique items
  FillUniqueItems<T><<<grid, block, 0, dev_ctx.stream()>>>(
      input,
      num_input,
      len_hashtable,
      unique_items_data,
      thrust::raw_pointer_cast(item_count.data()),
      keys,
      values,
      key_index);
  return unique_items;
}

template <typename T, typename Context>
void FillBufferHashTable(const Context& dev_ctx,
                         const T* input,
                         int num_input,
                         thrust::device_vector<T>* unique_items,
                         int* values,
                         int* key_index) {
  int block = 1024;
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
  int block = 1024;
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
void ReindexSrc(const Context& dev_ctx,
                T* edges_src,
                T* keys,
                int* values,
                int64_t num_edges,
                int64_t table_size) {
  // Fill outputs with reindex result.
  int block = 1024;
  int max_grid_dimx = dev_ctx.GetCUDAMaxGridDimSize()[0];
  int grid_tmp = (num_edges + block - 1) / block;
  int grid = grid_tmp < max_grid_dimx ? grid_tmp : max_grid_dimx;
  ReindexSrcOutput<T><<<grid, block, 0, dev_ctx.stream()>>>(
      edges_src, num_edges, table_size, keys, values);
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

  // Fill hash table
  int64_t num = out_nodes->size();
  int64_t log_num = 1 << static_cast<size_t>(1 + std::log2(num >> 1));
  int64_t table_size = log_num << 1;

  auto keys =
      phi::memory_utils::Alloc(dev_ctx.GetPlace(), table_size * sizeof(T));
  auto values =
      phi::memory_utils::Alloc(dev_ctx.GetPlace(), table_size * sizeof(int));
  auto key_index =
      phi::memory_utils::Alloc(dev_ctx.GetPlace(), table_size * sizeof(int));
  T* keys_ptr = reinterpret_cast<T*>(keys->ptr());
  int* values_ptr = reinterpret_cast<int*>(values->ptr());
  int* key_index_ptr = reinterpret_cast<int*>(key_index->ptr());

  InitializeHashTable<T>
      <<<GET_BLOCKS(table_size), CUDA_NUM_THREADS, 0, dev_ctx.stream()>>>(
          keys_ptr, table_size);
  InitializeHashTable<int>
      <<<GET_BLOCKS(table_size), CUDA_NUM_THREADS, 0, dev_ctx.stream()>>>(
          values_ptr, table_size);
  InitializeHashTable<int>
      <<<GET_BLOCKS(table_size), CUDA_NUM_THREADS, 0, dev_ctx.stream()>>>(
          key_index_ptr, table_size);

  int unique_len = 0;
  std::shared_ptr<phi::Allocation> unique_items =
      FillHashTable<T, Context>(dev_ctx,
                                thrust::raw_pointer_cast(out_nodes->data()),
                                out_nodes->size(),
                                table_size,
                                keys_ptr,
                                values_ptr,
                                key_index_ptr,
                                &unique_len);
  out_nodes->resize(unique_len);
  T* unique_items_data = reinterpret_cast<T*>(unique_items->ptr());
  thrust::copy(thrust::device_pointer_cast(unique_items_data),
               thrust::device_pointer_cast(unique_items_data) + unique_len,
               out_nodes->begin());

  ReindexSrc<T, Context>(dev_ctx,
                         thrust::raw_pointer_cast(src_outputs),
                         keys_ptr,
                         values_ptr,
                         num_edges,
                         table_size);
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
  int block = 1024;
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
void ReindexDst(const Context& dev_ctx,
                T* reindex_dst_data,
                int* scan_dst_data,
                const int* count_data,
                int num_edge_types,
                int node_len) {
  constexpr int BLOCK_WARPS = 128 / WARP_SIZE;
  constexpr int TILE_SIZE = BLOCK_WARPS * 16;
  const dim3 block(WARP_SIZE, BLOCK_WARPS);
  const dim3 grid((node_len + TILE_SIZE - 1) / TILE_SIZE);

  int begin = 0, count_i = 0;
  thrust::device_vector<int> dst_ptr(node_len + 1, 0);
  for (int i = 0; i < num_edge_types; i++) {
    thrust::inclusive_scan(
        thrust::device_pointer_cast(count_data) + i * node_len,
        thrust::device_pointer_cast(count_data) + (i + 1) * node_len,
        dst_ptr.begin() + 1);
    GetDstEdgeCUDAKernel<T, BLOCK_WARPS, TILE_SIZE>
        <<<grid, block, 0, dev_ctx.stream()>>>(
            node_len,
            scan_dst_data,
            count_data + i * node_len,
            thrust::raw_pointer_cast(dst_ptr.data()),
            reindex_dst_data + begin);
#ifdef PADDLE_WITH_HIP
    hipMemcpy(&count_i,
              thrust::raw_pointer_cast(dst_ptr.data()) + node_len,
              sizeof(int),
              hipMemcpyDeviceToHost);
#else
    cudaMemcpy(&count_i,
               thrust::raw_pointer_cast(dst_ptr.data()) + node_len,
               sizeof(int),
               cudaMemcpyDeviceToHost);
#endif
    begin += count_i;
  }
}

template <typename T, typename Context>
void GraphReindexKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& neighbors,
                        const DenseTensor& count,
                        const paddle::optional<DenseTensor>& hashtable_value,
                        const paddle::optional<DenseTensor>& hashtable_index,
                        DenseTensor* reindex_src,
                        DenseTensor* reindex_dst,
                        DenseTensor* out_nodes) {
  bool flag_buffer_hashtable =
      hashtable_value.is_initialized() && hashtable_index.is_initialized();
  const T* x_data = x.data<T>();
  const T* neighbors_data = neighbors.data<T>();
  const int* count_data = count.data<int>();
  const int bs = x.dims()[0];
  PADDLE_ENFORCE_NE(
      0,
      bs,
      errors::InvalidArgument("The first of dims should not be equal to 0."));
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
  reindex_dst->Resize({num_edges});
  T* reindex_dst_data = dev_ctx.template Alloc<T>(reindex_dst);

  ReindexDst<T, Context>(dev_ctx,
                         reindex_dst_data,
                         thrust::raw_pointer_cast(unique_dst_reindex.data()),
                         count_data,
                         num_edge_types,
                         bs);
  out_nodes->Resize({static_cast<int>(unique_nodes.size())});
  T* out_nodes_data = dev_ctx.template Alloc<T>(out_nodes);
  thrust::copy(unique_nodes.begin(), unique_nodes.end(), out_nodes_data);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    graph_reindex, GPU, ALL_LAYOUT, phi::GraphReindexKernel, int, int64_t) {}
