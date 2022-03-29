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

#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

#ifdef PADDLE_WITH_HETERPS
//#include "paddle/fluid/framework/fleet/heter_ps/graph_gpu_ps_table.h"
namespace paddle {
namespace framework {

constexpr int WARP_SIZE = 32;

/*
comment 0
this kernel just serves as an example of how to sample nodes' neighbors.
feel free to modify it
index[0,len) saves the nodes' index
actual_size[0,len) is to save the sample size of each node.
for ith node in index, actual_size[i] = min(node i's neighbor size, sample size)
sample_result is to save the neighbor sampling result, its size is len *
sample_size;

*/

struct MaxFunctor {
  int sample_size;
  HOSTDEVICE explicit inline MaxFunctor(int sample_size) {
    this->sample_size = sample_size;
  }
  HOSTDEVICE inline int operator()(int x) const {
    if (x > sample_size) {
      return sample_size;
    }
    return x;
  }
};

struct DegreeFunctor {
  GpuPsCommGraph graph;
  HOSTDEVICE explicit inline DegreeFunctor(GpuPsCommGraph graph) {
    this->graph = graph;
  }
  HOSTDEVICE inline int operator()(int i) const {
    return graph.node_list[i].neighbor_size;
  }
};

template <int BLOCK_WARPS, int TILE_SIZE>
__global__ void neighbor_sample(const uint64_t rand_seed, GpuPsCommGraph graph,
                                int sample_size, int* index, int len,
                                int64_t* sample_result, int* output_idx,
                                int* output_offset) {
  assert(blockDim.x == WARP_SIZE);
  assert(blockDim.y == BLOCK_WARPS);

  int i = blockIdx.x * TILE_SIZE + threadIdx.y;
  const int last_idx = min(static_cast<int>(blockIdx.x + 1) * TILE_SIZE, len);
  curandState rng;
  curand_init(rand_seed * gridDim.x + blockIdx.x,
              threadIdx.y * WARP_SIZE + threadIdx.x, 0, &rng);

  while (i < last_idx) {
    auto node_index = index[i];
    int degree = graph.node_list[node_index].neighbor_size;
    const int offset = graph.node_list[node_index].neighbor_offset;
    int output_start = output_offset[i];

    if (degree <= sample_size) {
      // Just copy
      for (int j = threadIdx.x; j < degree; j += WARP_SIZE) {
        sample_result[output_start + j] = graph.neighbor_list[offset + j];
      }
    } else {
      for (int j = threadIdx.x; j < degree; j += WARP_SIZE) {
        output_idx[output_start + j] = j;
      }

      __syncwarp();

      for (int j = sample_size + threadIdx.x; j < degree; j += WARP_SIZE) {
        const int num = curand(&rng) % (j + 1);
        if (num < sample_size) {
          atomicMax(
              reinterpret_cast<unsigned int*>(output_idx + output_start + num),
              static_cast<unsigned int>(j));
        }
      }

      __syncwarp();

      for (int j = threadIdx.x; j < sample_size; j += WARP_SIZE) {
        const int perm_idx = output_idx[output_start + j] + offset;
        sample_result[output_start + j] = graph.neighbor_list[perm_idx];
      }
    }

    i += BLOCK_WARPS;
  }
}

int GpuPsGraphTable::init_cpu_table(
    const paddle::distributed::GraphParameter& graph) {
  cpu_graph_table.reset(new paddle::distributed::GraphTable);
  cpu_table_status = cpu_graph_table->initialize(graph);
  if (cpu_table_status != 0) return cpu_table_status;
  std::function<void(std::vector<GpuPsCommGraph>&)> callback =
      [this](std::vector<GpuPsCommGraph>& res) {
        pthread_rwlock_wrlock(this->rw_lock.get());
        this->clear_graph_info();
        this->build_graph_from_cpu(res);
        pthread_rwlock_unlock(this->rw_lock.get());
        cv_.notify_one();
      };
  cpu_graph_table->set_graph_sample_callback(callback);
  return cpu_table_status;
}

int GpuPsGraphTable::load(const std::string& path, const std::string& param) {
  int status = cpu_graph_table->load(path, param);
  if (status != 0) {
    return status;
  }
  std::unique_lock<std::mutex> lock(mutex_);
  cpu_graph_table->start_graph_sampling();
  cv_.wait(lock);
  return 0;
}
/*
 comment 1

 gpu i triggers a neighbor_sample task,
 when this task is done,
 this function is called to move the sample result on other gpu back
 to gpu i and aggragate the result.
 the sample_result is saved on src_sample_res and the actual sample size for
 each node is saved on actual_sample_size.
 the number of actual sample_result for
 key[x] (refer to comment 2 for definition of key)
 is saved on  actual_sample_size[x], since the neighbor size of key[x] might be
 smaller than sample_size,
 is saved on src_sample_res [x*sample_size, x*sample_size +
 actual_sample_size[x])

 since before each gpu runs the neighbor_sample task,the key array is shuffled,
 but we have the idx array to save the original order.
 when the gpu i gets all the sample results from other gpus, it relies on
 idx array to recover the original order.
 that's what fill_dvals does.

*/
void GpuPsGraphTable::move_neighbor_sample_size_to_source_gpu(
    int gpu_id, int gpu_num, int* h_left, int* h_right, int* actual_sample_size,
    int* total_sample_size) {
  // This function copyed actual_sample_size to source_gpu,
  // and calculate total_sample_size of each gpu sample number.
  for (int i = 0; i < gpu_num; i++) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    auto shard_len = h_right[i] - h_left[i] + 1;
    auto& node = path_[gpu_id][i].nodes_.front();
    cudaMemcpyAsync(reinterpret_cast<char*>(actual_sample_size + h_left[i]),
                    node.val_storage + sizeof(int) * shard_len,
                    sizeof(int) * shard_len, cudaMemcpyDefault,
                    node.out_stream);
  }
  for (int i = 0; i < gpu_num; ++i) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      total_sample_size[i] = 0;
      continue;
    }
    auto& node = path_[gpu_id][i].nodes_.front();
    cudaStreamSynchronize(node.out_stream);

    auto shard_len = h_right[i] - h_left[i] + 1;
    thrust::device_vector<int> t_actual_sample_size(shard_len);
    thrust::copy(actual_sample_size + h_left[i],
                 actual_sample_size + h_left[i] + shard_len,
                 t_actual_sample_size.begin());
    total_sample_size[i] = thrust::reduce(t_actual_sample_size.begin(),
                                          t_actual_sample_size.end());
  }
}

void GpuPsGraphTable::move_neighbor_sample_result_to_source_gpu(
    int gpu_id, int gpu_num, int* h_left, int* h_right, int64_t* src_sample_res,
    thrust::host_vector<int>& total_sample_size) {
  /*
  if total_sample_size is [4, 5, 1, 6],
  then cumsum_total_sample_size is [0, 4, 9, 10];
  */
  thrust::host_vector<int> cumsum_total_sample_size(gpu_num, 0);
  thrust::exclusive_scan(total_sample_size.begin(), total_sample_size.end(),
                         cumsum_total_sample_size.begin(), 0);
  for (int i = 0; i < gpu_num; i++) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    auto shard_len = h_right[i] - h_left[i] + 1;
    // int cur_step = path_[gpu_id][i].nodes_.size() - 1;
    // auto& node = path_[gpu_id][i].nodes_[cur_step];
    auto& node = path_[gpu_id][i].nodes_.front();
    cudaMemcpyAsync(
        reinterpret_cast<char*>(src_sample_res + cumsum_total_sample_size[i]),
        node.val_storage + sizeof(int64_t) * shard_len,
        sizeof(int64_t) * total_sample_size[i], cudaMemcpyDefault,
        node.out_stream);
  }
  for (int i = 0; i < gpu_num; ++i) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    auto& node = path_[gpu_id][i].nodes_.front();
    cudaStreamSynchronize(node.out_stream);
  }
}

/*
TODO:
how to optimize it to eliminate the for loop
*/
__global__ void fill_dvalues_actual_sample_size(int* d_shard_actual_sample_size,
                                                int* d_actual_sample_size,
                                                int* idx, int len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    d_actual_sample_size[idx[i]] = d_shard_actual_sample_size[i];
  }
}

template <int BLOCK_WARPS, int TILE_SIZE>
__global__ void fill_dvalues_sample_result(int64_t* d_shard_vals,
                                           int64_t* d_vals,
                                           int* d_actual_sample_size, int* idx,
                                           int* offset, int* d_offset,
                                           int len) {
  assert(blockDim.x == WARP_SIZE);
  assert(blockDim.y == BLOCK_WARPS);

  int i = blockIdx.x * TILE_SIZE + threadIdx.y;
  const int last_idx = min(static_cast<int>(blockIdx.x + 1) * TILE_SIZE, len);
  while (i < last_idx) {
    const int sample_size = d_actual_sample_size[idx[i]];
    for (int j = threadIdx.x; j < sample_size; j += WARP_SIZE) {
      d_vals[offset[idx[i]] + j] = d_shard_vals[d_offset[i] + j];
    }
#ifdef PADDLE_WITH_CUDA
    __syncwarp();
#endif
    i += BLOCK_WARPS;
  }
}

__global__ void node_query_example(GpuPsCommGraph graph, int start, int size,
                                   int64_t* res) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    res[i] = graph.node_list[start + i].node_id;
  }
}

void GpuPsGraphTable::clear_graph_info() {
  if (tables_.size()) {
    for (auto table : tables_) delete table;
  }
  tables_.clear();
  for (auto graph : gpu_graph_list) {
    if (graph.neighbor_list != NULL) {
      cudaFree(graph.neighbor_list);
    }
    if (graph.node_list != NULL) {
      cudaFree(graph.node_list);
    }
  }
  gpu_graph_list.clear();
}
/*
the parameter std::vector<GpuPsCommGraph> cpu_graph_list is generated by cpu.
it saves the graph to be saved on each gpu.

for the ith GpuPsCommGraph, any the node's key satisfies that key % gpu_number
== i

In this function, memory is allocated on each gpu to save the graphs,
gpu i saves the ith graph from cpu_graph_list
*/

void GpuPsGraphTable::build_graph_from_cpu(
    std::vector<GpuPsCommGraph>& cpu_graph_list) {
  PADDLE_ENFORCE_EQ(
      cpu_graph_list.size(), resource_->total_gpu(),
      platform::errors::InvalidArgument("the cpu node list size doesn't match "
                                        "the number of gpu on your machine."));
  clear_graph_info();
  for (int i = 0; i < cpu_graph_list.size(); i++) {
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    gpu_graph_list.push_back(GpuPsCommGraph());
    auto table =
        new Table(std::max(1, cpu_graph_list[i].node_size) / load_factor_);
    tables_.push_back(table);
    if (cpu_graph_list[i].node_size > 0) {
      std::vector<int64_t> keys;
      std::vector<int> offset;
      cudaMalloc((void**)&gpu_graph_list[i].node_list,
                 cpu_graph_list[i].node_size * sizeof(GpuPsGraphNode));
      cudaMemcpy(gpu_graph_list[i].node_list, cpu_graph_list[i].node_list,
                 cpu_graph_list[i].node_size * sizeof(GpuPsGraphNode),
                 cudaMemcpyHostToDevice);
      for (int j = 0; j < cpu_graph_list[i].node_size; j++) {
        keys.push_back(cpu_graph_list[i].node_list[j].node_id);
        offset.push_back(j);
      }
      build_ps(i, keys.data(), offset.data(), keys.size(), 1024, 8);
      gpu_graph_list[i].node_size = cpu_graph_list[i].node_size;
    } else {
      gpu_graph_list[i].node_list = NULL;
      gpu_graph_list[i].node_size = 0;
    }
    if (cpu_graph_list[i].neighbor_size) {
      cudaMalloc((void**)&gpu_graph_list[i].neighbor_list,
                 cpu_graph_list[i].neighbor_size * sizeof(int64_t));
      cudaMemcpy(gpu_graph_list[i].neighbor_list,
                 cpu_graph_list[i].neighbor_list,
                 cpu_graph_list[i].neighbor_size * sizeof(int64_t),
                 cudaMemcpyHostToDevice);
      gpu_graph_list[i].neighbor_size = cpu_graph_list[i].neighbor_size;
    } else {
      gpu_graph_list[i].neighbor_list = NULL;
      gpu_graph_list[i].neighbor_size = 0;
    }
  }
  cudaDeviceSynchronize();
}
NeighborSampleResult* GpuPsGraphTable::graph_neighbor_sample(int gpu_id,
                                                             int64_t* key,
                                                             int sample_size,
                                                             int len) {
  /*
 comment 2
  this function shares some kernels with heter_comm_inl.h
  arguments definitions:
  gpu_id:the id of gpu.
  len:how many keys are used,(the length of array key)
  sample_size:how many neighbors should be sampled for each node in key.

  the code below shuffle the key array to make the keys
    that belong to a gpu-card stay together,
    the shuffled result is saved on d_shard_keys,
    if ith element in d_shard_keys_ptr is
    from jth element in the original key array, then idx[i] = j,
    idx could be used to recover the original array.
    if keys in range [a,b] belong to ith-gpu, then h_left[i] = a, h_right[i] =
 b,
    if no keys are allocated for ith-gpu, then h_left[i] == h_right[i] == -1

    for example, suppose key = [0,1,2,3,4,5,6,7,8], gpu_num = 2
    when we run this neighbor_sample function,
    the key is shuffled to [0,2,4,6,8,1,3,5,7]
    the first part (0,2,4,6,8) % 2 == 0,thus should be handled by gpu 0,
    the rest part should be handled by gpu1, because (1,3,5,7) % 2 == 1,
    h_left = [0,5],h_right = [4,8]

  */

  NeighborSampleResult* result = new NeighborSampleResult(sample_size, len);
  if (len == 0) {
    return result;
  }

  int total_gpu = resource_->total_gpu();
  int dev_id = resource_->dev_id(gpu_id);
  platform::CUDAPlace place = platform::CUDAPlace(dev_id);
  platform::CUDADeviceGuard guard(dev_id);
  auto stream = resource_->local_stream(gpu_id, 0);

  int grid_size = (len - 1) / block_size_ + 1;

  int h_left[total_gpu];   // NOLINT
  int h_right[total_gpu];  // NOLINT

  auto d_left = memory::Alloc(place, total_gpu * sizeof(int));
  auto d_right = memory::Alloc(place, total_gpu * sizeof(int));
  int* d_left_ptr = reinterpret_cast<int*>(d_left->ptr());
  int* d_right_ptr = reinterpret_cast<int*>(d_right->ptr());

  cudaMemsetAsync(d_left_ptr, -1, total_gpu * sizeof(int), stream);
  cudaMemsetAsync(d_right_ptr, -1, total_gpu * sizeof(int), stream);
  //
  auto d_idx = memory::Alloc(place, len * sizeof(int));
  int* d_idx_ptr = reinterpret_cast<int*>(d_idx->ptr());

  auto d_shard_keys = memory::Alloc(place, len * sizeof(int64_t));
  int64_t* d_shard_keys_ptr = reinterpret_cast<int64_t*>(d_shard_keys->ptr());

  split_input_to_shard(key, d_idx_ptr, len, d_left_ptr, d_right_ptr, gpu_id);

  fill_shard_key<<<grid_size, block_size_, 0, stream>>>(d_shard_keys_ptr, key,
                                                        d_idx_ptr, len);

  cudaStreamSynchronize(stream);

  cudaMemcpy(h_left, d_left_ptr, total_gpu * sizeof(int),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_right, d_right_ptr, total_gpu * sizeof(int),
             cudaMemcpyDeviceToHost);

  for (int i = 0; i < total_gpu; ++i) {
    int shard_len = h_left[i] == -1 ? 0 : h_right[i] - h_left[i] + 1;
    if (shard_len == 0) {
      continue;
    }
    /*
   comment 3
    shard_len denotes the size of keys on i-th gpu here,
    when we sample  on i-th gpu, we allocate shard_len * (1 + sample_size)
   int64_t units
    of memory, we use alloc_mem_i to denote it, the range [0,shard_len) is saved
   for the respective nodes' indexes
    and acutal sample_size.
    with nodes' indexes we could get the nodes to sample.
    since size of int64_t is 8 bits, while size of int is 4,
    the range of [0,shard_len) contains shard_len * 2 int uinits;
    The values of the first half of this range will be updated by
    the k-v map on i-th-gpu.
    The second half of this range is saved for actual sample size of each node.
    For node x,
    its sampling result is saved on the range
    [shard_len + sample_size * x,shard_len + sample_size * x +
   actual_sample_size_of_x)
    of alloc_mem_i, actual_sample_size_of_x equals ((int
   *)alloc_mem_i)[shard_len + x]
    */

    create_storage(gpu_id, i, shard_len * sizeof(int64_t),
                   shard_len * (1 + sample_size) * sizeof(int64_t));
  }
  walk_to_dest(gpu_id, total_gpu, h_left, h_right, d_shard_keys_ptr, NULL);

  for (int i = 0; i < total_gpu; ++i) {
    if (h_left[i] == -1) {
      continue;
    }
    // auto& node = path_[gpu_id][i].nodes_.back();
    auto& node = path_[gpu_id][i].nodes_.front();
    cudaStreamSynchronize(node.in_stream);
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    // use the key-value map to update alloc_mem_i[0,shard_len)
    tables_[i]->rwlock_->RDLock();
    tables_[i]->get(reinterpret_cast<int64_t*>(node.key_storage),
                    reinterpret_cast<int*>(node.val_storage),
                    h_right[i] - h_left[i] + 1,
                    resource_->remote_stream(i, gpu_id));
  }

  for (int i = 0; i < total_gpu; ++i) {
    if (h_left[i] == -1) {
      continue;
    }
    // cudaStreamSynchronize(resource_->remote_stream(i, num));
    // tables_[i]->rwlock_->UNLock();
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    auto& node = path_[gpu_id][i].nodes_.front();
    auto shard_len = h_right[i] - h_left[i] + 1;
    auto graph = gpu_graph_list[i];
    int* res_array = reinterpret_cast<int*>(node.val_storage);
    int* actual_size_array = res_array + shard_len;
    int64_t* sample_array = (int64_t*)(res_array + shard_len * 2);

    // 1. get actual_size_array.
    // 2. get sum of actual_size.
    // 3. get offset ptr
    thrust::device_vector<int> t_res_array(shard_len);
    thrust::copy(res_array, res_array + shard_len, t_res_array.begin());
    thrust::device_vector<int> t_actual_size_array(shard_len);
    thrust::transform(t_res_array.begin(), t_res_array.end(),
                      t_actual_size_array.begin(), DegreeFunctor(graph));

    if (sample_size >= 0) {
      thrust::transform(t_actual_size_array.begin(), t_actual_size_array.end(),
                        t_actual_size_array.begin(), MaxFunctor(sample_size));
    }

    thrust::copy(t_actual_size_array.begin(), t_actual_size_array.end(),
                 actual_size_array);

    int total_sample_sum =
        thrust::reduce(t_actual_size_array.begin(), t_actual_size_array.end());

    thrust::device_vector<int> output_idx(total_sample_sum);
    thrust::device_vector<int> output_offset(shard_len);
    thrust::exclusive_scan(t_actual_size_array.begin(),
                           t_actual_size_array.end(), output_offset.begin(), 0);

    constexpr int BLOCK_WARPS = 128 / WARP_SIZE;
    constexpr int TILE_SIZE = BLOCK_WARPS * 16;
    const dim3 block_(WARP_SIZE, BLOCK_WARPS);
    const dim3 grid_((shard_len + TILE_SIZE - 1) / TILE_SIZE);
    neighbor_sample<
        BLOCK_WARPS,
        TILE_SIZE><<<grid_, block_, 0, resource_->remote_stream(i, gpu_id)>>>(
        0, graph, sample_size, res_array, shard_len, sample_array,
        thrust::raw_pointer_cast(output_idx.data()),
        thrust::raw_pointer_cast(output_offset.data()));
  }

  for (int i = 0; i < total_gpu; ++i) {
    if (h_left[i] == -1) {
      continue;
    }
    cudaStreamSynchronize(resource_->remote_stream(i, gpu_id));
    tables_[i]->rwlock_->UNLock();
  }
  // walk_to_src(num, total_gpu, h_left, h_right, d_shard_vals_ptr);

  auto d_shard_actual_sample_size = memory::Alloc(place, len * sizeof(int));
  int* d_shard_actual_sample_size_ptr =
      reinterpret_cast<int*>(d_shard_actual_sample_size->ptr());
  // Store total sample number of each gpu.
  thrust::host_vector<int> d_shard_total_sample_size(total_gpu, 0);
  move_neighbor_sample_size_to_source_gpu(
      gpu_id, total_gpu, h_left, h_right, d_shard_actual_sample_size_ptr,
      thrust::raw_pointer_cast(d_shard_total_sample_size.data()));
  int allocate_sample_num = 0;
  for (int i = 0; i < total_gpu; ++i) {
    allocate_sample_num += d_shard_total_sample_size[i];
  }
  auto d_shard_vals =
      memory::Alloc(place, allocate_sample_num * sizeof(int64_t));
  int64_t* d_shard_vals_ptr = reinterpret_cast<int64_t*>(d_shard_vals->ptr());
  move_neighbor_sample_result_to_source_gpu(gpu_id, total_gpu, h_left, h_right,
                                            d_shard_vals_ptr,
                                            d_shard_total_sample_size);

  cudaMalloc((void**)&result->val, allocate_sample_num * sizeof(int64_t));
  cudaMalloc((void**)&result->actual_sample_size, len * sizeof(int));
  cudaMalloc((void**)&result->offset, len * sizeof(int));
  int64_t* val = result->val;
  int* actual_sample_size = result->actual_sample_size;
  int* offset = result->offset;

  fill_dvalues_actual_sample_size<<<grid_size, block_size_, 0, stream>>>(
      d_shard_actual_sample_size_ptr, actual_sample_size, d_idx_ptr, len);
  thrust::device_vector<int> t_actual_sample_size(len);
  thrust::copy(actual_sample_size, actual_sample_size + len,
               t_actual_sample_size.begin());
  thrust::exclusive_scan(t_actual_sample_size.begin(),
                         t_actual_sample_size.end(), offset, 0);
  int* d_offset;
  cudaMalloc(&d_offset, len * sizeof(int));
  thrust::copy(d_shard_actual_sample_size_ptr,
               d_shard_actual_sample_size_ptr + len,
               t_actual_sample_size.begin());
  thrust::exclusive_scan(t_actual_sample_size.begin(),
                         t_actual_sample_size.end(), d_offset, 0);
  constexpr int BLOCK_WARPS_ = 128 / WARP_SIZE;
  constexpr int TILE_SIZE_ = BLOCK_WARPS_ * 16;
  const dim3 block__(WARP_SIZE, BLOCK_WARPS_);
  const dim3 grid__((len + TILE_SIZE_ - 1) / TILE_SIZE_);
  fill_dvalues_sample_result<BLOCK_WARPS_,
                             TILE_SIZE_><<<grid__, block__, 0, stream>>>(
      d_shard_vals_ptr, val, actual_sample_size, d_idx_ptr, offset, d_offset,
      len);

  cudaStreamSynchronize(stream);
  for (int i = 0; i < total_gpu; ++i) {
    int shard_len = h_left[i] == -1 ? 0 : h_right[i] - h_left[i] + 1;
    if (shard_len == 0) {
      continue;
    }
    destroy_storage(gpu_id, i);
  }
  cudaFree(d_offset);
  return result;
}

NodeQueryResult* GpuPsGraphTable::graph_node_sample(int gpu_id,
                                                    int sample_size) {}

NodeQueryResult* GpuPsGraphTable::query_node_list(int gpu_id, int start,
                                                  int query_size) {
  NodeQueryResult* result = new NodeQueryResult();
  if (query_size <= 0) return result;
  int& actual_size = result->actual_sample_size;
  actual_size = 0;
  cudaMalloc((void**)&result->val, query_size * sizeof(int64_t));
  int64_t* val = result->val;
  int dev_id = resource_->dev_id(gpu_id);
  platform::CUDADeviceGuard guard(dev_id);
  std::vector<int> idx, gpu_begin_pos, local_begin_pos, sample_size;
  int size = 0;
  /*
  if idx[i] = a, gpu_begin_pos[i] = p1,
  gpu_local_begin_pos[i] = p2;
  sample_size[i] = s;
  then on gpu a, the nodes of positions [p1,p1 + s) should be returned
  and saved from the p2 position on the sample_result array

  for example:
  suppose
  gpu 0 saves [0,2,4,6,8], gpu1 saves [1,3,5,7]
  start = 3, query_size = 5
  we know [6,8,1,3,5] should be returned;
  idx = [0,1]
  gpu_begin_pos = [3,0]
  local_begin_pos = [0,3]
  sample_size = [2,3]

  */
  for (int i = 0; i < gpu_graph_list.size() && query_size != 0; i++) {
    auto graph = gpu_graph_list[i];
    if (graph.node_size == 0) {
      continue;
    }
    if (graph.node_size + size > start) {
      int cur_size = min(query_size, graph.node_size + size - start);
      query_size -= cur_size;
      idx.emplace_back(i);
      gpu_begin_pos.emplace_back(start - size);
      local_begin_pos.emplace_back(actual_size);
      start += cur_size;
      actual_size += cur_size;
      sample_size.emplace_back(cur_size);
      create_storage(gpu_id, i, 1, cur_size * sizeof(int64_t));
    }
    size += graph.node_size;
  }
  for (int i = 0; i < idx.size(); i++) {
    int dev_id_i = resource_->dev_id(idx[i]);
    platform::CUDADeviceGuard guard(dev_id_i);
    auto& node = path_[gpu_id][idx[i]].nodes_.front();
    int grid_size = (sample_size[i] - 1) / block_size_ + 1;
    node_query_example<<<grid_size, block_size_, 0,
                         resource_->remote_stream(idx[i], gpu_id)>>>(
        gpu_graph_list[idx[i]], gpu_begin_pos[i], sample_size[i],
        (int64_t*)node.val_storage);
  }

  for (int i = 0; i < idx.size(); i++) {
    cudaStreamSynchronize(resource_->remote_stream(idx[i], gpu_id));
    auto& node = path_[gpu_id][idx[i]].nodes_.front();
    cudaMemcpyAsync(reinterpret_cast<char*>(val + local_begin_pos[i]),
                    node.val_storage, node.val_bytes_len, cudaMemcpyDefault,
                    node.out_stream);
  }
  for (int i = 0; i < idx.size(); i++) {
    auto& node = path_[gpu_id][idx[i]].nodes_.front();
    cudaStreamSynchronize(node.out_stream);
  }
  return result;
}
}
};
#endif
