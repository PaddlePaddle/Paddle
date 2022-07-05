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

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

#include <functional>
#pragma once
#ifdef PADDLE_WITH_HETERPS
#include "paddle/fluid/framework/fleet/heter_ps/graph_gpu_ps_table.h"
namespace paddle {
namespace framework {
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

__global__ void get_cpu_id_index(int64_t* key,
                                 int* actual_sample_size,
                                 int64_t* cpu_key,
                                 int* sum,
                                 int* index,
                                 int len) {
  CUDA_KERNEL_LOOP(i, len) {
    if (actual_sample_size[i] == -1) {
      int old = atomicAdd(sum, 1);
      cpu_key[old] = key[i];
      index[old] = i;
      // printf("old %d i-%d key:%lld\n",old,i,key[i]);
    }
  }
}

__global__ void get_actual_gpu_ac(int* gpu_ac, int number_on_cpu) {
  CUDA_KERNEL_LOOP(i, number_on_cpu) { gpu_ac[i] /= sizeof(int64_t); }
}

template <int WARP_SIZE, int BLOCK_WARPS, int TILE_SIZE>
__global__ void copy_buffer_ac_to_final_place(int64_t* gpu_buffer,
                                              int* gpu_ac,
                                              int64_t* val,
                                              int* actual_sample_size,
                                              int* index,
                                              int* cumsum_gpu_ac,
                                              int number_on_cpu,
                                              int sample_size) {
  assert(blockDim.x == WARP_SIZE);
  assert(blockDim.y == BLOCK_WARPS);

  int i = blockIdx.x * TILE_SIZE + threadIdx.y;
  const int last_idx =
      min(static_cast<int>(blockIdx.x + 1) * TILE_SIZE, number_on_cpu);
  while (i < last_idx) {
    actual_sample_size[index[i]] = gpu_ac[i];
    for (int j = threadIdx.x; j < gpu_ac[i]; j += WARP_SIZE) {
      val[index[i] * sample_size + j] = gpu_buffer[cumsum_gpu_ac[i] + j];
    }
    i += BLOCK_WARPS;
  }
}

template <int WARP_SIZE, int BLOCK_WARPS, int TILE_SIZE>
__global__ void neighbor_sample_example_v2(GpuPsCommGraph graph,
                                           int64_t* node_index,
                                           int* actual_size,
                                           int64_t* res,
                                           int sample_len,
                                           int n,
                                           int default_value) {
  assert(blockDim.x == WARP_SIZE);
  assert(blockDim.y == BLOCK_WARPS);

  int i = blockIdx.x * TILE_SIZE + threadIdx.y;
  const int last_idx = min(static_cast<int>(blockIdx.x + 1) * TILE_SIZE, n);
  curandState rng;
  curand_init(blockIdx.x, threadIdx.y * WARP_SIZE + threadIdx.x, 0, &rng);

  while (i < last_idx) {
    if (node_index[i] == -1) {
      actual_size[i] = default_value;
      i += BLOCK_WARPS;
      continue;
    }
    int neighbor_len = (int)graph.node_list[node_index[i]].neighbor_size;
    int64_t data_offset = graph.node_list[node_index[i]].neighbor_offset;
    int offset = i * sample_len;
    int64_t* data = graph.neighbor_list;
    if (neighbor_len <= sample_len) {
      for (int j = threadIdx.x; j < neighbor_len; j += WARP_SIZE) {
        res[offset + j] = data[data_offset + j];
      }
      actual_size[i] = neighbor_len;
    } else {
      for (int j = threadIdx.x; j < sample_len; j += WARP_SIZE) {
        res[offset + j] = j;
      }
      __syncwarp();
      for (int j = sample_len + threadIdx.x; j < neighbor_len; j += WARP_SIZE) {
        const int num = curand(&rng) % (j + 1);
        if (num < sample_len) {
          atomicMax(reinterpret_cast<unsigned int*>(res + offset + num),
                    static_cast<unsigned int>(j));
        }
      }
      __syncwarp();
      for (int j = threadIdx.x; j < sample_len; j += WARP_SIZE) {
        const int64_t perm_idx = res[offset + j] + data_offset;
        res[offset + j] = data[perm_idx];
      }
      actual_size[i] = sample_len;
    }
    i += BLOCK_WARPS;
  }
}

__global__ void neighbor_sample_example(GpuPsCommGraph graph,
                                        int64_t* node_index,
                                        int* actual_size,
                                        int64_t* res,
                                        int sample_len,
                                        int* sample_status,
                                        int n,
                                        int from) {
  int id = blockIdx.x * blockDim.y + threadIdx.y;
  if (id < n) {
    if (node_index[id] == -1) {
      actual_size[id] = 0;
      return;
    }
    curandState rng;
    curand_init(blockIdx.x, threadIdx.x, threadIdx.y, &rng);
    int64_t index = threadIdx.x;
    int64_t offset = id * sample_len;
    int64_t* data = graph.neighbor_list;
    int64_t data_offset = graph.node_list[node_index[id]].neighbor_offset;
    int64_t neighbor_len = graph.node_list[node_index[id]].neighbor_size;
    int ac_len;
    if (sample_len > neighbor_len)
      ac_len = neighbor_len;
    else {
      ac_len = sample_len;
    }
    if (4 * ac_len >= 3 * neighbor_len) {
      if (index == 0) {
        res[offset] = curand(&rng) % (neighbor_len - ac_len + 1);
      }
      __syncwarp();
      int start = res[offset];
      while (index < ac_len) {
        res[offset + index] = data[data_offset + start + index];
        index += blockDim.x;
      }
      actual_size[id] = ac_len;
    } else {
      while (index < ac_len) {
        int num = curand(&rng) % neighbor_len;
        int* addr = sample_status + data_offset + num;
        int expected = *addr;
        if (!(expected & (1 << from))) {
          int old = atomicCAS(addr, expected, expected | (1 << from));
          if (old == expected) {
            res[offset + index] = num;
            index += blockDim.x;
          }
        }
      }
      __syncwarp();
      index = threadIdx.x;
      while (index < ac_len) {
        int* addr = sample_status + data_offset + res[offset + index];
        int expected, old = *addr;
        do {
          expected = old;
          old = atomicCAS(addr, expected, expected & (~(1 << from)));
        } while (old != expected);
        res[offset + index] = data[data_offset + res[offset + index]];
        index += blockDim.x;
      }
      actual_size[id] = ac_len;
    }
  }
  // const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  // if (i < n) {
  //   auto node_index = index[i];
  //   actual_size[i] = graph.node_list[node_index].neighbor_size < sample_size
  //                        ? graph.node_list[node_index].neighbor_size
  //                        : sample_size;
  //   int offset = graph.node_list[node_index].neighbor_offset;
  //   for (int j = 0; j < actual_size[i]; j++) {
  //     sample_result[sample_size * i + j] = graph.neighbor_list[offset + j];
  //   }
  // }
}

int GpuPsGraphTable::init_cpu_table(
    const paddle::distributed::GraphParameter& graph) {
  cpu_graph_table.reset(new paddle::distributed::GraphTable);
  cpu_table_status = cpu_graph_table->Initialize(graph);
  // if (cpu_table_status != 0) return cpu_table_status;
  // std::function<void(std::vector<GpuPsCommGraph>&)> callback =
  //     [this](std::vector<GpuPsCommGraph>& res) {
  //       pthread_rwlock_wrlock(this->rw_lock.get());
  //       this->clear_graph_info();
  //       this->build_graph_from_cpu(res);
  //       pthread_rwlock_unlock(this->rw_lock.get());
  //       cv_.notify_one();
  //     };
  // cpu_graph_table->set_graph_sample_callback(callback);
  return cpu_table_status;
}

// int GpuPsGraphTable::load(const std::string& path, const std::string& param)
// {
//   int status = cpu_graph_table->load(path, param);
//   if (status != 0) {
//     return status;
//   }
//   std::unique_lock<std::mutex> lock(mutex_);
//   cpu_graph_table->start_graph_sampling();
//   cv_.wait(lock);
//   return 0;
// }
/*
 comment 1
 gpu i triggers a neighbor_sample task,
 when this task is done,
 this function is called to move the sample result on other gpu back
 to gup i and aggragate the result.
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

void GpuPsGraphTable::display_sample_res(void* key,
                                         void* val,
                                         int len,
                                         int sample_len) {
  char key_buffer[len * sizeof(int64_t)];
  char val_buffer[sample_len * sizeof(int64_t) * len +
                  (len + len % 2) * sizeof(int) + len * sizeof(int64_t)];
  cudaMemcpy(key_buffer, key, sizeof(int64_t) * len, cudaMemcpyDeviceToHost);
  cudaMemcpy(val_buffer,
             val,
             sample_len * sizeof(int64_t) * len +
                 (len + len % 2) * sizeof(int) + len * sizeof(int64_t),
             cudaMemcpyDeviceToHost);
  int64_t* sample_val = (int64_t*)(val_buffer + (len + len % 2) * sizeof(int) +
                                   len * sizeof(int64_t));
  for (int i = 0; i < len; i++) {
    printf("key %lld\n", *(int64_t*)(key_buffer + i * sizeof(int64_t)));
    printf("index %lld\n", *(int64_t*)(val_buffer + i * sizeof(int64_t)));
    int ac_size = *(int*)(val_buffer + i * sizeof(int) + len * sizeof(int64_t));
    printf("sampled %d neigbhors\n", ac_size);
    for (int j = 0; j < ac_size; j++) {
      printf("%lld ", sample_val[i * sample_len + j]);
    }
    printf("\n");
  }
}
void GpuPsGraphTable::move_neighbor_sample_result_to_source_gpu(
    int start_index,
    int gpu_num,
    int sample_size,
    int* h_left,
    int* h_right,
    int64_t* src_sample_res,
    int* actual_sample_size) {
  int shard_len[gpu_num];
  for (int i = 0; i < gpu_num; i++) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    shard_len[i] = h_right[i] - h_left[i] + 1;
    int cur_step = (int)path_[start_index][i].nodes_.size() - 1;
    for (int j = cur_step; j > 0; j--) {
      cudaMemcpyAsync(path_[start_index][i].nodes_[j - 1].val_storage,
                      path_[start_index][i].nodes_[j].val_storage,
                      path_[start_index][i].nodes_[j - 1].val_bytes_len,
                      cudaMemcpyDefault,
                      path_[start_index][i].nodes_[j - 1].out_stream);
    }
    auto& node = path_[start_index][i].nodes_.front();
    cudaMemcpyAsync(
        reinterpret_cast<char*>(src_sample_res + h_left[i] * sample_size),
        node.val_storage + sizeof(int64_t) * shard_len[i] +
            sizeof(int) * (shard_len[i] + shard_len[i] % 2),
        sizeof(int64_t) * shard_len[i] * sample_size,
        cudaMemcpyDefault,
        node.out_stream);
    cudaMemcpyAsync(reinterpret_cast<char*>(actual_sample_size + h_left[i]),
                    node.val_storage + sizeof(int64_t) * shard_len[i],
                    sizeof(int) * shard_len[i],
                    cudaMemcpyDefault,
                    node.out_stream);
  }
  for (int i = 0; i < gpu_num; ++i) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    auto& node = path_[start_index][i].nodes_.front();
    cudaStreamSynchronize(node.out_stream);
    // cudaStreamSynchronize(resource_->remote_stream(i, start_index));
  }
  /*
    std::queue<CopyTask> que;
    // auto& node = path_[gpu_id][i].nodes_.front();
    // cudaMemcpyAsync(
    //     reinterpret_cast<char*>(src_sample_res + h_left[i] * sample_size),
    //     node.val_storage + sizeof(int64_t) * shard_len,
    //     node.val_bytes_len - sizeof(int64_t) * shard_len, cudaMemcpyDefault,
    //     node.out_stream);
    // cudaMemcpyAsync(reinterpret_cast<char*>(actual_sample_size + h_left[i]),
    //                 node.val_storage + sizeof(int) * shard_len,
    //                 sizeof(int) * shard_len, cudaMemcpyDefault,
    //                 node.out_stream);
    int cur_step = path_[start_index][i].nodes_.size() - 1;
    auto& node = path_[start_index][i].nodes_[cur_step];
    if (cur_step == 0) {
      // cudaMemcpyAsync(reinterpret_cast<char*>(src_val + h_left[i]),
      //                 node.val_storage, node.val_bytes_len,
      //                 cudaMemcpyDefault,
      //                 node.out_stream);
     // VLOG(0)<<"copy "<<node.gpu_num<<" to "<<start_index;
      cudaMemcpyAsync(
          reinterpret_cast<char*>(src_sample_res + h_left[i] * sample_size),
          node.val_storage + sizeof(int64_t) * shard_len[i],
          node.val_bytes_len - sizeof(int64_t) * shard_len[i],
          cudaMemcpyDefault,
          node.out_stream);
          //resource_->remote_stream(i, start_index));
      cudaMemcpyAsync(reinterpret_cast<char*>(actual_sample_size + h_left[i]),
                      node.val_storage + sizeof(int) * shard_len[i],
                      sizeof(int) * shard_len[i], cudaMemcpyDefault,
                      node.out_stream);
                      //resource_->remote_stream(i, start_index));
    } else {
      CopyTask t(&path_[start_index][i], cur_step - 1);
      que.push(t);
       //     VLOG(0)<<"copy "<<node.gpu_num<<" to
  "<<path_[start_index][i].nodes_[cur_step - 1].gpu_num;
      cudaMemcpyAsync(path_[start_index][i].nodes_[cur_step - 1].val_storage,
                      node.val_storage,
                      path_[start_index][i].nodes_[cur_step - 1].val_bytes_len,
                      cudaMemcpyDefault,
                     path_[start_index][i].nodes_[cur_step - 1].out_stream);
                     //resource_->remote_stream(i, start_index));
    }
  }
  while (!que.empty()) {
    CopyTask& cur_task = que.front();
    que.pop();
    int cur_step = cur_task.step;
    if (cur_task.path->nodes_[cur_step].sync) {
      cudaStreamSynchronize(cur_task.path->nodes_[cur_step].out_stream);
      //cudaStreamSynchronize(resource_->remote_stream(cur_task.path->nodes_.back().gpu_num,
  start_index));
    }
    if (cur_step > 0) {
      CopyTask c(cur_task.path, cur_step - 1);
      que.push(c);
      cudaMemcpyAsync(cur_task.path->nodes_[cur_step - 1].val_storage,
                      cur_task.path->nodes_[cur_step].val_storage,
                      cur_task.path->nodes_[cur_step - 1].val_bytes_len,
                      cudaMemcpyDefault,
                      cur_task.path->nodes_[cur_step - 1].out_stream);
                      //resource_->remote_stream(cur_task.path->nodes_.back().gpu_num,
  start_index));
    } else if (cur_step == 0) {
      int end_index = cur_task.path->nodes_.back().gpu_num;
      // cudaMemcpyAsync(reinterpret_cast<char*>(src_val + h_left[end_index]),
      //                 cur_task.path->nodes_[cur_step].val_storage,
      //                 cur_task.path->nodes_[cur_step].val_bytes_len,
      //                 cudaMemcpyDefault,
      //                 cur_task.path->nodes_[cur_step].out_stream);
      //VLOG(0)<<"copy "<<cur_task.path->nodes_[cur_step].gpu_num<< " to
  "<<start_index;
      cudaMemcpyAsync(reinterpret_cast<char*>(src_sample_res +
                                              h_left[end_index] * sample_size),
                      cur_task.path->nodes_[cur_step].val_storage +
                          sizeof(int64_t) * shard_len[end_index],
                      cur_task.path->nodes_[cur_step].val_bytes_len -
                          sizeof(int64_t) * shard_len[end_index],
                      cudaMemcpyDefault,
                      cur_task.path->nodes_[cur_step].out_stream);
                      //resource_->remote_stream(cur_task.path->nodes_.back().gpu_num,
  start_index));
      cudaMemcpyAsync(
          reinterpret_cast<char*>(actual_sample_size + h_left[end_index]),
          cur_task.path->nodes_[cur_step].val_storage +
              sizeof(int) * shard_len[end_index],
          sizeof(int) * shard_len[end_index], cudaMemcpyDefault,
          cur_task.path->nodes_[cur_step].out_stream);
          //resource_->remote_stream(cur_task.path->nodes_.back().gpu_num,
  start_index));
    }
  }
  for (int i = 0; i < gpu_num; ++i) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    auto& node = path_[start_index][i].nodes_.front();
    cudaStreamSynchronize(node.out_stream);
    //cudaStreamSynchronize(resource_->remote_stream(i, start_index));
  }
  */
}

/*
TODO:
how to optimize it to eliminate the for loop
*/
__global__ void fill_dvalues(int64_t* d_shard_vals,
                             int64_t* d_vals,
                             int* d_shard_actual_sample_size,
                             int* d_actual_sample_size,
                             int* idx,
                             int sample_size,
                             int len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    d_actual_sample_size[idx[i]] = d_shard_actual_sample_size[i];
    for (int j = 0; j < sample_size; j++) {
      d_vals[idx[i] * sample_size + j] = d_shard_vals[i * sample_size + j];
    }
  }
}

__global__ void fill_actual_vals(int64_t* vals,
                                 int64_t* actual_vals,
                                 int* actual_sample_size,
                                 int* cumsum_actual_sample_size,
                                 int sample_size,
                                 int len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    for (int j = 0; j < actual_sample_size[i]; j++) {
      actual_vals[cumsum_actual_sample_size[i] + j] = vals[sample_size * i + j];
    }
  }
}

__global__ void node_query_example(GpuPsCommGraph graph,
                                   int start,
                                   int size,
                                   int64_t* res) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    res[i] = graph.node_list[start + i].node_id;
  }
}

void GpuPsGraphTable::clear_graph_info(int gpu_id) {
  if (tables_.size() && tables_[gpu_id] != NULL) {
    delete tables_[gpu_id];
  }
  auto& graph = gpu_graph_list[gpu_id];
  if (graph.neighbor_list != NULL) {
    cudaFree(graph.neighbor_list);
  }
  if (graph.node_list != NULL) {
    cudaFree(graph.node_list);
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

void GpuPsGraphTable::build_graph_on_single_gpu(GpuPsCommGraph& g, int i) {
  clear_graph_info(i);
  platform::CUDADeviceGuard guard(resource_->dev_id(i));
  // platform::CUDADeviceGuard guard(i);
  gpu_graph_list[i] = GpuPsCommGraph();
  sample_status[i] = NULL;
  tables_[i] = new Table(std::max((int64_t)1, g.node_size) / load_factor_);
  if (g.node_size > 0) {
    std::vector<int64_t> keys;
    std::vector<int64_t> offset;
    cudaMalloc((void**)&gpu_graph_list[i].node_list,
               g.node_size * sizeof(GpuPsGraphNode));
    cudaMemcpy(gpu_graph_list[i].node_list,
               g.node_list,
               g.node_size * sizeof(GpuPsGraphNode),
               cudaMemcpyHostToDevice);
    for (int64_t j = 0; j < g.node_size; j++) {
      keys.push_back(g.node_list[j].node_id);
      offset.push_back(j);
    }
    build_ps(i, (uint64_t*)keys.data(), offset.data(), keys.size(), 1024, 8);
    gpu_graph_list[i].node_size = g.node_size;
  } else {
    build_ps(i, NULL, NULL, 0, 1024, 8);
    gpu_graph_list[i].node_list = NULL;
    gpu_graph_list[i].node_size = 0;
  }
  if (g.neighbor_size) {
    cudaError_t cudaStatus =
        cudaMalloc((void**)&gpu_graph_list[i].neighbor_list,
                   g.neighbor_size * sizeof(int64_t));
    PADDLE_ENFORCE_EQ(cudaStatus,
                      cudaSuccess,
                      platform::errors::InvalidArgument(
                          "ailed to allocate memory for graph on gpu "));
    VLOG(0) << "sucessfully allocate " << g.neighbor_size * sizeof(int64_t)
            << " bytes of memory for graph-edges on gpu "
            << resource_->dev_id(i);
    cudaMemcpy(gpu_graph_list[i].neighbor_list,
               g.neighbor_list,
               g.neighbor_size * sizeof(int64_t),
               cudaMemcpyHostToDevice);
    gpu_graph_list[i].neighbor_size = g.neighbor_size;
  } else {
    gpu_graph_list[i].neighbor_list = NULL;
    gpu_graph_list[i].neighbor_size = 0;
  }
}

void GpuPsGraphTable::init_sample_status() {
  for (int i = 0; i < gpu_num; i++) {
    if (gpu_graph_list[i].neighbor_size) {
      platform::CUDADeviceGuard guard(resource_->dev_id(i));
      int* addr;
      cudaMalloc((void**)&addr, gpu_graph_list[i].neighbor_size * sizeof(int));
      cudaMemset(addr, 0, gpu_graph_list[i].neighbor_size * sizeof(int));
      sample_status[i] = addr;
    }
  }
}

void GpuPsGraphTable::free_sample_status() {
  for (int i = 0; i < gpu_num; i++) {
    if (sample_status[i] != NULL) {
      platform::CUDADeviceGuard guard(resource_->dev_id(i));
      cudaFree(sample_status[i]);
    }
  }
}
void GpuPsGraphTable::build_graph_from_cpu(
    std::vector<GpuPsCommGraph>& cpu_graph_list) {
  VLOG(0) << "in build_graph_from_cpu cpu_graph_list size = "
          << cpu_graph_list.size();
  PADDLE_ENFORCE_EQ(
      cpu_graph_list.size(),
      resource_->total_device(),
      platform::errors::InvalidArgument("the cpu node list size doesn't match "
                                        "the number of gpu on your machine."));
  clear_graph_info();
  for (int i = 0; i < cpu_graph_list.size(); i++) {
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    gpu_graph_list[i] = GpuPsCommGraph();
    sample_status[i] = NULL;
    tables_[i] = new Table(std::max((int64_t)1, cpu_graph_list[i].node_size) /
                           load_factor_);
    if (cpu_graph_list[i].node_size > 0) {
      std::vector<int64_t> keys;
      std::vector<int64_t> offset;
      cudaMalloc((void**)&gpu_graph_list[i].node_list,
                 cpu_graph_list[i].node_size * sizeof(GpuPsGraphNode));
      cudaMemcpy(gpu_graph_list[i].node_list,
                 cpu_graph_list[i].node_list,
                 cpu_graph_list[i].node_size * sizeof(GpuPsGraphNode),
                 cudaMemcpyHostToDevice);
      for (int64_t j = 0; j < cpu_graph_list[i].node_size; j++) {
        keys.push_back(cpu_graph_list[i].node_list[j].node_id);
        offset.push_back(j);
      }
      build_ps(
          i, (uint64_t*)(keys.data()), offset.data(), keys.size(), 1024, 8);
      gpu_graph_list[i].node_size = cpu_graph_list[i].node_size;
    } else {
      build_ps(i, NULL, NULL, 0, 1024, 8);
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

NeighborSampleResult GpuPsGraphTable::graph_neighbor_sample_v3(
    NeighborSampleQuery q, bool cpu_switch) {
  return graph_neighbor_sample_v2(
      global_device_map[q.gpu_id], q.key, q.sample_size, q.len, cpu_switch);
}
NeighborSampleResult GpuPsGraphTable::graph_neighbor_sample(int gpu_id,
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

  NeighborSampleResult result;
  result.initialize(sample_size, len, resource_->dev_id(gpu_id));
  if (len == 0) {
    return result;
  }
  platform::CUDAPlace place = platform::CUDAPlace(resource_->dev_id(gpu_id));
  platform::CUDADeviceGuard guard(resource_->dev_id(gpu_id));
  int* actual_sample_size = result.actual_sample_size;
  int64_t* val = result.val;
  int total_gpu = resource_->total_device();
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
  auto d_shard_vals = memory::Alloc(place, sample_size * len * sizeof(int64_t));
  int64_t* d_shard_vals_ptr = reinterpret_cast<int64_t*>(d_shard_vals->ptr());
  auto d_shard_actual_sample_size = memory::Alloc(place, len * sizeof(int));
  int* d_shard_actual_sample_size_ptr =
      reinterpret_cast<int*>(d_shard_actual_sample_size->ptr());

  split_input_to_shard(
      (uint64_t*)(key), d_idx_ptr, len, d_left_ptr, d_right_ptr, gpu_id);

  heter_comm_kernel_->fill_shard_key(
      d_shard_keys_ptr, key, d_idx_ptr, len, stream);
  cudaStreamSynchronize(stream);

  cudaMemcpy(
      h_left, d_left_ptr, total_gpu * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(
      h_right, d_right_ptr, total_gpu * sizeof(int), cudaMemcpyDeviceToHost);
  // auto start1 = std::chrono::steady_clock::now();
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

    create_storage(gpu_id,
                   i,
                   shard_len * sizeof(int64_t),
                   shard_len * (1 + sample_size) * sizeof(int64_t) +
                       sizeof(int) * (shard_len + shard_len % 2));
    // auto& node = path_[gpu_id][i].nodes_[0];
  }
  walk_to_dest(
      gpu_id, total_gpu, h_left, h_right, (uint64_t*)(d_shard_keys_ptr), NULL);

  for (int i = 0; i < total_gpu; ++i) {
    if (h_left[i] == -1) {
      continue;
    }
    int shard_len = h_left[i] == -1 ? 0 : h_right[i] - h_left[i] + 1;
    auto& node = path_[gpu_id][i].nodes_.back();
    cudaMemsetAsync(
        node.val_storage, -1, shard_len * sizeof(int64_t), node.in_stream);
    cudaStreamSynchronize(node.in_stream);
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    tables_[i]->get(reinterpret_cast<uint64_t*>(node.key_storage),
                    reinterpret_cast<int64_t*>(node.val_storage),
                    h_right[i] - h_left[i] + 1,
                    resource_->remote_stream(i, gpu_id));
    // node.in_stream);
    auto graph = gpu_graph_list[i];
    int64_t* id_array = reinterpret_cast<int64_t*>(node.val_storage);
    int* actual_size_array = (int*)(id_array + shard_len);
    int64_t* sample_array =
        (int64_t*)(actual_size_array + shard_len + shard_len % 2);
    int sample_grid_size = (shard_len - 1) / dim_y + 1;
    dim3 block(parallel_sample_size, dim_y);
    dim3 grid(sample_grid_size);
    neighbor_sample_example<<<grid,
                              block,
                              0,
                              resource_->remote_stream(i, gpu_id)>>>(
        graph,
        id_array,
        actual_size_array,
        sample_array,
        sample_size,
        sample_status[i],
        shard_len,
        gpu_id);
  }

  for (int i = 0; i < total_gpu; ++i) {
    if (h_left[i] == -1) {
      continue;
    }
    cudaStreamSynchronize(resource_->remote_stream(i, gpu_id));
  }
  move_neighbor_sample_result_to_source_gpu(gpu_id,
                                            total_gpu,
                                            sample_size,
                                            h_left,
                                            h_right,
                                            d_shard_vals_ptr,
                                            d_shard_actual_sample_size_ptr);
  fill_dvalues<<<grid_size, block_size_, 0, stream>>>(
      d_shard_vals_ptr,
      val,
      d_shard_actual_sample_size_ptr,
      actual_sample_size,
      d_idx_ptr,
      sample_size,
      len);
  for (int i = 0; i < total_gpu; ++i) {
    int shard_len = h_left[i] == -1 ? 0 : h_right[i] - h_left[i] + 1;
    if (shard_len == 0) {
      continue;
    }
    destroy_storage(gpu_id, i);
  }
  cudaStreamSynchronize(stream);
  return result;
}

NeighborSampleResult GpuPsGraphTable::graph_neighbor_sample_v2(
    int gpu_id, int64_t* key, int sample_size, int len, bool cpu_query_switch) {
  NeighborSampleResult result;
  result.initialize(sample_size, len, resource_->dev_id(gpu_id));

  if (len == 0) {
    return result;
  }

  platform::CUDAPlace place = platform::CUDAPlace(resource_->dev_id(gpu_id));
  platform::CUDADeviceGuard guard(resource_->dev_id(gpu_id));
  int* actual_sample_size = result.actual_sample_size;
  int64_t* val = result.val;
  int total_gpu = resource_->total_device();
  auto stream = resource_->local_stream(gpu_id, 0);

  int grid_size = (len - 1) / block_size_ + 1;

  int h_left[total_gpu];   // NOLINT
  int h_right[total_gpu];  // NOLINT

  auto d_left = memory::Alloc(place, total_gpu * sizeof(int));
  auto d_right = memory::Alloc(place, total_gpu * sizeof(int));
  int* d_left_ptr = reinterpret_cast<int*>(d_left->ptr());
  int* d_right_ptr = reinterpret_cast<int*>(d_right->ptr());
  int default_value = 0;
  if (cpu_query_switch) {
    default_value = -1;
  }

  cudaMemsetAsync(d_left_ptr, -1, total_gpu * sizeof(int), stream);
  cudaMemsetAsync(d_right_ptr, -1, total_gpu * sizeof(int), stream);
  //
  auto d_idx = memory::Alloc(place, len * sizeof(int));
  int* d_idx_ptr = reinterpret_cast<int*>(d_idx->ptr());

  auto d_shard_keys = memory::Alloc(place, len * sizeof(int64_t));
  int64_t* d_shard_keys_ptr = reinterpret_cast<int64_t*>(d_shard_keys->ptr());
  auto d_shard_vals = memory::Alloc(place, sample_size * len * sizeof(int64_t));
  int64_t* d_shard_vals_ptr = reinterpret_cast<int64_t*>(d_shard_vals->ptr());
  auto d_shard_actual_sample_size = memory::Alloc(place, len * sizeof(int));
  int* d_shard_actual_sample_size_ptr =
      reinterpret_cast<int*>(d_shard_actual_sample_size->ptr());

  split_input_to_shard(
      (uint64_t*)(key), d_idx_ptr, len, d_left_ptr, d_right_ptr, gpu_id);

  heter_comm_kernel_->fill_shard_key(
      d_shard_keys_ptr, key, d_idx_ptr, len, stream);

  cudaStreamSynchronize(stream);

  cudaMemcpy(
      h_left, d_left_ptr, total_gpu * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(
      h_right, d_right_ptr, total_gpu * sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < total_gpu; ++i) {
    int shard_len = h_left[i] == -1 ? 0 : h_right[i] - h_left[i] + 1;
    if (shard_len == 0) {
      continue;
    }
    create_storage(gpu_id,
                   i,
                   shard_len * sizeof(int64_t),
                   shard_len * (1 + sample_size) * sizeof(int64_t) +
                       sizeof(int) * (shard_len + shard_len % 2));
  }
  walk_to_dest(
      gpu_id, total_gpu, h_left, h_right, (uint64_t*)(d_shard_keys_ptr), NULL);

  for (int i = 0; i < total_gpu; ++i) {
    if (h_left[i] == -1) {
      continue;
    }
    int shard_len = h_left[i] == -1 ? 0 : h_right[i] - h_left[i] + 1;
    auto& node = path_[gpu_id][i].nodes_.back();
    cudaMemsetAsync(
        node.val_storage, -1, shard_len * sizeof(int64_t), node.in_stream);
    cudaStreamSynchronize(node.in_stream);
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    // If not found, val is -1.
    tables_[i]->get(reinterpret_cast<uint64_t*>(node.key_storage),
                    reinterpret_cast<int64_t*>(node.val_storage),
                    h_right[i] - h_left[i] + 1,
                    resource_->remote_stream(i, gpu_id));

    auto graph = gpu_graph_list[i];
    int64_t* id_array = reinterpret_cast<int64_t*>(node.val_storage);
    int* actual_size_array = (int*)(id_array + shard_len);
    int64_t* sample_array =
        (int64_t*)(actual_size_array + shard_len + shard_len % 2);
    constexpr int WARP_SIZE = 32;
    constexpr int BLOCK_WARPS = 128 / WARP_SIZE;
    constexpr int TILE_SIZE = BLOCK_WARPS * 16;
    const dim3 block(WARP_SIZE, BLOCK_WARPS);
    const dim3 grid((shard_len + TILE_SIZE - 1) / TILE_SIZE);
    neighbor_sample_example_v2<WARP_SIZE, BLOCK_WARPS, TILE_SIZE>
        <<<grid, block, 0, resource_->remote_stream(i, gpu_id)>>>(
            graph,
            id_array,
            actual_size_array,
            sample_array,
            sample_size,
            shard_len,
            default_value);
  }

  for (int i = 0; i < total_gpu; ++i) {
    if (h_left[i] == -1) {
      continue;
    }
    cudaStreamSynchronize(resource_->remote_stream(i, gpu_id));
  }

  move_neighbor_sample_result_to_source_gpu(gpu_id,
                                            total_gpu,
                                            sample_size,
                                            h_left,
                                            h_right,
                                            d_shard_vals_ptr,
                                            d_shard_actual_sample_size_ptr);
  fill_dvalues<<<grid_size, block_size_, 0, stream>>>(
      d_shard_vals_ptr,
      val,
      d_shard_actual_sample_size_ptr,
      actual_sample_size,
      d_idx_ptr,
      sample_size,
      len);

  cudaStreamSynchronize(stream);

  if (cpu_query_switch) {
    // Get cpu keys and corresponding position.
    thrust::device_vector<int64_t> t_cpu_keys(len);
    thrust::device_vector<int> t_index(len + 1, 0);
    get_cpu_id_index<<<grid_size, block_size_, 0, stream>>>(
        key,
        actual_sample_size,
        thrust::raw_pointer_cast(t_cpu_keys.data()),
        thrust::raw_pointer_cast(t_index.data()),
        thrust::raw_pointer_cast(t_index.data()) + 1,
        len);

    cudaStreamSynchronize(stream);

    int number_on_cpu = 0;
    cudaMemcpy(&number_on_cpu,
               thrust::raw_pointer_cast(t_index.data()),
               sizeof(int),
               cudaMemcpyDeviceToHost);
    if (number_on_cpu > 0) {
      int64_t* cpu_keys = new int64_t[number_on_cpu];
      cudaMemcpy(cpu_keys,
                 thrust::raw_pointer_cast(t_cpu_keys.data()),
                 number_on_cpu * sizeof(int64_t),
                 cudaMemcpyDeviceToHost);

      std::vector<std::shared_ptr<char>> buffers(number_on_cpu);
      std::vector<int> ac(number_on_cpu);

      auto status = cpu_graph_table->random_sample_neighbors(
          0, cpu_keys, sample_size, buffers, ac, false);

      int total_cpu_sample_size = std::accumulate(ac.begin(), ac.end(), 0);
      total_cpu_sample_size /= sizeof(int64_t);

      // Merge buffers into one int64_t vector.
      int64_t* merge_buffers = new int64_t[total_cpu_sample_size];
      int start = 0;
      for (int j = 0; j < number_on_cpu; j++) {
        memcpy(merge_buffers + start, (int64_t*)(buffers[j].get()), ac[j]);
        start += ac[j] / sizeof(int64_t);
      }

      // Copy merge_buffers to gpu.
      thrust::device_vector<int64_t> gpu_buffers(total_cpu_sample_size);
      thrust::device_vector<int> gpu_ac(number_on_cpu);
      int64_t* gpu_buffers_ptr = thrust::raw_pointer_cast(gpu_buffers.data());
      int* gpu_ac_ptr = thrust::raw_pointer_cast(gpu_ac.data());
      cudaMemcpyAsync(gpu_buffers_ptr,
                      merge_buffers,
                      total_cpu_sample_size * sizeof(int64_t),
                      cudaMemcpyHostToDevice,
                      stream);
      cudaMemcpyAsync(gpu_ac_ptr,
                      ac.data(),
                      number_on_cpu * sizeof(int),
                      cudaMemcpyHostToDevice,
                      stream);

      // Copy gpu_buffers and gpu_ac using kernel.
      // Kernel divide for gpu_ac_ptr.
      int grid_size2 = (number_on_cpu - 1) / block_size_ + 1;
      get_actual_gpu_ac<<<grid_size2, block_size_, 0, stream>>>(gpu_ac_ptr,
                                                                number_on_cpu);

      cudaStreamSynchronize(stream);

      thrust::device_vector<int> cumsum_gpu_ac(number_on_cpu);
      thrust::exclusive_scan(
          gpu_ac.begin(), gpu_ac.end(), cumsum_gpu_ac.begin(), 0);

      constexpr int WARP_SIZE_ = 32;
      constexpr int BLOCK_WARPS_ = 128 / WARP_SIZE_;
      constexpr int TILE_SIZE_ = BLOCK_WARPS_ * 16;
      const dim3 block2(WARP_SIZE_, BLOCK_WARPS_);
      const dim3 grid2((number_on_cpu + TILE_SIZE_ - 1) / TILE_SIZE_);
      copy_buffer_ac_to_final_place<WARP_SIZE_, BLOCK_WARPS_, TILE_SIZE_>
          <<<grid2, block2, 0, stream>>>(
              gpu_buffers_ptr,
              gpu_ac_ptr,
              val,
              actual_sample_size,
              thrust::raw_pointer_cast(t_index.data()) + 1,
              thrust::raw_pointer_cast(cumsum_gpu_ac.data()),
              number_on_cpu,
              sample_size);

      delete[] merge_buffers;
      delete[] cpu_keys;
    }
  }

  {
    cudaStreamSynchronize(stream);
    platform::CUDAPlace place = platform::CUDAPlace(resource_->dev_id(gpu_id));
    platform::CUDADeviceGuard guard(resource_->dev_id(gpu_id));

    thrust::device_vector<int> t_actual_sample_size(len);
    thrust::copy(actual_sample_size,
                 actual_sample_size + len,
                 t_actual_sample_size.begin());
    int total_sample_size = thrust::reduce(t_actual_sample_size.begin(),
                                           t_actual_sample_size.end());

    result.actual_val_mem =
        memory::AllocShared(place, total_sample_size * sizeof(int64_t));
    result.actual_val = (int64_t*)(result.actual_val_mem)->ptr();

    result.set_total_sample_size(total_sample_size);

    thrust::device_vector<int> cumsum_actual_sample_size(len);
    thrust::exclusive_scan(t_actual_sample_size.begin(),
                           t_actual_sample_size.end(),
                           cumsum_actual_sample_size.begin(),
                           0);
    fill_actual_vals<<<grid_size, block_size_, 0, stream>>>(
        val,
        result.actual_val,
        actual_sample_size,
        thrust::raw_pointer_cast(cumsum_actual_sample_size.data()),
        sample_size,
        len);
  }
  for (int i = 0; i < total_gpu; ++i) {
    int shard_len = h_left[i] == -1 ? 0 : h_right[i] - h_left[i] + 1;
    if (shard_len == 0) {
      continue;
    }
    destroy_storage(gpu_id, i);
  }

  cudaStreamSynchronize(stream);
  return result;
}

NodeQueryResult GpuPsGraphTable::graph_node_sample(int gpu_id,
                                                   int sample_size) {
  return NodeQueryResult();
}

NodeQueryResult GpuPsGraphTable::query_node_list(int gpu_id,
                                                 int start,
                                                 int query_size) {
  NodeQueryResult result;
  if (query_size <= 0) return result;
  int& actual_size = result.actual_sample_size;
  actual_size = 0;
  // int dev_id = resource_->dev_id(gpu_id);
  // platform::CUDADeviceGuard guard(dev_id);
  std::vector<int> idx, gpu_begin_pos, local_begin_pos;
  int sample_size;
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
  std::function<int(int, int, int, int, int&, int&)> range_check =
      [](int x, int y, int x1, int y1, int& x2, int& y2) {
        if (y <= x1 || x >= y1) return 0;
        y2 = min(y, y1);
        x2 = max(x1, x);
        return y2 - x2;
      };
  auto graph = gpu_graph_list[gpu_id];
  if (graph.node_size == 0) {
    return result;
  }
  int x2, y2;
  int len = range_check(start, start + query_size, 0, graph.node_size, x2, y2);

  if (len == 0) {
    return result;
  }
  int64_t* val;
  sample_size = len;
  result.initialize(len, resource_->dev_id(gpu_id));
  actual_size = len;
  val = result.val;
  int dev_id_i = resource_->dev_id(gpu_id);
  platform::CUDADeviceGuard guard(dev_id_i);
  // platform::CUDADeviceGuard guard(i);
  int grid_size = (len - 1) / block_size_ + 1;
  node_query_example<<<grid_size,
                       block_size_,
                       0,
                       resource_->remote_stream(gpu_id, gpu_id)>>>(
      gpu_graph_list[gpu_id], x2, len, (int64_t*)val);
  cudaStreamSynchronize(resource_->remote_stream(gpu_id, gpu_id));
  return result;
  /*
  for (int i = 0; i < gpu_graph_list.size() && query_size != 0; i++) {
    auto graph = gpu_graph_list[i];
    if (graph.node_size == 0) {
      continue;
    }
    int x2, y2;
    int len = range_check(start, start + query_size, size,
                          size + graph.node_size, x2, y2);
    if (len > 0) {
      idx.push_back(i);
      gpu_begin_pos.emplace_back(x2 - size);
      local_begin_pos.emplace_back(actual_size);
      sample_size.push_back(len);
      actual_size += len;
      create_storage(gpu_id, i, 1, len * sizeof(int64_t));
    }
    size += graph.node_size;
  }
  for (int i = 0; i < idx.size(); i++) {
    int dev_id_i = resource_->dev_id(idx[i]);
    platform::CUDADeviceGuard guard(dev_id_i);
    // platform::CUDADeviceGuard guard(i);
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
  for (auto x : idx) {
    destroy_storage(gpu_id, x);
  }
  return result;
  */
}
}  // namespace framework
};  // namespace paddle
#endif
