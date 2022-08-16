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
#include "paddle/fluid/framework/fleet/heter_ps/gpu_graph_utils.h"
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

__global__ void get_cpu_id_index(uint64_t* key,
                                 int* actual_sample_size,
                                 uint64_t* cpu_key,
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
  CUDA_KERNEL_LOOP(i, number_on_cpu) { gpu_ac[i] /= sizeof(uint64_t); }
}

template <int WARP_SIZE, int BLOCK_WARPS, int TILE_SIZE>
__global__ void copy_buffer_ac_to_final_place(uint64_t* gpu_buffer,
                                              int* gpu_ac,
                                              uint64_t* val,
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

__global__ void get_features_kernel(GpuPsCommGraphFea graph,
                                    GpuPsFeaInfo* fea_info_array,
                                    int* actual_size,
                                    uint64_t* feature,
                                    int slot_num,
                                    int n) {
  int idx = blockIdx.x * blockDim.y + threadIdx.y;
  if (idx < n) {
    int feature_size = fea_info_array[idx].feature_size;
    int offset = idx * slot_num;
    if (feature_size == 0) {
      for (int k = 0; k < slot_num; ++k) {
        feature[offset + k] = 0;
      }
      actual_size[idx] = slot_num;
      return;
    }

    uint64_t* feature_start =
        &(graph.feature_list[fea_info_array[idx].feature_offset]);
    uint8_t* slot_id_start =
        &(graph.slot_id_list[fea_info_array[idx].feature_offset]);
    int m = 0;
    for (int k = 0; k < slot_num; ++k) {
      if (m >= fea_info_array[idx].feature_size || k < slot_id_start[m]) {
        feature[offset + k] = 0;
      } else if (k == slot_id_start[m]) {
        feature[offset + k] = feature_start[m];
        ++m;
      } else {
        assert(0);
      }
    }
    actual_size[idx] = slot_num;
  }
}

template <int WARP_SIZE, int BLOCK_WARPS, int TILE_SIZE>
__global__ void neighbor_sample_kernel(GpuPsCommGraph graph,
                                       GpuPsNodeInfo* node_info_list,
                                       int* actual_size,
                                       uint64_t* res,
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
    if (node_info_list[i].neighbor_size == 0) {
      actual_size[i] = default_value;
      i += BLOCK_WARPS;
      continue;
    }
    int neighbor_len = (int)node_info_list[i].neighbor_size;
    uint32_t data_offset = node_info_list[i].neighbor_offset;
    int offset = i * sample_len;
    uint64_t* data = graph.neighbor_list;
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

int GpuPsGraphTable::init_cpu_table(
    const paddle::distributed::GraphParameter& graph) {
  cpu_graph_table_.reset(new paddle::distributed::GraphTable);
  cpu_table_status = cpu_graph_table_->Initialize(graph);
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
  char key_buffer[len * sizeof(uint64_t)];
  char val_buffer[sample_len * sizeof(int64_t) * len +
                  (len + len % 2) * sizeof(int) + len * sizeof(uint64_t)];
  cudaMemcpy(key_buffer, key, sizeof(uint64_t) * len, cudaMemcpyDeviceToHost);
  cudaMemcpy(val_buffer,
             val,
             sample_len * sizeof(int64_t) * len +
                 (len + len % 2) * sizeof(int) + len * sizeof(uint64_t),
             cudaMemcpyDeviceToHost);
  uint64_t* sample_val =
      (uint64_t*)(val_buffer + (len + len % 2) * sizeof(int) +
                  len * sizeof(int64_t));
  for (int i = 0; i < len; i++) {
    printf("key %llu\n", *(int64_t*)(key_buffer + i * sizeof(uint64_t)));
    printf("index %llu\n", *(int64_t*)(val_buffer + i * sizeof(uint64_t)));
    int ac_size = *(int*)(val_buffer + i * sizeof(int) + len * sizeof(int64_t));
    printf("sampled %d neigbhors\n", ac_size);
    for (int j = 0; j < ac_size; j++) {
      printf("%llu ", sample_val[i * sample_len + j]);
    }
    printf("\n");
  }
}

void GpuPsGraphTable::move_result_to_source_gpu(int start_index,
                                                int gpu_num,
                                                int sample_size,
                                                int* h_left,
                                                int* h_right,
                                                uint64_t* src_sample_res,
                                                int* actual_sample_size) {
  int shard_len[gpu_num];
  for (int i = 0; i < gpu_num; i++) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    shard_len[i] = h_right[i] - h_left[i] + 1;
    int cur_step = (int)path_[start_index][i].nodes_.size() - 1;
    for (int j = cur_step; j > 0; j--) {
      CUDA_CHECK(
          cudaMemcpyAsync(path_[start_index][i].nodes_[j - 1].val_storage,
                          path_[start_index][i].nodes_[j].val_storage,
                          path_[start_index][i].nodes_[j - 1].val_bytes_len,
                          cudaMemcpyDefault,
                          path_[start_index][i].nodes_[j - 1].out_stream));
    }
    auto& node = path_[start_index][i].nodes_.front();
    CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<char*>(src_sample_res + h_left[i] * sample_size),
        node.val_storage + sizeof(int64_t) * shard_len[i] +
            sizeof(int) * (shard_len[i] + shard_len[i] % 2),
        sizeof(uint64_t) * shard_len[i] * sample_size,
        cudaMemcpyDefault,
        node.out_stream));
    CUDA_CHECK(
        cudaMemcpyAsync(reinterpret_cast<char*>(actual_sample_size + h_left[i]),
                        node.val_storage + sizeof(int64_t) * shard_len[i],
                        sizeof(int) * shard_len[i],
                        cudaMemcpyDefault,
                        node.out_stream));
  }
  for (int i = 0; i < gpu_num; ++i) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    auto& node = path_[start_index][i].nodes_.front();
    CUDA_CHECK(cudaStreamSynchronize(node.out_stream));
    // cudaStreamSynchronize(resource_->remote_stream(i, start_index));
  }
}

/*
TODO:
how to optimize it to eliminate the for loop
*/
__global__ void fill_dvalues(uint64_t* d_shard_vals,
                             uint64_t* d_vals,
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

__global__ void fill_dvalues(uint64_t* d_shard_vals,
                             uint64_t* d_vals,
                             int* d_shard_actual_sample_size,
                             int* idx,
                             int sample_size,
                             int len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    for (int j = 0; j < sample_size; j++) {
      d_vals[idx[i] * sample_size + j] = d_shard_vals[i * sample_size + j];
    }
  }
}

__global__ void fill_actual_vals(uint64_t* vals,
                                 uint64_t* actual_vals,
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
                                   uint64_t* res) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    res[i] = graph.node_list[start + i];
  }
}

void GpuPsGraphTable::clear_feature_info(int gpu_id) {
  int idx = 0;
  if (idx >= feature_table_num_) return;
  int offset = get_table_offset(gpu_id, GraphTableType::FEATURE_TABLE, idx);
  if (offset < tables_.size()) {
    delete tables_[offset];
    tables_[offset] = NULL;
  }

  int graph_fea_idx = gpu_id * feature_table_num_ + idx;
  if (graph_fea_idx >= gpu_graph_fea_list_.size()) {
    return;
  }
  auto& graph = gpu_graph_fea_list_[graph_fea_idx];
  if (graph.feature_list != NULL) {
    cudaFree(graph.feature_list);
    graph.feature_list = NULL;
  }

  if (graph.slot_id_list != NULL) {
    cudaFree(graph.slot_id_list);
    graph.slot_id_list = NULL;
  }
}

void GpuPsGraphTable::clear_graph_info(int gpu_id, int idx) {
  if (idx >= graph_table_num_) return;
  int offset = get_table_offset(gpu_id, GraphTableType::EDGE_TABLE, idx);
  if (offset < tables_.size()) {
    delete tables_[offset];
    tables_[offset] = NULL;
  }
  auto& graph = gpu_graph_list_[gpu_id * graph_table_num_ + idx];
  if (graph.neighbor_list != NULL) {
    cudaFree(graph.neighbor_list);
    graph.neighbor_list = nullptr;
  }
  if (graph.node_list != NULL) {
    cudaFree(graph.node_list);
    graph.node_list = nullptr;
  }
}
void GpuPsGraphTable::clear_graph_info(int idx) {
  for (int i = 0; i < gpu_num; i++) clear_graph_info(i, idx);
}
/*
the parameter std::vector<GpuPsCommGraph> cpu_graph_list is generated by cpu.
it saves the graph to be saved on each gpu.
for the ith GpuPsCommGraph, any the node's key satisfies that key % gpu_number
== i
In this function, memory is allocated on each gpu to save the graphs,
gpu i saves the ith graph from cpu_graph_list
*/
void GpuPsGraphTable::build_graph_fea_on_single_gpu(const GpuPsCommGraphFea& g,
                                                    int gpu_id) {
  clear_feature_info(gpu_id);
  int ntype_id = 0;

  platform::CUDADeviceGuard guard(resource_->dev_id(gpu_id));

  int offset = gpu_id * feature_table_num_ + ntype_id;
  gpu_graph_fea_list_[offset] = GpuPsCommGraphFea();

  int table_offset =
      get_table_offset(gpu_id, GraphTableType::FEATURE_TABLE, ntype_id);

  size_t capacity = std::max((uint64_t)1, g.node_size) / load_factor_;
  tables_[table_offset] = new Table(capacity);
  if (g.node_size > 0) {
    build_ps(gpu_id,
             g.node_list,
             (uint64_t*)g.fea_info_list,
             g.node_size,
             1024,
             8,
             table_offset);
    gpu_graph_fea_list_[offset].node_list = NULL;
    gpu_graph_fea_list_[offset].node_size = g.node_size;
  } else {
    build_ps(gpu_id, NULL, NULL, 0, 1024, 8, table_offset);
    gpu_graph_fea_list_[offset].node_list = NULL;
    gpu_graph_fea_list_[offset].node_size = 0;
  }
  if (g.feature_size) {
    // TODO
    cudaError_t cudaStatus =
        cudaMalloc((void**)&gpu_graph_fea_list_[offset].feature_list,
                   g.feature_size * sizeof(uint64_t));
    PADDLE_ENFORCE_EQ(
        cudaStatus,
        cudaSuccess,
        platform::errors::InvalidArgument(
            "ailed to allocate memory for graph-feature on gpu "));
    VLOG(0) << "sucessfully allocate " << g.feature_size * sizeof(uint64_t)
            << " bytes of memory for graph-feature on gpu "
            << resource_->dev_id(gpu_id);
    CUDA_CHECK(cudaMemcpy(gpu_graph_fea_list_[offset].feature_list,
                          g.feature_list,
                          g.feature_size * sizeof(uint64_t),
                          cudaMemcpyHostToDevice));

    // TODO
    cudaStatus = cudaMalloc((void**)&gpu_graph_fea_list_[offset].slot_id_list,
                            g.feature_size * sizeof(uint8_t));
    PADDLE_ENFORCE_EQ(
        cudaStatus,
        cudaSuccess,
        platform::errors::InvalidArgument(
            "ailed to allocate memory for graph-feature on gpu "));
    VLOG(0) << "sucessfully allocate " << g.feature_size * sizeof(uint8_t)
            << " bytes of memory for graph-feature on gpu "
            << resource_->dev_id(gpu_id);
    cudaMemcpy(gpu_graph_fea_list_[offset].slot_id_list,
               g.slot_id_list,
               g.feature_size * sizeof(uint8_t),
               cudaMemcpyHostToDevice);

    gpu_graph_fea_list_[offset].feature_size = g.feature_size;
  } else {
    gpu_graph_fea_list_[offset].feature_list = NULL;
    gpu_graph_fea_list_[offset].slot_id_list = NULL;
    gpu_graph_fea_list_[offset].feature_size = 0;
  }
  VLOG(0) << "gpu node_feature info card :" << gpu_id << " ,node_size is "
          << gpu_graph_fea_list_[offset].node_size << ", feature_size is "
          << gpu_graph_fea_list_[offset].feature_size;
}

/*
the parameter std::vector<GpuPsCommGraph> cpu_graph_list is generated by cpu.
it saves the graph to be saved on each gpu.
for the ith GpuPsCommGraph, any the node's key satisfies that key % gpu_number
== i
In this function, memory is allocated on each gpu to save the graphs,
gpu i saves the ith graph from cpu_graph_list
*/
void GpuPsGraphTable::build_graph_on_single_gpu(const GpuPsCommGraph& g,
                                                int i,
                                                int idx) {
  clear_graph_info(i, idx);
  platform::CUDADeviceGuard guard(resource_->dev_id(i));
  int offset = i * graph_table_num_ + idx;
  gpu_graph_list_[offset] = GpuPsCommGraph();
  int table_offset = get_table_offset(i, GraphTableType::EDGE_TABLE, idx);
  size_t capacity = std::max((uint64_t)1, (uint64_t)g.node_size) / load_factor_;
  tables_[table_offset] = new Table(capacity);
  if (g.node_size > 0) {
    if (FLAGS_gpugraph_load_node_list_into_hbm) {
      CUDA_CHECK(cudaMalloc((void**)&gpu_graph_list_[offset].node_list,
                            g.node_size * sizeof(uint64_t)));
      CUDA_CHECK(cudaMemcpy(gpu_graph_list_[offset].node_list,
                            g.node_list,
                            g.node_size * sizeof(uint64_t),
                            cudaMemcpyHostToDevice));
    }

    build_ps(i,
             g.node_list,
             (uint64_t*)(g.node_info_list),
             g.node_size,
             1024,
             8,
             table_offset);
    gpu_graph_list_[offset].node_size = g.node_size;
  } else {
    build_ps(i, NULL, NULL, 0, 1024, 8, table_offset);
    gpu_graph_list_[offset].node_list = NULL;
    gpu_graph_list_[offset].node_size = 0;
  }
  if (g.neighbor_size) {
    cudaError_t cudaStatus =
        cudaMalloc((void**)&gpu_graph_list_[offset].neighbor_list,
                   g.neighbor_size * sizeof(uint64_t));
    PADDLE_ENFORCE_EQ(cudaStatus,
                      cudaSuccess,
                      platform::errors::InvalidArgument(
                          "ailed to allocate memory for graph on gpu "));
    VLOG(0) << "sucessfully allocate " << g.neighbor_size * sizeof(uint64_t)
            << " bytes of memory for graph-edges on gpu "
            << resource_->dev_id(i);
    CUDA_CHECK(cudaMemcpy(gpu_graph_list_[offset].neighbor_list,
                          g.neighbor_list,
                          g.neighbor_size * sizeof(uint64_t),
                          cudaMemcpyHostToDevice));
    gpu_graph_list_[offset].neighbor_size = g.neighbor_size;
  } else {
    gpu_graph_list_[offset].neighbor_list = NULL;
    gpu_graph_list_[offset].neighbor_size = 0;
  }
  VLOG(0) << " gpu node_neighbor info card: " << i << " ,node_size is "
          << gpu_graph_list_[offset].node_size << ", neighbor_size is "
          << gpu_graph_list_[offset].neighbor_size;
}

void GpuPsGraphTable::build_graph_fea_from_cpu(
    const std::vector<GpuPsCommGraphFea>& cpu_graph_fea_list, int ntype_id) {
  PADDLE_ENFORCE_EQ(
      cpu_graph_fea_list.size(),
      resource_->total_device(),
      platform::errors::InvalidArgument("the cpu node list size doesn't match "
                                        "the number of gpu on your machine."));
  clear_feature_info(ntype_id);
  for (int i = 0; i < cpu_graph_fea_list.size(); i++) {
    int table_offset =
        get_table_offset(i, GraphTableType::FEATURE_TABLE, ntype_id);
    int offset = i * feature_table_num_ + ntype_id;
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    gpu_graph_fea_list_[offset] = GpuPsCommGraphFea();
    tables_[table_offset] = new Table(
        std::max((uint64_t)1, (uint64_t)cpu_graph_fea_list[i].node_size) /
        load_factor_);
    if (cpu_graph_fea_list[i].node_size > 0) {
      build_ps(i,
               cpu_graph_fea_list[i].node_list,
               (uint64_t*)cpu_graph_fea_list[i].fea_info_list,
               cpu_graph_fea_list[i].node_size,
               1024,
               8,
               table_offset);
      gpu_graph_fea_list_[offset].node_size = cpu_graph_fea_list[i].node_size;
    } else {
      build_ps(i, NULL, NULL, 0, 1024, 8, table_offset);
      gpu_graph_fea_list_[offset].node_list = NULL;
      gpu_graph_fea_list_[offset].node_size = 0;
    }
    if (cpu_graph_fea_list[i].feature_size) {
      // TODO
      CUDA_CHECK(
          cudaMalloc((void**)&gpu_graph_fea_list_[offset].feature_list,
                     cpu_graph_fea_list[i].feature_size * sizeof(uint64_t)));

      CUDA_CHECK(
          cudaMemcpy(gpu_graph_fea_list_[offset].feature_list,
                     cpu_graph_fea_list[i].feature_list,
                     cpu_graph_fea_list[i].feature_size * sizeof(uint64_t),
                     cudaMemcpyHostToDevice));

      // TODO
      CUDA_CHECK(
          cudaMalloc((void**)&gpu_graph_fea_list_[offset].slot_id_list,
                     cpu_graph_fea_list[i].feature_size * sizeof(uint8_t)));

      CUDA_CHECK(
          cudaMemcpy(gpu_graph_fea_list_[offset].slot_id_list,
                     cpu_graph_fea_list[i].slot_id_list,
                     cpu_graph_fea_list[i].feature_size * sizeof(uint8_t),
                     cudaMemcpyHostToDevice));

      gpu_graph_fea_list_[offset].feature_size =
          cpu_graph_fea_list[i].feature_size;
    } else {
      gpu_graph_fea_list_[offset].feature_list = NULL;
      gpu_graph_fea_list_[offset].slot_id_list = NULL;
      gpu_graph_fea_list_[offset].feature_size = 0;
    }
  }
  cudaDeviceSynchronize();
}

void GpuPsGraphTable::build_graph_from_cpu(
    const std::vector<GpuPsCommGraph>& cpu_graph_list, int idx) {
  VLOG(0) << "in build_graph_from_cpu cpu_graph_list size = "
          << cpu_graph_list.size();
  PADDLE_ENFORCE_EQ(
      cpu_graph_list.size(),
      resource_->total_device(),
      platform::errors::InvalidArgument("the cpu node list size doesn't match "
                                        "the number of gpu on your machine."));
  clear_graph_info(idx);
  for (int i = 0; i < cpu_graph_list.size(); i++) {
    int table_offset = get_table_offset(i, GraphTableType::EDGE_TABLE, idx);
    int offset = i * graph_table_num_ + idx;
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    gpu_graph_list_[offset] = GpuPsCommGraph();
    tables_[table_offset] =
        new Table(std::max((uint64_t)1, (uint64_t)cpu_graph_list[i].node_size) /
                  load_factor_);
    if (cpu_graph_list[i].node_size > 0) {
      CUDA_CHECK(cudaMalloc((void**)&gpu_graph_list_[offset].node_list,
                            cpu_graph_list[i].node_size * sizeof(uint64_t)));
      CUDA_CHECK(cudaMemcpy(gpu_graph_list_[offset].node_list,
                            cpu_graph_list[i].node_list,
                            cpu_graph_list[i].node_size * sizeof(uint64_t),
                            cudaMemcpyHostToDevice));
      build_ps(i,
               cpu_graph_list[i].node_list,
               (uint64_t*)(cpu_graph_list[i].node_info_list),
               cpu_graph_list[i].node_size,
               1024,
               8,
               table_offset);
      gpu_graph_list_[offset].node_size = cpu_graph_list[i].node_size;
    } else {
      build_ps(i, NULL, NULL, 0, 1024, 8, table_offset);
      gpu_graph_list_[offset].node_list = NULL;
      gpu_graph_list_[offset].node_size = 0;
    }
    if (cpu_graph_list[i].neighbor_size) {
      CUDA_CHECK(
          cudaMalloc((void**)&gpu_graph_list_[offset].neighbor_list,
                     cpu_graph_list[i].neighbor_size * sizeof(uint64_t)));

      CUDA_CHECK(cudaMemcpy(gpu_graph_list_[offset].neighbor_list,
                            cpu_graph_list[i].neighbor_list,
                            cpu_graph_list[i].neighbor_size * sizeof(uint64_t),
                            cudaMemcpyHostToDevice));
      gpu_graph_list_[offset].neighbor_size = cpu_graph_list[i].neighbor_size;
    } else {
      gpu_graph_list_[offset].neighbor_list = NULL;
      gpu_graph_list_[offset].neighbor_size = 0;
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());
}

NeighborSampleResult GpuPsGraphTable::graph_neighbor_sample_v3(
    NeighborSampleQuery q, bool cpu_switch) {
  return graph_neighbor_sample_v2(global_device_map[q.gpu_id],
                                  q.table_idx,
                                  q.src_nodes,
                                  q.sample_size,
                                  q.len,
                                  cpu_switch);
}

NeighborSampleResult GpuPsGraphTable::graph_neighbor_sample(int gpu_id,
                                                            uint64_t* key,
                                                            int sample_size,
                                                            int len) {
  return graph_neighbor_sample_v2(gpu_id, 0, key, sample_size, len, false);
}

NeighborSampleResult GpuPsGraphTable::graph_neighbor_sample_v2(
    int gpu_id,
    int idx,
    uint64_t* key,
    int sample_size,
    int len,
    bool cpu_query_switch) {
  NeighborSampleResult result;
  result.initialize(sample_size, len, resource_->dev_id(gpu_id));

  if (len == 0) {
    return result;
  }

  platform::CUDAPlace place = platform::CUDAPlace(resource_->dev_id(gpu_id));
  platform::CUDADeviceGuard guard(resource_->dev_id(gpu_id));

  int* actual_sample_size = result.actual_sample_size;
  uint64_t* val = result.val;
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

  CUDA_CHECK(cudaMemsetAsync(d_left_ptr, -1, total_gpu * sizeof(int), stream));
  CUDA_CHECK(cudaMemsetAsync(d_right_ptr, -1, total_gpu * sizeof(int), stream));
  //
  auto d_idx = memory::Alloc(place, len * sizeof(int));
  int* d_idx_ptr = reinterpret_cast<int*>(d_idx->ptr());

  auto d_shard_keys = memory::Alloc(place, len * sizeof(uint64_t));
  uint64_t* d_shard_keys_ptr = reinterpret_cast<uint64_t*>(d_shard_keys->ptr());
  auto d_shard_vals =
      memory::Alloc(place, sample_size * len * sizeof(uint64_t));
  uint64_t* d_shard_vals_ptr = reinterpret_cast<uint64_t*>(d_shard_vals->ptr());
  auto d_shard_actual_sample_size = memory::Alloc(place, len * sizeof(int));
  int* d_shard_actual_sample_size_ptr =
      reinterpret_cast<int*>(d_shard_actual_sample_size->ptr());

  split_input_to_shard(
      (uint64_t*)(key), d_idx_ptr, len, d_left_ptr, d_right_ptr, gpu_id);

  heter_comm_kernel_->fill_shard_key(
      d_shard_keys_ptr, key, d_idx_ptr, len, stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  CUDA_CHECK(cudaMemcpy(
      h_left, d_left_ptr, total_gpu * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(
      h_right, d_right_ptr, total_gpu * sizeof(int), cudaMemcpyDeviceToHost));
  for (int i = 0; i < total_gpu; ++i) {
    int shard_len = h_left[i] == -1 ? 0 : h_right[i] - h_left[i] + 1;
    if (shard_len == 0) {
      continue;
    }
    create_storage(gpu_id,
                   i,
                   shard_len * sizeof(uint64_t),
                   shard_len * sample_size * sizeof(uint64_t) +
                       shard_len * sizeof(uint64_t) +
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

    CUDA_CHECK(cudaMemsetAsync(
        node.val_storage, 0, shard_len * sizeof(int64_t), node.in_stream));
    CUDA_CHECK(cudaStreamSynchronize(node.in_stream));
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    // If not found, val is -1.
    int table_offset = get_table_offset(i, GraphTableType::EDGE_TABLE, idx);
    int offset = i * graph_table_num_ + idx;
    tables_[table_offset]->get(reinterpret_cast<uint64_t*>(node.key_storage),
                               reinterpret_cast<uint64_t*>(node.val_storage),
                               (size_t)(h_right[i] - h_left[i] + 1),
                               resource_->remote_stream(i, gpu_id));

    auto graph = gpu_graph_list_[offset];
    GpuPsNodeInfo* node_info_list =
        reinterpret_cast<GpuPsNodeInfo*>(node.val_storage);
    int* actual_size_array = (int*)(node_info_list + shard_len);
    uint64_t* sample_array =
        (uint64_t*)(actual_size_array + shard_len + shard_len % 2);
    constexpr int WARP_SIZE = 32;
    constexpr int BLOCK_WARPS = 128 / WARP_SIZE;
    constexpr int TILE_SIZE = BLOCK_WARPS * 16;
    const dim3 block(WARP_SIZE, BLOCK_WARPS);
    const dim3 grid((shard_len + TILE_SIZE - 1) / TILE_SIZE);

    neighbor_sample_kernel<WARP_SIZE, BLOCK_WARPS, TILE_SIZE>
        <<<grid, block, 0, resource_->remote_stream(i, gpu_id)>>>(
            graph,
            node_info_list,
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
    CUDA_CHECK(cudaStreamSynchronize(resource_->remote_stream(i, gpu_id)));
  }
  move_result_to_source_gpu(gpu_id,
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

  CUDA_CHECK(cudaStreamSynchronize(stream));

  if (cpu_query_switch) {
    // Get cpu keys and corresponding position.
    thrust::device_vector<uint64_t> t_cpu_keys(len);
    thrust::device_vector<int> t_index(len + 1, 0);
    get_cpu_id_index<<<grid_size, block_size_, 0, stream>>>(
        key,
        actual_sample_size,
        thrust::raw_pointer_cast(t_cpu_keys.data()),
        thrust::raw_pointer_cast(t_index.data()),
        thrust::raw_pointer_cast(t_index.data()) + 1,
        len);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    int number_on_cpu = 0;
    CUDA_CHECK(cudaMemcpy(&number_on_cpu,
                          thrust::raw_pointer_cast(t_index.data()),
                          sizeof(int),
                          cudaMemcpyDeviceToHost));
    if (number_on_cpu > 0) {
      uint64_t* cpu_keys = new uint64_t[number_on_cpu];
      CUDA_CHECK(cudaMemcpy(cpu_keys,
                            thrust::raw_pointer_cast(t_cpu_keys.data()),
                            number_on_cpu * sizeof(uint64_t),
                            cudaMemcpyDeviceToHost));

      std::vector<std::shared_ptr<char>> buffers(number_on_cpu);
      std::vector<int> ac(number_on_cpu);

      auto status = cpu_graph_table_->random_sample_neighbors(
          idx, cpu_keys, sample_size, buffers, ac, false);

      int total_cpu_sample_size = std::accumulate(ac.begin(), ac.end(), 0);
      total_cpu_sample_size /= sizeof(uint64_t);

      // Merge buffers into one uint64_t vector.
      uint64_t* merge_buffers = new uint64_t[total_cpu_sample_size];
      int start = 0;
      for (int j = 0; j < number_on_cpu; j++) {
        memcpy(merge_buffers + start, (uint64_t*)(buffers[j].get()), ac[j]);
        start += ac[j] / sizeof(uint64_t);
      }

      // Copy merge_buffers to gpu.
      thrust::device_vector<uint64_t> gpu_buffers(total_cpu_sample_size);
      thrust::device_vector<int> gpu_ac(number_on_cpu);
      uint64_t* gpu_buffers_ptr = thrust::raw_pointer_cast(gpu_buffers.data());
      int* gpu_ac_ptr = thrust::raw_pointer_cast(gpu_ac.data());
      CUDA_CHECK(cudaMemcpyAsync(gpu_buffers_ptr,
                                 merge_buffers,
                                 total_cpu_sample_size * sizeof(uint64_t),
                                 cudaMemcpyHostToDevice,
                                 stream));
      CUDA_CHECK(cudaMemcpyAsync(gpu_ac_ptr,
                                 ac.data(),
                                 number_on_cpu * sizeof(int),
                                 cudaMemcpyHostToDevice,
                                 stream));

      // Copy gpu_buffers and gpu_ac using kernel.
      // Kernel divide for gpu_ac_ptr.
      int grid_size2 = (number_on_cpu - 1) / block_size_ + 1;
      get_actual_gpu_ac<<<grid_size2, block_size_, 0, stream>>>(gpu_ac_ptr,
                                                                number_on_cpu);

      CUDA_CHECK(cudaStreamSynchronize(stream));

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
    CUDA_CHECK(cudaStreamSynchronize(stream));
    platform::CUDAPlace place = platform::CUDAPlace(resource_->dev_id(gpu_id));
    platform::CUDADeviceGuard guard(resource_->dev_id(gpu_id));

    thrust::device_vector<int> t_actual_sample_size(len);
    thrust::copy(actual_sample_size,
                 actual_sample_size + len,
                 t_actual_sample_size.begin());
    int total_sample_size = thrust::reduce(t_actual_sample_size.begin(),
                                           t_actual_sample_size.end());

    result.actual_val_mem =
        memory::AllocShared(place, total_sample_size * sizeof(uint64_t));
    result.actual_val = (uint64_t*)(result.actual_val_mem)->ptr();

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
                                                 int idx,
                                                 int start,
                                                 int query_size) {
  NodeQueryResult result;
  result.actual_sample_size = 0;
  if (query_size <= 0) return result;
  std::vector<int> gpu_begin_pos, local_begin_pos;
  std::function<int(int, int, int, int, int&, int&)> range_check =
      [](int x, int y, int x1, int y1, int& x2, int& y2) {
        if (y <= x1 || x >= y1) return 0;
        y2 = min(y, y1);
        x2 = max(x1, x);
        return y2 - x2;
      };

  int offset = gpu_id * graph_table_num_ + idx;
  const auto& graph = gpu_graph_list_[offset];
  if (graph.node_size == 0) {
    return result;
  }
  int x2, y2;
  int len = range_check(start, start + query_size, 0, graph.node_size, x2, y2);

  if (len == 0) {
    return result;
  }

  result.initialize(len, resource_->dev_id(gpu_id));
  result.actual_sample_size = len;
  uint64_t* val = result.val;

  int dev_id_i = resource_->dev_id(gpu_id);
  platform::CUDADeviceGuard guard(dev_id_i);
  int grid_size = (len - 1) / block_size_ + 1;
  node_query_example<<<grid_size,
                       block_size_,
                       0,
                       resource_->remote_stream(gpu_id, gpu_id)>>>(
      graph, x2, len, (uint64_t*)val);
  CUDA_CHECK(cudaStreamSynchronize(resource_->remote_stream(gpu_id, gpu_id)));
  return result;
}

int GpuPsGraphTable::get_feature_of_nodes(int gpu_id,
                                          uint64_t* d_nodes,
                                          uint64_t* d_feature,
                                          int node_num,
                                          int slot_num) {
  if (node_num == 0) {
    return -1;
  }

  platform::CUDAPlace place = platform::CUDAPlace(resource_->dev_id(gpu_id));
  platform::CUDADeviceGuard guard(resource_->dev_id(gpu_id));
  int total_gpu = resource_->total_device();
  auto stream = resource_->local_stream(gpu_id, 0);

  auto d_left = memory::Alloc(place, total_gpu * sizeof(int));
  auto d_right = memory::Alloc(place, total_gpu * sizeof(int));
  int* d_left_ptr = reinterpret_cast<int*>(d_left->ptr());
  int* d_right_ptr = reinterpret_cast<int*>(d_right->ptr());

  CUDA_CHECK(cudaMemsetAsync(d_left_ptr, -1, total_gpu * sizeof(int), stream));
  CUDA_CHECK(cudaMemsetAsync(d_right_ptr, -1, total_gpu * sizeof(int), stream));
  //
  auto d_idx = memory::Alloc(place, node_num * sizeof(int));
  int* d_idx_ptr = reinterpret_cast<int*>(d_idx->ptr());

  auto d_shard_keys = memory::Alloc(place, node_num * sizeof(uint64_t));
  uint64_t* d_shard_keys_ptr = reinterpret_cast<uint64_t*>(d_shard_keys->ptr());
  auto d_shard_vals =
      memory::Alloc(place, slot_num * node_num * sizeof(uint64_t));
  uint64_t* d_shard_vals_ptr = reinterpret_cast<uint64_t*>(d_shard_vals->ptr());
  auto d_shard_actual_size = memory::Alloc(place, node_num * sizeof(int));
  int* d_shard_actual_size_ptr =
      reinterpret_cast<int*>(d_shard_actual_size->ptr());

  split_input_to_shard(
      d_nodes, d_idx_ptr, node_num, d_left_ptr, d_right_ptr, gpu_id);

  heter_comm_kernel_->fill_shard_key(
      d_shard_keys_ptr, d_nodes, d_idx_ptr, node_num, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  int h_left[total_gpu];  // NOLINT
  CUDA_CHECK(cudaMemcpy(
      h_left, d_left_ptr, total_gpu * sizeof(int), cudaMemcpyDeviceToHost));
  int h_right[total_gpu];  // NOLINT
  CUDA_CHECK(cudaMemcpy(
      h_right, d_right_ptr, total_gpu * sizeof(int), cudaMemcpyDeviceToHost));
  for (int i = 0; i < total_gpu; ++i) {
    int shard_len = h_left[i] == -1 ? 0 : h_right[i] - h_left[i] + 1;
    if (shard_len == 0) {
      continue;
    }
    create_storage(gpu_id,
                   i,
                   shard_len * sizeof(uint64_t),
                   shard_len * slot_num * sizeof(uint64_t) +
                       shard_len * sizeof(uint64_t) +
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

    CUDA_CHECK(cudaMemsetAsync(
        node.val_storage, 0, shard_len * sizeof(uint64_t), node.in_stream));
    CUDA_CHECK(cudaStreamSynchronize(node.in_stream));
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    // If not found, val is -1.
    int table_offset = get_table_offset(i, GraphTableType::FEATURE_TABLE, 0);
    tables_[table_offset]->get(reinterpret_cast<uint64_t*>(node.key_storage),
                               reinterpret_cast<uint64_t*>(node.val_storage),
                               (size_t)(h_right[i] - h_left[i] + 1),
                               resource_->remote_stream(i, gpu_id));

    int offset = i * feature_table_num_;
    auto graph = gpu_graph_fea_list_[offset];

    GpuPsFeaInfo* val_array = reinterpret_cast<GpuPsFeaInfo*>(node.val_storage);
    int* actual_size_array = (int*)(val_array + shard_len);
    uint64_t* feature_array =
        (uint64_t*)(actual_size_array + shard_len + shard_len % 2);
    dim3 grid((shard_len - 1) / dim_y + 1);
    dim3 block(1, dim_y);
    get_features_kernel<<<grid,
                          block,
                          0,
                          resource_->remote_stream(i, gpu_id)>>>(
        graph,
        val_array,
        actual_size_array,
        feature_array,
        slot_num,
        shard_len);
  }

  for (int i = 0; i < total_gpu; ++i) {
    if (h_left[i] == -1) {
      continue;
    }
    CUDA_CHECK(cudaStreamSynchronize(resource_->remote_stream(i, gpu_id)));
  }

  move_result_to_source_gpu(gpu_id,
                            total_gpu,
                            slot_num,
                            h_left,
                            h_right,
                            d_shard_vals_ptr,
                            d_shard_actual_size_ptr);

  int grid_size = (node_num - 1) / block_size_ + 1;
  fill_dvalues<<<grid_size, block_size_, 0, stream>>>(d_shard_vals_ptr,
                                                      d_feature,
                                                      d_shard_actual_size_ptr,
                                                      d_idx_ptr,
                                                      slot_num,
                                                      node_num);

  for (int i = 0; i < total_gpu; ++i) {
    int shard_len = h_left[i] == -1 ? 0 : h_right[i] - h_left[i] + 1;
    if (shard_len == 0) {
      continue;
    }
    destroy_storage(gpu_id, i);
  }

  CUDA_CHECK(cudaStreamSynchronize(stream));

  return 0;
}
}  // namespace framework
};  // namespace paddle
#endif
