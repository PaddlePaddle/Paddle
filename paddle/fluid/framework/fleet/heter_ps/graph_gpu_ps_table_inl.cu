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

#include <math.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <functional>
#include "cub/cub.cuh"
#pragma once
#ifdef PADDLE_WITH_HETERPS
#include "cudf/block_radix_topk.cuh"
#include "cudf/random.cuh"
#include "paddle/common/flags.h"
#include "paddle/fluid/framework/fleet/heter_ps/gpu_graph_utils.h"
#include "paddle/fluid/framework/fleet/heter_ps/graph_gpu_ps_table.h"

#define ALIGN_INT64(LEN) (uint64_t((LEN) + 7) & uint64_t(~7))
#define HBMPS_MAX_BUFF 1024 * 1024
#define SAMPLE_SIZE_THRESHOLD 1024

COMMON_DECLARE_bool(enable_neighbor_list_use_uva);
COMMON_DECLARE_bool(enable_graph_multi_node_sampling);

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

__global__ void get_features_size(GpuPsFeaInfo* fea_info_array,
                                  uint32_t* feature_size,
                                  int n) {
  int idx = blockIdx.x * blockDim.y + threadIdx.y;
  if (idx < n) {
    feature_size[idx] = fea_info_array[idx].feature_size;
  }
}

__global__ void get_features_kernel(GpuPsCommGraphFea graph,
                                    GpuPsFeaInfo* fea_info_array,
                                    uint32_t* fea_size_prefix_sum,
                                    uint64_t* feature_array,
                                    uint8_t* slot_array,
                                    int n) {
  int idx = blockIdx.x * blockDim.y + threadIdx.y;
  if (idx < n) {
    uint32_t feature_size = fea_info_array[idx].feature_size;
    if (feature_size == 0) {
      return;
    }
    uint32_t src_offset = fea_info_array[idx].feature_offset;
    uint32_t dst_offset = fea_size_prefix_sum[idx];
    for (uint32_t j = 0; j < feature_size; ++j) {
      feature_array[dst_offset + j] = graph.feature_list[src_offset + j];
      slot_array[dst_offset + j] = graph.slot_id_list[src_offset + j];
    }
  }
}

__global__ void get_float_features_kernel(GpuPsCommGraphFloatFea graph,
                                          GpuPsFeaInfo* fea_info_array,
                                          uint32_t* fea_size_prefix_sum,
                                          float* feature_array,
                                          uint8_t* slot_array,
                                          int n) {
  int idx = blockIdx.x * blockDim.y + threadIdx.y;
  if (idx < n) {
    uint32_t feature_size = fea_info_array[idx].feature_size;
    if (feature_size == 0) {
      return;
    }
    uint32_t src_offset = fea_info_array[idx].feature_offset;
    uint32_t dst_offset = fea_size_prefix_sum[idx];
    for (uint32_t j = 0; j < feature_size; ++j) {
      feature_array[dst_offset + j] = graph.feature_list[src_offset + j];
      slot_array[dst_offset + j] = graph.slot_id_list[src_offset + j];
    }
  }
}

__global__ void get_features_kernel(GpuPsCommGraphFea graph,
                                    GpuPsFeaInfo* fea_info_array,
                                    int* actual_size,
                                    uint64_t* feature,
                                    int* slot_feature_num_map,
                                    int slot_num,
                                    int n,
                                    int fea_num_per_node) {
  int idx = blockIdx.x * blockDim.y + threadIdx.y;
  if (idx < n) {
    int feature_size = fea_info_array[idx].feature_size;
    int src_offset = fea_info_array[idx].feature_offset;
    int dst_offset = idx * fea_num_per_node;
    uint64_t* dst_feature = &feature[dst_offset];
    if (feature_size == 0) {
      for (int k = 0; k < fea_num_per_node; ++k) {
        dst_feature[k] = 0;
      }
      actual_size[idx] = fea_num_per_node;
      return;
    }

    uint64_t* feature_start = &(graph.feature_list[src_offset]);
    uint8_t* slot_id_start = &(graph.slot_id_list[src_offset]);
    for (int slot_id = 0, dst_fea_idx = 0, src_fea_idx = 0; slot_id < slot_num;
         slot_id++) {
      int feature_num = slot_feature_num_map[slot_id];
      if (src_fea_idx >= feature_size || slot_id < slot_id_start[src_fea_idx]) {
        for (int j = 0; j < feature_num; ++j, ++dst_fea_idx) {
          dst_feature[dst_fea_idx] = 0;
        }
      } else if (slot_id == slot_id_start[src_fea_idx]) {
        for (int j = 0; j < feature_num; ++j, ++dst_fea_idx) {
          if (slot_id == slot_id_start[src_fea_idx]) {
            dst_feature[dst_fea_idx] = feature_start[src_fea_idx++];
          } else {
            dst_feature[dst_fea_idx] = 0;
          }
        }
      } else {
        assert(0);
      }
    }
    actual_size[idx] = fea_num_per_node;
  }
}

__global__ void get_node_degree_kernel(GpuPsNodeInfo* node_info_list,
                                       int* node_degree,
                                       int n) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    node_degree[i] = node_info_list[i].neighbor_size;
  }
}

template <int WARP_SIZE, int BLOCK_WARPS, int TILE_SIZE>
__global__ void neighbor_sample_kernel_walking(GpuPsCommGraph graph,
                                               GpuPsNodeInfo* node_info_list,
                                               int* actual_size,
                                               uint64_t* res,
                                               int sample_len,
                                               int n,
                                               int neighbor_size_limit,
                                               int default_value) {
  // graph: The corresponding edge table.
  // node_info_list: The input node query, duplicate nodes allowed.
  // actual_size: The actual sample size of the input nodes.
  // res: The output sample neighbors of the input nodes.
  // sample_len: The fix sample size.
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
    int neighbor_len = node_info_list[i].neighbor_size;
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
      int neighbor_num;
      if (neighbor_len > neighbor_size_limit) {
        neighbor_num = neighbor_size_limit;
      } else {
        neighbor_num = neighbor_len;
      }
      for (int j = sample_len + threadIdx.x; j < neighbor_num; j += WARP_SIZE) {
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

/*__global__ void neighbor_sample_kernel_all_edge_type(
    GpuPsCommGraph* graphs,
    GpuPsNodeInfo* node_info_base,
    int* actual_size_base,
    uint64_t* sample_array_base,
    float* weight_array_base,
    int sample_len,
    int n,  // edge_type * shard_len
    int default_value,
    int shard_len,
    bool return_weight) {
  // graph: All edge tables.
  // node_info_list: The input node query, must be unique, otherwise the
  // randomness gets worse. actual_size_base: The begin position of actual
  // sample size of the input nodes. sample_array_base: The begin position of
  // sample neighbors of the input nodes. sample_len: The fix sample size.
  curandState rng;
  curand_init(blockIdx.x, threadIdx.x, 0, &rng);
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    int edge_idx = i / shard_len, node_i = i % shard_len;

    GpuPsNodeInfo* node_info_list = node_info_base + edge_idx * shard_len;
    int* actual_size_array = actual_size_base + edge_idx * shard_len;

    if (node_info_list[node_i].neighbor_size == 0) {
      actual_size_array[node_i] = default_value;
    } else {
      uint64_t* sample_array =
          sample_array_base + edge_idx * shard_len * sample_len;
      float* weight_array = nullptr;
      if (return_weight) {
        weight_array = weight_array_base + edge_idx * shard_len * sample_len;
      }
      int neighbor_len = node_info_list[node_i].neighbor_size;
      uint32_t data_offset = node_info_list[node_i].neighbor_offset;
      int offset = node_i * sample_len;
      uint64_t* data = graphs[edge_idx].neighbor_list;
      half* weight = graphs[edge_idx].weight_list;
      uint64_t tmp;
      int split, begin;
      if (neighbor_len <= sample_len) {
        actual_size_array[node_i] = neighbor_len;
        for (int j = 0; j < neighbor_len; j++) {
          sample_array[offset + j] = data[data_offset + j];
          if (return_weight) {
            weight_array[offset + j] = (float)weight[data_offset + j];
          }
        }
      } else {
        actual_size_array[node_i] = sample_len;
        if (neighbor_len < 2 * sample_len) {
          split = sample_len;
          begin = 0;
        } else {
          split = neighbor_len - sample_len;
          begin = neighbor_len - sample_len;
        }
        // there is a bug need fix
        //for (int idx = split; idx <= neighbor_len - 1; idx++) {
        //  const int num = curand(&rng) % (idx + 1);
        //  data[data_offset + idx] = atomicExch(
        //      reinterpret_cast<unsigned long long int*>(data +  // NOLINT
        //                                              data_offset + num),
        //    static_cast<unsigned long long int>(  // NOLINT
        //    data[data_offset + idx]));
        //  if (return_weight) {
        //    weight[data_offset + idx] = atomicExch(
        //    weight + data_offset + num, weight[data_offset + idx]);
        //  }
        //}
        for (int idx = 0; idx < sample_len; idx++) {
          sample_array[offset + idx] = data[data_offset + begin + idx];
          if (return_weight) {
            weight_array[offset + idx] = (float)weight[data_offset + begin +
idx];
          }
        }
      }
    }
  }
}*/

// For Weighted Sample

template <bool NeedNeighbor = false>
__global__ void get_actual_size_and_neighbor_count(
    GpuPsNodeInfo* node_info_list,
    int* actual_size,
    int* neighbor_count,
    int sample_len,
    int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n) return;
  int neighbor_len = node_info_list[i].neighbor_size;
  int k = neighbor_len;
  if (sample_len > 0) {
    k = min(neighbor_len, sample_len);
  }
  actual_size[i] = k;
  if (NeedNeighbor) {
    neighbor_count[i] = (neighbor_len <= sample_len) ? 0 : neighbor_len;
  }
}

__device__ __forceinline__ float gen_key_from_weight(
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

template <unsigned int BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE) __global__
    void weighted_sample_large_kernel(GpuPsCommGraph graph,
                                      GpuPsNodeInfo* node_info_list,
                                      uint64_t* res,
                                      const int* target_neighbor_offset,
                                      float* weight_keys_buff,
                                      int n,
                                      int sample_len,
                                      uint64_t random_seed,
                                      float* weight_array,
                                      bool return_weight) {
  int i = blockIdx.x;
  if (i >= n) return;
  int gidx = threadIdx.x + blockIdx.x * BLOCK_SIZE;
  int neighbor_len = node_info_list[i].neighbor_size;
  uint32_t data_offset = node_info_list[i].neighbor_offset;
  int offset = i * sample_len;
  uint64_t* data = graph.neighbor_list;
  half* weight = graph.weight_list;
  float* weight_keys_local_buff = weight_keys_buff + target_neighbor_offset[i];
  if (neighbor_len <= sample_len) {  // directly copy
    for (int j = threadIdx.x; j < neighbor_len; j += BLOCK_SIZE) {
      res[offset + j] = data[data_offset + j];
      if (return_weight) {
        weight_array[offset + j] = static_cast<float>(weight[data_offset + j]);
      }
    }
  } else {
    RandomNumGen rng(gidx, random_seed);  // get weight threshold
    for (int j = threadIdx.x; j < neighbor_len; j += BLOCK_SIZE) {
      float thread_weight = static_cast<float>(weight[data_offset + j]);
      weight_keys_local_buff[j] =
          static_cast<float>(gen_key_from_weight(thread_weight, rng));
    }
    __syncthreads();

    float topk_val;
    bool topk_is_unique;

    using BlockRadixSelectT =
        BlockRadixTopKGlobalMemory<float, BLOCK_SIZE, true>;
    __shared__ typename BlockRadixSelectT::TempStorage share_storage;

    BlockRadixSelectT{share_storage}.radixTopKGetThreshold(
        weight_keys_local_buff,
        sample_len,
        neighbor_len,
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
      for (int j = threadIdx.x; j < neighbor_len; j += BLOCK_SIZE) {
        float key = weight_keys_local_buff[j];
        bool has_topk = (key >= topk_val);  // diff 1
        if (has_topk) {
          int write_index = atomicAdd(&cnt, 1);
          res[offset + write_index] = data[data_offset + j];
          if (return_weight) {
            weight_array[offset + write_index] =
                static_cast<float>(weight[data_offset + j]);
          }
        }
      }
    } else {
      for (int j = threadIdx.x; j < neighbor_len; j += BLOCK_SIZE) {
        float key = weight_keys_local_buff[j];
        bool has_topk = (key > topk_val);  // diff 1
        if (has_topk) {
          int write_index = atomicAdd(&cnt, 1);
          res[offset + write_index] = data[data_offset + j];
          if (return_weight) {
            weight_array[offset + write_index] =
                static_cast<float>(weight[data_offset + j]);
          }
        }
      }
      __syncthreads();

      for (int j = threadIdx.x; j < neighbor_len; j += BLOCK_SIZE) {
        float key = weight_keys_local_buff[j];
        bool has_topk = (key == topk_val);
        if (has_topk) {
          int write_index = atomicAdd(&cnt, 1);
          if (write_index >= sample_len) {
            break;
          }
          res[offset + write_index] = data[data_offset + j];
          if (return_weight) {
            weight_array[offset + write_index] =
                static_cast<float>(weight[data_offset + j]);
          }
        }
      }
    }
  }
}

// A-RES algorithm
template <unsigned int ITEMS_PER_THREAD, unsigned int BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE) __global__
    void weighted_sample_kernel(GpuPsCommGraph graph,
                                GpuPsNodeInfo* node_info_list,
                                uint64_t* res,
                                int n,
                                int sample_len,
                                uint64_t random_seed,
                                float* weight_array,
                                bool return_weight) {
  int i = blockIdx.x;
  if (i >= n) return;
  int gidx = threadIdx.x + blockIdx.x * BLOCK_SIZE;
  int neighbor_len = node_info_list[i].neighbor_size;
  uint32_t data_offset = node_info_list[i].neighbor_offset;
  int offset = i * sample_len;
  uint64_t* data = graph.neighbor_list;
  half* weight = graph.weight_list;

  if (neighbor_len <= sample_len) {
    for (int j = threadIdx.x; j < neighbor_len; j += BLOCK_SIZE) {
      res[offset + j] = data[data_offset + j];
      if (return_weight) {
        weight_array[offset + j] = static_cast<float>(weight[data_offset + j]);
      }
    }
  } else {
    RandomNumGen rng(gidx, random_seed);
    float weight_keys[ITEMS_PER_THREAD];
    int neighbor_idxs[ITEMS_PER_THREAD];

    using BlockRadixTopKT =
        BlockRadixTopKRegister<float, BLOCK_SIZE, ITEMS_PER_THREAD, true, int>;
    __shared__ typename BlockRadixTopKT::TempStorage sort_tmp_storage;

    const int tx = threadIdx.x;
#pragma unroll
    for (int j = 0; j < ITEMS_PER_THREAD; j++) {
      int idx = BLOCK_SIZE * j + tx;
      if (idx < neighbor_len) {
        float thread_weight = static_cast<float>(weight[data_offset + idx]);
        weight_keys[j] = gen_key_from_weight(thread_weight, rng);
        neighbor_idxs[j] = idx;
      }
    }
    const int valid_count = (neighbor_len < (BLOCK_SIZE * ITEMS_PER_THREAD))
                                ? neighbor_len
                                : (BLOCK_SIZE * ITEMS_PER_THREAD);
    BlockRadixTopKT{sort_tmp_storage}.radixTopKToStriped(
        weight_keys, neighbor_idxs, sample_len, valid_count);
    __syncthreads();
    const int stride = BLOCK_SIZE * ITEMS_PER_THREAD - sample_len;

    for (int idx_offset = ITEMS_PER_THREAD * BLOCK_SIZE;
         idx_offset < neighbor_len;
         idx_offset += stride) {
#pragma unroll
      for (int j = 0; j < ITEMS_PER_THREAD; j++) {
        int local_idx = BLOCK_SIZE * j + tx - sample_len;
        int target_idx = idx_offset + local_idx;
        if (local_idx >= 0 && target_idx < neighbor_len) {
          float thread_weight =
              static_cast<float>(weight[data_offset + target_idx]);
          weight_keys[j] = gen_key_from_weight(thread_weight, rng);
          neighbor_idxs[j] = target_idx;
        }
      }
      const int iter_valid_count =
          ((neighbor_len - idx_offset) >= stride)
              ? (BLOCK_SIZE * ITEMS_PER_THREAD)
              : (sample_len + neighbor_len - idx_offset);
      BlockRadixTopKT{sort_tmp_storage}.radixTopKToStriped(
          weight_keys, neighbor_idxs, sample_len, iter_valid_count);
      __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < ITEMS_PER_THREAD; j++) {
      int idx = j * BLOCK_SIZE + tx;
      if (idx < sample_len) {
        res[offset + idx] = data[data_offset + neighbor_idxs[j]];
        if (return_weight) {
          weight_array[offset + idx] =
              static_cast<float>(weight[data_offset + neighbor_idxs[j]]);
        }
      }
    }
  }
}

// almost the same as walking kernel
__global__ void unweighted_sample_large_kernel(GpuPsCommGraph graph,
                                               GpuPsNodeInfo* node_info_list,
                                               uint64_t* res,
                                               int* actual_size,
                                               int n,
                                               int sample_len,
                                               uint64_t random_seed,
                                               float* weight_array,
                                               bool return_weight) {
  int i = blockIdx.x;
  if (i >= n) return;
  int gidx = threadIdx.x + blockIdx.x * blockDim.x;
  RandomNumGen rng(gidx, random_seed);
  rng.NextValue();
  int neighbor_len = node_info_list[i].neighbor_size;
  uint32_t data_offset = node_info_list[i].neighbor_offset;
  int offset = i * sample_len;
  uint64_t* data = graph.neighbor_list;
  half* weight = graph.weight_list;
  if (neighbor_len <= sample_len) {  // directly copy
    actual_size[i] = neighbor_len;
    for (int j = threadIdx.x; j < neighbor_len; j += blockDim.x) {
      res[offset + j] = data[data_offset + j];
      if (return_weight) {
        weight_array[offset + j] = static_cast<float>(weight[data_offset + j]);
      }
    }
  } else {
    actual_size[i] = sample_len;
    for (int j = threadIdx.x; j < sample_len; j += blockDim.x) {
      res[offset + j] = j;
    }
    __syncthreads();
    for (int j = sample_len + threadIdx.x; j < neighbor_len; j += blockDim.x) {
      const int rand_num = rng.RandomMod(j + 1);
      if (rand_num < sample_len) {
        atomicMax(reinterpret_cast<unsigned int*>(res + offset + rand_num),
                  static_cast<unsigned int>(j));
      }
    }
    __syncthreads();
    for (int j = threadIdx.x; j < sample_len; j += blockDim.x) {
      const int64_t perm_idx = res[offset + j] + data_offset;
      res[offset + j] = data[perm_idx];
    }
  }
}

__device__ __forceinline__ int Log2UpCUDA(int x) {
  if (x <= 2) return x - 1;
  return 32 - __clz(x - 1);
}

template <int BLOCK_DIM, int ITEMS_PER_THREAD>
__global__ void unweighted_sample_kernel(GpuPsCommGraph graph,
                                         GpuPsNodeInfo* node_info_list,
                                         uint64_t* res,
                                         int* actual_size,
                                         int n,
                                         int sample_len,
                                         uint64_t random_seed,
                                         float* weight_array,
                                         bool return_weight) {
  int i = blockIdx.x;
  if (i >= n) return;
  int gidx = threadIdx.x + blockIdx.x * blockDim.x;
  RandomNumGen rng(gidx, random_seed);
  rng.NextValue();
  int neighbor_len = node_info_list[i].neighbor_size;
  uint32_t data_offset = node_info_list[i].neighbor_offset;
  int offset = i * sample_len;
  uint64_t* data = graph.neighbor_list;
  half* weight = graph.weight_list;

  if (neighbor_len <= sample_len) {
    actual_size[i] = neighbor_len;
    for (int j = threadIdx.x; j < neighbor_len; j += blockDim.x) {
      res[offset + j] = data[data_offset + j];
      if (return_weight) {
        weight_array[offset + j] = static_cast<float>(weight[data_offset + j]);
      }
    }
  } else {
    actual_size[i] = sample_len;
    uint64_t sa_p[ITEMS_PER_THREAD];
    int M = sample_len;
    int N = neighbor_len;
    typedef cub::BlockRadixSort<uint64_t, BLOCK_DIM, ITEMS_PER_THREAD>
        BlockRadixSort;
    struct IntArray {
      int value[BLOCK_DIM * ITEMS_PER_THREAD];
    };
    struct SampleSharedData {
      IntArray s;
      IntArray p;
      IntArray q;
      IntArray chain;
      IntArray last_chain_tmp;
    };
    __shared__ union {
      typename BlockRadixSort::TempStorage temp_storage;
      SampleSharedData sample_shared_data;
    } shared_data;
#pragma unroll
    for (int j = 0; j < ITEMS_PER_THREAD; j++) {
      uint32_t idx = j * BLOCK_DIM + threadIdx.x;
      uint32_t r = idx < M ? rng.RandomMod(N - idx) : N;
      sa_p[j] = ((uint64_t)r << 32UL) | idx;
    }
    __syncthreads();
    BlockRadixSort(shared_data.temp_storage).SortBlockedToStriped(sa_p);
    __syncthreads();
#pragma unroll
    for (int j = 0; j < ITEMS_PER_THREAD; j++) {
      int idx = j * BLOCK_DIM + threadIdx.x;
      int s = static_cast<int>((sa_p[j] >> 32UL));
      shared_data.sample_shared_data.s.value[idx] = s;
      int p = sa_p[j] & 0xFFFFFFFF;
      shared_data.sample_shared_data.p.value[idx] = p;
      if (idx < M) shared_data.sample_shared_data.q.value[p] = idx;
      shared_data.sample_shared_data.chain.value[idx] = idx;
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < ITEMS_PER_THREAD; j++) {
      int idx = j * BLOCK_DIM + threadIdx.x;
      int si = shared_data.sample_shared_data.s.value[idx];
      int si1 = shared_data.sample_shared_data.s.value[idx + 1];
      if (idx < M && (idx == M - 1 || si != si1) && si >= N - M) {
        shared_data.sample_shared_data.chain.value[N - si - 1] =
            shared_data.sample_shared_data.p.value[idx];
      }
    }
    __syncthreads();
    for (int step = 0; step < Log2UpCUDA(M); ++step) {
#pragma unroll
      for (int j = 0; j < ITEMS_PER_THREAD; j++) {
        int idx = j * BLOCK_DIM + threadIdx.x;
        shared_data.sample_shared_data.last_chain_tmp.value[idx] =
            shared_data.sample_shared_data.chain.value[idx];
      }
      __syncthreads();
#pragma unroll
      for (int j = 0; j < ITEMS_PER_THREAD; j++) {
        int idx = j * BLOCK_DIM + threadIdx.x;
        if (idx < M) {
          shared_data.sample_shared_data.chain.value[idx] =
              shared_data.sample_shared_data.last_chain_tmp.value
                  [shared_data.sample_shared_data.last_chain_tmp.value[idx]];
        }
      }
      __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < ITEMS_PER_THREAD; j++) {
      int idx = j * BLOCK_DIM + threadIdx.x;
      shared_data.sample_shared_data.last_chain_tmp.value[idx] =
          N - shared_data.sample_shared_data.chain.value[idx] - 1;
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < ITEMS_PER_THREAD; j++) {
      int idx = j * BLOCK_DIM + threadIdx.x;
      int ai;
      if (idx < M) {
        int qi = shared_data.sample_shared_data.q.value[idx];
        if (idx == 0 || qi == 0 ||
            shared_data.sample_shared_data.s.value[qi] !=
                shared_data.sample_shared_data.s.value[qi - 1]) {
          ai = shared_data.sample_shared_data.s.value[qi];
        } else {
          int prev_i = shared_data.sample_shared_data.p.value[qi - 1];
          ai = shared_data.sample_shared_data.last_chain_tmp.value[prev_i];
        }
        sa_p[j] = ai;
      }
    }
    // Output
#pragma unroll
    for (int j = 0; j < ITEMS_PER_THREAD; j++) {
      int idx = j * BLOCK_DIM + threadIdx.x;
      int ai = sa_p[j];
      if (idx < M) {
        res[offset + idx] = data[data_offset + ai];
        if (return_weight) {
          weight_array[offset + idx] =
              static_cast<float>(weight[data_offset + ai]);
        }
      }
    }
  }
}

void GpuPsGraphTable::weighted_sample(GpuPsCommGraph& graph,
                                      GpuPsNodeInfo* node_info_list,
                                      int* actual_size_array,
                                      uint64_t* sample_array,
                                      int* neighbor_count_ptr,
                                      int cur_gpu_id,
                                      int remote_gpu_id,
                                      int sample_size,
                                      int shard_len,
                                      bool need_neighbor_count,
                                      uint64_t random_seed,
                                      float* weight_array,
                                      bool return_weight) {
  platform::CUDADeviceGuard guard(resource_->dev_id(remote_gpu_id));
  phi::GPUPlace place = phi::GPUPlace(resource_->dev_id(cur_gpu_id));
  auto cur_stream = resource_->remote_stream(remote_gpu_id, cur_gpu_id);
  constexpr int BLOCK_SIZE = 256;

  int grid_size = (shard_len + 127) / 128;
  if (need_neighbor_count) {
    get_actual_size_and_neighbor_count<true>
        <<<grid_size, 128, 0, cur_stream>>>(node_info_list,
                                            actual_size_array,
                                            neighbor_count_ptr,
                                            sample_size,
                                            shard_len);
  } else {
    get_actual_size_and_neighbor_count<false>
        <<<grid_size, 128, 0, cur_stream>>>(
            node_info_list, actual_size_array, nullptr, sample_size, shard_len);
  }
  CUDA_CHECK(cudaStreamSynchronize(cur_stream));

  paddle::memory::ThrustAllocator<cudaStream_t> allocator(place, cur_stream);
  if (sample_size > SAMPLE_SIZE_THRESHOLD) {
    // to be optimized
    thrust::exclusive_scan(thrust::cuda::par(allocator).on(cur_stream),
                           neighbor_count_ptr,
                           neighbor_count_ptr + shard_len + 1,
                           neighbor_count_ptr);
    int* neighbor_offset = neighbor_count_ptr;
    int target_neighbor_counts = 0;
    cudaMemcpyAsync(&target_neighbor_counts,
                    neighbor_offset + shard_len,
                    sizeof(int),
                    cudaMemcpyDeviceToHost,
                    cur_stream);
    CUDA_CHECK(cudaStreamSynchronize(cur_stream));

    auto target_weights_key_buf =
        memory::Alloc(place,
                      target_neighbor_counts * sizeof(float),
                      phi::Stream(reinterpret_cast<phi::StreamId>(cur_stream)));
    float* target_weights_key_buf_ptr =
        reinterpret_cast<float*>(target_weights_key_buf->ptr());
    weighted_sample_large_kernel<BLOCK_SIZE>
        <<<shard_len, BLOCK_SIZE, 0, cur_stream>>>(graph,
                                                   node_info_list,
                                                   sample_array,
                                                   neighbor_offset,
                                                   target_weights_key_buf_ptr,
                                                   shard_len,
                                                   sample_size,
                                                   random_seed,
                                                   weight_array,
                                                   return_weight);
    CUDA_CHECK(cudaStreamSynchronize(cur_stream));
  } else {
    using WeightedSampleFuncType = void (*)(GpuPsCommGraph,
                                            GpuPsNodeInfo*,
                                            uint64_t*,
                                            int,
                                            int,
                                            uint64_t,
                                            float*,
                                            bool);
    static const WeightedSampleFuncType func_array[7] = {
        weighted_sample_kernel<4, 128>,
        weighted_sample_kernel<6, 128>,
        weighted_sample_kernel<4, 256>,
        weighted_sample_kernel<5, 256>,
        weighted_sample_kernel<6, 256>,
        weighted_sample_kernel<8, 256>,
        weighted_sample_kernel<8, 512>,
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
    func_array[func_idx]<<<shard_len, block_size, 0, cur_stream>>>(
        graph,
        node_info_list,
        sample_array,
        shard_len,
        sample_size,
        random_seed,
        weight_array,
        return_weight);
    CUDA_CHECK(cudaStreamSynchronize(cur_stream));
  }
}

void GpuPsGraphTable::unweighted_sample(GpuPsCommGraph& graph,
                                        GpuPsNodeInfo* node_info_list,
                                        int* actual_size_array,
                                        uint64_t* sample_array,
                                        int cur_gpu_id,
                                        int remote_gpu_id,
                                        int sample_size,
                                        int shard_len,
                                        uint64_t random_seed,
                                        float* weight_array,
                                        bool return_weight) {
  platform::CUDADeviceGuard guard(resource_->dev_id(remote_gpu_id));
  phi::GPUPlace place = phi::GPUPlace(resource_->dev_id(cur_gpu_id));
  auto cur_stream = resource_->remote_stream(remote_gpu_id, cur_gpu_id);

  if (sample_size > SAMPLE_SIZE_THRESHOLD) {
    unweighted_sample_large_kernel<<<shard_len, 32, 0, cur_stream>>>(
        graph,
        node_info_list,
        sample_array,
        actual_size_array,
        shard_len,
        sample_size,
        random_seed,
        weight_array,
        return_weight);
  } else {
    using UnWeightedSampleFuncType = void (*)(GpuPsCommGraph,
                                              GpuPsNodeInfo*,
                                              uint64_t*,
                                              int*,
                                              int,
                                              int,
                                              uint64_t,
                                              float*,
                                              bool);
    static const UnWeightedSampleFuncType func_array[32] = {
        unweighted_sample_kernel<32, 1>,  unweighted_sample_kernel<32, 2>,
        unweighted_sample_kernel<32, 3>,  unweighted_sample_kernel<64, 2>,
        unweighted_sample_kernel<64, 3>,  unweighted_sample_kernel<64, 3>,
        unweighted_sample_kernel<128, 2>, unweighted_sample_kernel<128, 2>,
        unweighted_sample_kernel<128, 3>, unweighted_sample_kernel<128, 3>,
        unweighted_sample_kernel<128, 3>, unweighted_sample_kernel<128, 3>,
        unweighted_sample_kernel<256, 2>, unweighted_sample_kernel<256, 2>,
        unweighted_sample_kernel<256, 2>, unweighted_sample_kernel<256, 2>,
        unweighted_sample_kernel<256, 3>, unweighted_sample_kernel<256, 3>,
        unweighted_sample_kernel<256, 3>, unweighted_sample_kernel<256, 3>,
        unweighted_sample_kernel<256, 3>, unweighted_sample_kernel<256, 3>,
        unweighted_sample_kernel<256, 3>, unweighted_sample_kernel<256, 3>,
        unweighted_sample_kernel<256, 4>, unweighted_sample_kernel<256, 4>,
        unweighted_sample_kernel<256, 4>, unweighted_sample_kernel<256, 4>,
        unweighted_sample_kernel<256, 4>, unweighted_sample_kernel<256, 4>,
        unweighted_sample_kernel<256, 4>, unweighted_sample_kernel<256, 4>,
    };
    static const int warp_count_array[32] = {1, 1, 1, 2, 2, 2, 4, 4, 4, 4, 4,
                                             4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
                                             8, 8, 8, 8, 8, 8, 8, 8, 8, 8};
    int func_idx = (sample_size - 1) / 32;
    func_array[func_idx]<<<shard_len,
                           warp_count_array[func_idx] * 32,
                           0,
                           cur_stream>>>(graph,
                                         node_info_list,
                                         sample_array,
                                         actual_size_array,
                                         shard_len,
                                         sample_size,
                                         random_seed,
                                         weight_array,
                                         return_weight);
  }
  CUDA_CHECK(cudaStreamSynchronize(cur_stream));
}

int GpuPsGraphTable::init_cpu_table(
    const paddle::distributed::GraphParameter& graph, int gpu_num) {
  cpu_graph_table_.reset(new paddle::distributed::GraphTable);
  cpu_table_status = cpu_graph_table_->Initialize(graph);
  cpu_graph_table_->init_worker_poll(gpu_num);
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
 to gup i and aggregate the result.
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

void GpuPsGraphTable::display_sample_res(
    void* key, void* val, int len, int sample_len, int gpu_id) {
  platform::CUDADeviceGuard guard(resource_->dev_id(gpu_id));
  char key_buffer[len * sizeof(uint64_t)];  // NOLINT
  char val_buffer[sample_len * sizeof(int64_t) * len +
                  (len + len % 2) * sizeof(int) + len * sizeof(uint64_t)];
  cudaMemcpy(key_buffer, key, sizeof(uint64_t) * len, cudaMemcpyDeviceToHost);
  cudaMemcpy(val_buffer,
             val,
             sample_len * sizeof(int64_t) * len +
                 (len + len % 2) * sizeof(int) + len * sizeof(uint64_t),
             cudaMemcpyDeviceToHost);
  uint64_t* sample_val = reinterpret_cast<uint64_t*>(
      val_buffer + (len + len % 2) * sizeof(int) + len * sizeof(int64_t));
  for (int i = 0; i < len; i++) {
    printf("key %llu\n",
           *reinterpret_cast<int64_t*>(key_buffer + i * sizeof(uint64_t)));
    printf("index %llu\n",
           *reinterpret_cast<int64_t*>(val_buffer + i * sizeof(uint64_t)));
    int ac_size = *reinterpret_cast<int*>(val_buffer + i * sizeof(int) +
                                          len * sizeof(int64_t));
    printf("sampled %d neighbors\n", ac_size);
    for (int j = 0; j < ac_size; j++) {
      printf("%llu ", sample_val[i * sample_len + j]);
    }
    printf("\n");
  }
}

// 用模板参数
template <typename FeatureType>
void GpuPsGraphTable::move_result_to_source_gpu(int start_index,
                                                int gpu_num,
                                                int* h_left,
                                                int* h_right,
                                                int* fea_left,
                                                uint32_t* fea_num_list,
                                                uint32_t* actual_feature_size,
                                                FeatureType* feature_list,
                                                uint8_t* slot_list) {
  int shard_len[gpu_num];  // NOLINT
  for (int i = 0; i < gpu_num; i++) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    shard_len[i] = h_right[i] - h_left[i] + 1;
    int cur_step = path_[start_index][i].nodes_.size() - 1;
    for (int j = cur_step; j > 0; j--) {
      auto& dst_node = path_[start_index][i].nodes_[j - 1];
      auto& src_node = path_[start_index][i].nodes_[j];
      MemcpyPeerAsync(dst_node.val_storage,
                      src_node.val_storage,
                      dst_node.val_bytes_len,
                      src_node.out_stream);
      if (src_node.sync) {
        CUDA_CHECK(cudaStreamSynchronize(src_node.out_stream));
      }
    }
    auto& node = path_[start_index][i].nodes_.front();

    if (fea_num_list[i] > 0) {
      MemcpyPeerAsync(reinterpret_cast<char*>(feature_list + fea_left[i]),
                      node.val_storage +
                          sizeof(uint32_t) * (shard_len[i] + shard_len[i] % 2),
                      sizeof(FeatureType) * fea_num_list[i],
                      node.out_stream);
      MemcpyPeerAsync(reinterpret_cast<char*>(slot_list + fea_left[i]),
                      node.val_storage +
                          sizeof(uint32_t) * (shard_len[i] + shard_len[i] % 2) +
                          sizeof(FeatureType) * fea_num_list[i],
                      sizeof(uint8_t) * fea_num_list[i],
                      node.out_stream);
    }
    if (shard_len[i] > 0) {
      MemcpyPeerAsync(reinterpret_cast<char*>(actual_feature_size + h_left[i]),
                      node.val_storage,
                      sizeof(uint32_t) * shard_len[i],
                      node.out_stream);
    }
  }
  for (int i = 0; i < gpu_num; ++i) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    auto& node = path_[start_index][i].nodes_.front();
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    CUDA_CHECK(cudaStreamSynchronize(node.out_stream));
  }
}

void GpuPsGraphTable::move_result_to_source_gpu(int start_index,
                                                int gpu_num,
                                                int sample_size,
                                                int* h_left,
                                                int* h_right,
                                                uint64_t* src_sample_res,
                                                int* actual_sample_size) {
  int shard_len[gpu_num];  // NOLINT
  for (int i = 0; i < gpu_num; i++) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    shard_len[i] = h_right[i] - h_left[i] + 1;
    int cur_step = path_[start_index][i].nodes_.size() - 1;
    for (int j = cur_step; j > 0; j--) {
      auto& dst_node = path_[start_index][i].nodes_[j - 1];
      auto& src_node = path_[start_index][i].nodes_[j];
      MemcpyPeerAsync(dst_node.val_storage,
                      src_node.val_storage,
                      dst_node.val_bytes_len,
                      src_node.out_stream);
      if (src_node.sync) {
        CUDA_CHECK(cudaStreamSynchronize(src_node.out_stream));
      }
    }
    auto& node = path_[start_index][i].nodes_.front();
    MemcpyPeerAsync(
        reinterpret_cast<char*>(src_sample_res + h_left[i] * sample_size),
        node.val_storage + sizeof(int64_t) * shard_len[i] +
            sizeof(int) * (shard_len[i] + shard_len[i] % 2),
        sizeof(uint64_t) * shard_len[i] * sample_size,
        node.out_stream);
    MemcpyPeerAsync(reinterpret_cast<char*>(actual_sample_size + h_left[i]),
                    node.val_storage + sizeof(int64_t) * shard_len[i],
                    sizeof(int) * shard_len[i],
                    node.out_stream);
  }
  for (int i = 0; i < gpu_num; ++i) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    auto& node = path_[start_index][i].nodes_.front();
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    CUDA_CHECK(cudaStreamSynchronize(node.out_stream));
    // cudaStreamSynchronize(resource_->remote_stream(i, start_index));
  }
}

void GpuPsGraphTable::move_degree_to_source_gpu(
    int start_index, int gpu_num, int* h_left, int* h_right, int* node_degree) {
  std::vector<int> shard_len(gpu_num, 0);
  for (int i = 0; i < gpu_num; i++) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    shard_len[i] = h_right[i] - h_left[i] + 1;
    int cur_step = static_cast<int>(path_[start_index][i].nodes_.size()) - 1;
    for (int j = cur_step; j > 0; j--) {
      auto& dst_node = path_[start_index][i].nodes_[j - 1];
      auto& src_node = path_[start_index][i].nodes_[j];
      MemcpyPeerAsync(dst_node.val_storage,
                      src_node.val_storage,
                      dst_node.val_bytes_len,
                      src_node.out_stream);
      if (src_node.sync) {
        CUDA_CHECK(cudaStreamSynchronize(src_node.out_stream));
      }
    }
    auto& node = path_[start_index][i].nodes_.front();
    MemcpyPeerAsync(reinterpret_cast<char*>(node_degree + h_left[i]),
                    node.val_storage + sizeof(int64_t) * shard_len[i],
                    sizeof(int) * shard_len[i],
                    node.out_stream);
  }

  for (int i = 0; i < gpu_num; ++i) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    auto& node = path_[start_index][i].nodes_.front();
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    CUDA_CHECK(cudaStreamSynchronize(node.out_stream));
  }
}

void GpuPsGraphTable::move_result_to_source_gpu_all_edge_type(
    int start_index,
    int gpu_num,
    int sample_size,
    int* h_left,
    int* h_right,
    uint64_t* src_sample_res,
    int* actual_sample_size,
    float* edge_weight,
    int edge_type_len,
    int len,
    bool return_weight) {
  int shard_len[gpu_num];  // NOLINT

  for (int i = 0; i < gpu_num; i++) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    shard_len[i] = h_right[i] - h_left[i] + 1;
    int cur_step = path_[start_index][i].nodes_.size() - 1;
    for (int j = cur_step; j > 0; j--) {
      auto& dst_node = path_[start_index][i].nodes_[j - 1];
      auto& src_node = path_[start_index][i].nodes_[j];
      MemcpyPeerAsync(dst_node.val_storage,
                      src_node.val_storage,
                      dst_node.val_bytes_len,
                      src_node.out_stream);
      if (src_node.sync) {
        CUDA_CHECK(cudaStreamSynchronize(src_node.out_stream));
      }
    }
  }

  for (int i = 0; i < edge_type_len; i++) {
    for (int j = 0; j < gpu_num; j++) {
      if (h_left[j] == -1 || h_right[j] == -1) {
        continue;
      }
      auto& node = path_[start_index][j].nodes_.front();
      platform::CUDADeviceGuard guard(resource_->dev_id(j));
      if (!return_weight) {  // sample array
        MemcpyPeerAsync(
            reinterpret_cast<char*>(src_sample_res + i * len * sample_size +
                                    h_left[j] * sample_size),
            node.val_storage + sizeof(int64_t) * shard_len[j] * edge_type_len +
                sizeof(int) * (shard_len[j] * edge_type_len +
                               (shard_len[j] * edge_type_len) % 2) +
                sizeof(uint64_t) * i * shard_len[j] * sample_size,
            sizeof(uint64_t) * shard_len[j] * sample_size,
            node.out_stream);
      } else {
        MemcpyPeerAsync(  // edge weight
            reinterpret_cast<char*>(edge_weight + i * len * sample_size +
                                    h_left[j] * sample_size),
            node.val_storage + sizeof(int64_t) * shard_len[j] * edge_type_len +
                sizeof(int) * (shard_len[j] * edge_type_len) +
                sizeof(float) * i * shard_len[j] * sample_size,
            sizeof(float) * shard_len[j] * sample_size,
            node.out_stream);
        MemcpyPeerAsync(  // sample array
            reinterpret_cast<char*>(src_sample_res + i * len * sample_size +
                                    h_left[j] * sample_size),
            node.val_storage + sizeof(int64_t) * shard_len[j] * edge_type_len +
                sizeof(int) * (shard_len[j] * edge_type_len) +
                sizeof(float) * (shard_len[j] * sample_size * edge_type_len) +
                sizeof(int) *
                    ((shard_len[j] * edge_type_len) * (sample_size + 1)) % 2 +
                sizeof(uint64_t) * i * shard_len[j] * sample_size,
            sizeof(uint64_t) * shard_len[j] * sample_size,
            node.out_stream);
      }
      MemcpyPeerAsync(  // actual sample size
          reinterpret_cast<char*>(actual_sample_size + i * len + h_left[j]),
          node.val_storage + sizeof(int64_t) * shard_len[j] * edge_type_len +
              sizeof(int) * i * shard_len[j],
          sizeof(int) * shard_len[j],
          node.out_stream);
    }
  }

  for (int i = 0; i < gpu_num; i++) {
    if (h_left[i] == -1 || h_right[i] == -1) {
      continue;
    }
    auto& node = path_[start_index][i].nodes_.front();
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    CUDA_CHECK(cudaStreamSynchronize(node.out_stream));
  }
}

__global__ void fill_size(uint32_t* d_actual_size_list,
                          uint32_t* d_shard_size_list,
                          int* idx,
                          int len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    d_actual_size_list[idx[i]] = d_shard_size_list[i];
  }
}
// 搞成模板
template <typename T>
__global__ void fill_feature_and_slot(T* dst_feature_list,
                                      uint8_t* dst_slot_list,
                                      uint32_t* dst_size_prefix_sum_list,
                                      T* src_feature_list,
                                      uint8_t* src_slot_list,
                                      uint32_t* src_size_prefix_sum_list,
                                      uint32_t* src_size_list,
                                      int* idx,
                                      int len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    uint32_t dst_index = dst_size_prefix_sum_list[idx[i]];
    uint32_t src_index = src_size_prefix_sum_list[i];
    for (uint32_t j = 0; j < src_size_list[i]; j++) {
      dst_feature_list[dst_index + j] = src_feature_list[src_index + j];
      dst_slot_list[dst_index + j] = src_slot_list[src_index + j];
    }
  }
}

template <typename T>
__global__ void fill_vari_feature_and_slot(T* dst_feature_list,
                                           uint8_t* dst_slot_list,
                                           T* src_feature_list,
                                           uint8_t* src_slot_list,
                                           uint32_t* dst_size_prefix_sum_list,
                                           uint32_t* src_size_prefix_sum_list,
                                           uint32_t* src_size_list,
                                           uint32_t* idx,
                                           int len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    uint32_t dst_index = dst_size_prefix_sum_list[idx[i]];
    uint32_t src_index = src_size_prefix_sum_list[i];
    for (uint32_t j = 0; j < src_size_list[i]; j++) {
      dst_feature_list[dst_index + j] = src_feature_list[src_index + j];
      dst_slot_list[dst_index + j] = src_slot_list[src_index + j];
    }
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
    size_t offset1 = idx[i] * sample_size;
    size_t offset2 = i * sample_size;
    for (int j = 0; j < d_shard_actual_sample_size[i]; j++) {
      d_vals[offset1 + j] = d_shard_vals[offset2 + j];
    }
  }
}

__global__ void fill_dvalues(int* d_shard_degree,
                             int* d_degree,
                             int* idx,
                             int len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    d_degree[idx[i]] = d_shard_degree[i];
  }
}

__global__ void fill_dvalues_with_edge_type(uint64_t* d_shard_vals,
                                            uint64_t* d_vals,
                                            int* d_shard_actual_sample_size,
                                            int* d_actual_sample_size,
                                            float* d_shard_weights,
                                            float* d_weights,
                                            int* idx,
                                            int sample_size,
                                            int len,  // len * edge_type_len
                                            int mod,  // len
                                            bool return_weight) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    int a = i % mod,
        b = i - i % mod;  // a: get actual pos, b: get fill in which edge_type
    d_actual_sample_size[b + idx[a]] = d_shard_actual_sample_size[i];
    size_t offset1 = (b + idx[a]) * sample_size;
    size_t offset2 = i * sample_size;
    for (int j = 0; j < d_shard_actual_sample_size[i]; j++) {
      d_vals[offset1 + j] = d_shard_vals[offset2 + j];
      if (return_weight) {
        d_weights[offset1 + j] = d_shard_weights[offset2 + j];
      }
    }
  }
}

__global__ void fill_dvalues_with_edge_type_for_all2all(
    uint64_t* d_shard_vals,
    uint64_t* d_vals,
    int* d_shard_actual_sample_size,
    int* d_actual_sample_size,
    float* d_shard_weights,
    float* d_weights,
    int* idx,
    int sample_size,
    int n,  // len * edge_type_len
    int len,
    int edge_type_len,
    bool return_weight) {  // len
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int pos = idx[i % len] * edge_type_len +
              static_cast<int>(i) / len;  // pos: real position
    d_actual_sample_size[pos] = d_shard_actual_sample_size[i];
    size_t offset1 = pos * sample_size;
    size_t offset2 = i * sample_size;
    for (int j = 0; j < d_shard_actual_sample_size[i]; j++) {
      d_vals[offset1 + j] = d_shard_vals[offset2 + j];
      if (return_weight) {
        d_weights[offset1 + j] = d_shard_weights[offset2 + j];
      }
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
    int offset1 = cumsum_actual_sample_size[i];
    int offset2 = sample_size * i;
    for (int j = 0; j < actual_sample_size[i]; j++) {
      actual_vals[offset1 + j] = vals[offset2 + j];
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
  // slot fea
  int offset = get_table_offset(gpu_id, GraphTableType::FEATURE_TABLE, idx);
  if (offset < tables_.size()) {
    delete tables_[offset];
    tables_[offset] = NULL;
  }

  int graph_fea_idx = get_graph_fea_list_offset(gpu_id);
  platform::CUDADeviceGuard guard(resource_->dev_id(gpu_id));
  auto& graph = gpu_graph_fea_list_[graph_fea_idx];
  if (graph.feature_list != NULL) {
    cudaFree(graph.feature_list);
    graph.feature_list = NULL;
  }

  if (graph.slot_id_list != NULL) {
    cudaFree(graph.slot_id_list);
    graph.slot_id_list = NULL;
  }
  graph.feature_capacity = 0;

  if (float_feature_table_num_ > 0) {
    // float fea
    idx = 1;
    int float_offset =
        get_table_offset(gpu_id, GraphTableType::FEATURE_TABLE, idx);
    if (float_offset < tables_.size()) {
      delete tables_[float_offset];
      tables_[float_offset] = NULL;
    }

    int graph_float_fea_idx = get_graph_float_fea_list_offset(gpu_id);
    auto& float_graph = gpu_graph_float_fea_list_[graph_float_fea_idx];
    if (float_graph.feature_list != NULL) {
      cudaFree(float_graph.feature_list);
      float_graph.feature_list = NULL;
    }

    if (float_graph.slot_id_list != NULL) {
      cudaFree(float_graph.slot_id_list);
      float_graph.slot_id_list = NULL;
    }
    float_graph.feature_capacity = 0;
  }
}

void GpuPsGraphTable::reset_feature_info(int gpu_id,
                                         size_t capacity,
                                         size_t feature_size) {
  platform::CUDADeviceGuard guard(resource_->dev_id(gpu_id));
  int idx = 0;
  auto stream = get_local_stream(gpu_id);
  int offset = get_table_offset(gpu_id, GraphTableType::FEATURE_TABLE, idx);
  if (offset < tables_.size()) {
    delete tables_[offset];
    tables_[offset] = new Table(capacity, stream);
  }
  int graph_fea_idx = get_graph_fea_list_offset(gpu_id);
  auto& graph = gpu_graph_fea_list_[graph_fea_idx];
  graph.node_list = NULL;
  if (graph.feature_list == NULL) {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&graph.feature_list),
                          feature_size * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&graph.slot_id_list),
                          ALIGN_INT64(feature_size * sizeof(uint8_t))));
    graph.feature_capacity = feature_size;
  } else if (graph.feature_capacity < feature_size) {
    cudaFree(graph.feature_list);
    cudaFree(graph.slot_id_list);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&graph.feature_list),
                          feature_size * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&graph.slot_id_list),
                          ALIGN_INT64(feature_size * sizeof(uint8_t))));
    graph.feature_capacity = feature_size;
  } else {
    CUDA_CHECK(cudaMemsetAsync(
        graph.feature_list, 0, feature_size * sizeof(uint64_t), stream));
    CUDA_CHECK(cudaMemsetAsync(
        graph.slot_id_list, 0, feature_size * sizeof(uint8_t), stream));
    cudaStreamSynchronize(stream);
  }
}

void GpuPsGraphTable::reset_float_feature_info(int gpu_id,
                                               size_t capacity,
                                               size_t feature_size) {
  platform::CUDADeviceGuard guard(resource_->dev_id(gpu_id));
  int idx = 1;
  auto stream = get_local_stream(gpu_id);
  int offset = get_table_offset(gpu_id, GraphTableType::FEATURE_TABLE, idx);
  if (offset < tables_.size()) {
    delete tables_[offset];
    tables_[offset] = new Table(capacity, stream);
  }
  int graph_float_fea_idx = get_graph_float_fea_list_offset(gpu_id);
  auto& graph = gpu_graph_float_fea_list_[graph_float_fea_idx];

  graph.node_list = NULL;

  if (graph.feature_list == NULL) {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&graph.feature_list),
                          feature_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&graph.slot_id_list),
                          ALIGN_INT64(feature_size * sizeof(uint8_t))));
    graph.feature_capacity = feature_size;
  } else if (graph.feature_capacity < feature_size) {
    cudaFree(graph.feature_list);
    cudaFree(graph.slot_id_list);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&graph.feature_list),
                          feature_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&graph.slot_id_list),
                          ALIGN_INT64(feature_size * sizeof(uint8_t))));
    graph.feature_capacity = feature_size;
  } else {
    CUDA_CHECK(cudaMemsetAsync(
        graph.feature_list, 0, feature_size * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(
        graph.slot_id_list, 0, feature_size * sizeof(uint8_t), stream));
    cudaStreamSynchronize(stream);
  }
}

void GpuPsGraphTable::reset_rank_info(int gpu_id,
                                      size_t capacity,
                                      size_t feature_size) {
  platform::CUDADeviceGuard guard(resource_->dev_id(gpu_id));
  auto stream = get_local_stream(gpu_id);
  int offset = get_rank_list_offset(gpu_id);
  if (offset < rank_tables_.size()) {
    delete rank_tables_[offset];
    rank_tables_[offset] = new RankTable(capacity, stream);
    cudaStreamSynchronize(stream);
  }
}

void GpuPsGraphTable::clear_graph_info(int gpu_id, int idx) {
  if (idx >= graph_table_num_) return;
  platform::CUDADeviceGuard guard(resource_->dev_id(gpu_id));
  int offset = get_table_offset(gpu_id, GraphTableType::EDGE_TABLE, idx);
  if (offset < tables_.size()) {
    delete tables_[offset];
    tables_[offset] = NULL;
  }
  auto& graph = gpu_graph_list_[get_graph_list_offset(gpu_id, idx)];
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
  platform::CUDADeviceGuard guard(resource_->dev_id(gpu_id));
  size_t capacity = std::max((uint64_t)1, g.node_size) / load_factor_;
  int ntype_id = 0;  // slot feature
  reset_feature_info(gpu_id, capacity, g.feature_size);
  int offset = get_graph_fea_list_offset(gpu_id);
  int table_offset =
      get_table_offset(gpu_id, GraphTableType::FEATURE_TABLE, ntype_id);
  if (g.node_size > 0) {
    build_ps(gpu_id,
             g.node_list,
             reinterpret_cast<uint64_t*>(g.fea_info_list),
             g.node_size,
             HBMPS_MAX_BUFF,
             8,
             table_offset);
    gpu_graph_fea_list_[offset].node_size = g.node_size;
  } else {
    build_ps(gpu_id, NULL, NULL, 0, HBMPS_MAX_BUFF, 8, table_offset);
    gpu_graph_fea_list_[offset].node_size = 0;
  }
  if (g.feature_size) {
    auto stream = get_local_stream(gpu_id);
    CUDA_CHECK(cudaMemcpyAsync(gpu_graph_fea_list_[offset].feature_list,
                               g.feature_list,
                               g.feature_size * sizeof(uint64_t),
                               cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(gpu_graph_fea_list_[offset].slot_id_list,
                               g.slot_id_list,
                               g.feature_size * sizeof(uint8_t),
                               cudaMemcpyHostToDevice,
                               stream));
    cudaStreamSynchronize(stream);

    gpu_graph_fea_list_[offset].feature_size = g.feature_size;
  } else {
    gpu_graph_fea_list_[offset].feature_size = 0;
  }
  VLOG(1) << "gpu node_feature info card :" << gpu_id << " ,node_size is "
          << gpu_graph_fea_list_[offset].node_size << ", feature_size is "
          << gpu_graph_fea_list_[offset].feature_size;
}

void GpuPsGraphTable::build_graph_float_fea_on_single_gpu(
    const GpuPsCommGraphFloatFea& g, int gpu_id) {
  platform::CUDADeviceGuard guard(resource_->dev_id(gpu_id));
  size_t capacity = std::max((uint64_t)1, g.node_size) / load_factor_;
  int ntype_id = 1;  // float feature
  reset_float_feature_info(gpu_id, capacity, g.feature_size);
  int offset = get_graph_float_fea_list_offset(gpu_id);
  int table_offset =
      get_table_offset(gpu_id, GraphTableType::FEATURE_TABLE, ntype_id);
  if (g.node_size > 0) {
    build_ps(gpu_id,
             g.node_list,
             reinterpret_cast<uint64_t*>(g.fea_info_list),
             g.node_size,
             HBMPS_MAX_BUFF,
             8,
             table_offset);
    gpu_graph_float_fea_list_[offset].node_size = g.node_size;
  } else {
    build_ps(gpu_id, NULL, NULL, 0, HBMPS_MAX_BUFF, 8, table_offset);
    gpu_graph_float_fea_list_[offset].node_size = 0;
  }
  if (g.feature_size) {
    auto stream = get_local_stream(gpu_id);
    CUDA_CHECK(cudaMemcpyAsync(gpu_graph_float_fea_list_[offset].feature_list,
                               g.feature_list,
                               g.feature_size * sizeof(float),
                               cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(gpu_graph_float_fea_list_[offset].slot_id_list,
                               g.slot_id_list,
                               g.feature_size * sizeof(uint8_t),
                               cudaMemcpyHostToDevice,
                               stream));
    cudaStreamSynchronize(stream);

    gpu_graph_float_fea_list_[offset].feature_size = g.feature_size;
  } else {
    gpu_graph_float_fea_list_[offset].feature_size = 0;
  }
  VLOG(0) << "gpu node_float_feature info card :" << gpu_id << " ,node_size is "
          << gpu_graph_float_fea_list_[offset].node_size << ", feature_size is "
          << gpu_graph_float_fea_list_[offset].feature_size;
}

std::vector<std::shared_ptr<phi::Allocation>>
GpuPsGraphTable::get_edge_type_graph(int gpu_id, int edge_type_len) {
  int total_gpu = resource_->total_device();
  auto stream = resource_->local_stream(gpu_id, 0);

  phi::GPUPlace place = phi::GPUPlace(resource_->dev_id(gpu_id));
  platform::CUDADeviceGuard guard(resource_->dev_id(gpu_id));

  std::vector<std::shared_ptr<phi::Allocation>> graphs_vec;
  for (int i = 0; i < total_gpu; i++) {
    GpuPsCommGraph graphs[edge_type_len];  // NOLINT
    for (int idx = 0; idx < edge_type_len; idx++) {
      int table_offset = get_table_offset(i, GraphTableType::EDGE_TABLE, idx);
      int offset = get_graph_list_offset(i, idx);
      graphs[idx] = gpu_graph_list_[offset];
    }
    auto d_commgraph_mem = memory::AllocShared(
        place,
        edge_type_len * sizeof(GpuPsCommGraph),
        phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
    GpuPsCommGraph* d_commgraph_ptr =
        reinterpret_cast<GpuPsCommGraph*>(d_commgraph_mem->ptr());
    CUDA_CHECK(cudaMemcpyAsync(d_commgraph_ptr,
                               graphs,
                               sizeof(GpuPsCommGraph) * edge_type_len,
                               cudaMemcpyHostToDevice,
                               stream));
    cudaStreamSynchronize(stream);
    graphs_vec.emplace_back(d_commgraph_mem);
  }

  return graphs_vec;
}

void GpuPsGraphTable::rank_build_ps(int dev_num,
                                    uint64_t* h_keys,
                                    uint32_t* h_vals,
                                    size_t len,
                                    size_t chunk_size,
                                    int stream_num) {
  if (len <= 0) {
    return;
  }
  int dev_id = resource_->dev_id(dev_num);

  std::vector<std::shared_ptr<phi::Allocation>> d_key_bufs;
  std::vector<std::shared_ptr<phi::Allocation>> d_val_bufs;

  // auto adjust stream num by data length
  int max_stream = (len + chunk_size - 1) / chunk_size;
  if (max_stream < stream_num) {
    stream_num = max_stream;
  }
  if (stream_num > device_num_) {
    stream_num = device_num_;
  }

  DevPlace place = DevPlace(dev_id);
  AnyDeviceGuard guard(dev_id);
  ppStream streams[stream_num];  // NOLINT

  d_key_bufs.resize(stream_num);
  d_val_bufs.resize(stream_num);
  for (int i = 0; i < stream_num; ++i) {
    streams[i] = resource_->local_stream(dev_num, i);
    d_key_bufs[i] = MemoryAlloc(place, chunk_size * sizeof(uint64_t));
    d_val_bufs[i] = MemoryAlloc(place, chunk_size * sizeof(uint32_t));
  }

  int cur_len = 0;
  int cur_stream = 0;
  while (static_cast<size_t>(cur_len) < len) {
    cur_stream = cur_stream % stream_num;
    auto cur_use_stream = streams[cur_stream];

    int tmp_len = cur_len + chunk_size > len ? len - cur_len : chunk_size;

    auto dst_place = place;
    auto src_place = phi::CPUPlace();
    memory_copy(dst_place,
                reinterpret_cast<char*>(d_key_bufs[cur_stream]->ptr()),
                src_place,
                h_keys + cur_len,
                sizeof(uint64_t) * tmp_len,
                cur_use_stream);
    memory_copy(dst_place,
                reinterpret_cast<char*>(d_val_bufs[cur_stream]->ptr()),
                src_place,
                h_vals + cur_len,
                sizeof(uint32_t) * tmp_len,
                cur_use_stream);
    rank_tables_[dev_num]->insert(
        reinterpret_cast<uint64_t*>(d_key_bufs[cur_stream]->ptr()),
        reinterpret_cast<uint32_t*>(d_val_bufs[cur_stream]->ptr()),
        static_cast<size_t>(tmp_len),
        cur_use_stream);

    cur_stream += 1;
    cur_len += tmp_len;
  }
  for (int i = 0; i < stream_num; ++i) {
    sync_stream(streams[i]);
  }
}

void GpuPsGraphTable::build_rank_fea_on_single_gpu(const GpuPsCommRankFea& g,
                                                   int gpu_id) {
  platform::CUDADeviceGuard guard(resource_->dev_id(gpu_id));
  size_t capacity = std::max((uint64_t)1, g.feature_size) / load_factor_;
  reset_rank_info(gpu_id, capacity, g.feature_size);
  int offset = get_rank_list_offset(gpu_id);
  if (g.feature_size > 0) {
    rank_build_ps(gpu_id,
                  g.node_list,
                  reinterpret_cast<uint32_t*>(g.rank_list),
                  g.feature_size,
                  HBMPS_MAX_BUFF,
                  8);
  } else {
    rank_build_ps(gpu_id, NULL, NULL, 0, HBMPS_MAX_BUFF, 8);
  }
  VLOG(1) << "gpu rank_feature info card :" << gpu_id
          << " finish, size:" << g.feature_size;
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
                                                int gpu_id,
                                                int edge_idx) {
  clear_graph_info(gpu_id, edge_idx);
  platform::CUDADeviceGuard guard(resource_->dev_id(gpu_id));
  int offset = get_graph_list_offset(gpu_id, edge_idx);
  gpu_graph_list_[offset] = GpuPsCommGraph();
  int table_offset =
      get_table_offset(gpu_id, GraphTableType::EDGE_TABLE, edge_idx);
  size_t capacity = std::max((uint64_t)1, (uint64_t)g.node_size) / load_factor_;
  auto stream = get_local_stream(gpu_id);
  tables_[table_offset] = new Table(capacity, stream);
  if (g.node_size > 0) {
    if (FLAGS_gpugraph_load_node_list_into_hbm) {
      CUDA_CHECK(cudaMalloc(&gpu_graph_list_[offset].node_list,
                            g.node_size * sizeof(uint64_t)));
      CUDA_CHECK(cudaMemcpyAsync(gpu_graph_list_[offset].node_list,
                                 g.node_list,
                                 g.node_size * sizeof(uint64_t),
                                 cudaMemcpyHostToDevice,
                                 stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    build_ps(gpu_id,
             g.node_list,
             reinterpret_cast<uint64_t*>(g.node_info_list),
             g.node_size,
             HBMPS_MAX_BUFF,
             8,
             table_offset);
    gpu_graph_list_[offset].node_size = g.node_size;
  } else {
    build_ps(gpu_id, NULL, NULL, 0, HBMPS_MAX_BUFF, 8, table_offset);
    gpu_graph_list_[offset].node_list = NULL;
    gpu_graph_list_[offset].node_size = 0;
  }
  if (g.neighbor_size) {
    cudaError_t cudaStatus;
    if (!FLAGS_enable_neighbor_list_use_uva) {
      cudaStatus = cudaMalloc(&gpu_graph_list_[offset].neighbor_list,
                              g.neighbor_size * sizeof(uint64_t));
    } else {
      cudaStatus = cudaMallocManaged(&gpu_graph_list_[offset].neighbor_list,
                                     g.neighbor_size * sizeof(uint64_t));
    }
    PADDLE_ENFORCE_EQ(cudaStatus,
                      cudaSuccess,
                      common::errors::InvalidArgument(
                          "failed to allocate memory for graph on gpu %d",
                          resource_->dev_id(gpu_id)));
    VLOG(0) << "successfully allocate " << g.neighbor_size * sizeof(uint64_t)
            << " bytes of memory for graph-edges on gpu "
            << resource_->dev_id(gpu_id);
    CUDA_CHECK(cudaMemcpyAsync(gpu_graph_list_[offset].neighbor_list,
                               g.neighbor_list,
                               g.neighbor_size * sizeof(uint64_t),
                               cudaMemcpyHostToDevice,
                               stream));
    gpu_graph_list_[offset].neighbor_size = g.neighbor_size;

    if (g.is_weighted) {
      cudaError_t cudaStatus = cudaMalloc(&gpu_graph_list_[offset].weight_list,
                                          g.neighbor_size * sizeof(half));
      PADDLE_ENFORCE_EQ(
          cudaStatus,
          cudaSuccess,
          common::errors::InvalidArgument(
              "failed to allocate memory for graph edge weight on gpu %d",
              resource_->dev_id(gpu_id)));
      VLOG(0) << "successfully allocate " << g.neighbor_size * sizeof(float)
              << " bytes of memory for graph-edges-weight on gpu "
              << resource_->dev_id(gpu_id);
      CUDA_CHECK(cudaMemcpyAsync(gpu_graph_list_[offset].weight_list,
                                 g.weight_list,
                                 g.neighbor_size * sizeof(half),
                                 cudaMemcpyHostToDevice,
                                 stream));
    }

  } else {
    gpu_graph_list_[offset].neighbor_list = NULL;
    gpu_graph_list_[offset].neighbor_size = 0;
    gpu_graph_list_[offset].weight_list = NULL;
  }
  cudaStreamSynchronize(stream);
  VLOG(0) << " gpu node_neighbor info card: " << gpu_id << " ,node_size is "
          << gpu_graph_list_[offset].node_size << ", neighbor_size is "
          << gpu_graph_list_[offset].neighbor_size;
}

void GpuPsGraphTable::build_graph_from_cpu(
    const std::vector<GpuPsCommGraph>& cpu_graph_list, int edge_idx) {
  VLOG(0) << "in build_graph_from_cpu cpu_graph_list size = "
          << cpu_graph_list.size();
  PADDLE_ENFORCE_EQ(
      cpu_graph_list.size(),
      resource_->total_device(),
      common::errors::InvalidArgument("the cpu node list size doesn't match "
                                      "the number of gpu on your machine."));
  clear_graph_info(edge_idx);
  for (int i = 0; i < cpu_graph_list.size(); i++) {
    int table_offset =
        get_table_offset(i, GraphTableType::EDGE_TABLE, edge_idx);
    int offset = get_graph_list_offset(i, edge_idx);
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    gpu_graph_list_[offset] = GpuPsCommGraph();
    auto stream = get_local_stream(i);
    tables_[table_offset] =
        new Table(std::max((uint64_t)1, (uint64_t)cpu_graph_list[i].node_size) /
                      load_factor_,
                  stream);
    if (cpu_graph_list[i].node_size > 0) {
      CUDA_CHECK(cudaMalloc(&gpu_graph_list_[offset].node_list,
                            cpu_graph_list[i].node_size * sizeof(uint64_t)));
      CUDA_CHECK(cudaMemcpyAsync(gpu_graph_list_[offset].node_list,
                                 cpu_graph_list[i].node_list,
                                 cpu_graph_list[i].node_size * sizeof(uint64_t),
                                 cudaMemcpyHostToDevice,
                                 stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      build_ps(i,
               cpu_graph_list[i].node_list,
               reinterpret_cast<uint64_t*>(cpu_graph_list[i].node_info_list),
               cpu_graph_list[i].node_size,
               HBMPS_MAX_BUFF,
               8,
               table_offset);
      gpu_graph_list_[offset].node_size = cpu_graph_list[i].node_size;
    } else {
      build_ps(i, NULL, NULL, 0, HBMPS_MAX_BUFF, 8, table_offset);
      gpu_graph_list_[offset].node_list = NULL;
      gpu_graph_list_[offset].node_size = 0;
    }
    if (cpu_graph_list[i].neighbor_size) {
      if (!FLAGS_enable_neighbor_list_use_uva) {
        CUDA_CHECK(
            cudaMalloc(&gpu_graph_list_[offset].neighbor_list,
                       cpu_graph_list[i].neighbor_size * sizeof(uint64_t)));
      } else {
        CUDA_CHECK(cudaMallocManaged(
            &gpu_graph_list_[offset].neighbor_list,
            cpu_graph_list[i].neighbor_size * sizeof(uint64_t)));
      }

      CUDA_CHECK(
          cudaMemcpyAsync(gpu_graph_list_[offset].neighbor_list,
                          cpu_graph_list[i].neighbor_list,
                          cpu_graph_list[i].neighbor_size * sizeof(uint64_t),
                          cudaMemcpyHostToDevice,
                          stream));
      gpu_graph_list_[offset].neighbor_size = cpu_graph_list[i].neighbor_size;
    } else {
      gpu_graph_list_[offset].neighbor_list = NULL;
      gpu_graph_list_[offset].neighbor_size = 0;
    }
    cudaStreamSynchronize(stream);
  }
}

NeighborSampleResult GpuPsGraphTable::graph_neighbor_sample_v3(
    NeighborSampleQuery q, bool cpu_switch, bool compress, bool weighted) {
  if (multi_node_ && FLAGS_enable_graph_multi_node_sampling) {
    // multi node mode
    if (q.sample_step == 1) {
      auto result = graph_neighbor_sample_v2(global_device_map[q.gpu_id],
                                             q.table_idx,
                                             q.src_nodes,
                                             q.sample_size,
                                             q.len,
                                             q.neighbor_size_limit,
                                             cpu_switch,
                                             compress,
                                             weighted);
      return result;
    } else {
      auto result = graph_neighbor_sample_all2all(global_device_map[q.gpu_id],
                                                  q.sample_step,
                                                  q.table_idx,
                                                  q.src_nodes,
                                                  q.sample_size,
                                                  q.len,
                                                  q.neighbor_size_limit,
                                                  cpu_switch,
                                                  compress,
                                                  weighted);
      return result;
    }
  } else {
    // single node mode
    auto result = graph_neighbor_sample_v2(global_device_map[q.gpu_id],
                                           q.table_idx,
                                           q.src_nodes,
                                           q.sample_size,
                                           q.len,
                                           q.neighbor_size_limit,
                                           cpu_switch,
                                           compress,
                                           weighted);
    return result;
  }
}

NeighborSampleResult GpuPsGraphTable::graph_neighbor_sample(
    int gpu_id,
    uint64_t* key,
    int sample_size,
    int len,
    int neighbor_size_limit) {
  return graph_neighbor_sample_v2(gpu_id,
                                  0,
                                  key,
                                  sample_size,
                                  len,
                                  neighbor_size_limit,
                                  false,
                                  true,
                                  false);
}

NeighborSampleResult GpuPsGraphTable::graph_neighbor_sample_all2all(
    int gpu_id,
    int sample_step,
    int table_idx,
    uint64_t* d_keys,
    int sample_size,
    int len,
    int neighbor_size_limit,
    bool cpu_query_switch,
    bool compress,
    bool weighted) {
  platform::CUDADeviceGuard guard(gpu_id);
  auto& loc = storage_[gpu_id];
  auto stream = resource_->local_stream(gpu_id, 0);

  loc.alloc(len, sizeof(uint64_t) /*value_bytes*/);

  // all2all mode begins. init resource, partition keys, pull vals by all2all
  auto pull_size = gather_inter_keys_by_all2all(gpu_id, len, d_keys, stream);
  VLOG(2) << "gather_inter_keys_by_all2all finish, pull_size=" << pull_size
          << ", len=" << len;

  // do single-node multi-card sampling
  auto result = graph_neighbor_sample_v2(gpu_id,
                                         table_idx,
                                         loc.d_merged_keys,
                                         sample_size,
                                         pull_size,
                                         neighbor_size_limit,
                                         cpu_query_switch,
                                         compress,
                                         weighted);
  VLOG(2) << "graph_neighbor_sample_v2 local finish"
          << ", gpu_id=" << gpu_id << ", pull_size=" << pull_size
          << ", total_sample_size=" << result.total_sample_size;

  // init neighbor result
  NeighborSampleResult final;
  final.set_stream(stream);
  final.initialize(sample_size, len, gpu_id);

  // all2all mode finish, scatter sample values by all2all
  scatter_inter_vals_by_all2all_common(
      gpu_id,
      len,
      sizeof(uint64_t),                                // value_bytes
      reinterpret_cast<const uint64_t*>(result.val),   // in
      reinterpret_cast<uint64_t*>(final.val),          // out
      reinterpret_cast<uint64_t*>(loc.d_merged_vals),  // tmp hbm
      stream);
  VLOG(2) << "scatter_inter_vals_by_all2all val finish"
          << " gpu_id=" << gpu_id;

  // all2all mode finish, scatter sample sizes of every node by all2all
  scatter_inter_vals_by_all2all_common(
      gpu_id,
      len,
      sizeof(int),                                              // value_bytes
      reinterpret_cast<const int*>(result.actual_sample_size),  // in
      reinterpret_cast<int*>(final.actual_sample_size),         // out
      reinterpret_cast<int*>(loc.d_merged_vals),                // temp hbm
      stream);
  VLOG(2) << "scatter_inter_vals_by_all2all actual_sample_size finish"
          << " gpu_id=" << gpu_id;

  // build final.actual_val
  if (compress) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    phi::GPUPlace place = phi::GPUPlace(resource_->dev_id(gpu_id));
    platform::CUDADeviceGuard guard(resource_->dev_id(gpu_id));
    size_t temp_storage_bytes = 0;
    int total_sample_size = 0;
    auto cumsum_actual_sample_size =
        memory::Alloc(place,
                      (len + 1) * sizeof(int),
                      phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
    int* cumsum_actual_sample_size_p =
        reinterpret_cast<int*>(cumsum_actual_sample_size->ptr());
    CUDA_CHECK(
        cudaMemsetAsync(cumsum_actual_sample_size_p, 0, sizeof(int), stream));
    // VLOG(0) << "InclusiveSum begin";
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(NULL,
                                             temp_storage_bytes,
                                             final.actual_sample_size,
                                             cumsum_actual_sample_size_p + 1,
                                             len,
                                             stream));
    auto d_temp_storage =
        memory::Alloc(place,
                      temp_storage_bytes,
                      phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(d_temp_storage->ptr(),
                                             temp_storage_bytes,
                                             final.actual_sample_size,
                                             cumsum_actual_sample_size_p + 1,
                                             len,
                                             stream));
    CUDA_CHECK(cudaMemcpyAsync(&total_sample_size,
                               cumsum_actual_sample_size_p + len,
                               sizeof(int),
                               cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    final.set_total_sample_size(total_sample_size);

    final.actual_val_mem = memory::AllocShared(
        place,
        total_sample_size * sizeof(uint64_t),
        phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
    final.actual_val =
        reinterpret_cast<uint64_t*>((final.actual_val_mem)->ptr());

    VLOG(2) << "sample step:" << sample_step
            << ", total_sample_size:" << total_sample_size << ", len=" << len
            << ", final.val=" << final.val
            << ", final.actual_val=" << final.actual_val
            << ", final.actual_sample_size=" << final.actual_sample_size;

    int grid_size = (len - 1) / block_size_ + 1;
    fill_actual_vals<<<grid_size, block_size_, 0, stream>>>(
        final.val,
        final.actual_val,
        final.actual_sample_size,
        cumsum_actual_sample_size_p,
        sample_size,
        len);
    CUDA_CHECK(cudaStreamSynchronize(stream));  // hbm safe
  }

  return final;
}

NeighborSampleResult GpuPsGraphTable::graph_neighbor_sample_v2(
    int gpu_id,
    int idx,
    uint64_t* key,
    int sample_size,
    int len,
    int neighbor_size_limit,
    bool cpu_query_switch,
    bool compress,
    bool weighted) {
  NeighborSampleResult result;
  auto stream = resource_->local_stream(gpu_id, 0);
  result.set_stream(stream);
  result.initialize(sample_size, len, resource_->dev_id(gpu_id));

  if (len == 0) {
    return result;
  }

  phi::GPUPlace place = phi::GPUPlace(resource_->dev_id(gpu_id));
  platform::CUDADeviceGuard guard(resource_->dev_id(gpu_id));

  int* actual_sample_size = result.actual_sample_size;
  uint64_t* val = result.val;
  int total_gpu = resource_->total_device();

  int grid_size = (len - 1) / block_size_ + 1;

  int h_left[total_gpu];   // NOLINT
  int h_right[total_gpu];  // NOLINT

  auto d_left =
      memory::Alloc(place,
                    total_gpu * sizeof(int),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  auto d_right =
      memory::Alloc(place,
                    total_gpu * sizeof(int),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  int* d_left_ptr = reinterpret_cast<int*>(d_left->ptr());
  int* d_right_ptr = reinterpret_cast<int*>(d_right->ptr());
  int default_value = 0;
  if (cpu_query_switch) {
    default_value = -1;
  }

  CUDA_CHECK(cudaMemsetAsync(d_left_ptr, -1, total_gpu * sizeof(int), stream));
  CUDA_CHECK(cudaMemsetAsync(d_right_ptr, -1, total_gpu * sizeof(int), stream));
  //
  auto d_idx =
      memory::Alloc(place,
                    len * sizeof(int),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  int* d_idx_ptr = reinterpret_cast<int*>(d_idx->ptr());

  auto d_shard_keys =
      memory::Alloc(place,
                    len * sizeof(uint64_t),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  uint64_t* d_shard_keys_ptr = reinterpret_cast<uint64_t*>(d_shard_keys->ptr());
  auto d_shard_vals =
      memory::Alloc(place,
                    sample_size * len * sizeof(uint64_t),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  uint64_t* d_shard_vals_ptr = reinterpret_cast<uint64_t*>(d_shard_vals->ptr());
  auto d_shard_actual_sample_size =
      memory::Alloc(place,
                    len * sizeof(int),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  int* d_shard_actual_sample_size_ptr =
      reinterpret_cast<int*>(d_shard_actual_sample_size->ptr());

  split_idx_to_shard(reinterpret_cast<uint64_t*>(key),
                     d_idx_ptr,
                     len,
                     d_left_ptr,
                     d_right_ptr,
                     gpu_id,
                     stream);

  heter_comm_kernel_->fill_shard_key(
      d_shard_keys_ptr, key, d_idx_ptr, len, stream, gpu_id);

  CUDA_CHECK(cudaMemcpyAsync(h_left,
                             d_left_ptr,
                             total_gpu * sizeof(int),
                             cudaMemcpyDeviceToHost,
                             stream));
  CUDA_CHECK(cudaMemcpyAsync(h_right,
                             d_right_ptr,
                             total_gpu * sizeof(int),
                             cudaMemcpyDeviceToHost,
                             stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  device_mutex_[gpu_id]->lock();
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
  walk_to_dest(gpu_id,
               total_gpu,
               h_left,
               h_right,
               reinterpret_cast<uint64_t*>(d_shard_keys_ptr),
               NULL);

  for (int i = 0; i < total_gpu; ++i) {
    if (h_left[i] == -1) {
      continue;
    }
    int shard_len = h_left[i] == -1 ? 0 : h_right[i] - h_left[i] + 1;
    auto& node = path_[gpu_id][i].nodes_.back();
    //    CUDA_CHECK(cudaStreamSynchronize(node.in_stream));
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    auto cur_stream = resource_->remote_stream(i, gpu_id);
    CUDA_CHECK(cudaMemsetAsync(
        node.val_storage, 0, shard_len * sizeof(uint64_t), cur_stream));
    // If not found, val is -1.
    int table_offset = get_table_offset(i, GraphTableType::EDGE_TABLE, idx);
    int offset = get_graph_list_offset(i, idx);
    tables_[table_offset]->get(reinterpret_cast<uint64_t*>(node.key_storage),
                               reinterpret_cast<uint64_t*>(node.val_storage),
                               static_cast<size_t>(h_right[i] - h_left[i] + 1),
                               cur_stream);

    auto graph = gpu_graph_list_[offset];
    GpuPsNodeInfo* node_info_list =
        reinterpret_cast<GpuPsNodeInfo*>(node.val_storage);
    int* actual_size_array = reinterpret_cast<int*>(node_info_list + shard_len);
    uint64_t* sample_array = reinterpret_cast<uint64_t*>(
        actual_size_array + shard_len + shard_len % 2);

    if (!weighted) {
      constexpr int WARP_SIZE = 32;
      constexpr int BLOCK_WARPS = 128 / WARP_SIZE;
      constexpr int TILE_SIZE = BLOCK_WARPS * 16;
      const dim3 block(WARP_SIZE, BLOCK_WARPS);
      const dim3 grid((shard_len + TILE_SIZE - 1) / TILE_SIZE);
      neighbor_sample_kernel_walking<WARP_SIZE, BLOCK_WARPS, TILE_SIZE>
          <<<grid, block, 0, cur_stream>>>(graph,
                                           node_info_list,
                                           actual_size_array,
                                           sample_array,
                                           sample_size,
                                           shard_len,
                                           neighbor_size_limit,
                                           default_value);
    } else {
      // Weighted sample.
      thread_local std::random_device rd;
      thread_local std::mt19937 gen(rd());
      thread_local std::uniform_int_distribution<uint64_t> distrib;
      uint64_t random_seed = distrib(gen);

      const bool need_neighbor_count = sample_size > SAMPLE_SIZE_THRESHOLD;
      int* neighbor_count_ptr = nullptr;
      std::shared_ptr<phi::Allocation> neighbor_count;
      if (need_neighbor_count) {
        neighbor_count = memory::AllocShared(
            place,
            (shard_len + 1) * sizeof(int),
            phi::Stream(reinterpret_cast<phi::StreamId>(cur_stream)));
        neighbor_count_ptr = reinterpret_cast<int*>(neighbor_count->ptr());
      }

      PADDLE_ENFORCE_GT(sample_size,
                        0,
                        common::errors::InvalidArgument(
                            "sample_size should be greater than 0."));
      weighted_sample(graph,
                      node_info_list,
                      actual_size_array,
                      sample_array,
                      neighbor_count_ptr,
                      gpu_id,
                      i,
                      sample_size,
                      shard_len,
                      need_neighbor_count,
                      random_seed,
                      nullptr,
                      false);
    }
  }

  for (int i = 0; i < total_gpu; ++i) {
    if (h_left[i] == -1) {
      continue;
    }
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    CUDA_CHECK(cudaStreamSynchronize(resource_->remote_stream(i, gpu_id)));
  }

  move_result_to_source_gpu(gpu_id,
                            total_gpu,
                            sample_size,
                            h_left,
                            h_right,
                            d_shard_vals_ptr,
                            d_shard_actual_sample_size_ptr);

  for (int i = 0; i < total_gpu; ++i) {
    int shard_len = h_left[i] == -1 ? 0 : h_right[i] - h_left[i] + 1;
    if (shard_len == 0) {
      continue;
    }
    destroy_storage(gpu_id, i);
  }
  device_mutex_[gpu_id]->unlock();

  platform::CUDADeviceGuard guard2(resource_->dev_id(gpu_id));
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
        memcpy(merge_buffers + start,
               reinterpret_cast<uint64_t*>(buffers[j].get()),
               ac[j]);
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
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  if (compress) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    phi::GPUPlace place = phi::GPUPlace(resource_->dev_id(gpu_id));
    platform::CUDADeviceGuard guard(resource_->dev_id(gpu_id));
    size_t temp_storage_bytes = 0;
    int total_sample_size = 0;
    auto cumsum_actual_sample_size =
        memory::Alloc(place,
                      (len + 1) * sizeof(int),
                      phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
    int* cumsum_actual_sample_size_p =
        reinterpret_cast<int*>(cumsum_actual_sample_size->ptr());
    CUDA_CHECK(
        cudaMemsetAsync(cumsum_actual_sample_size_p, 0, sizeof(int), stream));
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(NULL,
                                             temp_storage_bytes,
                                             actual_sample_size,
                                             cumsum_actual_sample_size_p + 1,
                                             len,
                                             stream));
    auto d_temp_storage =
        memory::Alloc(place,
                      temp_storage_bytes,
                      phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(d_temp_storage->ptr(),
                                             temp_storage_bytes,
                                             actual_sample_size,
                                             cumsum_actual_sample_size_p + 1,
                                             len,
                                             stream));
    CUDA_CHECK(cudaMemcpyAsync(&total_sample_size,
                               cumsum_actual_sample_size_p + len,
                               sizeof(int),
                               cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    result.actual_val_mem = memory::AllocShared(
        place,
        total_sample_size * sizeof(uint64_t),
        phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
    result.actual_val =
        reinterpret_cast<uint64_t*>((result.actual_val_mem)->ptr());

    result.set_total_sample_size(total_sample_size);
    fill_actual_vals<<<grid_size, block_size_, 0, stream>>>(
        val,
        result.actual_val,
        actual_sample_size,
        cumsum_actual_sample_size_p,
        sample_size,
        len);
    CUDA_CHECK(cudaStreamSynchronize(stream));  // hbm safe
  }

  cudaStreamSynchronize(stream);
  return result;
}

NeighborSampleResultV2 GpuPsGraphTable::graph_neighbor_sample_sage(
    int gpu_id,
    int edge_type_len,
    const uint64_t* d_keys,
    int sample_size,
    int len,
    std::vector<std::shared_ptr<phi::Allocation>> edge_type_graphs,
    bool weighted,
    bool return_weight) {
  if (multi_node_ && FLAGS_enable_graph_multi_node_sampling) {
    // multi node mode
    auto result = graph_neighbor_sample_sage_all2all(gpu_id,
                                                     edge_type_len,
                                                     d_keys,
                                                     sample_size,
                                                     len,
                                                     edge_type_graphs,
                                                     weighted,
                                                     return_weight);
    return result;
  } else {
    auto result = graph_neighbor_sample_all_edge_type(gpu_id,
                                                      edge_type_len,
                                                      d_keys,
                                                      sample_size,
                                                      len,
                                                      edge_type_graphs,
                                                      weighted,
                                                      return_weight);
    return result;
  }
}

__global__ void rearange_neighbor_result(uint64_t* val,
                                         uint64_t* new_val,
                                         int* ac,
                                         int* new_ac,
                                         float* weight,
                                         float* new_weight,
                                         int sample_size,
                                         int len,
                                         int edge_type_len,
                                         int n,
                                         bool return_weight) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int pos = (i % edge_type_len) * len + static_cast<int>(i) / edge_type_len;
    new_ac[pos] = ac[i];
    size_t offset1 = pos * sample_size, offset2 = i * sample_size;
    for (int j = 0; j < ac[i]; j++) {
      new_val[offset1 + j] = val[offset2 + j];
      if (return_weight) {
        new_weight[offset1 + j] = weight[offset2 + j];
      }
    }
  }
}

NeighborSampleResultV2 GpuPsGraphTable::graph_neighbor_sample_sage_all2all(
    int gpu_id,
    int edge_type_len,
    const uint64_t* d_keys,
    int sample_size,
    int len,
    std::vector<std::shared_ptr<phi::Allocation>> edge_type_graphs,
    bool weighted,
    bool return_weight) {
  platform::CUDADeviceGuard guard(gpu_id);
  auto& loc = storage_[gpu_id];
  auto stream = resource_->local_stream(gpu_id, 0);

  loc.alloc(len, sizeof(uint64_t) * edge_type_len * sample_size);  // key_bytes

  // all2all mode begins, init resource, partition keys, pull vals by all2all.
  auto pull_size = gather_inter_keys_by_all2all(gpu_id, len, d_keys, stream);
  VLOG(2) << "gather_inter_keys_by_all2all sage finish, pull_size=" << pull_size
          << ", len=" << len;

  // do single-node multi-card sampling
  auto result = graph_neighbor_sample_all_edge_type(gpu_id,
                                                    edge_type_len,
                                                    loc.d_merged_keys,
                                                    sample_size,
                                                    pull_size,
                                                    edge_type_graphs,
                                                    weighted,
                                                    return_weight,
                                                    true);

  VLOG(2) << "graph_neighbor_sample_all_edge_type local finish"
          << ", gpu_id=" << gpu_id << ", pull_size=" << pull_size;

  // init neighbor result
  NeighborSampleResultV2 final;
  final.set_stream(stream);
  final.initialize(sample_size, len, edge_type_len, return_weight, gpu_id);

  VLOG(2) << "Begin scatter_inter_vals_by_all2all_common for val";
  // all2all mode finish, scatter sample values by all2all
  scatter_inter_vals_by_all2all_common(
      gpu_id,
      len,
      sizeof(uint64_t) * edge_type_len * sample_size,  // value_bytes
      reinterpret_cast<const uint64_t*>(result.val),   // in
      reinterpret_cast<uint64_t*>(final.val),          // out
      reinterpret_cast<uint64_t*>(loc.d_merged_vals),  // tmp hbm
      stream,
      true);
  VLOG(2) << "scatter_inter_vals_by_all2all sage val finish"
          << " gpu_id=" << gpu_id;

  // all2all mode finish, scatter sample sizes of every node by all2all
  scatter_inter_vals_by_all2all_common(
      gpu_id,
      len,
      sizeof(int) * edge_type_len,                              // value_bytes
      reinterpret_cast<const int*>(result.actual_sample_size),  // in
      reinterpret_cast<int*>(final.actual_sample_size),         // out
      reinterpret_cast<int*>(loc.d_merged_vals),                // tmp hbm
      stream,
      true);
  VLOG(2) << "scatter_inter_vals_by_all2all sage actual_sample_size finish"
          << " gpu_id=" << gpu_id;

  if (return_weight) {
    scatter_inter_vals_by_all2all_common(
        gpu_id,
        len,
        sizeof(float) * edge_type_len * sample_size,    // value_bytes
        reinterpret_cast<const float*>(result.weight),  // in
        reinterpret_cast<float*>(final.weight),         // out
        reinterpret_cast<float*>(loc.d_merged_vals),    // tmp hbm
        stream,
        true);
    VLOG(2) << "scatter_inter_vals_by_all2all sage weight finish"
            << " gpu_id=" << gpu_id;
  }

  // Rearange neighbor result.
  NeighborSampleResultV2 final2;
  final2.set_stream(stream);
  final2.initialize(sample_size, len, edge_type_len, return_weight, gpu_id);
  int grid_size_e = (len * edge_type_len - 1) / block_size_ + 1;
  rearange_neighbor_result<<<grid_size_e, block_size_, 0, stream>>>(
      reinterpret_cast<uint64_t*>(final.val),
      reinterpret_cast<uint64_t*>(final2.val),
      reinterpret_cast<int*>(final.actual_sample_size),
      reinterpret_cast<int*>(final2.actual_sample_size),
      reinterpret_cast<float*>(final.weight),
      reinterpret_cast<float*>(final2.weight),
      sample_size,
      len,
      edge_type_len,
      len * edge_type_len,
      return_weight);

  return final2;
}

// only for graphsage
NeighborSampleResultV2 GpuPsGraphTable::graph_neighbor_sample_all_edge_type(
    int gpu_id,
    int edge_type_len,
    const uint64_t* d_keys,
    int sample_size,
    int len,
    std::vector<std::shared_ptr<phi::Allocation>> edge_type_graphs,
    bool weighted,
    bool return_weight,
    bool for_all2all) {
  NeighborSampleResultV2 result;
  auto stream = resource_->local_stream(gpu_id, 0);
  result.set_stream(stream);
  result.initialize(sample_size,
                    len,
                    edge_type_len,
                    return_weight,
                    resource_->dev_id(gpu_id));
  if (len == 0) {
    return result;
  }

  phi::GPUPlace place = phi::GPUPlace(resource_->dev_id(gpu_id));
  platform::CUDADeviceGuard guard(resource_->dev_id(gpu_id));

  int* actual_sample_size = result.actual_sample_size;
  uint64_t* val = result.val;
  float* weight = result.weight;
  int total_gpu = resource_->total_device();

  int grid_size = (len - 1) / block_size_ + 1;
  int h_left[total_gpu];   // NOLINT
  int h_right[total_gpu];  // NOLINT
  auto d_left =
      memory::Alloc(place,
                    total_gpu * sizeof(int),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  auto d_right =
      memory::Alloc(place,
                    total_gpu * sizeof(int),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  int* d_left_ptr = reinterpret_cast<int*>(d_left->ptr());
  int* d_right_ptr = reinterpret_cast<int*>(d_right->ptr());
  int default_value = 0;
  CUDA_CHECK(cudaMemsetAsync(d_left_ptr, -1, total_gpu * sizeof(int), stream));
  CUDA_CHECK(cudaMemsetAsync(d_right_ptr, -1, total_gpu * sizeof(int), stream));

  auto d_idx =
      memory::Alloc(place,
                    len * sizeof(int),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  int* d_idx_ptr = reinterpret_cast<int*>(d_idx->ptr());
  auto d_shard_keys =
      memory::Alloc(place,
                    len * sizeof(uint64_t),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  uint64_t* d_shard_keys_ptr = reinterpret_cast<uint64_t*>(d_shard_keys->ptr());
  auto d_shard_vals =
      memory::Alloc(place,
                    sample_size * len * edge_type_len * sizeof(uint64_t),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  uint64_t* d_shard_vals_ptr = reinterpret_cast<uint64_t*>(d_shard_vals->ptr());
  auto d_shard_actual_sample_size =
      memory::Alloc(place,
                    len * edge_type_len * sizeof(int),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  int* d_shard_actual_sample_size_ptr =
      reinterpret_cast<int*>(d_shard_actual_sample_size->ptr());

  float* d_shard_weight_ptr = nullptr;
  std::shared_ptr<phi::Allocation> d_shard_weight;
  if (return_weight) {
    d_shard_weight = memory::AllocShared(
        place,
        sample_size * len * edge_type_len * sizeof(float),
        phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
    d_shard_weight_ptr = reinterpret_cast<float*>(d_shard_weight->ptr());
  }

  split_idx_to_shard(const_cast<uint64_t*>(d_keys),
                     d_idx_ptr,
                     len,
                     d_left_ptr,
                     d_right_ptr,
                     gpu_id,
                     stream);

  heter_comm_kernel_->fill_shard_key(d_shard_keys_ptr,
                                     const_cast<uint64_t*>(d_keys),
                                     d_idx_ptr,
                                     len,
                                     stream,
                                     gpu_id);

  CUDA_CHECK(cudaMemcpyAsync(h_left,
                             d_left_ptr,
                             total_gpu * sizeof(int),
                             cudaMemcpyDeviceToHost,
                             stream));
  CUDA_CHECK(cudaMemcpyAsync(h_right,
                             d_right_ptr,
                             total_gpu * sizeof(int),
                             cudaMemcpyDeviceToHost,
                             stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  device_mutex_[gpu_id]->lock();
  for (int i = 0; i < total_gpu; ++i) {
    int shard_len = h_left[i] == -1 ? 0 : h_right[i] - h_left[i] + 1;
    if (shard_len == 0) {
      continue;
    }
    if (!return_weight) {
      create_storage(
          gpu_id,
          i,
          shard_len * sizeof(uint64_t),
          shard_len * sizeof(uint64_t) * edge_type_len +  // key
              (shard_len * sample_size * sizeof(uint64_t)) *
                  edge_type_len +                        // sample
              shard_len * sizeof(int) * edge_type_len +  // actual sample size
              ((shard_len * edge_type_len) % 2) * sizeof(int));  // align
    } else {
      create_storage(
          gpu_id,
          i,
          shard_len * sizeof(uint64_t),
          shard_len * sizeof(uint64_t) * edge_type_len +  // key
              shard_len * sample_size * sizeof(uint64_t) *
                  edge_type_len +                        // sample
              shard_len * sizeof(int) * edge_type_len +  // actual sample size
              shard_len * sample_size * sizeof(float) *
                  edge_type_len +  // edge weight
              sizeof(int) * ((shard_len * edge_type_len * (1 + sample_size)) %
                             2));  // align, sizeof(int) == sizeof(float)
    }
  }
  walk_to_dest(gpu_id,
               total_gpu,
               h_left,
               h_right,
               reinterpret_cast<uint64_t*>(d_shard_keys_ptr),
               NULL);

  for (int i = 0; i < total_gpu; ++i) {
    if (h_left[i] == -1) {
      continue;
    }
    int shard_len = h_left[i] == -1 ? 0 : h_right[i] - h_left[i] + 1;
    auto& node = path_[gpu_id][i].nodes_.back();
    //    CUDA_CHECK(cudaStreamSynchronize(node.in_stream));
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    auto cur_stream = resource_->remote_stream(i, gpu_id);
    CUDA_CHECK(cudaMemsetAsync(node.val_storage,
                               0,
                               shard_len * edge_type_len * sizeof(uint64_t),
                               cur_stream));
    GpuPsNodeInfo* node_info_base =
        reinterpret_cast<GpuPsNodeInfo*>(node.val_storage);
    for (int idx = 0; idx < edge_type_len; idx++) {
      int table_offset = get_table_offset(i, GraphTableType::EDGE_TABLE, idx);
      int offset = get_graph_list_offset(i, idx);
      if (tables_[table_offset] == NULL) {
        continue;
      }
      tables_[table_offset]->get(
          reinterpret_cast<uint64_t*>(node.key_storage),
          reinterpret_cast<uint64_t*>(node_info_base + idx * shard_len),
          static_cast<size_t>(shard_len),
          cur_stream);
    }

    auto d_commgraph_mem = edge_type_graphs[i];
    GpuPsCommGraph* d_commgraph_ptr =
        reinterpret_cast<GpuPsCommGraph*>(d_commgraph_mem->ptr());
    int* actual_size_base =
        reinterpret_cast<int*>(node_info_base + shard_len * edge_type_len);

    float* weight_array_base = nullptr;
    uint64_t* sample_array_base = nullptr;
    if (return_weight) {
      weight_array_base = reinterpret_cast<float*>(actual_size_base +
                                                   shard_len * edge_type_len);
      sample_array_base = reinterpret_cast<uint64_t*>(
          weight_array_base + shard_len * edge_type_len * sample_size +
          (((shard_len * edge_type_len) * (1 + sample_size)) % 2));
    } else {
      sample_array_base = reinterpret_cast<uint64_t*>(
          actual_size_base + shard_len * edge_type_len +
          (shard_len * edge_type_len) % 2);
    }

    thread_local std::random_device rd;
    thread_local std::mt19937 gen(rd());
    thread_local std::uniform_int_distribution<uint64_t> distrib;
    uint64_t random_seed = distrib(gen);
    PADDLE_ENFORCE_GT(sample_size,
                      0,
                      common::errors::InvalidArgument(
                          "sample_size should be greater than 0."));

    if (!weighted) {
      /*int grid_size_ = (shard_len * edge_type_len - 1) / block_size_ + 1;
      neighbor_sample_kernel_all_edge_type<<<grid_size_,
                                             block_size_,
                                             0,
                                             cur_stream>>>(
          d_commgraph_ptr,
          node_info_base,
          actual_size_base,
          sample_array_base,
          weight_array_base,
          sample_size,
          shard_len * edge_type_len,
          default_value,
          shard_len,
          return_weight);*/
      for (int edge_idx = 0; edge_idx < edge_type_len; edge_idx++) {
        GpuPsNodeInfo* node_info_list = node_info_base + edge_idx * shard_len;
        int* actual_size_array = actual_size_base + edge_idx * shard_len;
        uint64_t* sample_array =
            sample_array_base + edge_idx * shard_len * sample_size;
        float* weight_array = nullptr;
        if (return_weight) {
          weight_array = weight_array_base + edge_idx * shard_len * sample_size;
        }
        int offset = get_graph_list_offset(i, edge_idx);
        auto graph = gpu_graph_list_[offset];
        unweighted_sample(graph,
                          node_info_list,
                          actual_size_array,
                          sample_array,
                          gpu_id,
                          i,
                          sample_size,
                          shard_len,
                          random_seed,
                          weight_array,
                          return_weight);
      }
    } else {
      // Weighted sample.
      const bool need_neighbor_count = sample_size > SAMPLE_SIZE_THRESHOLD;
      int* neighbor_count_ptr = nullptr;
      std::shared_ptr<phi::Allocation> neighbor_count;
      if (need_neighbor_count) {
        neighbor_count = memory::AllocShared(
            place,
            (shard_len + 1) * sizeof(int),
            phi::Stream(reinterpret_cast<phi::StreamId>(cur_stream)));
        neighbor_count_ptr = reinterpret_cast<int*>(neighbor_count->ptr());
      }
      for (int edge_idx = 0; edge_idx < edge_type_len; edge_idx++) {
        GpuPsNodeInfo* node_info_list = node_info_base + edge_idx * shard_len;
        int* actual_size_array = actual_size_base + edge_idx * shard_len;
        uint64_t* sample_array =
            sample_array_base + edge_idx * shard_len * sample_size;
        float* weight_array = nullptr;
        if (return_weight) {
          weight_array = weight_array_base + edge_idx * shard_len * sample_size;
        }
        int offset = get_graph_list_offset(i, edge_idx);
        auto graph = gpu_graph_list_[offset];

        weighted_sample(graph,
                        node_info_list,
                        actual_size_array,
                        sample_array,
                        neighbor_count_ptr,
                        gpu_id,
                        i,
                        sample_size,
                        shard_len,
                        need_neighbor_count,
                        random_seed,
                        weight_array,
                        return_weight);
      }
    }
  }

  for (int i = 0; i < total_gpu; ++i) {
    if (h_left[i] == -1) {
      continue;
    }
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    CUDA_CHECK(cudaStreamSynchronize(resource_->remote_stream(i, gpu_id)));
  }

  move_result_to_source_gpu_all_edge_type(gpu_id,
                                          total_gpu,
                                          sample_size,
                                          h_left,
                                          h_right,
                                          d_shard_vals_ptr,
                                          d_shard_actual_sample_size_ptr,
                                          d_shard_weight_ptr,
                                          edge_type_len,
                                          len,
                                          return_weight);

  int grid_size_e = (len * edge_type_len - 1) / block_size_ + 1;
  platform::CUDADeviceGuard guard2(resource_->dev_id(gpu_id));
  if (!for_all2all) {
    // vals: [e1, e2, e3], where e1 means all the sample res of len for
    // edge_type 1.
    fill_dvalues_with_edge_type<<<grid_size_e, block_size_, 0, stream>>>(
        d_shard_vals_ptr,
        val,
        d_shard_actual_sample_size_ptr,
        actual_sample_size,
        d_shard_weight_ptr,
        weight,
        d_idx_ptr,
        sample_size,
        len * edge_type_len,
        len,
        return_weight);
  } else {
    // different fill mode for all2all
    // vals: [node1, node2, node3], where node1 means all the sample res of
    // node1, including all edge_types.
    fill_dvalues_with_edge_type_for_all2all<<<grid_size_e,
                                              block_size_,
                                              0,
                                              stream>>>(
        d_shard_vals_ptr,
        val,
        d_shard_actual_sample_size_ptr,
        actual_sample_size,
        d_shard_weight_ptr,
        weight,
        d_idx_ptr,
        sample_size,
        len * edge_type_len,
        len,
        edge_type_len,
        return_weight);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  for (int i = 0; i < total_gpu; i++) {
    int shard_len = h_left[i] == -1 ? 0 : h_right[i] - h_left[i] + 1;
    if (shard_len == 0) {
      continue;
    }
    destroy_storage(gpu_id, i);
  }
  device_mutex_[gpu_id]->unlock();
  return result;
}

std::shared_ptr<phi::Allocation> GpuPsGraphTable::get_node_degree(int gpu_id,
                                                                  int edge_idx,
                                                                  uint64_t* key,
                                                                  int len) {
  if (multi_node_ && FLAGS_enable_graph_multi_node_sampling) {
    // multi node mode
    auto node_degree = get_node_degree_all2all(gpu_id, edge_idx, key, len);
    return node_degree;
  } else {
    auto node_degree = get_node_degree_single(gpu_id, edge_idx, key, len);
    return node_degree;
  }
}

std::shared_ptr<phi::Allocation> GpuPsGraphTable::get_node_degree_all2all(
    int gpu_id, int edge_idx, uint64_t* key, int len) {
  platform::CUDADeviceGuard guard(gpu_id);
  auto& loc = storage_[gpu_id];
  phi::GPUPlace place = phi::GPUPlace(resource_->dev_id(gpu_id));
  auto stream = resource_->local_stream(gpu_id, 0);

  loc.alloc(len, sizeof(int));

  // all2all mode begins, init resource, partition keys, pull vals by all2all.

  auto pull_size = gather_inter_keys_by_all2all(gpu_id, len, key, stream);
  VLOG(2) << "gather_inter_keys_by_all2all sage get_degree finish, pull_size="
          << pull_size << ", len=" << len;

  // do single-node multi-card get_node_degree
  auto result =
      get_node_degree_single(gpu_id, edge_idx, loc.d_merged_keys, pull_size);

  auto node_degree =
      memory::AllocShared(place,
                          len * sizeof(int),
                          phi::Stream(reinterpret_cast<phi::StreamId>(stream)));

  // all2all mode finish, scatter degree values by all2all
  scatter_inter_vals_by_all2all_common(
      gpu_id,
      len,
      sizeof(int),                                  // value_bytes
      reinterpret_cast<const int*>(result->ptr()),  // in
      reinterpret_cast<int*>(node_degree->ptr()),   // out
      reinterpret_cast<int*>(loc.d_merged_vals),    // tmp hbm
      stream);
  return node_degree;
}

std::shared_ptr<phi::Allocation> GpuPsGraphTable::get_node_degree_single(
    int gpu_id, int edge_idx, uint64_t* key, int len) {
  int total_gpu = resource_->total_device();
  phi::GPUPlace place = phi::GPUPlace(resource_->dev_id(gpu_id));
  platform::CUDADeviceGuard guard(resource_->dev_id(gpu_id));
  auto stream = resource_->local_stream(gpu_id, 0);

  auto node_degree =
      memory::AllocShared(place,
                          len * sizeof(int),
                          phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  int* node_degree_ptr = reinterpret_cast<int*>(node_degree->ptr());

  int grid_size = (len - 1) / block_size_ + 1;
  int h_left[total_gpu];   // NOLINT
  int h_right[total_gpu];  // NOLINT
  auto d_left =
      memory::Alloc(place,
                    total_gpu * sizeof(int),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  auto d_right =
      memory::Alloc(place,
                    total_gpu * sizeof(int),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  int* d_left_ptr = reinterpret_cast<int*>(d_left->ptr());
  int* d_right_ptr = reinterpret_cast<int*>(d_right->ptr());
  CUDA_CHECK(cudaMemsetAsync(d_left_ptr, -1, total_gpu * sizeof(int), stream));
  CUDA_CHECK(cudaMemsetAsync(d_right_ptr, -1, total_gpu * sizeof(int), stream));
  auto d_idx =
      memory::Alloc(place,
                    len * sizeof(int),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  int* d_idx_ptr = reinterpret_cast<int*>(d_idx->ptr());
  auto d_shard_keys =
      memory::Alloc(place,
                    len * sizeof(uint64_t),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  uint64_t* d_shard_keys_ptr = reinterpret_cast<uint64_t*>(d_shard_keys->ptr());
  auto d_shard_degree =
      memory::Alloc(place,
                    len * sizeof(int),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  int* d_shard_degree_ptr = reinterpret_cast<int*>(d_shard_degree->ptr());
  split_idx_to_shard(reinterpret_cast<uint64_t*>(key),
                     d_idx_ptr,
                     len,
                     d_left_ptr,
                     d_right_ptr,
                     gpu_id,
                     stream);
  heter_comm_kernel_->fill_shard_key(
      d_shard_keys_ptr, key, d_idx_ptr, len, stream, gpu_id);
  CUDA_CHECK(cudaMemcpyAsync(h_left,
                             d_left_ptr,
                             total_gpu * sizeof(int),
                             cudaMemcpyDeviceToHost,
                             stream));
  CUDA_CHECK(cudaMemcpyAsync(h_right,
                             d_right_ptr,
                             total_gpu * sizeof(int),
                             cudaMemcpyDeviceToHost,
                             stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  device_mutex_[gpu_id]->lock();
  for (int i = 0; i < total_gpu; ++i) {
    int shard_len = h_left[i] == -1 ? 0 : h_right[i] - h_left[i] + 1;
    if (shard_len == 0) {
      continue;
    }
    create_storage(
        gpu_id,
        i,
        shard_len * sizeof(uint64_t),
        shard_len * sizeof(uint64_t) + sizeof(int) * shard_len + shard_len % 2);
  }
  walk_to_dest(gpu_id,
               total_gpu,
               h_left,
               h_right,
               reinterpret_cast<uint64_t*>(d_shard_keys_ptr),
               NULL);
  for (int i = 0; i < total_gpu; ++i) {
    if (h_left[i] == -1) {
      continue;
    }
    int shard_len = h_left[i] == -1 ? 0 : h_right[i] - h_left[i] + 1;
    auto& node = path_[gpu_id][i].nodes_.back();
    //    CUDA_CHECK(cudaStreamSynchronize(node.in_stream));
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    auto cur_stream = resource_->remote_stream(i, gpu_id);
    CUDA_CHECK(cudaMemsetAsync(
        node.val_storage, 0, shard_len * sizeof(uint64_t), cur_stream));
    int table_offset =
        get_table_offset(i, GraphTableType::EDGE_TABLE, edge_idx);
    tables_[table_offset]->get(reinterpret_cast<uint64_t*>(node.key_storage),
                               reinterpret_cast<uint64_t*>(node.val_storage),
                               static_cast<size_t>(h_right[i] - h_left[i] + 1),
                               cur_stream);
    GpuPsNodeInfo* node_info_list =
        reinterpret_cast<GpuPsNodeInfo*>(node.val_storage);
    int* node_degree_array = reinterpret_cast<int*>(node_info_list + shard_len);
    int grid_size_ = (shard_len - 1) / block_size_ + 1;
    get_node_degree_kernel<<<grid_size_, block_size_, 0, cur_stream>>>(
        node_info_list, node_degree_array, shard_len);
  }
  for (int i = 0; i < total_gpu; ++i) {
    if (h_left[i] == -1) {
      continue;
    }
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    CUDA_CHECK(cudaStreamSynchronize(resource_->remote_stream(i, gpu_id)));
  }
  move_degree_to_source_gpu(
      gpu_id, total_gpu, h_left, h_right, d_shard_degree_ptr);
  fill_dvalues<<<grid_size, block_size_, 0, stream>>>(
      d_shard_degree_ptr, node_degree_ptr, d_idx_ptr, len);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  for (int i = 0; i < total_gpu; i++) {
    int shard_len = h_left[i] == -1 ? 0 : h_right[i] - h_left[i] + 1;
    if (shard_len == 0) {
      continue;
    }
    destroy_storage(gpu_id, i);
  }
  device_mutex_[gpu_id]->unlock();
  return node_degree;
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

  int offset = get_graph_list_offset(gpu_id, idx);
  const auto& graph = gpu_graph_list_[offset];
  if (graph.node_size == 0) {
    return result;
  }
  int x2, y2;
  int len = range_check(start, start + query_size, 0, graph.node_size, x2, y2);

  if (len == 0) {
    return result;
  }

  result.set_stream(resource_->local_stream(gpu_id, 0));
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
      graph, x2, len, reinterpret_cast<uint64_t*>(val));
  CUDA_CHECK(cudaStreamSynchronize(resource_->remote_stream(gpu_id, gpu_id)));
  return result;
}

// 外面调用这个函数用模板需要在头文件实现
int GpuPsGraphTable::get_feature_info_of_nodes(
    int gpu_id,
    uint64_t* d_nodes,
    int node_num,
    const std::shared_ptr<phi::Allocation>& size_list,
    const std::shared_ptr<phi::Allocation>& size_list_prefix_sum,
    std::shared_ptr<phi::Allocation>& feature_list,
    std::shared_ptr<phi::Allocation>& slot_list,
    bool sage_mode) {
  if (node_num == 0) {
    return 0;
  }

  int all_fea_num = 0;
  if (multi_node_) {
    if (infer_mode_ && sage_mode == false) {
      all_fea_num =
          get_feature_info_of_nodes_normal<uint64_t>(gpu_id,
                                                     d_nodes,
                                                     node_num,
                                                     size_list,
                                                     size_list_prefix_sum,
                                                     feature_list,
                                                     slot_list);
    } else {
      if (FLAGS_enable_graph_multi_node_sampling) {
        all_fea_num =
            get_feature_info_of_nodes_all2all<uint64_t>(gpu_id,
                                                        d_nodes,
                                                        node_num,
                                                        size_list,
                                                        size_list_prefix_sum,
                                                        feature_list,
                                                        slot_list,
                                                        sage_mode);
      }
    }
  } else {
    all_fea_num =
        get_feature_info_of_nodes_normal<uint64_t>(gpu_id,
                                                   d_nodes,
                                                   node_num,
                                                   size_list,
                                                   size_list_prefix_sum,
                                                   feature_list,
                                                   slot_list);
  }
  VLOG(2) << "end get feature info of nodes, all_fea_num: " << all_fea_num;
  return all_fea_num;
}

template <typename FeatureType>
int GpuPsGraphTable::get_feature_info_of_nodes_all2all(
    int gpu_id,
    uint64_t* d_nodes,
    int node_num,
    const std::shared_ptr<phi::Allocation>& size_list,
    const std::shared_ptr<phi::Allocation>& size_list_prefix_sum,
    std::shared_ptr<phi::Allocation>& feature_list,
    std::shared_ptr<phi::Allocation>& slot_list,
    bool sage_mode) {
  if (node_num == 0) {
    return 0;
  }

  phi::GPUPlace place = phi::GPUPlace(resource_->dev_id(gpu_id));
  platform::CUDADeviceGuard guard(resource_->dev_id(gpu_id));
  int total_gpu = resource_->total_device();
  auto stream = resource_->local_stream(gpu_id, 0);
  auto& loc = storage_[gpu_id];
  auto& res = loc.shard_res;

  loc.alloc(node_num, sizeof(uint32_t) /*key_bytes*/);
  size_t pull_size =
      gather_inter_keys_by_all2all(gpu_id, node_num, d_nodes, stream);
  VLOG(2) << "gather iner keys by all2all, pull size: " << pull_size
          << ", node num: " << node_num;

  std::shared_ptr<phi::Allocation> d_tmp_feature_list;
  std::shared_ptr<phi::Allocation> d_tmp_slot_list;
  std::shared_ptr<phi::Allocation> d_tmp_size_list;
  std::shared_ptr<phi::Allocation> d_tmp_size_prefixsum_list;

  d_tmp_size_list =
      memory::Alloc(place,
                    pull_size * sizeof(uint32_t),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  d_tmp_size_prefixsum_list =
      memory::Alloc(place,
                    pull_size * sizeof(uint32_t),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  int ret =
      get_feature_info_of_nodes_normal<FeatureType>(gpu_id,
                                                    loc.d_merged_keys,
                                                    pull_size,
                                                    d_tmp_size_list,
                                                    d_tmp_size_prefixsum_list,
                                                    d_tmp_feature_list,
                                                    d_tmp_slot_list);
  VLOG(2) << "finish feature info of nodes, ret: " << ret;

  FeatureType* d_tmp_feature_list_ptr =
      reinterpret_cast<FeatureType*>(d_tmp_feature_list->ptr());
  uint8_t* d_tmp_slot_list_ptr =
      reinterpret_cast<uint8_t*>(d_tmp_slot_list->ptr());
  uint32_t* d_tmp_size_list_ptr =
      reinterpret_cast<uint32_t*>(d_tmp_size_list->ptr());
  uint32_t* d_tmp_size_prefixsum_list_ptr =
      reinterpret_cast<uint32_t*>(d_tmp_size_prefixsum_list->ptr());

  uint32_t* size_list_ptr = reinterpret_cast<uint32_t*>(size_list->ptr());
  uint32_t* size_list_prefix_sum_ptr =
      reinterpret_cast<uint32_t*>(size_list_prefix_sum->ptr());

  VLOG(2) << "begin scatter size list";
  scatter_inter_vals_by_all2all_common(
      gpu_id,
      node_num,
      sizeof(uint32_t),
      reinterpret_cast<const uint32_t*>(d_tmp_size_list_ptr),
      reinterpret_cast<uint32_t*>(size_list_ptr),
      reinterpret_cast<uint32_t*>(loc.d_merged_vals),
      stream);
  VLOG(2) << "end scatter size list";
  std::shared_ptr<phi::Allocation> inter_size_list =
      memory::Alloc(place,
                    node_num * sizeof(uint32_t),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  uint32_t* inter_size_list_ptr =
      reinterpret_cast<uint32_t*>(inter_size_list->ptr());
  CUDA_CHECK(cudaMemcpyAsync(inter_size_list_ptr,
                             loc.d_merged_vals,
                             sizeof(uint32_t) * node_num,
                             cudaMemcpyDeviceToDevice,
                             stream));

  VLOG(2) << "begin calc size list prefix sum";
  size_t storage_bytes = 0;
  CUDA_CHECK(cub::DeviceScan::ExclusiveSum(NULL,
                                           storage_bytes,
                                           size_list_ptr,
                                           size_list_prefix_sum_ptr,
                                           node_num,
                                           stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  auto d_temp_storage_tmp =
      memory::Alloc(place,
                    storage_bytes,
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  CUDA_CHECK(cub::DeviceScan::ExclusiveSum(d_temp_storage_tmp->ptr(),
                                           storage_bytes,
                                           size_list_ptr,
                                           size_list_prefix_sum_ptr,
                                           node_num,
                                           stream));
  VLOG(2) << "end calc size list prefix sum";

  std::vector<uint32_t> h_feature_size_list(node_num, 0);
  CUDA_CHECK(
      cudaMemcpyAsync(reinterpret_cast<char*>(h_feature_size_list.data()),
                      size_list_ptr,
                      sizeof(uint32_t) * node_num,
                      cudaMemcpyDeviceToHost,
                      stream));
  int fea_num = 0;
  for (size_t i = 0; i < h_feature_size_list.size(); i++) {
    fea_num += h_feature_size_list[i];
  }
  VLOG(2) << "after calc, total fea num:" << fea_num;

  feature_list =
      memory::Alloc(place,
                    fea_num * sizeof(FeatureType),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  FeatureType* feature_list_ptr =
      reinterpret_cast<FeatureType*>(feature_list->ptr());
  slot_list =
      memory::Alloc(place,
                    fea_num * sizeof(uint8_t),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  uint8_t* slot_list_ptr = reinterpret_cast<uint8_t*>(slot_list->ptr());

  // calc new offset
  recalc_local_and_remote_size(gpu_id,
                               pull_size,
                               node_num,
                               d_tmp_size_list_ptr,
                               inter_size_list_ptr,
                               stream);

  VLOG(2) << "begin send feature list";
  std::shared_ptr<phi::Allocation> inter_feature_list =
      memory::Alloc(place,
                    fea_num * sizeof(FeatureType),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  FeatureType* inter_feature_list_ptr =
      reinterpret_cast<FeatureType*>(inter_feature_list->ptr());

  scatter_inter_vals_by_all2all_common(
      gpu_id,
      node_num,
      sizeof(FeatureType),
      reinterpret_cast<const FeatureType*>(d_tmp_feature_list_ptr),
      reinterpret_cast<FeatureType*>(feature_list_ptr),
      reinterpret_cast<FeatureType*>(inter_feature_list_ptr),
      stream,
      sage_mode,
      true);
  VLOG(2) << "end send feature list";

  VLOG(2) << "begin send slot list";
  std::shared_ptr<phi::Allocation> inter_slot_list =
      memory::Alloc(place,
                    fea_num * sizeof(uint8_t),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  uint8_t* inter_slot_list_ptr =
      reinterpret_cast<uint8_t*>(inter_slot_list->ptr());

  scatter_inter_vals_by_all2all_common(
      gpu_id,
      node_num,
      sizeof(uint8_t),
      reinterpret_cast<const uint8_t*>(d_tmp_slot_list_ptr),
      reinterpret_cast<uint8_t*>(slot_list_ptr),
      reinterpret_cast<uint8_t*>(inter_slot_list_ptr),
      stream,
      sage_mode,
      true);
  VLOG(2) << "end send slot list";

  auto inter_size_list_prefix_sum =
      memory::Alloc(place,
                    node_num * sizeof(uint32_t),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  uint32_t* inter_size_list_prefix_sum_ptr =
      reinterpret_cast<uint32_t*>(inter_size_list_prefix_sum->ptr());
  CUDA_CHECK(cub::DeviceScan::ExclusiveSum(NULL,
                                           storage_bytes,
                                           inter_size_list_ptr,
                                           inter_size_list_prefix_sum_ptr,
                                           node_num,
                                           stream));
  CUDA_CHECK(cub::DeviceScan::ExclusiveSum(d_temp_storage_tmp->ptr(),
                                           storage_bytes,
                                           inter_size_list_ptr,
                                           inter_size_list_prefix_sum_ptr,
                                           node_num,
                                           stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  int grid_size = (node_num - 1) / block_size_ + 1;
  fill_vari_feature_and_slot<<<grid_size, block_size_, 0, stream>>>(
      feature_list_ptr,
      slot_list_ptr,
      inter_feature_list_ptr,
      inter_slot_list_ptr,
      size_list_prefix_sum_ptr,
      inter_size_list_prefix_sum_ptr,
      inter_size_list_ptr,
      loc.shard_res.d_local_idx_parted,
      node_num);

  VLOG(2) << "end all2all get slot info, node_num: " << node_num
          << ", pull size: " << pull_size << ", fea num: " << fea_num;
  CUDA_CHECK(cudaStreamSynchronize(stream));

  return fea_num;
}

template <typename FeatureType>
int GpuPsGraphTable::get_feature_info_of_nodes_normal(
    int gpu_id,
    uint64_t* d_nodes,
    int node_num,
    const std::shared_ptr<phi::Allocation>& size_list,
    const std::shared_ptr<phi::Allocation>& size_list_prefix_sum,
    std::shared_ptr<phi::Allocation>& feature_list,
    std::shared_ptr<phi::Allocation>& slot_list) {
  if (node_num == 0) {
    return 0;
  }
  bool is_float_feature = false;
  if (std::is_same<FeatureType, float>::value) {
    is_float_feature = true;
  }

  phi::GPUPlace place = phi::GPUPlace(resource_->dev_id(gpu_id));
  platform::CUDADeviceGuard guard(resource_->dev_id(gpu_id));
  int total_gpu = resource_->total_device();
  auto stream = resource_->local_stream(gpu_id, 0);

  auto d_left =
      memory::Alloc(place,
                    total_gpu * sizeof(int),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  auto d_right =
      memory::Alloc(place,
                    total_gpu * sizeof(int),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  int* d_left_ptr = reinterpret_cast<int*>(d_left->ptr());
  int* d_right_ptr = reinterpret_cast<int*>(d_right->ptr());

  CUDA_CHECK(cudaMemsetAsync(d_left_ptr, -1, total_gpu * sizeof(int), stream));
  CUDA_CHECK(cudaMemsetAsync(d_right_ptr, -1, total_gpu * sizeof(int), stream));
  auto d_idx =
      memory::Alloc(place,
                    node_num * sizeof(int),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  int* d_idx_ptr = reinterpret_cast<int*>(d_idx->ptr());

  auto d_shard_keys =
      memory::Alloc(place,
                    node_num * sizeof(uint64_t),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  uint64_t* d_shard_keys_ptr = reinterpret_cast<uint64_t*>(d_shard_keys->ptr());
  split_idx_to_shard(
      d_nodes, d_idx_ptr, node_num, d_left_ptr, d_right_ptr, gpu_id, stream);

  heter_comm_kernel_->fill_shard_key(
      d_shard_keys_ptr, d_nodes, d_idx_ptr, node_num, stream, gpu_id);

  // slot feature
  std::vector<void*> d_fea_info(total_gpu, NULL);
  std::vector<void*> d_fea_size(total_gpu, NULL);
  std::vector<void*> d_fea_size_prefix_sum(total_gpu, NULL);
  std::vector<uint32_t> fea_num_list(total_gpu, 0);
  std::vector<int> fea_left(total_gpu, -1);

  int h_left[total_gpu];  // NOLINT
  CUDA_CHECK(cudaMemcpyAsync(h_left,
                             d_left_ptr,
                             total_gpu * sizeof(int),
                             cudaMemcpyDeviceToHost,
                             stream));
  int h_right[total_gpu];  // NOLINT
  CUDA_CHECK(cudaMemcpyAsync(h_right,
                             d_right_ptr,
                             total_gpu * sizeof(int),
                             cudaMemcpyDeviceToHost,
                             stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  device_mutex_[gpu_id]->lock();
  int shard_len[total_gpu];  // NOLINT
  void* d_temp_storage[total_gpu];
  std::vector<size_t> temp_storage_bytes(total_gpu, 0);

  for (int i = 0; i < total_gpu; ++i) {
    shard_len[i] = h_left[i] == -1 ? 0 : h_right[i] - h_left[i] + 1;
    d_temp_storage[i] = NULL;
    if (h_left[i] == -1) {
      continue;
    }
    create_storage(gpu_id, i, shard_len[i] * sizeof(uint64_t), 0);
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    auto& node = path_[gpu_id][i].nodes_.back();
    create_tmp_storage(
        d_fea_info[i], gpu_id, i, shard_len[i] * sizeof(uint64_t));
    CUDA_CHECK(cudaMemsetAsync(
        d_fea_info[i], 0, shard_len[i] * sizeof(uint64_t), node.in_stream));
    create_tmp_storage(
        d_fea_size[i], gpu_id, i, shard_len[i] * sizeof(uint32_t));
    create_tmp_storage(d_fea_size_prefix_sum[i],
                       gpu_id,
                       i,
                       (shard_len[i] + 1) * sizeof(uint32_t));
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        NULL,
        temp_storage_bytes[i],
        reinterpret_cast<uint32_t*>(d_fea_size[i]),
        reinterpret_cast<uint32_t*>(d_fea_size_prefix_sum[i] + 1),
        shard_len[i],
        resource_->remote_stream(i, gpu_id)));
  }

  for (int i = 0; i < total_gpu; ++i) {
    if (h_left[i] == -1) {
      continue;
    }
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    CUDA_CHECK(cudaStreamSynchronize(resource_->remote_stream(
        i, gpu_id)));  // wait for calc temp_storage_bytes
    create_tmp_storage(d_temp_storage[i], gpu_id, i, temp_storage_bytes[i]);
  }
  walk_to_dest(gpu_id,
               total_gpu,
               h_left,
               h_right,
               reinterpret_cast<uint64_t*>(d_shard_keys_ptr),
               NULL);

  // no sync so 8 card can parallel execute
  for (int i = 0; i < total_gpu; ++i) {
    if (h_left[i] == -1) {
      continue;
    }
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    auto& node = path_[gpu_id][i].nodes_.back();
    // If not found, val is -1.
    int table_offset;
    if (is_float_feature) {
      table_offset = get_table_offset(i, GraphTableType::FEATURE_TABLE, 1);
    } else {
      table_offset = get_table_offset(i, GraphTableType::FEATURE_TABLE, 0);
    }

    //    CUDA_CHECK(cudaStreamSynchronize(
    //        node.in_stream));  // wait for walk_to_dest and memset
    tables_[table_offset]->get(reinterpret_cast<uint64_t*>(node.key_storage),
                               reinterpret_cast<uint64_t*>(d_fea_info[i]),
                               static_cast<size_t>(h_right[i] - h_left[i] + 1),
                               resource_->remote_stream(i, gpu_id));
    dim3 grid((shard_len[i] - 1) / dim_y + 1);
    dim3 block(1, dim_y);

    get_features_size<<<grid, block, 0, resource_->remote_stream(i, gpu_id)>>>(
        reinterpret_cast<GpuPsFeaInfo*>(d_fea_info[i]),
        reinterpret_cast<uint32_t*>(d_fea_size[i]),
        shard_len[i]);
    CUDA_CHECK(cudaMemsetAsync(d_fea_size_prefix_sum[i],
                               0,
                               sizeof(uint32_t),
                               resource_->remote_stream(i, gpu_id)));
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        d_temp_storage[i],
        temp_storage_bytes[i],
        reinterpret_cast<uint32_t*>(d_fea_size[i]),
        reinterpret_cast<uint32_t*>(d_fea_size_prefix_sum[i]) + 1,
        shard_len[i],
        resource_->remote_stream(i, gpu_id)));
  }

  // wait for fea_num_list
  for (int i = 0; i < total_gpu; ++i) {
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    if (h_left[i] == -1) {
      continue;
    }
    auto& node = path_[gpu_id][i].nodes_.back();
    CUDA_CHECK(cudaMemcpyAsync(
        &fea_num_list[i],
        reinterpret_cast<uint32_t*>(d_fea_size_prefix_sum[i]) + shard_len[i],
        sizeof(uint32_t),
        cudaMemcpyDeviceToHost,
        resource_->remote_stream(i, gpu_id)));

    CUDA_CHECK(cudaStreamSynchronize(
        resource_->remote_stream(i, gpu_id)));  // wait for fea_num_list

    create_storage(gpu_id,
                   i,
                   0,
                   (shard_len[i] + shard_len[i] % 2) * sizeof(uint32_t) +
                       fea_num_list[i] * sizeof(FeatureType) +
                       fea_num_list[i] * sizeof(uint8_t));
    uint32_t* actual_size_array = reinterpret_cast<uint32_t*>(node.val_storage);
    CUDA_CHECK(cudaMemcpyAsync(actual_size_array,
                               d_fea_size[i],
                               sizeof(uint32_t) * shard_len[i],
                               cudaMemcpyDeviceToDevice,
                               resource_->remote_stream(i, gpu_id)));

    if (is_float_feature) {
      int offset = get_graph_float_fea_list_offset(i);
      auto& graph = gpu_graph_float_fea_list_[offset];

      float* feature_array = reinterpret_cast<float*>(
          actual_size_array + shard_len[i] + shard_len[i] % 2);
      uint8_t* slot_array =
          reinterpret_cast<uint8_t*>(feature_array + fea_num_list[i]);
      dim3 grid((shard_len[i] - 1) / dim_y + 1);
      dim3 block(1, dim_y);
      get_float_features_kernel<<<grid,
                                  block,
                                  0,
                                  resource_->remote_stream(i, gpu_id)>>>(
          graph,
          reinterpret_cast<GpuPsFeaInfo*>(d_fea_info[i]),
          reinterpret_cast<uint32_t*>(d_fea_size_prefix_sum[i]),
          feature_array,
          slot_array,
          shard_len[i]);
    } else {
      int offset = get_graph_fea_list_offset(i);
      auto& graph = gpu_graph_fea_list_[offset];

      uint64_t* feature_array = reinterpret_cast<uint64_t*>(
          actual_size_array + shard_len[i] + shard_len[i] % 2);
      uint8_t* slot_array =
          reinterpret_cast<uint8_t*>(feature_array + fea_num_list[i]);
      dim3 grid((shard_len[i] - 1) / dim_y + 1);
      dim3 block(1, dim_y);

      get_features_kernel<<<grid,
                            block,
                            0,
                            resource_->remote_stream(i, gpu_id)>>>(
          graph,
          reinterpret_cast<GpuPsFeaInfo*>(d_fea_info[i]),
          reinterpret_cast<uint32_t*>(d_fea_size_prefix_sum[i]),
          feature_array,
          slot_array,
          shard_len[i]);
    }
  }

  for (int i = 0; i < total_gpu; ++i) {
    if (h_left[i] == -1) {
      continue;
    }
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    CUDA_CHECK(cudaStreamSynchronize(resource_->remote_stream(i, gpu_id)));
  }

  uint32_t all_fea_num = 0;
  for (int i = 0; i < total_gpu; ++i) {
    fea_left[i] = all_fea_num;
    all_fea_num += fea_num_list[i];
  }

  platform::CUDADeviceGuard guard2(resource_->dev_id(gpu_id));
  auto feature_list_tmp =
      memory::Alloc(place,
                    all_fea_num * sizeof(FeatureType),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  FeatureType* d_feature_list_ptr =
      reinterpret_cast<FeatureType*>(feature_list_tmp->ptr());

  auto slot_list_tmp =
      memory::Alloc(place,
                    all_fea_num * sizeof(uint8_t),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  uint8_t* d_slot_list_ptr = reinterpret_cast<uint8_t*>(slot_list_tmp->ptr());

  auto size_list_tmp =
      memory::Alloc(place,
                    node_num * sizeof(uint32_t),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  uint32_t* d_size_list_ptr = reinterpret_cast<uint32_t*>(size_list_tmp->ptr());

  move_result_to_source_gpu(gpu_id,
                            total_gpu,
                            h_left,
                            h_right,
                            fea_left.data(),
                            fea_num_list.data(),
                            d_size_list_ptr,
                            d_feature_list_ptr,
                            d_slot_list_ptr);

  for (int i = 0; i < total_gpu; ++i) {
    if (shard_len[i] == 0) {
      continue;
    }
    destroy_storage(gpu_id, i);
    if (d_fea_info[i] != NULL) {
      destroy_tmp_storage(d_fea_info[i], gpu_id, i);
    }
    if (d_fea_size[i] != NULL) {
      destroy_tmp_storage(d_fea_size[i], gpu_id, i);
    }
    if (d_fea_size_prefix_sum[i] != NULL) {
      destroy_tmp_storage(d_fea_size_prefix_sum[i], gpu_id, i);
    }
    if (d_temp_storage[i] != NULL) {
      destroy_tmp_storage(d_temp_storage[i], gpu_id, i);
    }
  }

  d_fea_info.clear();
  d_fea_size.clear();
  d_fea_size_prefix_sum.clear();
  device_mutex_[gpu_id]->unlock();
  feature_list =
      memory::Alloc(place,
                    all_fea_num * sizeof(FeatureType),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));

  FeatureType* d_res_feature_list_ptr =
      reinterpret_cast<FeatureType*>(feature_list->ptr());

  slot_list =
      memory::Alloc(place,
                    all_fea_num * sizeof(uint8_t),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));

  uint8_t* d_res_slot_list_ptr = reinterpret_cast<uint8_t*>(slot_list->ptr());

  int grid_size = (node_num - 1) / block_size_ + 1;
  uint32_t* size_list_ptr = reinterpret_cast<uint32_t*>(size_list->ptr());
  uint32_t* size_list_prefix_sum_ptr =
      reinterpret_cast<uint32_t*>(size_list_prefix_sum->ptr());

  fill_size<<<grid_size, block_size_, 0, stream>>>(
      size_list_ptr, d_size_list_ptr, d_idx_ptr, node_num);
  size_t storage_bytes = 0;
  auto src_fea_size_prefix_sum =
      memory::Alloc(place,
                    node_num * sizeof(uint32_t),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));

  uint32_t* src_fea_size_prefix_sum_ptr =
      reinterpret_cast<uint32_t*>(src_fea_size_prefix_sum->ptr());
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cub::DeviceScan::ExclusiveSum(NULL,
                                           storage_bytes,
                                           size_list_ptr,
                                           size_list_prefix_sum_ptr,
                                           node_num,
                                           stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  auto d_temp_storage_tmp =
      memory::Alloc(place,
                    storage_bytes,
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  CUDA_CHECK(cub::DeviceScan::ExclusiveSum(d_temp_storage_tmp->ptr(),
                                           storage_bytes,
                                           size_list_ptr,
                                           size_list_prefix_sum_ptr,
                                           node_num,
                                           stream));

  CUDA_CHECK(cub::DeviceScan::ExclusiveSum(d_temp_storage_tmp->ptr(),
                                           storage_bytes,
                                           d_size_list_ptr,
                                           src_fea_size_prefix_sum_ptr,
                                           node_num,
                                           stream));
  fill_feature_and_slot<<<grid_size, block_size_, 0, stream>>>(
      d_res_feature_list_ptr,
      d_res_slot_list_ptr,
      size_list_prefix_sum_ptr,
      d_feature_list_ptr,
      d_slot_list_ptr,
      src_fea_size_prefix_sum_ptr,
      d_size_list_ptr,
      d_idx_ptr,
      node_num);

  CUDA_CHECK(cudaStreamSynchronize(stream));
  return all_fea_num;
}

int GpuPsGraphTable::get_rank_info_of_nodes(int gpu_id,
                                            const uint64_t* d_nodes,
                                            uint32_t* d_ranks,
                                            int node_len) {
  if (node_len == 0) {
    return -1;
  }

  phi::GPUPlace place = phi::GPUPlace(resource_->dev_id(gpu_id));
  platform::CUDADeviceGuard guard(resource_->dev_id(gpu_id));
  int total_gpu = resource_->total_device();
  auto stream = resource_->local_stream(gpu_id, 0);

  auto d_left =
      memory::Alloc(place,
                    total_gpu * sizeof(int),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  auto d_right =
      memory::Alloc(place,
                    total_gpu * sizeof(int),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  int* d_left_ptr = reinterpret_cast<int*>(d_left->ptr());
  int* d_right_ptr = reinterpret_cast<int*>(d_right->ptr());

  CUDA_CHECK(cudaMemsetAsync(d_left_ptr, -1, total_gpu * sizeof(int), stream));
  CUDA_CHECK(cudaMemsetAsync(d_right_ptr, -1, total_gpu * sizeof(int), stream));
  //
  auto d_idx =
      memory::Alloc(place,
                    node_len * sizeof(int),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  int* d_idx_ptr = reinterpret_cast<int*>(d_idx->ptr());

  auto d_shard_keys =
      memory::Alloc(place,
                    node_len * sizeof(uint64_t),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  uint64_t* d_shard_keys_ptr = reinterpret_cast<uint64_t*>(d_shard_keys->ptr());
  auto d_shard_vals =
      memory::Alloc(place,
                    node_len * sizeof(uint32_t),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  uint32_t* d_shard_vals_ptr = reinterpret_cast<uint32_t*>(d_shard_vals->ptr());
  auto d_shard_actual_size =
      memory::Alloc(place,
                    node_len * sizeof(int),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  int* d_shard_actual_size_ptr =
      reinterpret_cast<int*>(d_shard_actual_size->ptr());
  split_idx_to_shard(const_cast<uint64_t*>(d_nodes),
                     d_idx_ptr,
                     node_len,
                     d_left_ptr,
                     d_right_ptr,
                     gpu_id,
                     stream);

  heter_comm_kernel_->fill_shard_key(d_shard_keys_ptr,
                                     const_cast<uint64_t*>(d_nodes),
                                     d_idx_ptr,
                                     node_len,
                                     stream,
                                     gpu_id);

  int h_left[total_gpu];  // NOLINT
  CUDA_CHECK(cudaMemcpyAsync(h_left,
                             d_left_ptr,
                             total_gpu * sizeof(int),
                             cudaMemcpyDeviceToHost,
                             stream));
  int h_right[total_gpu];  // NOLINT
  CUDA_CHECK(cudaMemcpyAsync(h_right,
                             d_right_ptr,
                             total_gpu * sizeof(int),
                             cudaMemcpyDeviceToHost,
                             stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  device_mutex_[gpu_id]->lock();
  for (int i = 0; i < total_gpu; ++i) {
    int shard_len = h_left[i] == -1 ? 0 : h_right[i] - h_left[i] + 1;
    if (shard_len == 0) {
      continue;
    }
    create_storage(
        gpu_id, i, shard_len * sizeof(uint64_t), shard_len * sizeof(uint32_t));
  }

  walk_to_dest(gpu_id,
               total_gpu,
               h_left,
               h_right,
               reinterpret_cast<uint64_t*>(d_shard_keys_ptr),
               NULL);

  for (int i = 0; i < total_gpu; ++i) {
    if (h_left[i] == -1) {
      continue;
    }
    int shard_len = h_left[i] == -1 ? 0 : h_right[i] - h_left[i] + 1;
    auto& node = path_[gpu_id][i].nodes_.back();

    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    auto cur_stream = resource_->remote_stream(i, gpu_id);
    CUDA_CHECK(cudaMemsetAsync(
        node.val_storage, 0, shard_len * sizeof(uint32_t), cur_stream));
    // If not found, val is -1.
    int table_offset = get_rank_list_offset(i);
    rank_tables_[table_offset]->get(
        reinterpret_cast<uint64_t*>(node.key_storage),
        reinterpret_cast<uint32_t*>(node.val_storage),
        static_cast<size_t>(h_right[i] - h_left[i] + 1),
        cur_stream);
  }

  for (int i = 0; i < total_gpu; ++i) {
    if (h_left[i] == -1) {
      continue;
    }
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    CUDA_CHECK(cudaStreamSynchronize(resource_->remote_stream(i, gpu_id)));
  }

  walk_to_src(gpu_id,
              total_gpu,
              h_left,
              h_right,
              reinterpret_cast<char*>(d_shard_vals_ptr),
              sizeof(uint32_t));

  for (int i = 0; i < total_gpu; ++i) {
    int shard_len = h_left[i] == -1 ? 0 : h_right[i] - h_left[i] + 1;
    if (shard_len == 0) {
      continue;
    }
    destroy_storage(gpu_id, i);
  }
  device_mutex_[gpu_id]->unlock();

  platform::CUDADeviceGuard guard2(resource_->dev_id(gpu_id));
  int grid_size = (node_len - 1) / block_size_ + 1;
  heter_comm_kernel_->fill_dvals(
      d_shard_vals_ptr, d_ranks, d_idx_ptr, node_len, stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));
  return 0;
}

int GpuPsGraphTable::get_float_feature_info_of_nodes(
    int gpu_id,
    uint64_t* d_nodes,
    int node_num,
    const std::shared_ptr<phi::Allocation>& size_list,
    const std::shared_ptr<phi::Allocation>& size_list_prefix_sum,
    std::shared_ptr<phi::Allocation>& feature_list,
    std::shared_ptr<phi::Allocation>& slot_list,
    bool sage_mode) {
  if (node_num == 0) {
    return 0;
  }

  int all_fea_num = 0;
  if (multi_node_) {
    if (infer_mode_ && sage_mode == false) {
      all_fea_num =
          get_feature_info_of_nodes_normal<float>(gpu_id,
                                                  d_nodes,
                                                  node_num,
                                                  size_list,
                                                  size_list_prefix_sum,
                                                  feature_list,
                                                  slot_list);
    } else {
      if (FLAGS_enable_graph_multi_node_sampling) {
        all_fea_num =
            get_feature_info_of_nodes_all2all<float>(gpu_id,
                                                     d_nodes,
                                                     node_num,
                                                     size_list,
                                                     size_list_prefix_sum,
                                                     feature_list,
                                                     slot_list,
                                                     sage_mode);
      }
    }
  } else {
    all_fea_num = get_feature_info_of_nodes_normal<float>(gpu_id,
                                                          d_nodes,
                                                          node_num,
                                                          size_list,
                                                          size_list_prefix_sum,
                                                          feature_list,
                                                          slot_list);
  }
  VLOG(2) << "end get float feature info of nodes, all_fea_num: "
          << all_fea_num;
  return all_fea_num;
}

int GpuPsGraphTable::get_feature_of_nodes(int gpu_id,
                                          uint64_t* d_nodes,
                                          uint64_t* d_feature,
                                          int node_num,
                                          int slot_num,
                                          int* d_slot_feature_num_map,
                                          int fea_num_per_node) {
  if (node_num == 0) {
    return -1;
  }

  phi::GPUPlace place = phi::GPUPlace(resource_->dev_id(gpu_id));
  platform::CUDADeviceGuard guard(resource_->dev_id(gpu_id));
  int total_gpu = resource_->total_device();
  auto stream = resource_->local_stream(gpu_id, 0);

  auto d_left =
      memory::Alloc(place,
                    total_gpu * sizeof(int),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  auto d_right =
      memory::Alloc(place,
                    total_gpu * sizeof(int),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  int* d_left_ptr = reinterpret_cast<int*>(d_left->ptr());
  int* d_right_ptr = reinterpret_cast<int*>(d_right->ptr());

  CUDA_CHECK(cudaMemsetAsync(d_left_ptr, -1, total_gpu * sizeof(int), stream));
  CUDA_CHECK(cudaMemsetAsync(d_right_ptr, -1, total_gpu * sizeof(int), stream));
  //
  auto d_idx =
      memory::Alloc(place,
                    node_num * sizeof(int),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  int* d_idx_ptr = reinterpret_cast<int*>(d_idx->ptr());

  auto d_shard_keys =
      memory::Alloc(place,
                    node_num * sizeof(uint64_t),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  uint64_t* d_shard_keys_ptr = reinterpret_cast<uint64_t*>(d_shard_keys->ptr());
  auto d_shard_vals =
      memory::Alloc(place,
                    fea_num_per_node * node_num * sizeof(uint64_t),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  uint64_t* d_shard_vals_ptr = reinterpret_cast<uint64_t*>(d_shard_vals->ptr());
  auto d_shard_actual_size =
      memory::Alloc(place,
                    node_num * sizeof(int),
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  int* d_shard_actual_size_ptr =
      reinterpret_cast<int*>(d_shard_actual_size->ptr());

  split_idx_to_shard(
      d_nodes, d_idx_ptr, node_num, d_left_ptr, d_right_ptr, gpu_id, stream);

  heter_comm_kernel_->fill_shard_key(
      d_shard_keys_ptr, d_nodes, d_idx_ptr, node_num, stream, gpu_id);

  int h_left[total_gpu];  // NOLINT
  CUDA_CHECK(cudaMemcpyAsync(h_left,
                             d_left_ptr,
                             total_gpu * sizeof(int),
                             cudaMemcpyDeviceToHost,
                             stream));
  int h_right[total_gpu];  // NOLINT
  CUDA_CHECK(cudaMemcpyAsync(h_right,
                             d_right_ptr,
                             total_gpu * sizeof(int),
                             cudaMemcpyDeviceToHost,
                             stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  device_mutex_[gpu_id]->lock();
  for (int i = 0; i < total_gpu; ++i) {
    int shard_len = h_left[i] == -1 ? 0 : h_right[i] - h_left[i] + 1;
    if (shard_len == 0) {
      continue;
    }
    create_storage(gpu_id,
                   i,
                   shard_len * sizeof(uint64_t),
                   shard_len * fea_num_per_node * sizeof(uint64_t) +
                       shard_len * sizeof(uint64_t) +
                       sizeof(int) * (shard_len + shard_len % 2));
  }

  walk_to_dest(gpu_id,
               total_gpu,
               h_left,
               h_right,
               reinterpret_cast<uint64_t*>(d_shard_keys_ptr),
               NULL);

  for (int i = 0; i < total_gpu; ++i) {
    if (h_left[i] == -1) {
      continue;
    }
    int shard_len = h_left[i] == -1 ? 0 : h_right[i] - h_left[i] + 1;
    auto& node = path_[gpu_id][i].nodes_.back();

    //    CUDA_CHECK(cudaStreamSynchronize(node.in_stream));
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    auto cur_stream = resource_->remote_stream(i, gpu_id);
    CUDA_CHECK(cudaMemsetAsync(
        node.val_storage, 0, shard_len * sizeof(uint64_t), cur_stream));
    // If not found, val is -1.
    int table_offset = get_table_offset(i, GraphTableType::FEATURE_TABLE, 0);
    tables_[table_offset]->get(reinterpret_cast<uint64_t*>(node.key_storage),
                               reinterpret_cast<uint64_t*>(node.val_storage),
                               static_cast<size_t>(h_right[i] - h_left[i] + 1),
                               cur_stream);

    int offset = get_graph_fea_list_offset(i);
    auto graph = gpu_graph_fea_list_[offset];

    GpuPsFeaInfo* val_array = reinterpret_cast<GpuPsFeaInfo*>(node.val_storage);
    int* actual_size_array = reinterpret_cast<int*>(val_array + shard_len);
    uint64_t* feature_array = reinterpret_cast<uint64_t*>(
        actual_size_array + shard_len + shard_len % 2);
    dim3 grid((shard_len - 1) / dim_y + 1);
    dim3 block(1, dim_y);
    get_features_kernel<<<grid, block, 0, cur_stream>>>(graph,
                                                        val_array,
                                                        actual_size_array,
                                                        feature_array,
                                                        d_slot_feature_num_map,
                                                        slot_num,
                                                        shard_len,
                                                        fea_num_per_node);
  }

  for (int i = 0; i < total_gpu; ++i) {
    if (h_left[i] == -1) {
      continue;
    }
    platform::CUDADeviceGuard guard(resource_->dev_id(i));
    CUDA_CHECK(cudaStreamSynchronize(resource_->remote_stream(i, gpu_id)));
  }

  move_result_to_source_gpu(gpu_id,
                            total_gpu,
                            fea_num_per_node,
                            h_left,
                            h_right,
                            d_shard_vals_ptr,
                            d_shard_actual_size_ptr);
  for (int i = 0; i < total_gpu; ++i) {
    int shard_len = h_left[i] == -1 ? 0 : h_right[i] - h_left[i] + 1;
    if (shard_len == 0) {
      continue;
    }
    destroy_storage(gpu_id, i);
  }
  device_mutex_[gpu_id]->unlock();

  platform::CUDADeviceGuard guard2(resource_->dev_id(gpu_id));
  int grid_size = (node_num - 1) / block_size_ + 1;
  fill_dvalues<<<grid_size, block_size_, 0, stream>>>(d_shard_vals_ptr,
                                                      d_feature,
                                                      d_shard_actual_size_ptr,
                                                      d_idx_ptr,
                                                      fea_num_per_node,
                                                      node_num);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  return 0;
}

};  // namespace framework
};  // namespace paddle
#endif
