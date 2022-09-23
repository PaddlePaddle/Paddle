/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#if defined _WIN32 || defined __APPLE__
#else
#define _LINUX
#endif
#if defined(PADDLE_WITH_CUDA) && defined(PADDLE_WITH_HETERPS)

#include "paddle/fluid/framework/data_feed.h"
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>
#include <sstream>
#include "cub/cub.cuh"
#include "paddle/fluid/framework/fleet/heter_ps/gpu_graph_node.h"
#include "paddle/fluid/framework/fleet/heter_ps/graph_gpu_wrapper.h"

#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/unique_kernel.h"
#include "paddle/phi/kernels/graph_reindex_kernel.h"
#include "paddle/phi/kernels/gpu/graph_reindex_funcs.h"

DECLARE_bool(enable_opt_get_features);

namespace paddle {
namespace framework {

#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

// CUDA: use 512 threads per block
const int CUDA_NUM_THREADS = 512;
// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}
// fill slot values
__global__ void FillSlotValueOffsetKernel(const int ins_num,
                                          const int used_slot_num,
                                          size_t *slot_value_offsets,
                                          const int *uint64_offsets,
                                          const int uint64_slot_size,
                                          const int *float_offsets,
                                          const int float_slot_size,
                                          const UsedSlotGpuType *used_slots) {
  int col_num = ins_num + 1;
  int uint64_cols = uint64_slot_size + 1;
  int float_cols = float_slot_size + 1;

  CUDA_KERNEL_LOOP(slot_idx, used_slot_num) {
    int value_off = slot_idx * col_num;
    slot_value_offsets[value_off] = 0;

    auto &info = used_slots[slot_idx];
    if (info.is_uint64_value) {
      for (int k = 0; k < ins_num; ++k) {
        int pos = k * uint64_cols + info.slot_value_idx;
        int num = uint64_offsets[pos + 1] - uint64_offsets[pos];
        PADDLE_ENFORCE(num >= 0, "The number of slot size must be ge 0.");
        slot_value_offsets[value_off + k + 1] =
            slot_value_offsets[value_off + k] + num;
      }
    } else {
      for (int k = 0; k < ins_num; ++k) {
        int pos = k * float_cols + info.slot_value_idx;
        int num = float_offsets[pos + 1] - float_offsets[pos];
        PADDLE_ENFORCE(num >= 0, "The number of slot size must be ge 0.");
        slot_value_offsets[value_off + k + 1] =
            slot_value_offsets[value_off + k] + num;
      }
    }
  }
}

__global__ void fill_actual_neighbors(int64_t* vals,
                                      int64_t* actual_vals,
                                      int64_t* actual_vals_dst,
                                      int* actual_sample_size,
                                      int* cumsum_actual_sample_size,
                                      int sample_size,
                                      int len,
                                      int mod) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    int offset1 = cumsum_actual_sample_size[i];
    int offset2 = sample_size * i;
    int dst_id = i % mod;
    for (int j = 0; j < actual_sample_size[i]; j++) {
      actual_vals[offset1 + j] = vals[offset2 + j];
      actual_vals_dst[offset1 + j] = dst_id;
    }
  }
}

void SlotRecordInMemoryDataFeed::FillSlotValueOffset(
    const int ins_num,
    const int used_slot_num,
    size_t *slot_value_offsets,
    const int *uint64_offsets,
    const int uint64_slot_size,
    const int *float_offsets,
    const int float_slot_size,
    const UsedSlotGpuType *used_slots) {
  auto stream =
      dynamic_cast<phi::GPUContext *>(
          paddle::platform::DeviceContextPool::Instance().Get(this->place_))
          ->stream();
  FillSlotValueOffsetKernel<<<GET_BLOCKS(used_slot_num),
                              CUDA_NUM_THREADS,
                              0,
                              stream>>>(ins_num,
                                        used_slot_num,
                                        slot_value_offsets,
                                        uint64_offsets,
                                        uint64_slot_size,
                                        float_offsets,
                                        float_slot_size,
                                        used_slots);
  cudaStreamSynchronize(stream);
}

__global__ void CopyForTensorKernel(const int used_slot_num,
                                    const int ins_num,
                                    void **dest,
                                    const size_t *slot_value_offsets,
                                    const uint64_t *uint64_feas,
                                    const int *uint64_offsets,
                                    const int *uint64_ins_lens,
                                    const int uint64_slot_size,
                                    const float *float_feas,
                                    const int *float_offsets,
                                    const int *float_ins_lens,
                                    const int float_slot_size,
                                    const UsedSlotGpuType *used_slots) {
  int col_num = ins_num + 1;
  int uint64_cols = uint64_slot_size + 1;
  int float_cols = float_slot_size + 1;

  CUDA_KERNEL_LOOP(i, ins_num * used_slot_num) {
    int slot_idx = i / ins_num;
    int ins_idx = i % ins_num;

    uint32_t value_offset = slot_value_offsets[slot_idx * col_num + ins_idx];
    auto &info = used_slots[slot_idx];
    if (info.is_uint64_value) {
      uint64_t *up = reinterpret_cast<uint64_t *>(dest[slot_idx]);
      int index = info.slot_value_idx + uint64_cols * ins_idx;
      int old_off = uint64_offsets[index];
      int num = uint64_offsets[index + 1] - old_off;
      PADDLE_ENFORCE(num >= 0, "The number of slot size must be ge 0.");
      int uint64_value_offset = uint64_ins_lens[ins_idx];
      for (int k = 0; k < num; ++k) {
        up[k + value_offset] = uint64_feas[k + old_off + uint64_value_offset];
      }
    } else {
      float *fp = reinterpret_cast<float *>(dest[slot_idx]);
      int index = info.slot_value_idx + float_cols * ins_idx;
      int old_off = float_offsets[index];
      int num = float_offsets[index + 1] - old_off;
      PADDLE_ENFORCE(num >= 0, "The number of slot size must be ge 0.");
      int float_value_offset = float_ins_lens[ins_idx];
      for (int k = 0; k < num; ++k) {
        fp[k + value_offset] = float_feas[k + old_off + float_value_offset];
      }
    }
  }
}

void SlotRecordInMemoryDataFeed::CopyForTensor(
    const int ins_num,
    const int used_slot_num,
    void **dest,
    const size_t *slot_value_offsets,
    const uint64_t *uint64_feas,
    const int *uint64_offsets,
    const int *uint64_ins_lens,
    const int uint64_slot_size,
    const float *float_feas,
    const int *float_offsets,
    const int *float_ins_lens,
    const int float_slot_size,
    const UsedSlotGpuType *used_slots) {
  auto stream =
      dynamic_cast<phi::GPUContext *>(
          paddle::platform::DeviceContextPool::Instance().Get(this->place_))
          ->stream();

  CopyForTensorKernel<<<GET_BLOCKS(used_slot_num * ins_num),
                        CUDA_NUM_THREADS,
                        0,
                        stream>>>(used_slot_num,
                                  ins_num,
                                  dest,
                                  slot_value_offsets,
                                  uint64_feas,
                                  uint64_offsets,
                                  uint64_ins_lens,
                                  uint64_slot_size,
                                  float_feas,
                                  float_offsets,
                                  float_ins_lens,
                                  float_slot_size,
                                  used_slots);
  cudaStreamSynchronize(stream);
}

__global__ void GraphFillCVMKernel(int64_t *tensor, int len) {
  CUDA_KERNEL_LOOP(idx, len) { tensor[idx] = 1; }
}

__global__ void CopyDuplicateKeys(int64_t *dist_tensor,
                                  uint64_t *src_tensor,
                                  int len) {
  CUDA_KERNEL_LOOP(idx, len) {
    dist_tensor[idx * 2] = src_tensor[idx];
    dist_tensor[idx * 2 + 1] = src_tensor[idx];
  }
}

int GraphDataGenerator::AcquireInstance(BufState *state) {
  //
  if (state->GetNextStep()) {
    state->Debug();
    return state->len;
  } else if (state->GetNextCentrolWord()) {
    state->Debug();
    return state->len;
  } else if (state->GetNextBatch()) {
    state->Debug();
    return state->len;
  }
  return 0;
}

// TODO opt
__global__ void GraphFillFeatureKernel(uint64_t *id_tensor,
                                       int *fill_ins_num,
                                       uint64_t *walk,
                                       uint64_t *feature,
                                       int *row,
                                       int central_word,
                                       int step,
                                       int len,
                                       int col_num,
                                       int slot_num) {
  __shared__ int32_t local_key[CUDA_NUM_THREADS * 16];
  __shared__ int local_num;
  __shared__ int global_num;

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadIdx.x == 0) {
    local_num = 0;
  }
  __syncthreads();
  if (idx < len) {
    int src = row[idx] * col_num + central_word;
    if (walk[src] != 0 && walk[src + step] != 0) {
      size_t dst = atomicAdd(&local_num, 1);
      for (int i = 0; i < slot_num; ++i) {
        local_key[dst * 2 * slot_num + i * 2] = feature[src * slot_num + i];
        local_key[dst * 2 * slot_num + i * 2 + 1] =
            feature[(src + step) * slot_num + i];
      }
    }
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    global_num = atomicAdd(fill_ins_num, local_num);
  }
  __syncthreads();

  if (threadIdx.x < local_num) {
    for (int i = 0; i < slot_num; ++i) {
      id_tensor[(global_num * 2 + 2 * threadIdx.x) * slot_num + i] =
          local_key[(2 * threadIdx.x) * slot_num + i];
      id_tensor[(global_num * 2 + 2 * threadIdx.x + 1) * slot_num + i] =
          local_key[(2 * threadIdx.x + 1) * slot_num + i];
    }
  }
}

__global__ void GraphFillIdKernel(uint64_t *id_tensor,
                                  int *fill_ins_num,
                                  uint64_t *walk,
                                  int *row,
                                  int central_word,
                                  int step,
                                  int len,
                                  int col_num) {
  __shared__ uint64_t local_key[CUDA_NUM_THREADS * 2];
  __shared__ int local_num;
  __shared__ int global_num;

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadIdx.x == 0) {
    local_num = 0;
  }
  __syncthreads();
  // int dst = idx * 2;
  // id_tensor[dst] = walk[src];
  // id_tensor[dst + 1] = walk[src + step];
  if (idx < len) {
    int src = row[idx] * col_num + central_word;
    if (walk[src] != 0 && walk[src + step] != 0) {
      size_t dst = atomicAdd(&local_num, 1);
      local_key[dst * 2] = walk[src];
      local_key[dst * 2 + 1] = walk[src + step];
    }
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    global_num = atomicAdd(fill_ins_num, local_num);
  }
  __syncthreads();

  if (threadIdx.x < local_num) {
    id_tensor[global_num * 2 + 2 * threadIdx.x] = local_key[2 * threadIdx.x];
    id_tensor[global_num * 2 + 2 * threadIdx.x + 1] =
        local_key[2 * threadIdx.x + 1];
  }
}

__global__ void GraphFillSlotKernel(uint64_t *id_tensor,
                                    uint64_t *feature_buf,
                                    int len,
                                    int total_ins,
                                    int slot_num,
                                    int* slot_feature_num_map,
                                    int fea_num_per_node,
                                    int* actual_slot_id_map,
                                    int* fea_offset_map) {
  CUDA_KERNEL_LOOP(idx, len) {
    int fea_idx = idx / total_ins;
    int ins_idx = idx % total_ins;
    int actual_slot_id = actual_slot_id_map[fea_idx];
    int fea_offset = fea_offset_map[fea_idx];
    ((uint64_t *)(id_tensor[actual_slot_id]))[ins_idx * slot_feature_num_map[actual_slot_id] + fea_offset]
        = feature_buf[ins_idx * fea_num_per_node + fea_idx];
  }
}

__global__ void GraphFillSlotLodKernelOpt(uint64_t *id_tensor,
                                          int len,
                                          int total_ins,
                                          int* slot_feature_num_map) {
  CUDA_KERNEL_LOOP(idx, len) {
    int slot_idx = idx / total_ins;
    int ins_idx = idx % total_ins;
    ((uint64_t *)(id_tensor[slot_idx]))[ins_idx] = ins_idx * slot_feature_num_map[slot_idx];
  }
}

__global__ void GraphFillSlotLodKernel(int64_t *id_tensor, int len) {
  CUDA_KERNEL_LOOP(idx, len) { id_tensor[idx] = idx; }
}

int GraphDataGenerator::FillInsBuf() {
  if (ins_buf_pair_len_ >= batch_size_) {
    return batch_size_;
  }
  int total_instance = AcquireInstance(&buf_state_);

  VLOG(2) << "total_ins: " << total_instance;
  buf_state_.Debug();

  if (total_instance == 0) {
    int res = FillWalkBuf(d_walk_);
    if (!res) {
      // graph iterate complete
      return -1;
    } else {
      total_instance = buf_state_.len;
      VLOG(2) << "total_ins: " << total_instance;
      buf_state_.Debug();
      // if (total_instance == 0) {
      //  return -1;
      //}
    }

    if (!FLAGS_enable_opt_get_features && slot_num_ > 0) {
      FillFeatureBuf(d_walk_, d_feature_);
      if (debug_mode_) {
        int len = buf_size_ > 5000 ? 5000 : buf_size_;
        uint64_t h_walk[len];
        cudaMemcpy(h_walk,
                   d_walk_->ptr(),
                   len * sizeof(uint64_t),
                   cudaMemcpyDeviceToHost);
        uint64_t h_feature[len * slot_num_];
        cudaMemcpy(h_feature,
                   d_feature_->ptr(),
                   len * slot_num_ * sizeof(uint64_t),
                   cudaMemcpyDeviceToHost);
        for (int i = 0; i < len; ++i) {
          std::stringstream ss;
          for (int j = 0; j < slot_num_; ++j) {
            ss << h_feature[i * slot_num_ + j] << " ";
          }
          VLOG(2) << "aft FillFeatureBuf, gpu[" << gpuid_ << "] walk[" << i
                  << "] = " << (uint64_t)h_walk[i] << " feature["
                  << i * slot_num_ << ".." << (i + 1) * slot_num_
                  << "] = " << ss.str();
        }
      }
    }
  }

  uint64_t *walk = reinterpret_cast<uint64_t *>(d_walk_->ptr());
  uint64_t *ins_buf = reinterpret_cast<uint64_t *>(d_ins_buf_->ptr());
  int *random_row = reinterpret_cast<int *>(d_random_row_->ptr());
  int *d_pair_num = reinterpret_cast<int *>(d_pair_num_->ptr());
  cudaMemsetAsync(d_pair_num, 0, sizeof(int), stream_);
  int len = buf_state_.len;
  GraphFillIdKernel<<<GET_BLOCKS(len), CUDA_NUM_THREADS, 0, stream_>>>(
      ins_buf + ins_buf_pair_len_ * 2,
      d_pair_num,
      walk,
      random_row + buf_state_.cursor,
      buf_state_.central_word,
      window_step_[buf_state_.step],
      len,
      walk_len_);
  int h_pair_num;
  cudaMemcpyAsync(
      &h_pair_num, d_pair_num, sizeof(int), cudaMemcpyDeviceToHost, stream_);
  if (!FLAGS_enable_opt_get_features && slot_num_ > 0) {
    uint64_t *feature_buf = reinterpret_cast<uint64_t *>(d_feature_buf_->ptr());
    uint64_t *feature = reinterpret_cast<uint64_t *>(d_feature_->ptr());
    cudaMemsetAsync(d_pair_num, 0, sizeof(int), stream_);
    int len = buf_state_.len;
    VLOG(2) << "feature_buf start[" << ins_buf_pair_len_ * 2 * slot_num_
            << "] len[" << len << "]";
    GraphFillFeatureKernel<<<GET_BLOCKS(len), CUDA_NUM_THREADS, 0, stream_>>>(
        feature_buf + ins_buf_pair_len_ * 2 * slot_num_,
        d_pair_num,
        walk,
        feature,
        random_row + buf_state_.cursor,
        buf_state_.central_word,
        window_step_[buf_state_.step],
        len,
        walk_len_,
        slot_num_);
  }

  cudaStreamSynchronize(stream_);
  ins_buf_pair_len_ += h_pair_num;

  if (debug_mode_) {
    uint64_t h_ins_buf[ins_buf_pair_len_ * 2];
    cudaMemcpy(h_ins_buf,
               ins_buf,
               2 * ins_buf_pair_len_ * sizeof(uint64_t),
               cudaMemcpyDeviceToHost);
    VLOG(2) << "h_pair_num = " << h_pair_num
            << ", ins_buf_pair_len = " << ins_buf_pair_len_;
    for (int xx = 0; xx < 2 * ins_buf_pair_len_; xx++) {
      VLOG(2) << "h_ins_buf[" << xx << "]: " << h_ins_buf[xx];
    }
    delete[] h_ins_buf;

    if (!FLAGS_enable_opt_get_features && slot_num_ > 0) {
      uint64_t *feature_buf =
          reinterpret_cast<uint64_t *>(d_feature_buf_->ptr());
      uint64_t h_feature_buf[(batch_size_ * 2 * 2) * slot_num_];
      cudaMemcpy(h_feature_buf,
                 feature_buf,
                 (batch_size_ * 2 * 2) * slot_num_ * sizeof(uint64_t),
                 cudaMemcpyDeviceToHost);
      for (int xx = 0; xx < (batch_size_ * 2 * 2) * slot_num_; xx++) {
        VLOG(2) << "h_feature_buf[" << xx << "]: " << h_feature_buf[xx];
      }
    }
  }
  return ins_buf_pair_len_;
}

std::vector<std::shared_ptr<phi::Allocation>> GraphDataGenerator::SampleNeighbors(
    int64_t* uniq_nodes, int len, int sample_size,
    std::vector<int>& edges_split_num, int64_t* neighbor_len) {

  auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();
  auto edge_to_id = gpu_graph_ptr->edge_to_id;
  
  auto sample_res = gpu_graph_ptr->graph_neighbor_sample_all_edge_type(
      gpuid_, edge_to_id_len_, (uint64_t*)(uniq_nodes), sample_size, len,
      edge_type_graph_);

  int* all_sample_count_ptr =
      reinterpret_cast<int* >(sample_res.actual_sample_size_mem->ptr());

  auto cumsum_actual_sample_size =
      memory::Alloc(place_, (len * edge_to_id_len_ + 1) * sizeof(int));
  int* cumsum_actual_sample_size_ptr =
      reinterpret_cast<int*>(cumsum_actual_sample_size->ptr());
  cudaMemsetAsync(cumsum_actual_sample_size_ptr, 
                  0, 
                  (len * edge_to_id_len_ + 1) * sizeof(int),
                  stream_);

  size_t temp_storage_bytes = 0;
  CUDA_CHECK(cub::DeviceScan::InclusiveSum(NULL,
                                           temp_storage_bytes,
                                           all_sample_count_ptr,
                                           cumsum_actual_sample_size_ptr + 1,
                                           len * edge_to_id_len_,
                                           stream_));
  auto d_temp_storage = memory::Alloc(place_, temp_storage_bytes);
  CUDA_CHECK(cub::DeviceScan::InclusiveSum(d_temp_storage->ptr(),
                                           temp_storage_bytes,
                                           all_sample_count_ptr,
                                           cumsum_actual_sample_size_ptr + 1,
                                           len * edge_to_id_len_,
                                           stream_));
  cudaStreamSynchronize(stream_);

  edges_split_num.resize(edge_to_id_len_);
  for (int i = 0; i < edge_to_id_len_; i++) {
    cudaMemcpyAsync(
        edges_split_num.data() + i,
        cumsum_actual_sample_size_ptr + (i + 1) * len,
        sizeof(int),
        cudaMemcpyDeviceToHost,
        stream_);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  int all_sample_size = edges_split_num[edge_to_id_len_ - 1];
  auto final_sample_val =
      memory::AllocShared(place_, all_sample_size * sizeof(int64_t));
  auto final_sample_val_dst =
      memory::AllocShared(place_, all_sample_size * sizeof(int64_t));
  int64_t* final_sample_val_ptr =
      reinterpret_cast<int64_t* >(final_sample_val->ptr());
  int64_t* final_sample_val_dst_ptr =
      reinterpret_cast<int64_t* >(final_sample_val_dst->ptr());
  int64_t* all_sample_val_ptr =
      reinterpret_cast<int64_t* >(sample_res.val_mem->ptr());
  fill_actual_neighbors<<<GET_BLOCKS(len * edge_to_id_len_),
                          CUDA_NUM_THREADS,
                          0,
                          stream_>>>(all_sample_val_ptr,
                                     final_sample_val_ptr,
                                     final_sample_val_dst_ptr,
                                     all_sample_count_ptr,
                                     cumsum_actual_sample_size_ptr,
                                     sample_size,
                                     len * edge_to_id_len_,
                                     len);
  *neighbor_len = all_sample_size;
  cudaStreamSynchronize(stream_);

  std::vector<std::shared_ptr<phi::Allocation>> sample_results;
  sample_results.emplace_back(final_sample_val);
  sample_results.emplace_back(final_sample_val_dst);
  return sample_results; 
}

std::shared_ptr<phi::Allocation> GraphDataGenerator::GetReindexResult(
    int64_t* reindex_src_data, const int64_t* center_nodes, int* final_nodes_len,
    int node_len, int64_t neighbor_len) {

  VLOG(2) << gpuid_ << ": Enter GetReindexResult Function";
  const phi::GPUContext& dev_ctx_ = 
    *(static_cast<phi::GPUContext *>(
        platform::DeviceContextPool::Instance().Get(place_)));
  
  // Reset reindex table
  int64_t* d_reindex_table_key_ptr =
      reinterpret_cast<int64_t* >(d_reindex_table_key_->ptr());
  int* d_reindex_table_value_ptr =
      reinterpret_cast<int* >(d_reindex_table_value_->ptr());
  int* d_reindex_table_index_ptr =
      reinterpret_cast<int* >(d_reindex_table_index_->ptr());

  VLOG(2) << gpuid_ << ": ResetReindexTable With -1";
  // Fill table with -1.
  cudaMemsetAsync(d_reindex_table_key_ptr, -1, 
                  reindex_table_size_ * sizeof(int64_t), stream_);
  cudaMemsetAsync(d_reindex_table_value_ptr, -1,
                  reindex_table_size_ * sizeof(int), stream_);
  cudaMemsetAsync(d_reindex_table_index_ptr, -1,
                  reindex_table_size_ * sizeof(int), stream_);

  VLOG(2) << gpuid_ << ": Alloc all_nodes";
  auto all_nodes =
      memory::AllocShared(place_, (node_len + neighbor_len) * sizeof(int64_t));
  int64_t* all_nodes_data = reinterpret_cast<int64_t* >(all_nodes->ptr());

  VLOG(2) << gpuid_ << ": cudaMemcpy all_nodes_data";
  cudaMemcpy(all_nodes_data, center_nodes, sizeof(int64_t) * node_len,
             cudaMemcpyDeviceToDevice);
  cudaMemcpy(all_nodes_data + node_len, reindex_src_data, sizeof(int64_t) * neighbor_len,
             cudaMemcpyDeviceToDevice);

  cudaStreamSynchronize(stream_);
  VLOG(2) << gpuid_ << ": Run phi::FillHashTable";
  auto final_nodes =
      phi::FillHashTable<int64_t, phi::GPUContext>(dev_ctx_, all_nodes_data,
                                              node_len + neighbor_len,
                                              reindex_table_size_,
                                              d_reindex_table_key_ptr,
                                              d_reindex_table_value_ptr,
                                              d_reindex_table_index_ptr,
                                              final_nodes_len);

  VLOG(2) << gpuid_ << ": Run phi::ReindexSrcOutput";
  phi::ReindexSrcOutput<int64_t><<<GET_BLOCKS(neighbor_len), CUDA_NUM_THREADS, 0,
                              stream_>>>(reindex_src_data, neighbor_len,
                                         reindex_table_size_,
                                         d_reindex_table_key_ptr,
                                         d_reindex_table_value_ptr);
  return final_nodes;
}

std::shared_ptr<phi::Allocation> GraphDataGenerator::GenerateSampleGraph(
    uint64_t* node_ids, int len, int* final_len, phi::DenseTensor* inverse) {

  const phi::GPUContext& dev_ctx_ =
    *(static_cast<phi::GPUContext *>(
        platform::DeviceContextPool::Instance().Get(place_)));

  VLOG(2) << "Get Unique Nodes";
  phi::DenseTensor in_x = phi::Empty<int64_t>(dev_ctx_, {len});
  cudaMemcpy(in_x.data<int64_t>(), node_ids, len * sizeof(uint64_t),
             cudaMemcpyDeviceToDevice);

  phi::DenseTensor uniq_nodes, index;
  std::vector<int> axis;
  phi::UniqueKernel<int64_t, phi::GPUContext>(dev_ctx_, in_x, false, true,
      false, axis, phi::DataType::INT32, &uniq_nodes, &index, inverse, &index);

  int64_t* uniq_nodes_data = uniq_nodes.data<int64_t>();
  int uniq_len = uniq_nodes.numel();
  int len_samples = samples_.size();

  int *num_nodes_tensor_ptr_[len_samples];
  int *next_num_nodes_tensor_ptr_[len_samples];
  int64_t *edges_src_tensor_ptr_[len_samples];
  int64_t *edges_dst_tensor_ptr_[len_samples];
  int *edges_split_tensor_ptr_[len_samples];

  VLOG(2) << "Sample Neighbors and Reindex";
  std::vector<int> edges_split_num;
  std::vector<std::shared_ptr<phi::Allocation>> final_nodes_vec;
  std::vector<int> final_nodes_len_vec;

  for (int i = 0; i < len_samples; i++) {

    edges_split_num.clear();
    std::shared_ptr<phi::Allocation> neighbors, reindex_dst;
    int64_t neighbors_len = 0;
    if (i == 0) {
      auto sample_results =
          SampleNeighbors(uniq_nodes_data, uniq_len, samples_[i], edges_split_num,
                          &neighbors_len);
      neighbors = sample_results[0];
      reindex_dst = sample_results[1];
      edges_split_num.push_back(uniq_len);
    } else {
      int64_t* final_nodes_data =
          reinterpret_cast<int64_t* >(final_nodes_vec[i - 1]->ptr());
      auto sample_results =
          SampleNeighbors(final_nodes_data, final_nodes_len_vec[i - 1],
                          samples_[i], edges_split_num, &neighbors_len);
      neighbors = sample_results[0];
      reindex_dst = sample_results[1];
      edges_split_num.push_back(final_nodes_len_vec[i - 1]);
    }
    
    int64_t* reindex_src_data = reinterpret_cast<int64_t* >(neighbors->ptr());
    int64_t* reindex_dst_data = reinterpret_cast<int64_t* >(reindex_dst->ptr());
    int final_nodes_len = 0;
    if (i == 0) {
      auto tmp_final_nodes =
          GetReindexResult(reindex_src_data, uniq_nodes_data, &final_nodes_len,
                           uniq_len, neighbors_len);
      final_nodes_vec.emplace_back(tmp_final_nodes);
      final_nodes_len_vec.emplace_back(final_nodes_len);
    } else {
      int64_t* final_nodes_data =
          reinterpret_cast<int64_t* >(final_nodes_vec[i - 1]->ptr());
      auto tmp_final_nodes =
          GetReindexResult(reindex_src_data, final_nodes_data, &final_nodes_len,
                           final_nodes_len_vec[i - 1], neighbors_len);
      final_nodes_vec.emplace_back(tmp_final_nodes);
      final_nodes_len_vec.emplace_back(final_nodes_len);
    }
    
    int offset = 3 + 2 * slot_num_ + 5 * i;
    num_nodes_tensor_ptr_[i] =
        feed_vec_[offset]->mutable_data<int>({1}, this->place_);
    next_num_nodes_tensor_ptr_[i] =
        feed_vec_[offset + 1]->mutable_data<int>({1}, this->place_);
    edges_src_tensor_ptr_[i] =
        feed_vec_[offset + 2]->mutable_data<int64_t>({neighbors_len, 1}, this->place_);
    edges_dst_tensor_ptr_[i] =
        feed_vec_[offset + 3]->mutable_data<int64_t>({neighbors_len, 1}, this->place_);
    edges_split_tensor_ptr_[i] =
        feed_vec_[offset + 4]->mutable_data<int>({edge_to_id_len_}, this->place_);

    cudaMemcpyAsync(num_nodes_tensor_ptr_[i], final_nodes_len_vec.data() + i,
                    sizeof(int), cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(next_num_nodes_tensor_ptr_[i], edges_split_num.data() + edge_to_id_len_,
                    sizeof(int), cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(edges_split_tensor_ptr_[i], edges_split_num.data(),
                    sizeof(int) * edge_to_id_len_, cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(edges_src_tensor_ptr_[i], reindex_src_data,
                    sizeof(int64_t) * neighbors_len, cudaMemcpyDeviceToDevice, stream_);
    cudaMemcpyAsync(edges_dst_tensor_ptr_[i], reindex_dst_data,
                    sizeof(int64_t) * neighbors_len, cudaMemcpyDeviceToDevice, stream_);

    cudaStreamSynchronize(stream_);
  }

  *final_len = final_nodes_len_vec[len_samples - 1];
  return final_nodes_vec[len_samples - 1];
}

int GraphDataGenerator::GenerateBatch() {
  int total_instance = 0;
  platform::CUDADeviceGuard guard(gpuid_);
  int res = 0;
  if (!gpu_graph_training_) {
    while (cursor_ < h_device_keys_.size()) {
      size_t device_key_size = h_device_keys_[cursor_]->size();
      if (infer_node_type_start_[cursor_] >= device_key_size) {
        cursor_++;
        continue;
      }
      total_instance =
          (infer_node_type_start_[cursor_] + batch_size_ <= device_key_size)
              ? batch_size_
              : device_key_size - infer_node_type_start_[cursor_];
      uint64_t *d_type_keys =
          reinterpret_cast<uint64_t *>(d_device_keys_[cursor_]->ptr());
      d_type_keys += infer_node_type_start_[cursor_];
      infer_node_type_start_[cursor_] += total_instance;
      VLOG(1) << "in graph_data generator:batch_size = " << batch_size_
              << " instance = " << total_instance;
      total_instance *= 2;
      if (!sage_mode_) {
        id_tensor_ptr_ =
            feed_vec_[0]->mutable_data<int64_t>({total_instance, 1}, this->place_);
        show_tensor_ptr_ =
            feed_vec_[1]->mutable_data<int64_t>({total_instance}, this->place_);
        clk_tensor_ptr_ =
            feed_vec_[2]->mutable_data<int64_t>({total_instance}, this->place_);
        CopyDuplicateKeys<<<GET_BLOCKS(total_instance / 2),
                            CUDA_NUM_THREADS,
                            0,
                            stream_>>>(
            id_tensor_ptr_, d_type_keys, total_instance / 2);
        GraphFillCVMKernel<<<GET_BLOCKS(total_instance),
                             CUDA_NUM_THREADS,
                             0,
                             stream_>>>(show_tensor_ptr_, total_instance);
        GraphFillCVMKernel<<<GET_BLOCKS(total_instance),
                             CUDA_NUM_THREADS,
                             0,
                             stream_>>>(clk_tensor_ptr_, total_instance);
      } else {
        
        auto node_buf = memory::AllocShared(
            place_, total_instance * sizeof(uint64_t));
        int64_t* node_buf_ptr = reinterpret_cast<int64_t* >(node_buf->ptr());
        VLOG(1) << "copy center keys";
        CopyDuplicateKeys<<<GET_BLOCKS(total_instance / 2),
                            CUDA_NUM_THREADS,
                            0,
                            stream_>>>(
            node_buf_ptr, d_type_keys, total_instance / 2);
        phi::DenseTensor inverse_;
        VLOG(1) << "generate sample graph";
        uint64_t* node_buf_ptr_ = reinterpret_cast<uint64_t* >(node_buf->ptr());
        std::shared_ptr<phi::Allocation> final_infer_nodes =
            GenerateSampleGraph(node_buf_ptr_, total_instance, &uniq_instance_,
                                &inverse_);
        id_tensor_ptr_ =
            feed_vec_[0]->mutable_data<int64_t>({uniq_instance_, 1}, this->place_);
        show_tensor_ptr_ =
            feed_vec_[1]->mutable_data<int64_t>({uniq_instance_}, this->place_);
        clk_tensor_ptr_ =
            feed_vec_[2]->mutable_data<int64_t>({uniq_instance_}, this->place_);
        int index_offset = 3 + slot_num_ * 2 + 5 * samples_.size();
        index_tensor_ptr_ =
            feed_vec_[index_offset]->mutable_data<int>({total_instance}, this->place_);

        VLOG(1) << "copy id and index";
        cudaMemcpy(id_tensor_ptr_, final_infer_nodes->ptr(),
                   sizeof(int64_t) * uniq_instance_,
                   cudaMemcpyDeviceToDevice);
        cudaMemcpy(index_tensor_ptr_, inverse_.data<int>(), sizeof(int) * total_instance,
                   cudaMemcpyDeviceToDevice);
        GraphFillCVMKernel<<<GET_BLOCKS(uniq_instance_),
                             CUDA_NUM_THREADS,
                             0,
                             stream_>>>(
            show_tensor_ptr_, uniq_instance_);
        GraphFillCVMKernel<<<GET_BLOCKS(uniq_instance_),
                             CUDA_NUM_THREADS,
                             0,
                             stream_>>>(
            clk_tensor_ptr_, uniq_instance_);
      }
      break;
    }
    if (total_instance == 0) {
      return 0;
    }
  } else {
    while (ins_buf_pair_len_ < batch_size_) {
      res = FillInsBuf();
      if (res == -1) {
        if (ins_buf_pair_len_ == 0) {
          return 0;
        } else {
          break;
        }
      }
    }
    total_instance =
        ins_buf_pair_len_ < batch_size_ ? ins_buf_pair_len_ : batch_size_;
    total_instance *= 2;
  }

  int64_t *slot_tensor_ptr_[slot_num_];
  int64_t *slot_lod_tensor_ptr_[slot_num_];
  if (slot_num_ > 0) {
    for (int i = 0; i < slot_num_; ++i) {
      slot_tensor_ptr_[i] = feed_vec_[3 + 2 * i]->mutable_data<int64_t>(
          {total_instance * h_slot_feature_num_map_[i], 1}, this->place_);
      slot_lod_tensor_ptr_[i] = feed_vec_[3 + 2 * i + 1]->mutable_data<int64_t>(
          {total_instance + 1}, this->place_);
    }
    if (FLAGS_enable_opt_get_features || !gpu_graph_training_) {
      cudaMemcpyAsync(d_slot_tensor_ptr_->ptr(),
                      slot_tensor_ptr_,
                      sizeof(uint64_t *) * slot_num_,
                      cudaMemcpyHostToDevice,
                      stream_);
      cudaMemcpyAsync(d_slot_lod_tensor_ptr_->ptr(),
                      slot_lod_tensor_ptr_,
                      sizeof(uint64_t *) * slot_num_,
                      cudaMemcpyHostToDevice,
                      stream_);
    }
  }

  uint64_t *ins_cursor, *ins_buf;
  std::shared_ptr<phi::Allocation> final_nodes;
  phi::DenseTensor inverse;
  if (gpu_graph_training_) {
    VLOG(2) << "total_instance: " << total_instance
            << ", ins_buf_pair_len = " << ins_buf_pair_len_;
    ins_buf = reinterpret_cast<uint64_t *>(d_ins_buf_->ptr());
    ins_cursor = ins_buf + ins_buf_pair_len_ * 2 - total_instance;
    if (!sage_mode_) {
      id_tensor_ptr_ =
          feed_vec_[0]->mutable_data<int64_t>({total_instance, 1}, this->place_);
      show_tensor_ptr_ =
          feed_vec_[1]->mutable_data<int64_t>({total_instance}, this->place_);
      clk_tensor_ptr_ =
          feed_vec_[2]->mutable_data<int64_t>({total_instance}, this->place_);
      cudaMemcpyAsync(id_tensor_ptr_,
                      ins_cursor,
                      sizeof(uint64_t) * total_instance,
                      cudaMemcpyDeviceToDevice,
                      stream_);
      GraphFillCVMKernel<<<GET_BLOCKS(total_instance),
                           CUDA_NUM_THREADS,
                           0,
                           stream_>>>(show_tensor_ptr_, total_instance);
      GraphFillCVMKernel<<<GET_BLOCKS(total_instance),
                           CUDA_NUM_THREADS,
                           0,
                           stream_>>>(clk_tensor_ptr_, total_instance);
    } else {
      VLOG(2) << gpuid_ << " " << "Ready to enter GenerateSampleGraph";
      final_nodes = GenerateSampleGraph(ins_cursor, total_instance, &uniq_instance_,
                                        &inverse);
      VLOG(2) << "Copy Final Results";
      id_tensor_ptr_ =
          feed_vec_[0]->mutable_data<int64_t>({uniq_instance_, 1}, this->place_);
      show_tensor_ptr_ =
          feed_vec_[1]->mutable_data<int64_t>({uniq_instance_}, this->place_);
      clk_tensor_ptr_ =
          feed_vec_[2]->mutable_data<int64_t>({uniq_instance_}, this->place_);
      int index_offset = 3 + slot_num_ * 2 + 5 * samples_.size();
      index_tensor_ptr_ =
          feed_vec_[index_offset]->mutable_data<int>({total_instance}, this->place_);

      cudaMemcpyAsync(id_tensor_ptr_,
                      final_nodes->ptr(),
                      sizeof(int64_t) * uniq_instance_,
                      cudaMemcpyDeviceToDevice,
                      stream_);
      cudaMemcpyAsync(index_tensor_ptr_,
                      inverse.data<int>(),
                      sizeof(int) * total_instance,
                      cudaMemcpyDeviceToDevice,
                      stream_);
      GraphFillCVMKernel<<<GET_BLOCKS(uniq_instance_),
                           CUDA_NUM_THREADS,
                           0,
                           stream_>>>(show_tensor_ptr_, uniq_instance_);
      GraphFillCVMKernel<<<GET_BLOCKS(uniq_instance_),
                           CUDA_NUM_THREADS,
                           0,
                           stream_>>>(clk_tensor_ptr_, uniq_instance_);

    }
  } else {
    ins_cursor = (uint64_t *)id_tensor_ptr_;
  }

  if (slot_num_ > 0) {
    uint64_t *feature_buf = reinterpret_cast<uint64_t *>(d_feature_buf_->ptr());
    if (FLAGS_enable_opt_get_features || !gpu_graph_training_) {
      FillFeatureBuf(ins_cursor, feature_buf, total_instance);
      // FillFeatureBuf(id_tensor_ptr_, feature_buf, total_instance);
      if (debug_mode_) {
        uint64_t h_walk[total_instance];
        cudaMemcpy(h_walk,
                   ins_cursor,
                   total_instance * sizeof(uint64_t),
                   cudaMemcpyDeviceToHost);
        uint64_t h_feature[total_instance * fea_num_per_node_];
        cudaMemcpy(h_feature,
                   feature_buf,
                   total_instance * fea_num_per_node_ * sizeof(uint64_t),
                   cudaMemcpyDeviceToHost);
        for (int i = 0; i < total_instance; ++i) {
          std::stringstream ss;
          for (int j = 0; j < fea_num_per_node_; ++j) {
            ss << h_feature[i * fea_num_per_node_ + j] << " ";
          }
          VLOG(2) << "aft FillFeatureBuf, gpu[" << gpuid_ << "] walk[" << i
                  << "] = " << (uint64_t)h_walk[i] << " feature["
                  << i * fea_num_per_node_ << ".." << (i + 1) * fea_num_per_node_
                  << "] = " << ss.str();
        }
      }

      GraphFillSlotKernel<<<GET_BLOCKS(total_instance * fea_num_per_node_),
                            CUDA_NUM_THREADS,
                            0,
                            stream_>>>((uint64_t *)d_slot_tensor_ptr_->ptr(),
                                       feature_buf,
                                       total_instance * fea_num_per_node_,
                                       total_instance,
                                       slot_num_,
                                       (int*)d_slot_feature_num_map_->ptr(),
                                       fea_num_per_node_,
                                       (int*)d_actual_slot_id_map_->ptr(),
                                       (int*)d_fea_offset_map_->ptr());
      GraphFillSlotLodKernelOpt<<<GET_BLOCKS((total_instance + 1) * slot_num_),
                                  CUDA_NUM_THREADS,
                                  0,
                                  stream_>>>(
          (uint64_t *)d_slot_lod_tensor_ptr_->ptr(),
          (total_instance + 1) * slot_num_,
          total_instance + 1,
          (int*)d_slot_feature_num_map_->ptr());
    } else {
      for (int i = 0; i < slot_num_; ++i) {
        int feature_buf_offset =
            (ins_buf_pair_len_ * 2 - total_instance) * slot_num_ + i * 2;
        for (int j = 0; j < total_instance; j += 2) {
          VLOG(2) << "slot_tensor[" << i << "][" << j << "] <- feature_buf["
                  << feature_buf_offset + j * slot_num_ << "]";
          VLOG(2) << "slot_tensor[" << i << "][" << j + 1 << "] <- feature_buf["
                  << feature_buf_offset + j * slot_num_ + 1 << "]";
          cudaMemcpyAsync(slot_tensor_ptr_[i] + j,
                          &feature_buf[feature_buf_offset + j * slot_num_],
                          sizeof(uint64_t) * 2,
                          cudaMemcpyDeviceToDevice,
                          stream_);
        }
        GraphFillSlotLodKernel<<<GET_BLOCKS(total_instance),
                                 CUDA_NUM_THREADS,
                                 0,
                                 stream_>>>(slot_lod_tensor_ptr_[i],
                                            total_instance + 1);
      }
    }
  }

  offset_.clear();
  offset_.push_back(0);
  if (!sage_mode_) {
    offset_.push_back(total_instance);
  } else {
    offset_.push_back(uniq_instance_);
  }
  LoD lod{offset_};
  feed_vec_[0]->set_lod(lod);
  if (slot_num_ > 0) {
    for (int i = 0; i < slot_num_; ++i) {
      feed_vec_[3 + 2 * i]->set_lod(lod);
    }
  }

  cudaStreamSynchronize(stream_);
  if (!gpu_graph_training_) return 1;
  ins_buf_pair_len_ -= total_instance / 2;
  if (debug_mode_) {
    uint64_t h_slot_tensor[fea_num_per_node_][total_instance];
    uint64_t h_slot_lod_tensor[slot_num_][total_instance + 1];
    for (int i = 0; i < slot_num_; ++i) {
      cudaMemcpy(h_slot_tensor[i],
                 slot_tensor_ptr_[i],
                 total_instance * sizeof(uint64_t),
                 cudaMemcpyDeviceToHost);
      int len = total_instance > 5000 ? 5000 : total_instance;
      for (int j = 0; j < len; ++j) {
        VLOG(2) << "gpu[" << gpuid_ << "] slot_tensor[" << i << "][" << j
                << "] = " << h_slot_tensor[i][j];
      }

      cudaMemcpy(h_slot_lod_tensor[i],
                 slot_lod_tensor_ptr_[i],
                 (total_instance + 1) * sizeof(uint64_t),
                 cudaMemcpyDeviceToHost);
      len = total_instance + 1 > 5000 ? 5000 : total_instance + 1;
      for (int j = 0; j < len; ++j) {
        VLOG(2) << "gpu[" << gpuid_ << "] slot_lod_tensor[" << i << "][" << j
                << "] = " << h_slot_lod_tensor[i][j];
      }
    }
  }

  return 1;
}

__global__ void GraphFillSampleKeysKernel(uint64_t *neighbors,
                                          uint64_t *sample_keys,
                                          int *prefix_sum,
                                          int *sampleidx2row,
                                          int *tmp_sampleidx2row,
                                          int *actual_sample_size,
                                          int cur_degree,
                                          int len) {
  CUDA_KERNEL_LOOP(idx, len) {
    for (int k = 0; k < actual_sample_size[idx]; k++) {
      size_t offset = prefix_sum[idx] + k;
      sample_keys[offset] = neighbors[idx * cur_degree + k];
      tmp_sampleidx2row[offset] = sampleidx2row[idx] + k;
    }
  }
}

__global__ void GraphDoWalkKernel(uint64_t *neighbors,
                                  uint64_t *walk,
                                  int *d_prefix_sum,
                                  int *actual_sample_size,
                                  int cur_degree,
                                  int step,
                                  int len,
                                  int *id_cnt,
                                  int *sampleidx2row,
                                  int col_size) {
  CUDA_KERNEL_LOOP(i, len) {
    for (int k = 0; k < actual_sample_size[i]; k++) {
      // int idx = sampleidx2row[i];
      size_t row = sampleidx2row[k + d_prefix_sum[i]];
      // size_t row = idx * cur_degree + k;
      size_t col = step;
      size_t offset = (row * col_size + col);
      walk[offset] = neighbors[i * cur_degree + k];
    }
  }
}

// Fill keys to the first column of walk
__global__ void GraphFillFirstStepKernel(int *prefix_sum,
                                         int *sampleidx2row,
                                         uint64_t *walk,
                                         uint64_t *keys,
                                         int len,
                                         int walk_degree,
                                         int col_size,
                                         int *actual_sample_size,
                                         uint64_t *neighbors,
                                         uint64_t *sample_keys) {
  CUDA_KERNEL_LOOP(idx, len) {
    for (int k = 0; k < actual_sample_size[idx]; k++) {
      size_t row = prefix_sum[idx] + k;
      sample_keys[row] = neighbors[idx * walk_degree + k];
      sampleidx2row[row] = row;

      size_t offset = col_size * row;
      walk[offset] = keys[idx];
      walk[offset + 1] = neighbors[idx * walk_degree + k];
    }
  }
}

// Fill sample_res to the stepth column of walk
void GraphDataGenerator::FillOneStep(uint64_t *d_start_ids,
                                     uint64_t *walk,
                                     int len,
                                     NeighborSampleResult &sample_res,
                                     int cur_degree,
                                     int step,
                                     int *len_per_row) {
  size_t temp_storage_bytes = 0;
  int *d_actual_sample_size = sample_res.actual_sample_size;
  uint64_t *d_neighbors = sample_res.val;
  int *d_prefix_sum = reinterpret_cast<int *>(d_prefix_sum_->ptr());
  uint64_t *d_sample_keys = reinterpret_cast<uint64_t *>(d_sample_keys_->ptr());
  int *d_sampleidx2row =
      reinterpret_cast<int *>(d_sampleidx2rows_[cur_sampleidx2row_]->ptr());
  int *d_tmp_sampleidx2row =
      reinterpret_cast<int *>(d_sampleidx2rows_[1 - cur_sampleidx2row_]->ptr());

  CUDA_CHECK(cub::DeviceScan::InclusiveSum(NULL,
                                           temp_storage_bytes,
                                           d_actual_sample_size,
                                           d_prefix_sum + 1,
                                           len,
                                           stream_));
  auto d_temp_storage = memory::Alloc(place_, temp_storage_bytes);

  CUDA_CHECK(cub::DeviceScan::InclusiveSum(d_temp_storage->ptr(),
                                           temp_storage_bytes,
                                           d_actual_sample_size,
                                           d_prefix_sum + 1,
                                           len,
                                           stream_));

  cudaStreamSynchronize(stream_);

  if (step == 1) {
    GraphFillFirstStepKernel<<<GET_BLOCKS(len), CUDA_NUM_THREADS, 0, stream_>>>(
        d_prefix_sum,
        d_tmp_sampleidx2row,
        walk,
        d_start_ids,
        len,
        walk_degree_,
        walk_len_,
        d_actual_sample_size,
        d_neighbors,
        d_sample_keys);

  } else {
    GraphFillSampleKeysKernel<<<GET_BLOCKS(len),
                                CUDA_NUM_THREADS,
                                0,
                                stream_>>>(d_neighbors,
                                           d_sample_keys,
                                           d_prefix_sum,
                                           d_sampleidx2row,
                                           d_tmp_sampleidx2row,
                                           d_actual_sample_size,
                                           cur_degree,
                                           len);

    GraphDoWalkKernel<<<GET_BLOCKS(len), CUDA_NUM_THREADS, 0, stream_>>>(
        d_neighbors,
        walk,
        d_prefix_sum,
        d_actual_sample_size,
        cur_degree,
        step,
        len,
        len_per_row,
        d_tmp_sampleidx2row,
        walk_len_);
  }
  if (debug_mode_) {
    size_t once_max_sample_keynum = walk_degree_ * once_sample_startid_len_;
    int *h_prefix_sum = new int[len + 1];
    int *h_actual_size = new int[len];
    int *h_offset2idx = new int[once_max_sample_keynum];
    uint64_t h_sample_keys[once_max_sample_keynum];
    cudaMemcpy(h_offset2idx,
               d_tmp_sampleidx2row,
               once_max_sample_keynum * sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaMemcpy(h_prefix_sum,
               d_prefix_sum,
               (len + 1) * sizeof(int),
               cudaMemcpyDeviceToHost);
    for (int xx = 0; xx < once_max_sample_keynum; xx++) {
      VLOG(2) << "h_offset2idx[" << xx << "]: " << h_offset2idx[xx];
    }
    for (int xx = 0; xx < len + 1; xx++) {
      VLOG(2) << "h_prefix_sum[" << xx << "]: " << h_prefix_sum[xx];
    }
    delete[] h_prefix_sum;
    delete[] h_actual_size;
    delete[] h_offset2idx;
    delete[] h_sample_keys;
  }
  cudaStreamSynchronize(stream_);
  cur_sampleidx2row_ = 1 - cur_sampleidx2row_;
}

int GraphDataGenerator::FillFeatureBuf(uint64_t *d_walk,
                                       uint64_t *d_feature,
                                       size_t key_num) {
  platform::CUDADeviceGuard guard(gpuid_);

  auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();
  int ret = gpu_graph_ptr->get_feature_of_nodes(
      gpuid_, d_walk, d_feature, key_num, slot_num_,
      (int*)d_slot_feature_num_map_->ptr(), fea_num_per_node_);
  return ret;
}

int GraphDataGenerator::FillFeatureBuf(
    std::shared_ptr<phi::Allocation> d_walk,
    std::shared_ptr<phi::Allocation> d_feature) {
  platform::CUDADeviceGuard guard(gpuid_);

  auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();
  int ret = gpu_graph_ptr->get_feature_of_nodes(gpuid_,
                                                (uint64_t *)d_walk->ptr(),
                                                (uint64_t *)d_feature->ptr(),
                                                buf_size_,
                                                slot_num_,
                                                (int*)d_slot_feature_num_map_->ptr(),
                                                fea_num_per_node_);
  return ret;
}

int GraphDataGenerator::FillWalkBuf(std::shared_ptr<phi::Allocation> d_walk) {
  platform::CUDADeviceGuard guard(gpuid_);
  size_t once_max_sample_keynum = walk_degree_ * once_sample_startid_len_;
  ////////
  uint64_t *h_walk;
  uint64_t *h_sample_keys;
  int *h_offset2idx;
  int *h_len_per_row;
  uint64_t *h_prefix_sum;
  if (debug_mode_) {
    h_walk = new uint64_t[buf_size_];
    h_sample_keys = new uint64_t[once_max_sample_keynum];
    h_offset2idx = new int[once_max_sample_keynum];
    h_len_per_row = new int[once_max_sample_keynum];
    h_prefix_sum = new uint64_t[once_max_sample_keynum + 1];
  }
  ///////
  auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();
  uint64_t *walk = reinterpret_cast<uint64_t *>(d_walk->ptr());
  int *len_per_row = reinterpret_cast<int *>(d_len_per_row_->ptr());
  uint64_t *d_sample_keys = reinterpret_cast<uint64_t *>(d_sample_keys_->ptr());
  cudaMemsetAsync(walk, 0, buf_size_ * sizeof(uint64_t), stream_);
  cudaMemsetAsync(
      len_per_row, 0, once_max_sample_keynum * sizeof(int), stream_);
  int i = 0;
  int total_row = 0;
  size_t node_type_len = first_node_type_.size();
  int remain_size =
      buf_size_ - walk_degree_ * once_sample_startid_len_ * walk_len_;

  while (i <= remain_size) {
    int cur_node_idx = cursor_ % node_type_len;
    int node_type = first_node_type_[cur_node_idx];
    auto &path = meta_path_[cur_node_idx];
    size_t start = node_type_start_[node_type];
    // auto node_query_result = gpu_graph_ptr->query_node_list(
    //    gpuid_, node_type, start, once_sample_startid_len_);

    // int tmp_len = node_query_result.actual_sample_size;
    VLOG(2) << "choose start type: " << node_type;
    int type_index = type_to_index_[node_type];
    size_t device_key_size = h_device_keys_[type_index]->size();
    VLOG(2) << "type: " << node_type << " size: " << device_key_size
            << " start: " << start;
    uint64_t *d_type_keys =
        reinterpret_cast<uint64_t *>(d_device_keys_[type_index]->ptr());
    int tmp_len = start + once_sample_startid_len_ > device_key_size
                      ? device_key_size - start
                      : once_sample_startid_len_;
    node_type_start_[node_type] = tmp_len + start;
    if (tmp_len == 0) {
      finish_node_type_.insert(node_type);
      if (finish_node_type_.size() == node_type_start_.size()) {
        break;
      }
      cursor_ += 1;
      continue;
    }
    // if (tmp_len == 0) {
    //  break;
    //}
    VLOG(2) << "i = " << i << " buf_size_ = " << buf_size_
            << " tmp_len = " << tmp_len << " cursor = " << cursor_
            << " once_max_sample_keynum = " << once_max_sample_keynum;
    uint64_t *cur_walk = walk + i;

    NeighborSampleQuery q;
    q.initialize(gpuid_,
                 path[0],
                 (uint64_t)(d_type_keys + start),
                 walk_degree_,
                 tmp_len);
    auto sample_res = gpu_graph_ptr->graph_neighbor_sample_v3(q, false, true);

    int step = 1;
    VLOG(2) << "sample edge type: " << path[0] << " step: " << 1;
    jump_rows_ = sample_res.total_sample_size;
    FillOneStep(d_type_keys + start,
                cur_walk,
                tmp_len,
                sample_res,
                walk_degree_,
                step,
                len_per_row);
    VLOG(2) << "jump_row: " << jump_rows_;
    /////////
    if (debug_mode_) {
      cudaMemcpy(
          h_walk, walk, buf_size_ * sizeof(uint64_t), cudaMemcpyDeviceToHost);
      for (int xx = 0; xx < buf_size_; xx++) {
        VLOG(2) << "h_walk[" << xx << "]: " << h_walk[xx];
      }
    }
    /////////
    step++;
    size_t path_len = path.size();
    for (; step < walk_len_; step++) {
      if (sample_res.total_sample_size == 0) {
        break;
      }
      auto sample_key_mem = sample_res.actual_val_mem;
      uint64_t *sample_keys_ptr =
          reinterpret_cast<uint64_t *>(sample_key_mem->ptr());
      int edge_type_id = path[(step - 1) % path_len];
      VLOG(2) << "sample edge type: " << edge_type_id << " step: " << step;
      q.initialize(gpuid_,
                   edge_type_id,
                   (uint64_t)sample_keys_ptr,
                   1,
                   sample_res.total_sample_size);
      int sample_key_len =  sample_res.total_sample_size;
      sample_res = gpu_graph_ptr->graph_neighbor_sample_v3(q, false, true);

      FillOneStep(d_type_keys + start,
                  cur_walk,
                  sample_key_len,
                  sample_res,
                  1,
                  step,
                  len_per_row);
      if (debug_mode_) {
        cudaMemcpy(
            h_walk, walk, buf_size_ * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        for (int xx = 0; xx < buf_size_; xx++) {
          VLOG(2) << "h_walk[" << xx << "]: " << h_walk[xx];
        }
      }
    }
    // cursor_ += tmp_len;
    i += jump_rows_ * walk_len_;
    total_row += jump_rows_;
    cursor_ += 1;
  }
  buf_state_.Reset(total_row);
  int *d_random_row = reinterpret_cast<int *>(d_random_row_->ptr());

  thrust::random::default_random_engine engine(shuffle_seed_);
  const auto &exec_policy = thrust::cuda::par.on(stream_);
  thrust::counting_iterator<int> cnt_iter(0);
  thrust::shuffle_copy(exec_policy,
                       cnt_iter,
                       cnt_iter + total_row,
                       thrust::device_pointer_cast(d_random_row),
                       engine);

  cudaStreamSynchronize(stream_);
  shuffle_seed_ = engine();

  if (debug_mode_) {
    int *h_random_row = new int[total_row + 10];
    cudaMemcpy(h_random_row,
               d_random_row,
               total_row * sizeof(int),
               cudaMemcpyDeviceToHost);
    for (int xx = 0; xx < total_row; xx++) {
      VLOG(2) << "h_random_row[" << xx << "]: " << h_random_row[xx];
    }
    delete[] h_random_row;
    delete[] h_walk;
    delete[] h_sample_keys;
    delete[] h_offset2idx;
    delete[] h_len_per_row;
    delete[] h_prefix_sum;
  }
  return total_row != 0;
}

void GraphDataGenerator::AllocResource(const paddle::platform::Place &place,
                                       std::vector<LoDTensor *> feed_vec) {
  place_ = place;
  gpuid_ = place_.GetDeviceId();
  VLOG(3) << "gpuid " << gpuid_;
  stream_ = dynamic_cast<phi::GPUContext *>(
                platform::DeviceContextPool::Instance().Get(place))
                ->stream();
  feed_vec_ = feed_vec;
  if (!sage_mode_) {
    slot_num_ = (feed_vec_.size() - 3) / 2;
  } else {
    slot_num_ = (feed_vec_.size() - 4 - samples_.size() * 5) / 2;
  }

  auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();
  h_slot_feature_num_map_ = gpu_graph_ptr->slot_feature_num_map();
  fea_num_per_node_ = 0;
  for (int i = 0; i < slot_num_; ++i) {
    fea_num_per_node_ += h_slot_feature_num_map_[i];
  }
  std::vector<int> h_actual_slot_id_map, h_fea_offset_map;
  h_actual_slot_id_map.resize(fea_num_per_node_);
  h_fea_offset_map.resize(fea_num_per_node_);
  for (int slot_id = 0, fea_idx = 0; slot_id < slot_num_; ++slot_id) {
    for (int j = 0; j < h_slot_feature_num_map_[slot_id]; ++j, ++fea_idx) {
      h_actual_slot_id_map[fea_idx] = slot_id;
      h_fea_offset_map[fea_idx] = j;
    }
  }

  d_slot_feature_num_map_ = memory::Alloc(place, slot_num_ * sizeof(int));
  cudaMemcpy(d_slot_feature_num_map_->ptr(), h_slot_feature_num_map_.data(),
          sizeof(int) * slot_num_, cudaMemcpyHostToDevice);
  d_actual_slot_id_map_ = memory::Alloc(place, fea_num_per_node_ * sizeof(int));
  cudaMemcpy(d_actual_slot_id_map_->ptr(), h_actual_slot_id_map.data(),
          sizeof(int) * fea_num_per_node_, cudaMemcpyHostToDevice);
  d_fea_offset_map_ = memory::Alloc(place, fea_num_per_node_ * sizeof(int));
  cudaMemcpy(d_fea_offset_map_->ptr(), h_fea_offset_map.data(),
          sizeof(int) * fea_num_per_node_, cudaMemcpyHostToDevice);

  // d_device_keys_.resize(h_device_keys_.size());
  VLOG(2) << "h_device_keys size: " << h_device_keys_.size();
  infer_node_type_start_ = std::vector<int>(h_device_keys_.size(), 0);
  for (size_t i = 0; i < h_device_keys_.size(); i++) {
    for (size_t j = 0; j < h_device_keys_[i]->size(); j++) {
      VLOG(3) << "h_device_keys_[" << i << "][" << j
              << "] = " << (*(h_device_keys_[i]))[j];
    }
    auto buf = memory::AllocShared(
        place_, h_device_keys_[i]->size() * sizeof(uint64_t));
    d_device_keys_.push_back(buf);
    CUDA_CHECK(cudaMemcpyAsync(buf->ptr(),
                               h_device_keys_[i]->data(),
                               h_device_keys_[i]->size() * sizeof(uint64_t),
                               cudaMemcpyHostToDevice,
                               stream_));
  }
  // h_device_keys_ = h_device_keys;
  // device_key_size_ = h_device_keys_->size();
  // d_device_keys_ =
  //    memory::AllocShared(place_, device_key_size_ * sizeof(int64_t));
  // CUDA_CHECK(cudaMemcpyAsync(d_device_keys_->ptr(), h_device_keys_->data(),
  //                           device_key_size_ * sizeof(int64_t),
  //                           cudaMemcpyHostToDevice, stream_));
  size_t once_max_sample_keynum = walk_degree_ * once_sample_startid_len_;
  d_prefix_sum_ =
      memory::AllocShared(place_, (once_max_sample_keynum + 1) * sizeof(int));
  int *d_prefix_sum_ptr = reinterpret_cast<int *>(d_prefix_sum_->ptr());
  cudaMemsetAsync(
      d_prefix_sum_ptr, 0, (once_max_sample_keynum + 1) * sizeof(int), stream_);
  cursor_ = 0;
  jump_rows_ = 0;
  d_walk_ = memory::AllocShared(place_, buf_size_ * sizeof(uint64_t));
  cudaMemsetAsync(d_walk_->ptr(), 0, buf_size_ * sizeof(uint64_t), stream_);
  if (!FLAGS_enable_opt_get_features && slot_num_ > 0) {
    d_feature_ =
        memory::AllocShared(place_, buf_size_ * slot_num_ * sizeof(uint64_t));
    cudaMemsetAsync(
        d_feature_->ptr(), 0, buf_size_ * sizeof(uint64_t), stream_);
  }
  d_sample_keys_ =
      memory::AllocShared(place_, once_max_sample_keynum * sizeof(uint64_t));

  d_sampleidx2rows_.push_back(
      memory::AllocShared(place_, once_max_sample_keynum * sizeof(int)));
  d_sampleidx2rows_.push_back(
      memory::AllocShared(place_, once_max_sample_keynum * sizeof(int)));
  cur_sampleidx2row_ = 0;

  d_len_per_row_ =
      memory::AllocShared(place_, once_max_sample_keynum * sizeof(int));
  for (int i = -window_; i < 0; i++) {
    window_step_.push_back(i);
  }
  for (int i = 0; i < window_; i++) {
    window_step_.push_back(i + 1);
  }
  buf_state_.Init(batch_size_, walk_len_, &window_step_);
  d_random_row_ = memory::AllocShared(
      place_,
      (once_sample_startid_len_ * walk_degree_ * repeat_time_) * sizeof(int));
  shuffle_seed_ = 0;

  ins_buf_pair_len_ = 0;
  d_ins_buf_ =
      memory::AllocShared(place_, (batch_size_ * 2 * 2) * sizeof(uint64_t));
  if (slot_num_ > 0) {
    d_feature_buf_ = memory::AllocShared(
        place_, (batch_size_ * 2 * 2) * slot_num_ * sizeof(uint64_t));
  }
  d_pair_num_ = memory::AllocShared(place_, sizeof(int));
  if (FLAGS_enable_opt_get_features && slot_num_ > 0) {
    d_slot_tensor_ptr_ =
        memory::AllocShared(place_, slot_num_ * sizeof(uint64_t *));
    d_slot_lod_tensor_ptr_ =
        memory::AllocShared(place_, slot_num_ * sizeof(uint64_t *));
  }

  if (sage_mode_) {
    reindex_table_size_ = batch_size_ * 2;
    // get hashtable size
    for (int i = 0; i < samples_.size(); i++) {
      reindex_table_size_ *= (samples_[i] * edge_to_id_len_ + 1);
    }
    int64_t next_pow2 = 
        1 << static_cast<size_t>(1 + std::log2(reindex_table_size_ >> 1));
    reindex_table_size_ = next_pow2 << 1;

    d_reindex_table_key_ = 
        memory::AllocShared(place_, reindex_table_size_ * sizeof(int64_t));
    d_reindex_table_value_ =
        memory::AllocShared(place_, reindex_table_size_ * sizeof(int));
    d_reindex_table_index_ =
        memory::AllocShared(place_, reindex_table_size_ * sizeof(int));
    auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();
    edge_type_graph_ =
        gpu_graph_ptr->get_edge_type_graph(gpuid_, edge_to_id_len_);
  }

  cudaStreamSynchronize(stream_);
}

void GraphDataGenerator::SetConfig(
    const paddle::framework::DataFeedDesc &data_feed_desc) {
  auto graph_config = data_feed_desc.graph_config();
  walk_degree_ = graph_config.walk_degree();
  walk_len_ = graph_config.walk_len();
  window_ = graph_config.window();
  once_sample_startid_len_ = graph_config.once_sample_startid_len();
  debug_mode_ = graph_config.debug_mode();
  gpu_graph_training_ = graph_config.gpu_graph_training();
  if (debug_mode_ || !gpu_graph_training_) {
    batch_size_ = graph_config.batch_size();
  } else {
    batch_size_ = once_sample_startid_len_;
  }
  repeat_time_ = graph_config.sample_times_one_chunk();
  buf_size_ =
      once_sample_startid_len_ * walk_len_ * walk_degree_ * repeat_time_;
  VLOG(2) << "Confirm GraphConfig, walk_degree : " << walk_degree_
          << ", walk_len : " << walk_len_ << ", window : " << window_
          << ", once_sample_startid_len : " << once_sample_startid_len_
          << ", sample_times_one_chunk : " << repeat_time_
          << ", batch_size: " << batch_size_;
  std::string first_node_type = graph_config.first_node_type();
  std::string meta_path = graph_config.meta_path();
  sage_mode_ = graph_config.sage_mode();
  std::string str_samples = graph_config.samples();

  auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();
  auto edge_to_id = gpu_graph_ptr->edge_to_id;
  edge_to_id_len_ = edge_to_id.size();
  auto node_to_id = gpu_graph_ptr->feature_to_id;
  // parse first_node_type
  auto node_types =
      paddle::string::split_string<std::string>(first_node_type, ";");
  VLOG(2) << "node_types: " << first_node_type;
  finish_node_type_.clear();
  node_type_start_.clear();
  for (auto &type : node_types) {
    auto iter = node_to_id.find(type);
    PADDLE_ENFORCE_NE(
        iter,
        node_to_id.end(),
        platform::errors::NotFound("(%s) is not found in node_to_id.", type));
    VLOG(2) << "node_to_id[" << type << "] = " << iter->second;
    first_node_type_.push_back(iter->second);
    node_type_start_[iter->second] = 0;
  }
  meta_path_.resize(first_node_type_.size());
  auto meta_paths = paddle::string::split_string<std::string>(meta_path, ";");

  for (size_t i = 0; i < meta_paths.size(); i++) {
    auto path = meta_paths[i];
    auto nodes = paddle::string::split_string<std::string>(path, "-");
    for (auto &node : nodes) {
      auto iter = edge_to_id.find(node);
      PADDLE_ENFORCE_NE(
          iter,
          edge_to_id.end(),
          platform::errors::NotFound("(%s) is not found in edge_to_id.", node));
      VLOG(2) << "edge_to_id[" << node << "] = " << iter->second;
      meta_path_[i].push_back(iter->second);
    }
  }

  auto samples = paddle::string::split_string<std::string>(str_samples, ";");
  for (size_t i = 0; i < samples.size(); i++) {
    int sample_size = std::stoi(samples[i]);
    samples_.emplace_back(sample_size);
  }
};

}  // namespace framework
}  // namespace paddle
#endif
