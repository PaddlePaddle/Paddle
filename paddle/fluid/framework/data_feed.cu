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
#include <thrust/random.h>
#include <thrust/shuffle.h>
#include <sstream>
#include "cub/cub.cuh"
#include "paddle/fluid/framework/fleet/heter_ps/gpu_graph_node.h"
#include "paddle/fluid/framework/fleet/heter_ps/gpu_graph_utils.h"
#include "paddle/fluid/framework/fleet/heter_ps/graph_gpu_wrapper.h"
#include "paddle/fluid/framework/fleet/heter_ps/hashtable.h"
#include "paddle/fluid/framework/fleet/ps_gpu_wrapper.h"

DECLARE_bool(enable_opt_get_features);
DECLARE_int32(gpugraph_storage_mode);
DECLARE_double(gpugraph_hbm_table_load_factor);

namespace paddle {
namespace framework {

#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

#define DEBUG_STATE(state)                                             \
  VLOG(2) << "left: " << state->left << " right: " << state->right     \
          << " central_word: " << state->central_word                  \
          << " step: " << state->step << " cursor: " << state->cursor  \
          << " len: " << state->len << " row_num: " << state->row_num; \
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
    DEBUG_STATE(state);
    return state->len;
  } else if (state->GetNextCentrolWord()) {
    DEBUG_STATE(state);
    return state->len;
  } else if (state->GetNextBatch()) {
    DEBUG_STATE(state);
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

int GraphDataGenerator::FillIdShowClkTensor(int total_instance,
                                            bool gpu_graph_training,
                                            size_t cursor) {
  id_tensor_ptr_ =
      feed_vec_[0]->mutable_data<int64_t>({total_instance, 1}, this->place_);
  show_tensor_ptr_ =
      feed_vec_[1]->mutable_data<int64_t>({total_instance}, this->place_);
  clk_tensor_ptr_ =
      feed_vec_[2]->mutable_data<int64_t>({total_instance}, this->place_);
  auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();
  if (gpu_graph_training) {
    uint64_t *ins_cursor, *ins_buf;
    ins_buf = reinterpret_cast<uint64_t *>(d_ins_buf_->ptr());
    ins_cursor = ins_buf + ins_buf_pair_len_ * 2 - total_instance;
    cudaMemcpyAsync(id_tensor_ptr_,
                    ins_cursor,
                    sizeof(uint64_t) * total_instance,
                    cudaMemcpyDeviceToDevice,
                    train_stream_);
  } else {
    uint64_t *d_type_keys =
        reinterpret_cast<uint64_t *>(d_device_keys_[cursor]->ptr());
    d_type_keys += infer_node_start_;
    infer_node_start_ += total_instance / 2;
    CopyDuplicateKeys<<<GET_BLOCKS(total_instance / 2),
                        CUDA_NUM_THREADS,
                        0,
                        train_stream_>>>(
        id_tensor_ptr_, d_type_keys, total_instance / 2);
  }

  GraphFillCVMKernel<<<GET_BLOCKS(total_instance),
                       CUDA_NUM_THREADS,
                       0,
                       train_stream_>>>(show_tensor_ptr_, total_instance);
  GraphFillCVMKernel<<<GET_BLOCKS(total_instance),
                       CUDA_NUM_THREADS,
                       0,
                       train_stream_>>>(clk_tensor_ptr_, total_instance);
  return 0;
}

int GraphDataGenerator::FillGraphSlotFeature(int total_instance,
                                             bool gpu_graph_training) {
  int64_t *slot_tensor_ptr_[slot_num_];
  int64_t *slot_lod_tensor_ptr_[slot_num_];
  for (int i = 0; i < slot_num_; ++i) {
    slot_tensor_ptr_[i] = feed_vec_[3 + 2 * i]->mutable_data<int64_t>(
        {total_instance * h_slot_feature_num_map_[i], 1}, this->place_);
    slot_lod_tensor_ptr_[i] = feed_vec_[3 + 2 * i + 1]->mutable_data<int64_t>(
        {total_instance + 1}, this->place_);
  }
  uint64_t *ins_cursor, *ins_buf;
  if (gpu_graph_training) {
    ins_buf = reinterpret_cast<uint64_t *>(d_ins_buf_->ptr());
    ins_cursor = ins_buf + ins_buf_pair_len_ * 2 - total_instance;
  } else {
    id_tensor_ptr_ =
        feed_vec_[0]->mutable_data<int64_t>({total_instance, 1}, this->place_);
    ins_cursor = (uint64_t *)id_tensor_ptr_;
  }

  cudaMemcpyAsync(d_slot_tensor_ptr_->ptr(),
                  slot_tensor_ptr_,
                  sizeof(uint64_t *) * slot_num_,
                  cudaMemcpyHostToDevice,
                  train_stream_);
  cudaMemcpyAsync(d_slot_lod_tensor_ptr_->ptr(),
                  slot_lod_tensor_ptr_,
                  sizeof(uint64_t *) * slot_num_,
                  cudaMemcpyHostToDevice,
                  train_stream_);
  uint64_t *feature_buf = reinterpret_cast<uint64_t *>(d_feature_buf_->ptr());
  FillFeatureBuf(ins_cursor, feature_buf, total_instance);
  GraphFillSlotKernel<<<GET_BLOCKS(total_instance * fea_num_per_node_),
                        CUDA_NUM_THREADS,
                        0,
                        train_stream_>>>((uint64_t *)d_slot_tensor_ptr_->ptr(),
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
                              train_stream_>>>(
      (uint64_t *)d_slot_lod_tensor_ptr_->ptr(),
      (total_instance + 1) * slot_num_,
      total_instance + 1,
      (int*)d_slot_feature_num_map_->ptr());
  if (debug_mode_) {
    uint64_t h_walk[total_instance];
    cudaMemcpy(h_walk,
               ins_cursor,
               total_instance * sizeof(uint64_t),
               cudaMemcpyDeviceToHost);
    uint64_t h_feature[total_instance * slot_num_ * fea_num_per_node_];
    cudaMemcpy(h_feature,
               feature_buf,
               total_instance * fea_num_per_node_ * slot_num_ * sizeof(uint64_t),
               cudaMemcpyDeviceToHost);
    for (int i = 0; i < total_instance; ++i) {
      std::stringstream ss;
      for (int j = 0; j < fea_num_per_node_; ++j) {
        ss << h_feature[i * fea_num_per_node_ + j] << " ";
      }
      VLOG(2) << "aft FillFeatureBuf, gpu[" << gpuid_ << "] walk[" << i
              << "] = " << (uint64_t)h_walk[i] << " feature[" << i * fea_num_per_node_
              << ".." << (i + 1) * fea_num_per_node_ << "] = " << ss.str();
    }

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
  return 0;
}

int GraphDataGenerator::MakeInsPair() {
  uint64_t *walk = reinterpret_cast<uint64_t *>(d_walk_->ptr());
  uint64_t *ins_buf = reinterpret_cast<uint64_t *>(d_ins_buf_->ptr());
  int *random_row = reinterpret_cast<int *>(d_random_row_->ptr());
  int *d_pair_num = reinterpret_cast<int *>(d_pair_num_->ptr());
  cudaMemsetAsync(d_pair_num, 0, sizeof(int), train_stream_);
  int len = buf_state_.len;
  // make pair
  GraphFillIdKernel<<<GET_BLOCKS(len), CUDA_NUM_THREADS, 0, train_stream_>>>(
      ins_buf + ins_buf_pair_len_ * 2,
      d_pair_num,
      walk,
      random_row + buf_state_.cursor,
      buf_state_.central_word,
      window_step_[buf_state_.step],
      len,
      walk_len_);
  int h_pair_num;
  cudaMemcpyAsync(&h_pair_num,
                  d_pair_num,
                  sizeof(int),
                  cudaMemcpyDeviceToHost,
                  train_stream_);
  cudaStreamSynchronize(train_stream_);
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
  }
  return ins_buf_pair_len_;
}

int GraphDataGenerator::FillInsBuf() {
  if (ins_buf_pair_len_ >= batch_size_) {
    return batch_size_;
  }
  int total_instance = AcquireInstance(&buf_state_);

  VLOG(2) << "total_ins: " << total_instance;
  buf_state_.Debug();

  if (total_instance == 0) {
    return -1;
  }
  return MakeInsPair();
}

int GraphDataGenerator::GenerateBatch() {
  int total_instance = 0;
  platform::CUDADeviceGuard guard(gpuid_);
  int res = 0;
  auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();
  if (!gpu_graph_training_) {
    total_instance = (infer_node_start_ + batch_size_ <= infer_node_end_)
                         ? batch_size_
                         : infer_node_end_ - infer_node_start_;
    VLOG(1) << "in graph_data generator:batch_size = " << batch_size_
            << " instance = " << total_instance;
    total_instance *= 2;
    if (total_instance == 0) {
      return 0;
    }
    FillIdShowClkTensor(total_instance, gpu_graph_training_, cursor_);
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
    VLOG(2) << "total_instance: " << total_instance
            << ", ins_buf_pair_len = " << ins_buf_pair_len_;
    FillIdShowClkTensor(total_instance, gpu_graph_training_);
  }

  if (slot_num_ > 0) {
    FillGraphSlotFeature(total_instance, gpu_graph_training_);
  }
  offset_.clear();
  offset_.push_back(0);
  offset_.push_back(total_instance);
  LoD lod{offset_};
  feed_vec_[0]->set_lod(lod);
  if (slot_num_ > 0) {
    for (int i = 0; i < slot_num_; ++i) {
      feed_vec_[3 + 2 * i]->set_lod(lod);
    }
  }

  cudaStreamSynchronize(train_stream_);
  if (!gpu_graph_training_) return 1;
  ins_buf_pair_len_ -= total_instance / 2;
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

__global__ void GetUniqueFeaNum(uint64_t *d_in,
                                uint64_t *unique_num,
                                size_t len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint64_t local_num;
  if (threadIdx.x == 0) {
    local_num = 0;
  }
  __syncthreads();

  if (i < len - 1) {
    if (d_in[i] != d_in[i + 1]) {
      atomicAdd(&local_num, 1);
    }
  }
  if (i == len - 1) {
    atomicAdd(&local_num, 1);
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(unique_num, local_num);
  }
}

__global__ void UniqueFeature(uint64_t *d_in,
                              uint64_t *d_out,
                              uint64_t *unique_num,
                              size_t len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint64_t local_key[CUDA_NUM_THREADS];
  __shared__ uint64_t local_num;
  __shared__ uint64_t global_num;
  if (threadIdx.x == 0) {
    local_num = 0;
  }
  __syncthreads();

  if (i < len - 1) {
    if (d_in[i] != d_in[i + 1]) {
      size_t dst = atomicAdd(&local_num, 1);
      local_key[dst] = d_in[i];
    }
  }
  if (i == len - 1) {
    size_t dst = atomicAdd(&local_num, 1);
    local_key[dst] = d_in[i];
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    global_num = atomicAdd(unique_num, local_num);
  }
  __syncthreads();

  if (threadIdx.x < local_num) {
    d_out[global_num + threadIdx.x] = local_key[threadIdx.x];
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
                                           sample_stream_));
  auto d_temp_storage = memory::Alloc(
      place_,
      temp_storage_bytes,
      phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));

  CUDA_CHECK(cub::DeviceScan::InclusiveSum(d_temp_storage->ptr(),
                                           temp_storage_bytes,
                                           d_actual_sample_size,
                                           d_prefix_sum + 1,
                                           len,
                                           sample_stream_));

  cudaStreamSynchronize(sample_stream_);

  if (step == 1) {
    GraphFillFirstStepKernel<<<GET_BLOCKS(len),
                               CUDA_NUM_THREADS,
                               0,
                               sample_stream_>>>(d_prefix_sum,
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
                                sample_stream_>>>(d_neighbors,
                                                  d_sample_keys,
                                                  d_prefix_sum,
                                                  d_sampleidx2row,
                                                  d_tmp_sampleidx2row,
                                                  d_actual_sample_size,
                                                  cur_degree,
                                                  len);

    GraphDoWalkKernel<<<GET_BLOCKS(len), CUDA_NUM_THREADS, 0, sample_stream_>>>(
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
  }
  cudaStreamSynchronize(sample_stream_);
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

// 尝试插入table, 0表示插入成功
int GraphDataGenerator::InsertTable(
    const unsigned long *d_keys,
    unsigned long len,
    std::shared_ptr<phi::Allocation> d_uniq_node_num) {
  uint64_t h_uniq_node_num = 0;
  uint64_t *d_uniq_node_num_ptr =
      reinterpret_cast<uint64_t *>(d_uniq_node_num->ptr());
  cudaMemcpyAsync(&h_uniq_node_num,
                  d_uniq_node_num_ptr,
                  sizeof(uint64_t),
                  cudaMemcpyDeviceToHost,
                  sample_stream_);
  cudaStreamSynchronize(sample_stream_);
  // 产生了足够多的node，采样结束
  VLOG(2) << "table capcity: " << train_table_cap_ << ", " << h_uniq_node_num
          << " used";
  if (h_uniq_node_num + len >= train_table_cap_) {
    return 1;
  }
  table_->insert(d_keys, len, d_uniq_node_num_ptr, sample_stream_);
  CUDA_CHECK(cudaStreamSynchronize(sample_stream_));
  return 0;
}

void GraphDataGenerator::DoWalk() {
  int device_id = place_.GetDeviceId();
  debug_gpu_memory_info(device_id, "DoWalk start");
  if (gpu_graph_training_) {
    FillWalkBuf();
  } else {
    FillInferBuf();
  }
  debug_gpu_memory_info(device_id, "DoWalk end");
}

void GraphDataGenerator::clear_gpu_mem() {
  d_len_per_row_.reset();
  d_sample_keys_.reset();
  d_prefix_sum_.reset();
  for (size_t i = 0; i < d_sampleidx2rows_.size(); i++) {
    d_sampleidx2rows_[i].reset();
  }
  delete table_;
}

int GraphDataGenerator::FillInferBuf() {
  platform::CUDADeviceGuard guard(gpuid_);
  auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();
  auto &global_infer_node_type_start =
      gpu_graph_ptr->global_infer_node_type_start_[gpuid_];
  auto &infer_cursor = gpu_graph_ptr->infer_cursor_[thread_id_];
  total_row_ = 0;
  if (infer_cursor < h_device_keys_len_.size()) {
    if (global_infer_node_type_start[infer_cursor] >=
        h_device_keys_len_[infer_cursor]) {
      infer_cursor++;
      if (infer_cursor >= h_device_keys_len_.size()) {
        return 0;
      }
    }
    size_t device_key_size = h_device_keys_len_[infer_cursor];
    total_row_ =
        (global_infer_node_type_start[infer_cursor] + infer_table_cap_ <=
         device_key_size)
            ? infer_table_cap_
            : device_key_size - global_infer_node_type_start[infer_cursor];

    host_vec_.resize(total_row_);
    uint64_t *d_type_keys =
        reinterpret_cast<uint64_t *>(d_device_keys_[infer_cursor]->ptr());
    cudaMemcpyAsync(host_vec_.data(),
                    d_type_keys + global_infer_node_type_start[infer_cursor],
                    sizeof(uint64_t) * total_row_,
                    cudaMemcpyDeviceToHost,
                    sample_stream_);
    cudaStreamSynchronize(sample_stream_);
    VLOG(1) << "cursor: " << infer_cursor
            << " start: " << global_infer_node_type_start[infer_cursor]
            << " num: " << total_row_;
    infer_node_start_ = global_infer_node_type_start[infer_cursor];
    global_infer_node_type_start[infer_cursor] += total_row_;
    infer_node_end_ = global_infer_node_type_start[infer_cursor];
    cursor_ = infer_cursor;
  }
  return 0;
}

void GraphDataGenerator::ClearSampleState() {
  auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();
  auto &finish_node_type = gpu_graph_ptr->finish_node_type_[gpuid_];
  auto &node_type_start = gpu_graph_ptr->node_type_start_[gpuid_];
  finish_node_type.clear();
  for (auto iter = node_type_start.begin(); iter != node_type_start.end(); iter++) {
    iter->second = 0;
  }
}

int GraphDataGenerator::FillWalkBuf() {
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
  uint64_t *walk = reinterpret_cast<uint64_t *>(d_walk_->ptr());
  int *len_per_row = reinterpret_cast<int *>(d_len_per_row_->ptr());
  uint64_t *d_sample_keys = reinterpret_cast<uint64_t *>(d_sample_keys_->ptr());
  cudaMemsetAsync(walk, 0, buf_size_ * sizeof(uint64_t), sample_stream_);
  // cudaMemsetAsync(
  //     len_per_row, 0, once_max_sample_keynum * sizeof(int), sample_stream_);
  int sample_times = 0;
  int i = 0;
  total_row_ = 0;

  // 获取全局采样状态
  auto &first_node_type = gpu_graph_ptr->first_node_type_;
  auto &meta_path = gpu_graph_ptr->meta_path_;
  auto &node_type_start = gpu_graph_ptr->node_type_start_[gpuid_];
  auto &finish_node_type = gpu_graph_ptr->finish_node_type_[gpuid_];
  auto &type_to_index = gpu_graph_ptr->get_graph_type_to_index();
  auto &cursor = gpu_graph_ptr->cursor_[thread_id_];
  size_t node_type_len = first_node_type.size();
  int remain_size =
      buf_size_ - walk_degree_ * once_sample_startid_len_ * walk_len_;
  int total_samples = 0;

  while (i <= remain_size) {
    int cur_node_idx = cursor % node_type_len;
    int node_type = first_node_type[cur_node_idx];
    auto &path = meta_path[cur_node_idx];
    size_t start = node_type_start[node_type];
    VLOG(2) << "cur_node_idx = " << cur_node_idx
            << " meta_path.size = " << meta_path.size();
    // auto node_query_result = gpu_graph_ptr->query_node_list(
    //     gpuid_, node_type, start, once_sample_startid_len_);

    // int tmp_len = node_query_result.actual_sample_size;
    VLOG(2) << "choose start type: " << node_type;
    int type_index = type_to_index[node_type];
    size_t device_key_size = h_device_keys_len_[type_index];
    VLOG(2) << "type: " << node_type << " size: " << device_key_size
            << " start: " << start;
    uint64_t *d_type_keys =
        reinterpret_cast<uint64_t *>(d_device_keys_[type_index]->ptr());
    int tmp_len = start + once_sample_startid_len_ > device_key_size
                      ? device_key_size - start
                      : once_sample_startid_len_;
    bool update = true;
    if (tmp_len == 0) {
      finish_node_type.insert(node_type);
      if (finish_node_type.size() == node_type_start.size()) {
        cursor = 0;
        epoch_finish_ = true;
        break;
      }
      cursor += 1;
      continue;
    }

    VLOG(2) << "gpuid = " << gpuid_ << " path[0] = " << path[0];
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
    total_samples += sample_res.total_sample_size;
    VLOG(2) << "i = " << i << " start = " << start << " tmp_len = " << tmp_len
            << " cursor = " << node_type << " cur_node_idx = " << cur_node_idx
            << " jump row: " << jump_rows_;
    VLOG(2) << "jump_row: " << jump_rows_;
    if (jump_rows_ == 0) {
      node_type_start[node_type] = tmp_len + start;
      cursor += 1;
      continue;
    }

    if (FLAGS_gpugraph_storage_mode != GpuGraphStorageMode::WHOLE_HBM) {
      if (InsertTable(d_type_keys + start, tmp_len, d_uniq_node_num_) != 0) {
        VLOG(2) << "in step 0, insert key stage, table is full";
        update = false;
        break;
      }
      if (InsertTable(sample_res.actual_val, sample_res.total_sample_size, d_uniq_node_num_) !=
          0) {
        VLOG(2) << "in step 0, insert sample res stage, table is full";
        update = false;
        break;
      }
    }
    FillOneStep(d_type_keys + start,
                cur_walk,
                tmp_len,
                sample_res,
                walk_degree_,
                step,
                len_per_row);
    /////////
    if (debug_mode_) {
      cudaMemcpy(
          h_walk, walk, buf_size_ * sizeof(uint64_t), cudaMemcpyDeviceToHost);
      for (int xx = 0; xx < buf_size_; xx++) {
        VLOG(2) << "h_walk[" << xx << "]: " << h_walk[xx];
      }
    }

    VLOG(2) << "sample, step=" << step << " sample_keys=" << tmp_len
        << " sample_res_len=" << sample_res.total_sample_size;

    /////////
    step++;
    size_t path_len = path.size();
    for (; step < walk_len_; step++) {
      if (sample_res.total_sample_size == 0) {
        VLOG(2) << "sample finish, step=" << step;
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
      int sample_key_len = sample_res.total_sample_size;
      sample_res = gpu_graph_ptr->graph_neighbor_sample_v3(q, false, true);
      total_samples += sample_res.total_sample_size;
      if (FLAGS_gpugraph_storage_mode != GpuGraphStorageMode::WHOLE_HBM) {
        if (InsertTable(sample_res.actual_val, sample_res.total_sample_size, d_uniq_node_num_) !=
            0) {
          VLOG(2) << "in step: " << step << ", table is full";
          update = false;
          break;
        }
      }
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

      VLOG(2) << "sample, step=" << step << " sample_keys=" << sample_key_len
          << " sample_res_len=" << sample_res.total_sample_size;
    }
    // 此时更新全局采样状态
    if (update == true) {
      node_type_start[node_type] = tmp_len + start;
      i += jump_rows_ * walk_len_;
      total_row_ += jump_rows_;
      cursor += 1;
      sample_times++;
    } else {
      VLOG(2) << "table is full, not update stat!";
      break;
    }
  }
  buf_state_.Reset(total_row_);
  int *d_random_row = reinterpret_cast<int *>(d_random_row_->ptr());

  thrust::random::default_random_engine engine(shuffle_seed_);
  const auto &exec_policy = thrust::cuda::par.on(sample_stream_);
  thrust::counting_iterator<int> cnt_iter(0);
  thrust::shuffle_copy(exec_policy,
                       cnt_iter,
                       cnt_iter + total_row_,
                       thrust::device_pointer_cast(d_random_row),
                       engine);

  cudaStreamSynchronize(sample_stream_);
  shuffle_seed_ = engine();

  if (debug_mode_) {
    int *h_random_row = new int[total_row_ + 10];
    cudaMemcpy(h_random_row,
               d_random_row,
               total_row_ * sizeof(int),
               cudaMemcpyDeviceToHost);
    for (int xx = 0; xx < total_row_; xx++) {
      VLOG(2) << "h_random_row[" << xx << "]: " << h_random_row[xx];
    }
    delete[] h_random_row;
    delete[] h_walk;
    delete[] h_sample_keys;
    delete[] h_offset2idx;
    delete[] h_len_per_row;
    delete[] h_prefix_sum;
  }
  if (FLAGS_gpugraph_storage_mode != GpuGraphStorageMode::WHOLE_HBM) {
    // table_->prefetch(cudaCpuDeviceId, sample_stream_);
    // thrust::pair<uint64_t, uint64_t> *kv = table_->data();
    // size_t size = table_->size();
    // uint64_t unused_key = std::numeric_limits<uint64_t>::max();
    // for (size_t i = 0; i < size; i++) {
    //   if (kv[i].first == unused_key) {
    //     continue;
    //   }
    //   host_vec_.push_back(kv[i].first);
    // }

    uint64_t h_uniq_node_num = 0;
    uint64_t *d_uniq_node_num =
        reinterpret_cast<uint64_t *>(d_uniq_node_num_->ptr());
    cudaMemcpyAsync(&h_uniq_node_num,
                    d_uniq_node_num,
                    sizeof(uint64_t),
                    cudaMemcpyDeviceToHost,
                    sample_stream_);
    cudaStreamSynchronize(sample_stream_);
    VLOG(2) << "h_uniq_node_num: " << h_uniq_node_num;
    // 临时显存, 存储去重后的nodeid
    auto d_uniq_node = memory::AllocShared(
        place_,
        h_uniq_node_num * sizeof(uint64_t),
        phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));
    uint64_t *d_uniq_node_ptr =
        reinterpret_cast<uint64_t *>(d_uniq_node->ptr());

    auto d_node_cursor = memory::AllocShared(
        place_,
        sizeof(uint64_t),
        phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));

    uint64_t *d_node_cursor_ptr =
        reinterpret_cast<uint64_t *>(d_node_cursor->ptr());
    cudaMemsetAsync(d_node_cursor_ptr, 0, sizeof(uint64_t), sample_stream_);
    // uint64_t unused_key = std::numeric_limits<uint64_t>::max();
    table_->get_keys(d_uniq_node_ptr, d_node_cursor_ptr, sample_stream_);

    cudaStreamSynchronize(sample_stream_);

    host_vec_.resize(h_uniq_node_num);
    cudaMemcpyAsync(host_vec_.data(),
                    d_uniq_node_ptr,
                    sizeof(uint64_t) * h_uniq_node_num,
                    cudaMemcpyDeviceToHost,
                    sample_stream_);
    cudaStreamSynchronize(sample_stream_);
    
    VLOG(0) << "sample_times:" << sample_times
        << ", d_walk_size:" << buf_size_
        << ", d_walk_offset:" << i
        << ", total_rows:" << total_row_
        << ", total_samples:" << total_samples
        << ", h_uniq_node_num:" << h_uniq_node_num;
  }
  return total_row_ != 0;
}

void GraphDataGenerator::SetFeedVec(std::vector<LoDTensor *> feed_vec) {
  feed_vec_ = feed_vec;
}
void GraphDataGenerator::AllocResource(int thread_id,
                                       std::vector<LoDTensor *> feed_vec) {
  auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();
  gpuid_ = gpu_graph_ptr->device_id_mapping[thread_id];
  thread_id_ = thread_id;
  place_ = platform::CUDAPlace(gpuid_);
  debug_gpu_memory_info(gpuid_, "AllocResource start");

  platform::CUDADeviceGuard guard(gpuid_);
  if (FLAGS_gpugraph_storage_mode != GpuGraphStorageMode::WHOLE_HBM) {
    table_ = new HashTable<uint64_t, uint64_t>(
        train_table_cap_ / FLAGS_gpugraph_hbm_table_load_factor);
  }
  VLOG(1) << "AllocResource gpuid " << gpuid_
          << " feed_vec.size: " << feed_vec.size()
          << " table cap: " << train_table_cap_;
  sample_stream_ = gpu_graph_ptr->get_local_stream(gpuid_);
  train_stream_ = dynamic_cast<phi::GPUContext *>(
                      platform::DeviceContextPool::Instance().Get(place_))
                      ->stream();
  // feed_vec_ = feed_vec;
  slot_num_ = (feed_vec.size() - 3) / 2;

  // infer_node_type_start_ = std::vector<int>(h_device_keys_.size(), 0);
  // for (size_t i = 0; i < h_device_keys_.size(); i++) {
  //   for (size_t j = 0; j < h_device_keys_[i]->size(); j++) {
  //     VLOG(3) << "h_device_keys_[" << i << "][" << j
  //             << "] = " << (*(h_device_keys_[i]))[j];
  //   }
  //   auto buf = memory::AllocShared(
  //       place_, h_device_keys_[i]->size() * sizeof(uint64_t));
  //   d_device_keys_.push_back(buf);
  //   CUDA_CHECK(cudaMemcpyAsync(buf->ptr(),
  //                              h_device_keys_[i]->data(),
  //                              h_device_keys_[i]->size() * sizeof(uint64_t),
  //                              cudaMemcpyHostToDevice,
  //                              stream_));
  // }
  auto &d_graph_all_type_keys = gpu_graph_ptr->d_graph_all_type_total_keys_;
  auto &h_graph_all_type_keys_len = gpu_graph_ptr->h_graph_all_type_keys_len_;

  for (size_t i = 0; i < d_graph_all_type_keys.size(); i++) {
    d_device_keys_.push_back(d_graph_all_type_keys[i][thread_id]);
    h_device_keys_len_.push_back(h_graph_all_type_keys_len[i][thread_id]);
  }
  VLOG(2) << "h_device_keys size: " << h_device_keys_len_.size();
  


  size_t once_max_sample_keynum = walk_degree_ * once_sample_startid_len_;
  d_prefix_sum_ = memory::AllocShared(
      place_,
      (once_max_sample_keynum + 1) * sizeof(int),
      phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));
  int *d_prefix_sum_ptr = reinterpret_cast<int *>(d_prefix_sum_->ptr());
  cudaMemsetAsync(d_prefix_sum_ptr,
                  0,
                  (once_max_sample_keynum + 1) * sizeof(int),
                  sample_stream_);
  cursor_ = 0;
  jump_rows_ = 0;
  d_uniq_node_num_ = memory::AllocShared(
      place_,
      sizeof(uint64_t),
      phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));
  cudaMemsetAsync(d_uniq_node_num_->ptr(), 0, sizeof(uint64_t), sample_stream_);

  d_walk_ = memory::AllocShared(
      place_,
      buf_size_ * sizeof(uint64_t),
      phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));
  cudaMemsetAsync(
      d_walk_->ptr(), 0, buf_size_ * sizeof(uint64_t), sample_stream_);
  d_sample_keys_ = memory::AllocShared(
      place_,
      once_max_sample_keynum * sizeof(uint64_t),
      phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));

  d_sampleidx2rows_.push_back(memory::AllocShared(
      place_,
      once_max_sample_keynum * sizeof(int),
      phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_))));
  d_sampleidx2rows_.push_back(memory::AllocShared(
      place_,
      once_max_sample_keynum * sizeof(int),
      phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_))));
  cur_sampleidx2row_ = 0;

  d_len_per_row_ = memory::AllocShared(
      place_,
      once_max_sample_keynum * sizeof(int),
      phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));
  for (int i = -window_; i < 0; i++) {
    window_step_.push_back(i);
  }
  for (int i = 0; i < window_; i++) {
    window_step_.push_back(i + 1);
  }
  buf_state_.Init(batch_size_, walk_len_, &window_step_);
  d_random_row_ = memory::AllocShared(
      place_,
      (once_sample_startid_len_ * walk_degree_ * repeat_time_) * sizeof(int),
      phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));
  shuffle_seed_ = 0;

  ins_buf_pair_len_ = 0;
  d_ins_buf_ =
      memory::AllocShared(place_, (batch_size_ * 2 * 2) * sizeof(uint64_t));
  d_pair_num_ = memory::AllocShared(place_, sizeof(int));

  d_slot_tensor_ptr_ =
      memory::AllocShared(place_, slot_num_ * sizeof(uint64_t *));
  d_slot_lod_tensor_ptr_ =
      memory::AllocShared(place_, slot_num_ * sizeof(uint64_t *));

  cudaStreamSynchronize(sample_stream_);

  debug_gpu_memory_info(gpuid_, "AllocResource end");
}

void GraphDataGenerator::AllocTrainResource(int thread_id) {
  if (slot_num_ > 0) {
    platform::CUDADeviceGuard guard(gpuid_);
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
      
    d_slot_feature_num_map_ = memory::Alloc(place_, slot_num_ * sizeof(int));
    cudaMemcpy(d_slot_feature_num_map_->ptr(), h_slot_feature_num_map_.data(),
          sizeof(int) * slot_num_, cudaMemcpyHostToDevice);
    d_actual_slot_id_map_ = memory::Alloc(place_, fea_num_per_node_ * sizeof(int));
    cudaMemcpy(d_actual_slot_id_map_->ptr(), h_actual_slot_id_map.data(),
          sizeof(int) * fea_num_per_node_, cudaMemcpyHostToDevice);
    d_fea_offset_map_ = memory::Alloc(place_, fea_num_per_node_ * sizeof(int));
    cudaMemcpy(d_fea_offset_map_->ptr(), h_fea_offset_map.data(),
          sizeof(int) * fea_num_per_node_, cudaMemcpyHostToDevice);
    d_feature_buf_ = memory::AllocShared(
        place_, (batch_size_ * 2 * 2) * fea_num_per_node_ * sizeof(uint64_t));
  }
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
  train_table_cap_ = graph_config.train_table_cap();
  infer_table_cap_ = graph_config.infer_table_cap();
  epoch_finish_ = false;
  VLOG(0) << "Confirm GraphConfig, walk_degree : " << walk_degree_
          << ", walk_len : " << walk_len_ << ", window : " << window_
          << ", once_sample_startid_len : " << once_sample_startid_len_
          << ", sample_times_one_chunk : " << repeat_time_
          << ", batch_size: " << batch_size_
          << ", train_table_cap: " << train_table_cap_
          << ", infer_table_cap: " << infer_table_cap_;
  std::string first_node_type = graph_config.first_node_type();
  std::string meta_path = graph_config.meta_path();
  auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();
  debug_gpu_memory_info("init_conf start");
  gpu_graph_ptr->init_conf(first_node_type, meta_path);
  debug_gpu_memory_info("init_conf end");
};

}  // namespace framework
}  // namespace paddle
#endif
