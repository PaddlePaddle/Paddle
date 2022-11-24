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
<<<<<<< HEAD
#include "paddle/fluid/framework/fleet/heter_ps/gpu_graph_utils.h"
#include "paddle/fluid/framework/fleet/heter_ps/graph_gpu_wrapper.h"
#include "paddle/fluid/framework/fleet/heter_ps/hashtable.h"
#include "paddle/fluid/framework/fleet/ps_gpu_wrapper.h"

DECLARE_bool(enable_opt_get_features);
DECLARE_int32(gpugraph_storage_mode);
DECLARE_double(gpugraph_hbm_table_load_factor);
=======
#include "paddle/fluid/framework/fleet/heter_ps/graph_gpu_wrapper.h"

DECLARE_bool(enable_opt_get_features);
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f

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
<<<<<<< HEAD
    DEBUG_STATE(state);
    return state->len;
  } else if (state->GetNextCentrolWord()) {
    DEBUG_STATE(state);
    return state->len;
  } else if (state->GetNextBatch()) {
    DEBUG_STATE(state);
=======
    state->Debug();
    return state->len;
  } else if (state->GetNextCentrolWord()) {
    state->Debug();
    return state->len;
  } else if (state->GetNextBatch()) {
    state->Debug();
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
    return state->len;
  }
  return 0;
}

<<<<<<< HEAD
// TODO opt
=======
// TODO(fengdanlei): opt
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
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
<<<<<<< HEAD
                                    int slot_num,
                                    int *slot_feature_num_map,
                                    int fea_num_per_node,
                                    int *actual_slot_id_map,
                                    int *fea_offset_map) {
  CUDA_KERNEL_LOOP(idx, len) {
    int fea_idx = idx / total_ins;
    int ins_idx = idx % total_ins;
    int actual_slot_id = actual_slot_id_map[fea_idx];
    int fea_offset = fea_offset_map[fea_idx];
    ((uint64_t *)(id_tensor[actual_slot_id]))
        [ins_idx * slot_feature_num_map[actual_slot_id] + fea_offset] =
            feature_buf[ins_idx * fea_num_per_node + fea_idx];
=======
                                    int slot_num) {
  CUDA_KERNEL_LOOP(idx, len) {
    int slot_idx = idx / total_ins;
    int ins_idx = idx % total_ins;
    ((uint64_t *)(id_tensor[slot_idx]))[ins_idx] =  // NOLINT
        feature_buf[ins_idx * slot_num + slot_idx];
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
  }
}

__global__ void GraphFillSlotLodKernelOpt(uint64_t *id_tensor,
                                          int len,
<<<<<<< HEAD
                                          int total_ins,
                                          int *slot_feature_num_map) {
  CUDA_KERNEL_LOOP(idx, len) {
    int slot_idx = idx / total_ins;
    int ins_idx = idx % total_ins;
    ((uint64_t *)(id_tensor[slot_idx]))[ins_idx] =
        ins_idx * slot_feature_num_map[slot_idx];
=======
                                          int total_ins) {
  CUDA_KERNEL_LOOP(idx, len) {
    int slot_idx = idx / total_ins;
    int ins_idx = idx % total_ins;
    ((uint64_t *)(id_tensor[slot_idx]))[ins_idx] = ins_idx;  // NOLINT
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
  }
}

__global__ void GraphFillSlotLodKernel(int64_t *id_tensor, int len) {
  CUDA_KERNEL_LOOP(idx, len) { id_tensor[idx] = idx; }
}

<<<<<<< HEAD
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
  uint64_t *ins_cursor, *ins_buf;
  if (gpu_graph_training) {
    ins_buf = reinterpret_cast<uint64_t *>(d_ins_buf_->ptr());
    ins_cursor = ins_buf + ins_buf_pair_len_ * 2 - total_instance;
  } else {
    id_tensor_ptr_ =
        feed_vec_[0]->mutable_data<int64_t>({total_instance, 1}, this->place_);
    ins_cursor = (uint64_t *)id_tensor_ptr_;
  }
  return FillSlotFeature(ins_cursor, total_instance);
}

int GraphDataGenerator::MakeInsPair() {
=======
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
        uint64_t h_walk[len];  // NOLINT
        cudaMemcpy(h_walk,
                   d_walk_->ptr(),
                   len * sizeof(uint64_t),
                   cudaMemcpyDeviceToHost);
        uint64_t h_feature[len * slot_num_];  // NOLINT
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

>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
  uint64_t *walk = reinterpret_cast<uint64_t *>(d_walk_->ptr());
  uint64_t *ins_buf = reinterpret_cast<uint64_t *>(d_ins_buf_->ptr());
  int *random_row = reinterpret_cast<int *>(d_random_row_->ptr());
  int *d_pair_num = reinterpret_cast<int *>(d_pair_num_->ptr());
<<<<<<< HEAD
  cudaMemsetAsync(d_pair_num, 0, sizeof(int), train_stream_);
  int len = buf_state_.len;
  // make pair
  GraphFillIdKernel<<<GET_BLOCKS(len), CUDA_NUM_THREADS, 0, train_stream_>>>(
=======
  cudaMemsetAsync(d_pair_num, 0, sizeof(int), stream_);
  int len = buf_state_.len;
  GraphFillIdKernel<<<GET_BLOCKS(len), CUDA_NUM_THREADS, 0, stream_>>>(
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
      ins_buf + ins_buf_pair_len_ * 2,
      d_pair_num,
      walk,
      random_row + buf_state_.cursor,
      buf_state_.central_word,
      window_step_[buf_state_.step],
      len,
      walk_len_);
  int h_pair_num;
<<<<<<< HEAD
  cudaMemcpyAsync(&h_pair_num,
                  d_pair_num,
                  sizeof(int),
                  cudaMemcpyDeviceToHost,
                  train_stream_);
  cudaStreamSynchronize(train_stream_);
  ins_buf_pair_len_ += h_pair_num;

  if (debug_mode_) {
    uint64_t h_ins_buf[ins_buf_pair_len_ * 2];
=======
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
    uint64_t h_ins_buf[ins_buf_pair_len_ * 2];  // NOLINT
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
    cudaMemcpy(h_ins_buf,
               ins_buf,
               2 * ins_buf_pair_len_ * sizeof(uint64_t),
               cudaMemcpyDeviceToHost);
    VLOG(2) << "h_pair_num = " << h_pair_num
            << ", ins_buf_pair_len = " << ins_buf_pair_len_;
    for (int xx = 0; xx < 2 * ins_buf_pair_len_; xx++) {
      VLOG(2) << "h_ins_buf[" << xx << "]: " << h_ins_buf[xx];
    }
<<<<<<< HEAD
=======
    delete[] h_ins_buf;

    if (!FLAGS_enable_opt_get_features && slot_num_ > 0) {
      uint64_t *feature_buf =
          reinterpret_cast<uint64_t *>(d_feature_buf_->ptr());
      uint64_t h_feature_buf[(batch_size_ * 2 * 2) * slot_num_];  // NOLINT
      cudaMemcpy(h_feature_buf,
                 feature_buf,
                 (batch_size_ * 2 * 2) * slot_num_ * sizeof(uint64_t),
                 cudaMemcpyDeviceToHost);
      for (int xx = 0; xx < (batch_size_ * 2 * 2) * slot_num_; xx++) {
        VLOG(2) << "h_feature_buf[" << xx << "]: " << h_feature_buf[xx];
      }
    }
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
  }
  return ins_buf_pair_len_;
}

<<<<<<< HEAD
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

=======
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
int GraphDataGenerator::GenerateBatch() {
  int total_instance = 0;
  platform::CUDADeviceGuard guard(gpuid_);
  int res = 0;
<<<<<<< HEAD
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
=======
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
      id_tensor_ptr_ = feed_vec_[0]->mutable_data<int64_t>({total_instance, 1},
                                                           this->place_);
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
      break;
    }
    if (total_instance == 0) {
      return 0;
    }
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
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
<<<<<<< HEAD
    total_instance *= 2;
    VLOG(2) << "total_instance: " << total_instance
            << ", ins_buf_pair_len = " << ins_buf_pair_len_;
    FillIdShowClkTensor(total_instance, gpu_graph_training_);
  }

  if (slot_num_ > 0) {
    FillGraphSlotFeature(total_instance, gpu_graph_training_);
  }
=======

    total_instance *= 2;
    id_tensor_ptr_ =
        feed_vec_[0]->mutable_data<int64_t>({total_instance, 1}, this->place_);
    show_tensor_ptr_ =
        feed_vec_[1]->mutable_data<int64_t>({total_instance}, this->place_);
    clk_tensor_ptr_ =
        feed_vec_[2]->mutable_data<int64_t>({total_instance}, this->place_);
  }

  int64_t *slot_tensor_ptr_[slot_num_];
  int64_t *slot_lod_tensor_ptr_[slot_num_];
  if (slot_num_ > 0) {
    for (int i = 0; i < slot_num_; ++i) {
      slot_tensor_ptr_[i] = feed_vec_[3 + 2 * i]->mutable_data<int64_t>(
          {total_instance, 1}, this->place_);
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
  if (gpu_graph_training_) {
    VLOG(2) << "total_instance: " << total_instance
            << ", ins_buf_pair_len = " << ins_buf_pair_len_;
    // uint64_t *ins_buf = reinterpret_cast<uint64_t *>(d_ins_buf_->ptr());
    // uint64_t *ins_cursor = ins_buf + ins_buf_pair_len_ * 2 - total_instance;
    ins_buf = reinterpret_cast<uint64_t *>(d_ins_buf_->ptr());
    ins_cursor = ins_buf + ins_buf_pair_len_ * 2 - total_instance;
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
    ins_cursor = (uint64_t *)id_tensor_ptr_;  // NOLINT
  }

  if (slot_num_ > 0) {
    uint64_t *feature_buf = reinterpret_cast<uint64_t *>(d_feature_buf_->ptr());
    if (FLAGS_enable_opt_get_features || !gpu_graph_training_) {
      FillFeatureBuf(ins_cursor, feature_buf, total_instance);
      // FillFeatureBuf(id_tensor_ptr_, feature_buf, total_instance);
      if (debug_mode_) {
        uint64_t h_walk[total_instance];  // NOLINT
        cudaMemcpy(h_walk,
                   ins_cursor,
                   total_instance * sizeof(uint64_t),
                   cudaMemcpyDeviceToHost);
        uint64_t h_feature[total_instance * slot_num_];  // NOLINT
        cudaMemcpy(h_feature,
                   feature_buf,
                   total_instance * slot_num_ * sizeof(uint64_t),
                   cudaMemcpyDeviceToHost);
        for (int i = 0; i < total_instance; ++i) {
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

      GraphFillSlotKernel<<<GET_BLOCKS(total_instance * slot_num_),
                            CUDA_NUM_THREADS,
                            0,
                            stream_>>>(
          (uint64_t *)d_slot_tensor_ptr_->ptr(),  // NOLINT
          feature_buf,
          total_instance * slot_num_,
          total_instance,
          slot_num_);
      GraphFillSlotLodKernelOpt<<<GET_BLOCKS((total_instance + 1) * slot_num_),
                                  CUDA_NUM_THREADS,
                                  0,
                                  stream_>>>(
          (uint64_t *)d_slot_lod_tensor_ptr_->ptr(),  // NOLINT
          (total_instance + 1) * slot_num_,
          total_instance + 1);
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

>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
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

<<<<<<< HEAD
  cudaStreamSynchronize(train_stream_);
  if (!gpu_graph_training_) return 1;
  ins_buf_pair_len_ -= total_instance / 2;
=======
  cudaStreamSynchronize(stream_);
  if (!gpu_graph_training_) return 1;
  ins_buf_pair_len_ -= total_instance / 2;
  if (debug_mode_) {
    uint64_t h_slot_tensor[slot_num_][total_instance];
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

>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
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

<<<<<<< HEAD
__global__ void get_each_ins_info(uint8_t *slot_list,
                                  uint32_t *slot_size_list,
                                  uint32_t *slot_size_prefix,
                                  uint32_t *each_ins_slot_num,
                                  uint32_t *each_ins_slot_num_inner_prefix,
                                  size_t key_num,
                                  int slot_num) {
  const size_t i = blockIdx.x * blockDim.y + threadIdx.y;
  if (i < key_num) {
    uint32_t slot_index = slot_size_prefix[i];
    size_t each_ins_slot_index = i * slot_num;
    for (int j = 0; j < slot_size_list[i]; j++) {
      each_ins_slot_num[each_ins_slot_index + slot_list[slot_index + j]] += 1;
    }
    each_ins_slot_num_inner_prefix[each_ins_slot_index] = 0;
    for (int j = 1; j < slot_num; j++) {
      each_ins_slot_num_inner_prefix[each_ins_slot_index + j] =
          each_ins_slot_num[each_ins_slot_index + j - 1] +
          each_ins_slot_num_inner_prefix[each_ins_slot_index + j - 1];
    }
  }
}

__global__ void fill_slot_num(uint32_t *d_each_ins_slot_num_ptr,
                              uint64_t **d_ins_slot_num_vector_ptr,
                              size_t key_num,
                              int slot_num) {
  const size_t i = blockIdx.x * blockDim.y + threadIdx.y;
  if (i < key_num) {
    size_t d_each_index = i * slot_num;
    for (int j = 0; j < slot_num; j++) {
      d_ins_slot_num_vector_ptr[j][i] =
          d_each_ins_slot_num_ptr[d_each_index + j];
    }
  }
}

__global__ void fill_slot_tensor(uint64_t *feature_list,
                                 uint32_t *feature_size_prefixsum,
                                 uint32_t *each_ins_slot_num_inner_prefix,
                                 uint64_t *ins_slot_num,
                                 int64_t *slot_lod_tensor,
                                 int64_t *slot_tensor,
                                 int slot,
                                 int slot_num,
                                 size_t node_num) {
  const size_t i = blockIdx.x * blockDim.y + threadIdx.y;
  if (i < node_num) {
    size_t dst_index = slot_lod_tensor[i];
    size_t src_index = feature_size_prefixsum[i] +
                       each_ins_slot_num_inner_prefix[slot_num * i + slot];
    for (uint64_t j = 0; j < ins_slot_num[i]; j++) {
      slot_tensor[dst_index + j] = feature_list[src_index + j];
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
=======
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
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
<<<<<<< HEAD
                                           sample_stream_));
  auto d_temp_storage = memory::Alloc(
      place_,
      temp_storage_bytes,
      phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));
=======
                                           stream_));
  auto d_temp_storage = memory::Alloc(place_, temp_storage_bytes);
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f

  CUDA_CHECK(cub::DeviceScan::InclusiveSum(d_temp_storage->ptr(),
                                           temp_storage_bytes,
                                           d_actual_sample_size,
                                           d_prefix_sum + 1,
                                           len,
<<<<<<< HEAD
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
=======
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
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f

  } else {
    GraphFillSampleKeysKernel<<<GET_BLOCKS(len),
                                CUDA_NUM_THREADS,
                                0,
<<<<<<< HEAD
                                sample_stream_>>>(d_neighbors,
                                                  d_sample_keys,
                                                  d_prefix_sum,
                                                  d_sampleidx2row,
                                                  d_tmp_sampleidx2row,
                                                  d_actual_sample_size,
                                                  cur_degree,
                                                  len);

    GraphDoWalkKernel<<<GET_BLOCKS(len), CUDA_NUM_THREADS, 0, sample_stream_>>>(
=======
                                stream_>>>(d_neighbors,
                                           d_sample_keys,
                                           d_prefix_sum,
                                           d_sampleidx2row,
                                           d_tmp_sampleidx2row,
                                           d_actual_sample_size,
                                           cur_degree,
                                           len);

    GraphDoWalkKernel<<<GET_BLOCKS(len), CUDA_NUM_THREADS, 0, stream_>>>(
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
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
<<<<<<< HEAD
=======
    uint64_t h_sample_keys[once_max_sample_keynum];  // NOLINT
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
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
<<<<<<< HEAD
  }
  cudaStreamSynchronize(sample_stream_);
  cur_sampleidx2row_ = 1 - cur_sampleidx2row_;
}

int GraphDataGenerator::FillSlotFeature(uint64_t *d_walk, size_t key_num) {
  platform::CUDADeviceGuard guard(gpuid_);
  auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();
  std::shared_ptr<phi::Allocation> d_feature_list;
  std::shared_ptr<phi::Allocation> d_slot_list;
  uint32_t *d_feature_size_list_ptr =
      reinterpret_cast<uint32_t *>(d_feature_size_list_buf_->ptr());
  uint32_t *d_feature_size_prefixsum_ptr =
      reinterpret_cast<uint32_t *>(d_feature_size_prefixsum_buf_->ptr());
  int fea_num =
      gpu_graph_ptr->get_feature_info_of_nodes(gpuid_,
                                               d_walk,
                                               key_num,
                                               d_feature_size_list_ptr,
                                               d_feature_size_prefixsum_ptr,
                                               d_feature_list,
                                               d_slot_list);
  int64_t *slot_tensor_ptr_[slot_num_];
  int64_t *slot_lod_tensor_ptr_[slot_num_];
  if (fea_num == 0) {
    int64_t default_lod = 1;
    for (int i = 0; i < slot_num_; ++i) {
      slot_lod_tensor_ptr_[i] = feed_vec_[3 + 2 * i + 1]->mutable_data<int64_t>(
          {(long)key_num + 1}, this->place_);
      slot_tensor_ptr_[i] =
          feed_vec_[3 + 2 * i]->mutable_data<int64_t>({1, 1}, this->place_);
      CUDA_CHECK(cudaMemsetAsync(
          slot_tensor_ptr_[i], 0, sizeof(int64_t), train_stream_));
      CUDA_CHECK(cudaMemsetAsync(slot_lod_tensor_ptr_[i],
                                 0,
                                 sizeof(int64_t) * key_num,
                                 train_stream_));
      CUDA_CHECK(cudaMemcpyAsync(
          reinterpret_cast<char *>(slot_lod_tensor_ptr_[i] + key_num),
          &default_lod,
          sizeof(int64_t),
          cudaMemcpyHostToDevice,
          train_stream_));
    }
    CUDA_CHECK(cudaStreamSynchronize(train_stream_));
    return 0;
  }

  uint64_t *d_feature_list_ptr =
      reinterpret_cast<uint64_t *>(d_feature_list->ptr());
  uint8_t *d_slot_list_ptr = reinterpret_cast<uint8_t *>(d_slot_list->ptr());

  std::shared_ptr<phi::Allocation> d_each_ins_slot_num_inner_prefix =
      memory::AllocShared(place_, (slot_num_ * key_num) * sizeof(uint32_t));
  std::shared_ptr<phi::Allocation> d_each_ins_slot_num =
      memory::AllocShared(place_, (slot_num_ * key_num) * sizeof(uint32_t));
  uint32_t *d_each_ins_slot_num_ptr =
      reinterpret_cast<uint32_t *>(d_each_ins_slot_num->ptr());
  uint32_t *d_each_ins_slot_num_inner_prefix_ptr =
      reinterpret_cast<uint32_t *>(d_each_ins_slot_num_inner_prefix->ptr());
  CUDA_CHECK(cudaMemsetAsync(d_each_ins_slot_num_ptr,
                             0,
                             slot_num_ * key_num * sizeof(uint32_t),
                             train_stream_));

  dim3 grid((key_num - 1) / 256 + 1);
  dim3 block(1, 256);

  get_each_ins_info<<<grid, block, 0, train_stream_>>>(
      d_slot_list_ptr,
      d_feature_size_list_ptr,
      d_feature_size_prefixsum_ptr,
      d_each_ins_slot_num_ptr,
      d_each_ins_slot_num_inner_prefix_ptr,
      key_num,
      slot_num_);

  std::vector<std::shared_ptr<phi::Allocation>> ins_slot_num(slot_num_,
                                                             nullptr);
  std::vector<uint64_t *> ins_slot_num_vecotr(slot_num_, NULL);
  std::shared_ptr<phi::Allocation> d_ins_slot_num_vector =
      memory::AllocShared(place_, (slot_num_) * sizeof(uint64_t *));
  uint64_t **d_ins_slot_num_vector_ptr =
      reinterpret_cast<uint64_t **>(d_ins_slot_num_vector->ptr());
  for (int i = 0; i < slot_num_; i++) {
    ins_slot_num[i] = memory::AllocShared(place_, key_num * sizeof(uint64_t));
    ins_slot_num_vecotr[i] =
        reinterpret_cast<uint64_t *>(ins_slot_num[i]->ptr());
  }
  CUDA_CHECK(
      cudaMemcpyAsync(reinterpret_cast<char *>(d_ins_slot_num_vector_ptr),
                      ins_slot_num_vecotr.data(),
                      sizeof(uint64_t *) * slot_num_,
                      cudaMemcpyHostToDevice,
                      train_stream_));
  fill_slot_num<<<grid, block, 0, train_stream_>>>(
      d_each_ins_slot_num_ptr, d_ins_slot_num_vector_ptr, key_num, slot_num_);
  CUDA_CHECK(cudaStreamSynchronize(train_stream_));

  for (int i = 0; i < slot_num_; ++i) {
    slot_lod_tensor_ptr_[i] = feed_vec_[3 + 2 * i + 1]->mutable_data<int64_t>(
        {(long)key_num + 1}, this->place_);
  }
  size_t temp_storage_bytes = 0;
  CUDA_CHECK(cub::DeviceScan::InclusiveSum(NULL,
                                           temp_storage_bytes,
                                           ins_slot_num_vecotr[0],
                                           slot_lod_tensor_ptr_[0] + 1,
                                           key_num,
                                           train_stream_));
  CUDA_CHECK(cudaStreamSynchronize(train_stream_));
  auto d_temp_storage = memory::Alloc(
      this->place_,
      temp_storage_bytes,
      phi::Stream(reinterpret_cast<phi::StreamId>(train_stream_)));
  std::vector<int64_t> each_slot_fea_num(slot_num_, 0);
  for (int i = 0; i < slot_num_; ++i) {
    CUDA_CHECK(cudaMemsetAsync(
        slot_lod_tensor_ptr_[i], 0, sizeof(uint64_t), train_stream_));
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(d_temp_storage->ptr(),
                                             temp_storage_bytes,
                                             ins_slot_num_vecotr[i],
                                             slot_lod_tensor_ptr_[i] + 1,
                                             key_num,
                                             train_stream_));
    CUDA_CHECK(cudaMemcpyAsync(&each_slot_fea_num[i],
                               slot_lod_tensor_ptr_[i] + key_num,
                               sizeof(uint64_t),
                               cudaMemcpyDeviceToHost,
                               train_stream_));
  }
  CUDA_CHECK(cudaStreamSynchronize(train_stream_));
  for (int i = 0; i < slot_num_; ++i) {
    slot_tensor_ptr_[i] = feed_vec_[3 + 2 * i]->mutable_data<int64_t>(
        {each_slot_fea_num[i], 1}, this->place_);
  }
  int64_t default_lod = 1;
  for (int i = 0; i < slot_num_; ++i) {
    fill_slot_tensor<<<grid, block, 0, train_stream_>>>(
        d_feature_list_ptr,
        d_feature_size_prefixsum_ptr,
        d_each_ins_slot_num_inner_prefix_ptr,
        ins_slot_num_vecotr[i],
        slot_lod_tensor_ptr_[i],
        slot_tensor_ptr_[i],
        i,
        slot_num_,
        key_num);
    // trick for empty tensor
    if (each_slot_fea_num[i] == 0) {
      slot_tensor_ptr_[i] =
          feed_vec_[3 + 2 * i]->mutable_data<int64_t>({1, 1}, this->place_);
      CUDA_CHECK(cudaMemsetAsync(
          slot_tensor_ptr_[i], 0, sizeof(uint64_t), train_stream_));
      CUDA_CHECK(cudaMemcpyAsync(
          reinterpret_cast<char *>(slot_lod_tensor_ptr_[i] + key_num),
          &default_lod,
          sizeof(int64_t),
          cudaMemcpyHostToDevice,
          train_stream_));
    }
  }
  CUDA_CHECK(cudaStreamSynchronize(train_stream_));

  if (debug_mode_) {
    std::vector<uint32_t> h_feature_size_list(key_num, 0);
    std::vector<uint32_t> h_feature_size_list_prefixsum(key_num, 0);
    std::vector<uint64_t> node_list(key_num, 0);
    std::vector<uint64_t> h_feature_list(fea_num, 0);
    std::vector<uint8_t> h_slot_list(fea_num, 0);

    CUDA_CHECK(
        cudaMemcpyAsync(reinterpret_cast<char *>(h_feature_size_list.data()),
                        d_feature_size_list_ptr,
                        sizeof(uint32_t) * key_num,
                        cudaMemcpyDeviceToHost,
                        train_stream_));
    CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<char *>(h_feature_size_list_prefixsum.data()),
        d_feature_size_prefixsum_ptr,
        sizeof(uint32_t) * key_num,
        cudaMemcpyDeviceToHost,
        train_stream_));
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<char *>(node_list.data()),
                               d_walk,
                               sizeof(uint64_t) * key_num,
                               cudaMemcpyDeviceToHost,
                               train_stream_));

    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<char *>(h_feature_list.data()),
                               d_feature_list_ptr,
                               sizeof(uint64_t) * fea_num,
                               cudaMemcpyDeviceToHost,
                               train_stream_));
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<char *>(h_slot_list.data()),
                               d_slot_list_ptr,
                               sizeof(uint8_t) * fea_num,
                               cudaMemcpyDeviceToHost,
                               train_stream_));

    CUDA_CHECK(cudaStreamSynchronize(train_stream_));
    for (size_t i = 0; i < key_num; i++) {
      std::stringstream ss;
      ss << "node_id: " << node_list[i]
         << " fea_num: " << h_feature_size_list[i] << " offset "
         << h_feature_size_list_prefixsum[i] << " slot: ";
      for (uint32_t j = 0; j < h_feature_size_list[i]; j++) {
        ss << int(h_slot_list[h_feature_size_list_prefixsum[i] + j]) << " : "
           << h_feature_list[h_feature_size_list_prefixsum[i] + j] << "  ";
      }
      VLOG(0) << ss.str();
    }
    VLOG(0) << "all fea_num is " << fea_num << " calc fea_num is "
            << h_feature_size_list[key_num - 1] +
                   h_feature_size_list_prefixsum[key_num - 1];
    for (int i = 0; i < slot_num_; ++i) {
      std::vector<int64_t> h_slot_lod_tensor(key_num + 1, 0);
      CUDA_CHECK(
          cudaMemcpyAsync(reinterpret_cast<char *>(h_slot_lod_tensor.data()),
                          slot_lod_tensor_ptr_[i],
                          sizeof(int64_t) * (key_num + 1),
                          cudaMemcpyDeviceToHost,
                          train_stream_));
      CUDA_CHECK(cudaStreamSynchronize(train_stream_));
      std::stringstream ss_lod;
      std::stringstream ss_tensor;
      ss_lod << " slot " << i << " lod is [";
      for (size_t j = 0; j < key_num + 1; j++) {
        ss_lod << h_slot_lod_tensor[j] << ",";
      }
      ss_lod << "]";
      std::vector<int64_t> h_slot_tensor(h_slot_lod_tensor[key_num], 0);
      CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<char *>(h_slot_tensor.data()),
                                 slot_tensor_ptr_[i],
                                 sizeof(int64_t) * h_slot_lod_tensor[key_num],
                                 cudaMemcpyDeviceToHost,
                                 train_stream_));
      CUDA_CHECK(cudaStreamSynchronize(train_stream_));

      ss_tensor << " tensor is [ ";
      for (size_t j = 0; j < h_slot_lod_tensor[key_num]; j++) {
        ss_tensor << h_slot_tensor[j] << ",";
      }
      ss_tensor << "]";
      VLOG(0) << ss_lod.str() << "  " << ss_tensor.str();
    }
  }

  return 0;
}

=======
    delete[] h_sample_keys;
  }
  cudaStreamSynchronize(stream_);
  cur_sampleidx2row_ = 1 - cur_sampleidx2row_;
}

>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
int GraphDataGenerator::FillFeatureBuf(uint64_t *d_walk,
                                       uint64_t *d_feature,
                                       size_t key_num) {
  platform::CUDADeviceGuard guard(gpuid_);

  auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();
<<<<<<< HEAD
  int ret =
      gpu_graph_ptr->get_feature_of_nodes(gpuid_,
                                          d_walk,
                                          d_feature,
                                          key_num,
                                          slot_num_,
                                          (int *)d_slot_feature_num_map_->ptr(),
                                          fea_num_per_node_);
=======
  int ret = gpu_graph_ptr->get_feature_of_nodes(
      gpuid_, d_walk, d_feature, key_num, slot_num_);
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
  return ret;
}

int GraphDataGenerator::FillFeatureBuf(
    std::shared_ptr<phi::Allocation> d_walk,
    std::shared_ptr<phi::Allocation> d_feature) {
  platform::CUDADeviceGuard guard(gpuid_);

  auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();
<<<<<<< HEAD
  int ret =
      gpu_graph_ptr->get_feature_of_nodes(gpuid_,
                                          (uint64_t *)d_walk->ptr(),
                                          (uint64_t *)d_feature->ptr(),
                                          buf_size_,
                                          slot_num_,
                                          (int *)d_slot_feature_num_map_->ptr(),
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
  for (auto iter = node_type_start.begin(); iter != node_type_start.end();
       iter++) {
    iter->second = 0;
  }
}

int GraphDataGenerator::FillWalkBuf() {
=======
  int ret = gpu_graph_ptr->get_feature_of_nodes(
      gpuid_,
      (uint64_t *)d_walk->ptr(),     // NOLINT
      (uint64_t *)d_feature->ptr(),  // NOLINT
      buf_size_,
      slot_num_);
  return ret;
}

int GraphDataGenerator::FillWalkBuf(std::shared_ptr<phi::Allocation> d_walk) {
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
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
<<<<<<< HEAD
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
=======
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
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
    VLOG(2) << "type: " << node_type << " size: " << device_key_size
            << " start: " << start;
    uint64_t *d_type_keys =
        reinterpret_cast<uint64_t *>(d_device_keys_[type_index]->ptr());
    int tmp_len = start + once_sample_startid_len_ > device_key_size
                      ? device_key_size - start
                      : once_sample_startid_len_;
<<<<<<< HEAD
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
=======
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
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
    uint64_t *cur_walk = walk + i;

    NeighborSampleQuery q;
    q.initialize(gpuid_,
                 path[0],
                 (uint64_t)(d_type_keys + start),
                 walk_degree_,
                 tmp_len);
<<<<<<< HEAD
    auto sample_res = gpu_graph_ptr->graph_neighbor_sample_v3(q, false, true);
=======
    auto sample_res = gpu_graph_ptr->graph_neighbor_sample_v3(q, false);
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f

    int step = 1;
    VLOG(2) << "sample edge type: " << path[0] << " step: " << 1;
    jump_rows_ = sample_res.total_sample_size;
<<<<<<< HEAD
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
      if (InsertTable(sample_res.actual_val,
                      sample_res.total_sample_size,
                      d_uniq_node_num_) != 0) {
        VLOG(2) << "in step 0, insert sample res stage, table is full";
        update = false;
        break;
      }
    }
=======
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
    FillOneStep(d_type_keys + start,
                cur_walk,
                tmp_len,
                sample_res,
                walk_degree_,
                step,
                len_per_row);
<<<<<<< HEAD
=======
    VLOG(2) << "jump_row: " << jump_rows_;
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
    /////////
    if (debug_mode_) {
      cudaMemcpy(
          h_walk, walk, buf_size_ * sizeof(uint64_t), cudaMemcpyDeviceToHost);
      for (int xx = 0; xx < buf_size_; xx++) {
        VLOG(2) << "h_walk[" << xx << "]: " << h_walk[xx];
      }
    }
<<<<<<< HEAD

    VLOG(2) << "sample, step=" << step << " sample_keys=" << tmp_len
            << " sample_res_len=" << sample_res.total_sample_size;

=======
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
    /////////
    step++;
    size_t path_len = path.size();
    for (; step < walk_len_; step++) {
      if (sample_res.total_sample_size == 0) {
<<<<<<< HEAD
        VLOG(2) << "sample finish, step=" << step;
=======
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
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
<<<<<<< HEAD
      int sample_key_len = sample_res.total_sample_size;
      sample_res = gpu_graph_ptr->graph_neighbor_sample_v3(q, false, true);
      total_samples += sample_res.total_sample_size;
      if (FLAGS_gpugraph_storage_mode != GpuGraphStorageMode::WHOLE_HBM) {
        if (InsertTable(sample_res.actual_val,
                        sample_res.total_sample_size,
                        d_uniq_node_num_) != 0) {
          VLOG(2) << "in step: " << step << ", table is full";
          update = false;
          break;
        }
      }
      FillOneStep(d_type_keys + start,
                  cur_walk,
                  sample_key_len,
=======
      sample_res = gpu_graph_ptr->graph_neighbor_sample_v3(q, false);

      FillOneStep(d_type_keys + start,
                  cur_walk,
                  sample_res.total_sample_size,
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
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
<<<<<<< HEAD

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
=======
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
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
      VLOG(2) << "h_random_row[" << xx << "]: " << h_random_row[xx];
    }
    delete[] h_random_row;
    delete[] h_walk;
    delete[] h_sample_keys;
    delete[] h_offset2idx;
    delete[] h_len_per_row;
    delete[] h_prefix_sum;
  }
<<<<<<< HEAD
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

    VLOG(0) << "sample_times:" << sample_times << ", d_walk_size:" << buf_size_
            << ", d_walk_offset:" << i << ", total_rows:" << total_row_
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
=======
  return total_row != 0;
}

void GraphDataGenerator::AllocResource(
    const paddle::platform::Place &place,
    std::vector<phi::DenseTensor *> feed_vec) {
  place_ = place;
  gpuid_ = place_.GetDeviceId();
  VLOG(3) << "gpuid " << gpuid_;
  stream_ = dynamic_cast<phi::GPUContext *>(
                platform::DeviceContextPool::Instance().Get(place))
                ->stream();
  feed_vec_ = feed_vec;
  slot_num_ = (feed_vec_.size() - 3) / 2;

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
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
  for (int i = -window_; i < 0; i++) {
    window_step_.push_back(i);
  }
  for (int i = 0; i < window_; i++) {
    window_step_.push_back(i + 1);
  }
  buf_state_.Init(batch_size_, walk_len_, &window_step_);
  d_random_row_ = memory::AllocShared(
      place_,
<<<<<<< HEAD
      (once_sample_startid_len_ * walk_degree_ * repeat_time_) * sizeof(int),
      phi::Stream(reinterpret_cast<phi::StreamId>(sample_stream_)));
=======
      (once_sample_startid_len_ * walk_degree_ * repeat_time_) * sizeof(int));
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
  shuffle_seed_ = 0;

  ins_buf_pair_len_ = 0;
  d_ins_buf_ =
      memory::AllocShared(place_, (batch_size_ * 2 * 2) * sizeof(uint64_t));
<<<<<<< HEAD
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
    d_feature_size_list_buf_ =
        memory::AllocShared(place_, (batch_size_ * 2) * sizeof(uint32_t));
    d_feature_size_prefixsum_buf_ =
        memory::AllocShared(place_, (batch_size_ * 2 + 1) * sizeof(uint32_t));
  }
=======
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

  cudaStreamSynchronize(stream_);
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
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
<<<<<<< HEAD
  train_table_cap_ = graph_config.train_table_cap();
  infer_table_cap_ = graph_config.infer_table_cap();
  epoch_finish_ = false;
  VLOG(1) << "Confirm GraphConfig, walk_degree : " << walk_degree_
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
=======
  VLOG(2) << "Confirm GraphConfig, walk_degree : " << walk_degree_
          << ", walk_len : " << walk_len_ << ", window : " << window_
          << ", once_sample_startid_len : " << once_sample_startid_len_
          << ", sample_times_one_chunk : " << repeat_time_
          << ", batch_size: " << batch_size_;
  std::string first_node_type = graph_config.first_node_type();
  std::string meta_path = graph_config.meta_path();
  auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();
  auto edge_to_id = gpu_graph_ptr->edge_to_id;
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
}
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f

}  // namespace framework
}  // namespace paddle
#endif
