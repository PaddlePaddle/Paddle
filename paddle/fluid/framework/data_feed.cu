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

#include <thrust/device_ptr.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>
#include "cub/cub.cuh"
#include "paddle/fluid/framework/data_feed.h"
#include "paddle/fluid/framework/fleet/heter_ps/gpu_graph_node.h"
#include "paddle/fluid/framework/fleet/heter_ps/graph_gpu_wrapper.h"

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
__global__ void FillSlotValueOffsetKernel(
    const int ins_num, const int used_slot_num, size_t *slot_value_offsets,
    const int *uint64_offsets, const int uint64_slot_size,
    const int *float_offsets, const int float_slot_size,
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
    const int ins_num, const int used_slot_num, size_t *slot_value_offsets,
    const int *uint64_offsets, const int uint64_slot_size,
    const int *float_offsets, const int float_slot_size,
    const UsedSlotGpuType *used_slots) {
  auto stream =
      dynamic_cast<platform::CUDADeviceContext *>(
          paddle::platform::DeviceContextPool::Instance().Get(this->place_))
          ->stream();
  FillSlotValueOffsetKernel<<<GET_BLOCKS(used_slot_num), CUDA_NUM_THREADS, 0,
                              stream>>>(
      ins_num, used_slot_num, slot_value_offsets, uint64_offsets,
      uint64_slot_size, float_offsets, float_slot_size, used_slots);
  cudaStreamSynchronize(stream);
}

__global__ void CopyForTensorKernel(
    const int used_slot_num, const int ins_num, void **dest,
    const size_t *slot_value_offsets, const uint64_t *uint64_feas,
    const int *uint64_offsets, const int *uint64_ins_lens,
    const int uint64_slot_size, const float *float_feas,
    const int *float_offsets, const int *float_ins_lens,
    const int float_slot_size, const UsedSlotGpuType *used_slots) {
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
    const int ins_num, const int used_slot_num, void **dest,
    const size_t *slot_value_offsets, const uint64_t *uint64_feas,
    const int *uint64_offsets, const int *uint64_ins_lens,
    const int uint64_slot_size, const float *float_feas,
    const int *float_offsets, const int *float_ins_lens,
    const int float_slot_size, const UsedSlotGpuType *used_slots) {
  auto stream =
      dynamic_cast<platform::CUDADeviceContext *>(
          paddle::platform::DeviceContextPool::Instance().Get(this->place_))
          ->stream();

  CopyForTensorKernel<<<GET_BLOCKS(used_slot_num * ins_num), CUDA_NUM_THREADS,
                        0, stream>>>(
      used_slot_num, ins_num, dest, slot_value_offsets, uint64_feas,
      uint64_offsets, uint64_ins_lens, uint64_slot_size, float_feas,
      float_offsets, float_ins_lens, float_slot_size, used_slots);
  cudaStreamSynchronize(stream);
}

__global__ void GraphFillCVMKernel(int64_t *tensor, int len) {
  CUDA_KERNEL_LOOP(idx, len) { tensor[idx] = 1; }
}

__global__ void GraphFillIdKernel(int64_t *id_tensor, int64_t *walk, int *row,
                                  int central_word, int step, int len,
                                  int col_num) {
  CUDA_KERNEL_LOOP(idx, len) {
    int dst = idx * 2;
    int src = row[idx] * col_num + central_word;
    id_tensor[dst] = walk[src];
    id_tensor[dst + 1] = walk[src + step];
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

int GraphDataGenerator::GenerateBatch() {
  platform::CUDADeviceGuard guard(gpuid_);

  int total_instance = AcquireInstance(&buf_state_);

  VLOG(2) << "total_ins: " << total_instance;
  buf_state_.Debug();

  if (total_instance == 0) {
    int res = FillWalkBuf(d_walk_);
    if (!res) {
      return 0;
    } else {
      total_instance = buf_state_.len;
      VLOG(2) << "total_ins: " << total_instance;
      buf_state_.Debug();
      if (total_instance == 0) {
        return 0;
      }
    }
  }

  total_instance *= 2;
  id_tensor_ptr_ =
      feed_vec_[0]->mutable_data<int64_t>({total_instance, 1}, this->place_);
  show_tensor_ptr_ =
      feed_vec_[1]->mutable_data<int64_t>({total_instance}, this->place_);
  clk_tensor_ptr_ =
      feed_vec_[2]->mutable_data<int64_t>({total_instance}, this->place_);

  int64_t *walk = reinterpret_cast<int64_t *>(d_walk_->ptr());
  int *random_row = reinterpret_cast<int *>(d_random_row_->ptr());
  int len = buf_state_.len;
  GraphFillIdKernel<<<GET_BLOCKS(len), CUDA_NUM_THREADS, 0, stream_>>>(
      id_tensor_ptr_, walk, random_row + buf_state_.cursor,
      buf_state_.central_word, window_step_[buf_state_.step], len, walk_len_);
  GraphFillCVMKernel<<<GET_BLOCKS(len), CUDA_NUM_THREADS, 0, stream_>>>(
      show_tensor_ptr_, total_instance);
  GraphFillCVMKernel<<<GET_BLOCKS(len), CUDA_NUM_THREADS, 0, stream_>>>(
      clk_tensor_ptr_, total_instance);

  offset_.clear();
  offset_.push_back(0);
  offset_.push_back(total_instance);
  LoD lod{offset_};
  feed_vec_[0]->set_lod(lod);
  cudaStreamSynchronize(stream_);
  return 1;
}

__global__ void GraphFillSampleKeysKernel(
    int64_t *neighbors, int64_t *sample_keys, int64_t *prefix_sum,
    int *sampleidx2row, int *tmp_sampleidx2row, int *actual_sample_size,
    int cur_degree, int len) {
  CUDA_KERNEL_LOOP(idx, len) {
    for (int k = 0; k < actual_sample_size[idx]; k++) {
      size_t offset = prefix_sum[idx] + k;
      sample_keys[offset] = neighbors[idx * cur_degree + k];
      tmp_sampleidx2row[offset] = sampleidx2row[idx] + k;
    }
  }
}

__global__ void GraphDoWalkKernel(int64_t *neighbors, int64_t *walk,
                                  int64_t *d_prefix_sum,
                                  int *actual_sample_size, int cur_degree,
                                  int step, int len, int *id_cnt,
                                  int *sampleidx2row, int col_size) {
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
__global__ void GraphFillFirstStepKernel(
    int64_t *prefix_sum, int *sampleidx2row, int64_t *walk, int64_t *keys,
    int len, int walk_degree, int col_size, int *actual_sample_size,
    int64_t *neighbors, int64_t *sample_keys) {
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
void GraphDataGenerator::FillOneStep(int64_t *walk, int len,
                                     NeighborSampleResult &sample_res,
                                     int cur_degree, int step,
                                     int *len_per_row) {
  size_t temp_storage_bytes = 0;
  int *d_actual_sample_size = sample_res.actual_sample_size;
  int64_t *d_neighbors = sample_res.val;
  int64_t *d_prefix_sum = reinterpret_cast<int64_t *>(d_prefix_sum_->ptr());
  int64_t *d_sample_keys = reinterpret_cast<int64_t *>(d_sample_keys_->ptr());
  int *d_sampleidx2row =
      reinterpret_cast<int *>(d_sampleidx2rows_[cur_sampleidx2row_]->ptr());
  int *d_tmp_sampleidx2row =
      reinterpret_cast<int *>(d_sampleidx2rows_[1 - cur_sampleidx2row_]->ptr());

  CUDA_CHECK(cub::DeviceScan::InclusiveSum(NULL, temp_storage_bytes,
                                           d_actual_sample_size,
                                           d_prefix_sum + 1, len, stream_));
  auto d_temp_storage = memory::Alloc(place_, temp_storage_bytes);

  CUDA_CHECK(cub::DeviceScan::InclusiveSum(
      d_temp_storage->ptr(), temp_storage_bytes, d_actual_sample_size,
      d_prefix_sum + 1, len, stream_));

  int64_t next_start_id_len = 0;
  cudaMemcpyAsync(&next_start_id_len, d_prefix_sum + len, sizeof(int64_t),
                  cudaMemcpyDeviceToHost, stream_);
  cudaStreamSynchronize(stream_);

  if (step == 1) {
    GraphFillFirstStepKernel<<<GET_BLOCKS(len), CUDA_NUM_THREADS, 0, stream_>>>(
        d_prefix_sum, d_tmp_sampleidx2row, walk, device_keys_ + cursor_, len,
        walk_degree_, walk_len_, d_actual_sample_size, d_neighbors,
        d_sample_keys);
    jump_rows_ = next_start_id_len;

  } else {
    GraphFillSampleKeysKernel<<<GET_BLOCKS(len), CUDA_NUM_THREADS, 0,
                                stream_>>>(
        d_neighbors, d_sample_keys, d_prefix_sum, d_sampleidx2row,
        d_tmp_sampleidx2row, d_actual_sample_size, cur_degree, len);

    GraphDoWalkKernel<<<GET_BLOCKS(len), CUDA_NUM_THREADS, 0, stream_>>>(
        d_neighbors, walk, d_prefix_sum, d_actual_sample_size, cur_degree, step,
        len, len_per_row, d_tmp_sampleidx2row, walk_len_);
  }
  if (debug_mode_) {
    size_t once_max_sample_keynum = walk_degree_ * once_sample_startid_len_;
    int64_t *h_prefix_sum = new int64_t[len + 1];
    int *h_actual_size = new int[len];
    int *h_offset2idx = new int[once_max_sample_keynum];
    int64_t *h_sample_keys = new int64_t[once_max_sample_keynum];
    cudaMemcpy(h_offset2idx, d_tmp_sampleidx2row,
               once_max_sample_keynum * sizeof(int), cudaMemcpyDeviceToHost);

    cudaMemcpy(h_prefix_sum, d_prefix_sum, (len + 1) * sizeof(int64_t),
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
  sample_keys_len_ = next_start_id_len;
  cur_sampleidx2row_ = 1 - cur_sampleidx2row_;
  cudaStreamSynchronize(stream_);
}

int GraphDataGenerator::FillWalkBuf(std::shared_ptr<phi::Allocation> d_walk) {
  platform::CUDADeviceGuard guard(gpuid_);
  size_t once_max_sample_keynum = walk_degree_ * once_sample_startid_len_;
  ////////
  int64_t *h_walk;
  int64_t *h_sample_keys;
  int *h_offset2idx;
  int *h_len_per_row;
  int64_t *h_prefix_sum;
  if (debug_mode_) {
    h_walk = new int64_t[buf_size_];
    h_sample_keys = new int64_t[once_max_sample_keynum];
    h_offset2idx = new int[once_max_sample_keynum];
    h_len_per_row = new int[once_max_sample_keynum];
    h_prefix_sum = new int64_t[once_max_sample_keynum + 1];
  }
  ///////
  auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();
  int64_t *walk = reinterpret_cast<int64_t *>(d_walk->ptr());
  int *len_per_row = reinterpret_cast<int *>(d_len_per_row_->ptr());
  int64_t *d_sample_keys = reinterpret_cast<int64_t *>(d_sample_keys_->ptr());
  cudaMemsetAsync(walk, 0, buf_size_ * sizeof(sizeof(int64_t)), stream_);
  cudaMemsetAsync(len_per_row, 0, once_max_sample_keynum * sizeof(int),
                  stream_);
  int i = 0;
  int total_row = 0;
  int remain_size =
      buf_size_ - walk_degree_ * once_sample_startid_len_ * walk_len_;
  while (i <= remain_size) {
    int tmp_len = cursor_ + once_sample_startid_len_ > device_key_size_
                      ? device_key_size_ - cursor_
                      : once_sample_startid_len_;
    if (tmp_len == 0) {
      break;
    }
    VLOG(2) << "i = " << i << " buf_size_ = " << buf_size_
            << " tmp_len = " << tmp_len << " cursor = " << cursor_
            << " once_max_sample_keynum = " << once_max_sample_keynum;
    int64_t *cur_walk = walk + i;

    if (debug_mode_) {
      cudaMemcpy(h_walk, walk, buf_size_ * sizeof(int64_t),
                 cudaMemcpyDeviceToHost);
      for (int xx = 0; xx < buf_size_; xx++) {
        VLOG(2) << "h_walk[" << xx << "]: " << h_walk[xx];
      }
    }
    auto sample_res = gpu_graph_ptr->graph_neighbor_sample(
        gpuid_, device_keys_ + cursor_, walk_degree_, tmp_len);

    int step = 1;
    jump_rows_ = 0;
    FillOneStep(cur_walk, tmp_len, sample_res, walk_degree_, step, len_per_row);
    /////////
    if (debug_mode_) {
      cudaMemcpy(h_walk, walk, buf_size_ * sizeof(int64_t),
                 cudaMemcpyDeviceToHost);
      for (int xx = 0; xx < buf_size_; xx++) {
        VLOG(2) << "h_walk[" << xx << "]: " << h_walk[xx];
      }
    }
    /////////
    step++;
    for (; step < walk_len_; step++) {
      if (sample_keys_len_ == 0) {
        break;
      }
      sample_res = gpu_graph_ptr->graph_neighbor_sample(gpuid_, d_sample_keys,
                                                        1, sample_keys_len_);

      FillOneStep(cur_walk, sample_keys_len_, sample_res, 1, step, len_per_row);
      if (debug_mode_) {
        cudaMemcpy(h_walk, walk, buf_size_ * sizeof(int64_t),
                   cudaMemcpyDeviceToHost);
        for (int xx = 0; xx < buf_size_; xx++) {
          VLOG(2) << "h_walk[" << xx << "]: " << h_walk[xx];
        }
      }
    }
    cursor_ += tmp_len;
    i += jump_rows_ * walk_len_;
    total_row += jump_rows_;
  }
  buf_state_.Reset(total_row);
  int *d_random_row = reinterpret_cast<int *>(d_random_row_->ptr());

  thrust::random::default_random_engine engine(shuffle_seed_);
  const auto &exec_policy = thrust::cuda::par.on(stream_);
  thrust::counting_iterator<int> cnt_iter(0);
  thrust::shuffle_copy(exec_policy, cnt_iter, cnt_iter + total_row,
                       thrust::device_pointer_cast(d_random_row), engine);

  cudaStreamSynchronize(stream_);
  shuffle_seed_ = engine();

  if (debug_mode_) {
    int *h_random_row = new int[total_row + 10];
    cudaMemcpy(h_random_row, d_random_row, total_row * sizeof(int),
               cudaMemcpyDeviceToHost);
    for (int xx = 0; xx < total_row; xx++) {
      VLOG(2) << "h_random_row[" << xx << "]: " << h_random_row[xx];
    }
    delete h_random_row;
    delete[] h_walk;
    delete[] h_sample_keys;
    delete[] h_offset2idx;
    delete[] h_len_per_row;
    delete[] h_prefix_sum;
  }
  return total_row != 0;
}

}  // namespace framework
}  // namespace paddle
#endif
