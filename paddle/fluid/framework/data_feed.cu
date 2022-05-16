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

__global__ void GraphFillIdKernel(int64_t *id_tensor, int *actual_sample_size,
                                  int64_t *prefix_sum, int64_t *device_key,
                                  int64_t *neighbors, int sample_size,
                                  int len) {
  CUDA_KERNEL_LOOP(idx, len) {
    for (int k = 0; k < actual_sample_size[idx]; k++) {
      int offset = (prefix_sum[idx] + k) * 2;
      id_tensor[offset] = device_key[idx];
      id_tensor[offset + 1] = neighbors[idx * sample_size + k];
    }
  }
}

__global__ void GraphFillCVMKernel(int64_t *tensor, int len) {
  CUDA_KERNEL_LOOP(idx, len) { tensor[idx] = 1; }
}

void GraphDataGenerator::FeedGraphIns(size_t cursor, int len,
                                      NeighborSampleResult &sample_res) {
  size_t temp_storage_bytes = 0;
  int *d_actual_sample_size = sample_res.actual_sample_size;
  int64_t *d_neighbors = sample_res.val;
  int64_t *d_prefix_sum = reinterpret_cast<int64_t *>(d_prefix_sum_->ptr());
  CUDA_CHECK(cub::DeviceScan::InclusiveSum(NULL, temp_storage_bytes,
                                           d_actual_sample_size,
                                           d_prefix_sum + 1, len, stream_));
  auto d_temp_storage = memory::Alloc(place_, temp_storage_bytes);

  CUDA_CHECK(cub::DeviceScan::InclusiveSum(
      d_temp_storage->ptr(), temp_storage_bytes, d_actual_sample_size,
      d_prefix_sum + 1, len, stream_));
  cudaStreamSynchronize(stream_);
  int64_t total_ins = 0;
  cudaMemcpyAsync(&total_ins, d_prefix_sum + len, sizeof(int64_t),
                  cudaMemcpyDeviceToHost, stream_);

  total_ins *= 2;
  id_tensor_ptr_ =
      feed_vec_[0]->mutable_data<int64_t>({total_ins, 1}, this->place_);
  show_tensor_ptr_ =
      feed_vec_[1]->mutable_data<int64_t>({total_ins}, this->place_);
  clk_tensor_ptr_ =
      feed_vec_[2]->mutable_data<int64_t>({total_ins}, this->place_);

  GraphFillIdKernel<<<GET_BLOCKS(len), CUDA_NUM_THREADS, 0, stream_>>>(
      id_tensor_ptr_, d_actual_sample_size, d_prefix_sum,
      device_keys_ + cursor_, d_neighbors, walk_degree_, len);
  GraphFillCVMKernel<<<GET_BLOCKS(len), CUDA_NUM_THREADS, 0, stream_>>>(
      show_tensor_ptr_, total_ins);
  GraphFillCVMKernel<<<GET_BLOCKS(len), CUDA_NUM_THREADS, 0, stream_>>>(
      clk_tensor_ptr_, total_ins);

  offset_.clear();
  offset_.push_back(0);
  offset_.push_back(total_ins);
  LoD lod{offset_};
  feed_vec_[0]->set_lod(lod);
  // feed_vec_[1]->set_lod(lod);
  // feed_vec_[2]->set_lod(lod);
  cudaStreamSynchronize(stream_);
}

int GraphDataGenerator::GenerateBatch() {
  // GpuPsGraphTable *g = (GpuPsGraphTable *)(gpu_graph_ptr->graph_table);
  platform::CUDADeviceGuard guard(gpuid_);
  auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();
  int tmp_len = cursor_ + sample_key_size_ > device_key_size_
                    ? device_key_size_ - cursor_
                    : sample_key_size_;
  VLOG(3) << "device key size: " << device_key_size_
          << " this batch: " << tmp_len << " cursor: " << cursor_
          << " sample_key_size_: " << sample_key_size_;
  if (tmp_len == 0) {
    return 0;
  }
  int total_instance = 1;
  auto sample_res = gpu_graph_ptr->graph_neighbor_sample(
      gpuid_, device_keys_ + cursor_, walk_degree_, tmp_len);
  FeedGraphIns(cursor_, tmp_len, sample_res);
  cursor_ += tmp_len;
  return 1;
}

}  // namespace framework
}  // namespace paddle
#endif
