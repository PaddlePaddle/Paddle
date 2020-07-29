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

#include "paddle/fluid/framework/data_feed.h"

namespace paddle {
namespace framework {

#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

__global__ void CopyForTensorKernel(FeatureItem *src, void **dest,
                                    size_t *offset, char *type,
                                    size_t total_size, size_t row_size,
                                    size_t col_size) {
  CUDA_KERNEL_LOOP(i, row_size * col_size) {
    int row_id = i / col_size;
    int col_id = i % col_size;
    size_t left, right;
    if (row_id == 0) {
      left = offset[row_id * (col_size + 1) + col_id];
      right = offset[row_id * (col_size + 1) + col_id + 1];
    } else {
      left = offset[row_id * (col_size + 1) + col_id] -
             offset[(row_id - 1) * (col_size + 1) + col_id];
      right = offset[row_id * (col_size + 1) + col_id + 1] -
              offset[(row_id - 1) * (col_size + 1) + col_id + 1];
    }

    uint64_t *up = NULL;
    float *fp = NULL;
    if (type[row_id] == 'f') {
      fp = reinterpret_cast<float *>(dest[row_id]);
    } else {
      up = reinterpret_cast<uint64_t *>(
          *(reinterpret_cast<uint64_t **>(dest) + row_id));
    }
    size_t begin = offset[row_id * (col_size + 1) + col_id + 1] +
                   offset[(row_size - 1) * (col_size + 1) + col_id] -
                   offset[row_id * (col_size + 1) + col_id] - (right - left);
    PADDLE_ENFORCE(begin >= 0, "begin must be ge 0.");
    PADDLE_ENFORCE(
        begin < total_size,
        "begin must be lt total_size, but your begin[%lu], total_size[%lu]",
        begin, total_size);
    for (size_t k = left; k < right; ++k) {
      PADDLE_ENFORCE((begin + k - left) >= 0 && (begin + k - left) < total_size,
                     "begin+k-left must be in [0, total_size)");
      if (type[row_id] == 'f') {
        *(fp + k) = src[begin + k - left].sign().float_feasign_;
      } else {
        *(up + k) = src[begin + k - left].sign().uint64_feasign_;
      }
    }
  }
}

void MultiSlotInMemoryDataFeed::CopyForTensor(
    const paddle::platform::Place &place, FeatureItem *src, void **dest,
    size_t *offset, char *type, size_t total_size, size_t row_size,
    size_t col_size) {
  auto stream = dynamic_cast<platform::CUDADeviceContext *>(
                    platform::DeviceContextPool::Instance().Get(
                        boost::get<platform::CUDAPlace>(place)))
                    ->stream();
  CopyForTensorKernel<<<((row_size * (col_size - 1)) + 511) / 512, 512, 0,
                        stream>>>(src, dest, offset, type, total_size, row_size,
                                  col_size - 1);
  cudaStreamSynchronize(stream);
}

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
        PADDLE_ENFORCE(num >= 0, "num size must be ge 0.");
        slot_value_offsets[value_off + k + 1] =
            slot_value_offsets[value_off + k] + num;
      }
    } else {
      for (int k = 0; k < ins_num; ++k) {
        int pos = k * float_cols + info.slot_value_idx;
        int num = float_offsets[pos + 1] - float_offsets[pos];
        PADDLE_ENFORCE(num >= 0, "num size must be ge 0.");
        slot_value_offsets[value_off + k + 1] =
            slot_value_offsets[value_off + k] + num;
      }
    }
  }
}

void SlotPaddleBoxDataFeed::FillSlotValueOffset(
    const int ins_num, const int used_slot_num, size_t *slot_value_offsets,
    const int *uint64_offsets, const int uint64_slot_size,
    const int *float_offsets, const int float_slot_size,
    const UsedSlotGpuType *used_slots) {
  auto stream = dynamic_cast<platform::CUDADeviceContext *>(
                    platform::DeviceContextPool::Instance().Get(
                        boost::get<platform::CUDAPlace>(this->place_)))
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
      PADDLE_ENFORCE(num >= 0, "num size must be ge 0.");
      int uint64_value_offset = uint64_ins_lens[ins_idx];
      for (int k = 0; k < num; ++k) {
        up[k + value_offset] = uint64_feas[k + old_off + uint64_value_offset];
      }
    } else {
      float *fp = reinterpret_cast<float *>(dest[slot_idx]);
      int index = info.slot_value_idx + float_cols * ins_idx;
      int old_off = float_offsets[index];
      int num = float_offsets[index + 1] - old_off;
      PADDLE_ENFORCE(num >= 0, "num size must be ge 0.");
      int float_value_offset = float_ins_lens[ins_idx];
      for (int k = 0; k < num; ++k) {
        fp[k + value_offset] = float_feas[k + old_off + float_value_offset];
      }
    }
  }
}

void SlotPaddleBoxDataFeed::CopyForTensor(
    const int ins_num, const int used_slot_num, void **dest,
    const size_t *slot_value_offsets, const uint64_t *uint64_feas,
    const int *uint64_offsets, const int *uint64_ins_lens,
    const int uint64_slot_size, const float *float_feas,
    const int *float_offsets, const int *float_ins_lens,
    const int float_slot_size, const UsedSlotGpuType *used_slots) {
  auto stream = dynamic_cast<platform::CUDADeviceContext *>(
                    platform::DeviceContextPool::Instance().Get(
                        boost::get<platform::CUDAPlace>(this->place_)))
                    ->stream();

  CopyForTensorKernel<<<GET_BLOCKS(used_slot_num * ins_num), CUDA_NUM_THREADS,
                        0, stream>>>(
      used_slot_num, ins_num, dest, slot_value_offsets, uint64_feas,
      uint64_offsets, uint64_ins_lens, uint64_slot_size, float_feas,
      float_offsets, float_ins_lens, float_slot_size, used_slots);
  cudaStreamSynchronize(stream);
}

__global__ void CopyRankOffsetKernel(int *mat, const int ins_num,
                                     const int pv_num, const int max_rank,
                                     const int *ad_rank, const int *cmatch,
                                     const int *pv_offset, const int cols) {
  CUDA_KERNEL_LOOP(ins_idx, pv_num) {
    int pv_ad_num = pv_offset[ins_idx + 1] - pv_offset[ins_idx];
    int pv_ad_index = pv_offset[ins_idx];

    for (int j = 0; j < pv_ad_num; ++j) {
      int rank = -1;
      int ins_cmatch = cmatch[pv_ad_index + j];

      if ((ins_cmatch == 222 || ins_cmatch == 223) &&
          ad_rank[pv_ad_index + j] <= max_rank &&
          ad_rank[pv_ad_index + j] != 0) {
        rank = ad_rank[pv_ad_index + j];
      }

      if (!(rank <= max_rank)) {
        printf("check rank[%d] <= max_rank[%d] failed, ins_idx[%d] j[%d]\n",
               rank, max_rank, ins_idx, j);
        asm("trap;");
      }
      mat[(pv_ad_index + j) * cols] = rank;

      if (rank > 0) {
        for (int k = 0; k < pv_ad_num; ++k) {
          int fast_rank = -1;

          int pv_ins_cmatch = cmatch[pv_ad_index + k];
          if ((pv_ins_cmatch == 222 || pv_ins_cmatch == 223) &&
              ad_rank[pv_ad_index + k] <= max_rank &&
              ad_rank[pv_ad_index + k] != 0) {
            fast_rank = ad_rank[pv_ad_index + k];
          }
          if (!(fast_rank <= max_rank)) {
            printf("check fast_rank[%d] <= max_rank[%d] failed\n", fast_rank,
                   max_rank);
            asm("trap;");
          }
          if (fast_rank > 0) {
            int m = fast_rank - 1;
            mat[(pv_ad_index + j) * cols + (2 * m + 1)] =
                ad_rank[pv_ad_index + k];
            mat[(pv_ad_index + j) * cols + (2 * m + 2)] = pv_ad_index + k;
          }
        }
      }
    }
  }
}

void SlotPaddleBoxDataFeed::CopyRankOffset(int *dest, const int ins_num,
                                           const int pv_num, const int max_rank,
                                           const int *ranks, const int *cmatchs,
                                           const int *ad_offsets,
                                           const int cols) {
  auto stream = dynamic_cast<platform::CUDADeviceContext *>(
                    platform::DeviceContextPool::Instance().Get(
                        boost::get<platform::CUDAPlace>(this->place_)))
                    ->stream();
  cudaMemsetAsync(dest, -1, sizeof(int) * ins_num * cols, stream);
  CopyRankOffsetKernel<<<GET_BLOCKS(pv_num), CUDA_NUM_THREADS, 0, stream>>>(
      dest, ins_num, pv_num, max_rank, ranks, cmatchs, ad_offsets, cols);
  cudaStreamSynchronize(stream);
}

}  // namespace framework
}  // namespace paddle
