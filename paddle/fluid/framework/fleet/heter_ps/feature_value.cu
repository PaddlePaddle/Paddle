/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
  http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#ifdef PADDLE_WITH_HETERPS
#include "paddle/fluid/framework/fleet/heter_ps/feature_value.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/core/platform/cuda_device_guard.h"

namespace paddle {
namespace framework {

const int CUDA_NUM_THREADS = phi::PADDLE_CUDA_NUM_THREADS;
#define GET_BLOCK(N) ((N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS)
#define CUDA_BLOCK(N) GET_BLOCK(N), CUDA_NUM_THREADS, 0

template <typename GPUAccessor>
__global__ void PullCopy(float** dest,
                         const float* src,
                         const int64_t* len,
                         int slot_num,
                         int total_len,
                         uint64_t** keys,
                         uint64_t max_val_size,
                         int* gpu_dim,
                         GPUAccessor gpu_accessor) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int low = 0;
    int high = slot_num - 1;
    while (low < high) {
      int mid = (low + high) / 2;
      if (i < len[mid])
        high = mid;
      else
        low = mid + 1;
    }
    int x = low;
    int y = i - (x ? len[x - 1] : 0);
    float* feature_value_ptr =
        (float*)((char*)src + uint64_t(i) * uint64_t(max_val_size));  // NOLINT
    int mf_dim = gpu_dim[x] - 3;
    gpu_accessor.Select(
        dest[x] + y * (mf_dim + 3), feature_value_ptr, keys[x] + y, mf_dim);
  }
}

template <typename TAccess>
__global__ void PullDedupCopy(const size_t N,
                              const uint64_t* total_keys,
                              float** dest,
                              const float* src,
                              const int64_t* slot_lens,
                              uint64_t max_val_size,
                              const int* slot_dims,
                              const size_t hidden,
                              const int* key2slot,
                              const uint32_t* restore_idx,
                              TAccess accessor) {
  CUDA_KERNEL_LOOP_TYPE(idx, N, size_t) {
    int i = idx / hidden;
    int off = idx % hidden;

    int x = key2slot[i];
    int y = i - slot_lens[x];

    float* dest_ptr = dest[x] + y * hidden;
    // 0 key fill zero
    if (total_keys[i] == 0) {
      *(dest_ptr + off) = 0;
      return;
    }

    float* src_ptr = (float*)((char*)src + uint64_t(restore_idx[i]) *  // NOLINT
                                               uint64_t(max_val_size));
    switch (off) {
      case 0:
        *(dest_ptr + off) = src_ptr[accessor.ShowIndex()];
        break;
      case 1:
        *(dest_ptr + off) = src_ptr[accessor.ClickIndex()];
        break;
      case 2:
        *(dest_ptr + off) = src_ptr[accessor.EmbedWIndex()];
        break;
      default:
        int embedx_id = off - 3;
        if (embedx_id >= static_cast<int>(src_ptr[accessor.MfSizeIndex()])) {
          *(dest_ptr + off) = 0;
        } else {
          *(dest_ptr + off) = src_ptr[accessor.EmbedxWIndex() + embedx_id];
        }
        break;
    }
  }
}

template <typename GPUAccessor>
__global__ void PushCopyWithPool(float* dest,
                                 float** src,
                                 int64_t* len,
                                 int slot_num,
                                 uint64_t total_len,
                                 int bs,
                                 int* slot_vector,
                                 int* mf_dim_vector,
                                 size_t grad_value_size,
                                 GPUAccessor gpu_accessor) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int low = 0;
    int high = slot_num - 1;
    while (low < high) {
      int mid = (low + high) / 2;
      if (i < len[mid])
        high = mid;
      else
        low = mid + 1;
    }
    int x = low;
    int y = i - (x ? len[low - 1] : 0);
    float* cur = (float*)((char*)dest + i * grad_value_size);  // NOLINT

    cur[gpu_accessor.common_push_value.SlotIndex()] =
        static_cast<float>(slot_vector[x]);
    int mf_dim = mf_dim_vector[x];
    cur[gpu_accessor.common_push_value.MfDimIndex()] =
        static_cast<float>(mf_dim);

    cur[gpu_accessor.common_push_value.ShowIndex()] =
        *(src[x] + y * (mf_dim + 3));
    cur[gpu_accessor.common_push_value.ClickIndex()] =
        *(src[x] + y * (mf_dim + 3) + 1);
    cur[gpu_accessor.common_push_value.EmbedGIndex()] =
        *(src[x] + y * (mf_dim + 3) + 2) * -1. * bs;
    for (int j = 0; j < mf_dim; j++) {
      cur[gpu_accessor.common_push_value.EmbedxGIndex() + j] =
          *(src[x] + y * (mf_dim + 3) + 3 + j) * -1. * bs;
    }
  }
}

template <typename TAccess>
__global__ void PushMergeCopyAtomic(const size_t N,
                                    const uint64_t* total_keys,
                                    float* dest,
                                    float** src,
                                    const int hidden,
                                    const int bs,
                                    const int* slot_vector,
                                    const int* slot_dims,
                                    const int64_t* slot_lens,
                                    const int* key2slot,
                                    const uint32_t* d_restore_idx,
                                    size_t grad_value_size,
                                    TAccess accessor) {
  CUDA_KERNEL_LOOP_TYPE(idx, N, size_t) {
    int i = idx / hidden;
    int off = idx % hidden;
    // filter 0 keys
    if (total_keys[i] == 0) {
      return;
    }

    int x = key2slot[i];
    int y = i - slot_lens[x];

    const float* ptr = src[x] + y * hidden;
    float* cur =
        (float*)((char*)dest + d_restore_idx[i] * grad_value_size);  // NOLINT
    int mf_dim = slot_dims[x] - 3;
    switch (off) {
      case 0:
        cur[accessor.SlotIndex()] = static_cast<float>(slot_vector[x]);
        cur[accessor.MfDimIndex()] = static_cast<float>(mf_dim);
        phi::CudaAtomicAdd(&cur[accessor.ShowIndex()], *(ptr + off));
        break;
      case 1:
        phi::CudaAtomicAdd(&cur[accessor.ClickIndex()], *(ptr + off));
        break;
      case 2:
        phi::CudaAtomicAdd(&cur[accessor.EmbedGIndex()],
                           *(ptr + off) * -1. * bs);
        break;
      default:
        int embedx_idx = off - 3;
        if (embedx_idx < mf_dim) {
          phi::CudaAtomicAdd(&cur[accessor.EmbedxGIndex() + embedx_idx],
                             *(ptr + off) * -1. * bs);
        }
        break;
    }
  }
}

#define SUM_GRAD_VALUE                                             \
  for (uint32_t j = 0; j < count; ++j) {                           \
    const uint32_t& pos = d_sort_idx[start + j];                   \
    const int& x = key2slot[pos];                                  \
    y = pos - slot_lens[x];                                        \
    val += *(reinterpret_cast<float*>(src[x] + y * hidden + off)); \
  }

template <typename TAccess>
__global__ void PushMergeCopy(const size_t N,
                              const uint64_t* total_keys,
                              float* dest,
                              float** src,
                              const int hidden,
                              const int bs,
                              const int* slot_vector,
                              const int* slot_dims,
                              const int64_t* slot_lens,
                              const int* key2slot,
                              const uint32_t* d_sort_idx,
                              const uint32_t* d_sort_offset,
                              const uint32_t* d_sort_cnt,
                              size_t grad_value_size,
                              TAccess accessor) {
  CUDA_KERNEL_LOOP_TYPE(idx, N, size_t) {
    int i = idx / hidden;
    int off = idx % hidden;
    // filter 0 keys
    float* cur = (float*)((char*)dest + i * grad_value_size);  // NOLINT

    if (total_keys[i] == 0) {
      switch (off) {
        case 0:
          cur[accessor.SlotIndex()] = static_cast<float>(0);
          cur[accessor.MfDimIndex()] = static_cast<float>(0);
          cur[accessor.ShowIndex()] = 0.0;
          break;
        case 1:
          cur[accessor.ClickIndex()] = 0.0;
          break;
        case 2:
          cur[accessor.EmbedGIndex()] = 0.0;
          break;
        default:
          cur[accessor.EmbedxGIndex() + off - 3] = 0.0;
          break;
      }
      return;
    }

    const uint32_t& start = d_sort_offset[i];
    const uint32_t& count = d_sort_cnt[i];
    const uint32_t& pos = d_sort_idx[start];

    const int& x = key2slot[pos];
    int y = pos - slot_lens[x];
    int mf_dim = slot_dims[x] - 3;

    double val = 0.0;

    switch (off) {
      case 0:
        cur[accessor.SlotIndex()] = static_cast<float>(slot_vector[x]);
        cur[accessor.MfDimIndex()] = static_cast<float>(mf_dim);
        SUM_GRAD_VALUE
        cur[accessor.ShowIndex()] = val;
        break;
      case 1:
        SUM_GRAD_VALUE
        cur[accessor.ClickIndex()] = val;
        break;
      case 2:
        SUM_GRAD_VALUE
        cur[accessor.EmbedGIndex()] = val * -1. * bs;
        break;
      default:
        int embedx_idx = off - 3;
        if (embedx_idx < mf_dim) {
          SUM_GRAD_VALUE
          cur[accessor.EmbedxGIndex() + embedx_idx] = val * -1. * bs;
        } else {
          cur[accessor.EmbedxGIndex() + embedx_idx] = 0.0;
        }
        break;
    }
  }
}

template <typename GPUAccessor>
void AccessorWrapper<GPUAccessor>::CopyForPullImpl(
    const phi::Place& place,
    uint64_t** gpu_keys,
    const std::vector<float*>& values,
    const float* total_values_gpu,
    const int64_t* gpu_len,
    const int slot_num,
    const int hidden_size,
    const int64_t total_length,
    int* gpu_dim,
    int feature_value_size) {
  int device_id = place.GetDeviceId();
  platform::CUDADeviceGuard guard(device_id);
  auto stream = dynamic_cast<phi::GPUContext*>(
                    phi::DeviceContextPool::Instance().Get(place))
                    ->stream();
  auto buf_value = memory::Alloc(place, values.size() * sizeof(float*));
  float** gpu_values = reinterpret_cast<float**>(buf_value->ptr());
  cudaMemcpy(gpu_values,
             values.data(),
             values.size() * sizeof(float*),
             cudaMemcpyHostToDevice);
  PullCopy<<<(total_length + 1024 - 1) / 1024, 1024, 0, stream>>>(
      gpu_values,
      total_values_gpu,
      gpu_len,
      slot_num,
      total_length,
      gpu_keys,
      feature_value_size,
      gpu_dim,
      gpu_accessor_);
  cudaStreamSynchronize(stream);
}

template <typename GPUAccessor>
void AccessorWrapper<GPUAccessor>::CopyForPushImpl(
    const phi::Place& place,
    const std::vector<const float*>& grad_values,
    float* total_grad_values_gpu,
    const std::vector<int64_t>& slot_lengths,
    const uint64_t total_length,
    const int batch_size,
    size_t grad_value_size,
    std::vector<int>& slot_vector,           // NOLINT
    std::vector<int>& slot_mf_dim_vector) {  // NOLINT
  int device_id = place.GetDeviceId();
  platform::CUDADeviceGuard guard(device_id);
  auto stream = dynamic_cast<phi::GPUContext*>(
                    phi::DeviceContextPool::Instance().Get(place))
                    ->stream();
  auto slot_lengths_lod = slot_lengths;
  for (int i = 1; i < slot_lengths_lod.size(); i++) {
    slot_lengths_lod[i] += slot_lengths_lod[i - 1];
  }
  auto buf_grad_value =
      memory::Alloc(place, grad_values.size() * sizeof(float*));
  auto buf_length = memory::Alloc(place, slot_lengths.size() * sizeof(int64_t));
  auto buf_slot_vector =
      memory::Alloc(place, slot_lengths_lod.size() * sizeof(int));
  auto buf_mf_dim_vector =
      memory::Alloc(place, slot_lengths_lod.size() * sizeof(int));
  float** gpu_values = reinterpret_cast<float**>(buf_grad_value->ptr());
  int64_t* gpu_len = reinterpret_cast<int64_t*>(buf_length->ptr());
  int* d_slot_vector = reinterpret_cast<int*>(buf_slot_vector->ptr());
  int* d_mf_dim_vector = reinterpret_cast<int*>(buf_mf_dim_vector->ptr());
  cudaMemcpyAsync(gpu_values,
                  grad_values.data(),
                  grad_values.size() * sizeof(float*),
                  cudaMemcpyHostToDevice,
                  stream);
  cudaMemcpyAsync(gpu_len,
                  slot_lengths_lod.data(),
                  slot_lengths.size() * sizeof(int64_t),
                  cudaMemcpyHostToDevice,
                  stream);
  cudaMemcpyAsync(d_slot_vector,
                  slot_vector.data(),
                  slot_lengths_lod.size() * sizeof(int),
                  cudaMemcpyHostToDevice,
                  stream);
  cudaMemcpyAsync(d_mf_dim_vector,
                  slot_mf_dim_vector.data(),
                  slot_lengths_lod.size() * sizeof(int),
                  cudaMemcpyHostToDevice,
                  stream);
  PushCopyWithPool<<<(total_length + 1024 - 1) / 1024, 1024, 0, stream>>>(
      total_grad_values_gpu,
      gpu_values,
      gpu_len,
      slot_lengths.size(),
      total_length,
      batch_size,
      d_slot_vector,
      d_mf_dim_vector,
      grad_value_size,
      gpu_accessor_);
  cudaStreamSynchronize(stream);
}

template <typename GPUAccessor>
void AccessorWrapper<GPUAccessor>::CopyForPullDedupImpl(
    const phi::Place& place,
    const uint64_t* total_keys,
    float** gpu_values,
    const float* total_values_gpu,
    const int64_t* slot_lens,
    const int* key2slot,
    const int hidden_size,
    const int64_t total_length,
    const int* slot_dims,
    const uint32_t* gpu_restore_idx,
    int pull_value_size) {
  int device_id = place.GetDeviceId();
  platform::CUDADeviceGuard guard(device_id);
  auto stream = dynamic_cast<phi::GPUContext*>(
                    phi::DeviceContextPool::Instance().Get(place))
                    ->stream();
  size_t N = total_length * hidden_size;
  PullDedupCopy<<<CUDA_BLOCK(N), stream>>>(N,
                                           total_keys,
                                           gpu_values,
                                           total_values_gpu,
                                           slot_lens,
                                           pull_value_size,
                                           slot_dims,
                                           hidden_size,
                                           key2slot,
                                           gpu_restore_idx,
                                           gpu_accessor_.common_pull_value);
  cudaStreamSynchronize(stream);
}

template <typename GPUAccessor>
void AccessorWrapper<GPUAccessor>::CopyForPushDedupImpl(
    const phi::Place& place,
    const uint64_t* total_keys,
    float** grad_values,
    float* total_grad_values_gpu,
    const int* slots,
    const int64_t* slot_lens,
    const int hidden_size,
    const int64_t total_length,
    const int64_t dedup_length,
    const int batch_size,
    const int* slot_dims,
    const int* key2slot,
    const uint32_t* d_restore_idx,
    const size_t grad_value_size) {
  int device_id = place.GetDeviceId();
  platform::CUDADeviceGuard guard(device_id);
  auto stream = dynamic_cast<phi::GPUContext*>(
                    phi::DeviceContextPool::Instance().Get(place))
                    ->stream();
  cudaMemsetAsync(
      total_grad_values_gpu, 0, dedup_length * grad_value_size, stream);
  size_t N = total_length * hidden_size;
  PushMergeCopyAtomic<<<CUDA_BLOCK(N), stream>>>(
      N,
      total_keys,
      total_grad_values_gpu,
      grad_values,
      hidden_size,
      batch_size,
      slots,
      slot_dims,
      slot_lens,
      key2slot,
      d_restore_idx,
      grad_value_size,
      gpu_accessor_.common_push_value);

  cudaStreamSynchronize(stream);
}

template <typename GPUAccessor>
void AccessorWrapper<GPUAccessor>::CopyForPushDedupImpl(
    const phi::Place& place,
    const uint64_t* total_keys,
    float** grad_values,
    float* total_grad_values_gpu,
    const int* slots,
    const int64_t* slot_lens,
    const int hidden_size,
    const int64_t total_length,
    const int64_t dedup_length,
    const int batch_size,
    const int* slot_dims,
    const int* key2slot,
    const uint32_t* gpu_sort_idx,
    const uint32_t* gpu_sort_offset,
    const uint32_t* gpu_sort_lens,
    const size_t grad_value_size) {
  int device_id = place.GetDeviceId();
  platform::CUDADeviceGuard guard(device_id);
  auto stream = dynamic_cast<phi::GPUContext*>(
                    phi::DeviceContextPool::Instance().Get(place))
                    ->stream();
  // merge all grad to one
  size_t N = dedup_length * hidden_size;
  PushMergeCopy<<<CUDA_BLOCK(N), stream>>>(N,
                                           total_keys,
                                           total_grad_values_gpu,
                                           grad_values,
                                           hidden_size,
                                           batch_size,
                                           slots,
                                           slot_dims,
                                           slot_lens,
                                           key2slot,
                                           gpu_sort_idx,
                                           gpu_sort_offset,
                                           gpu_sort_lens,
                                           grad_value_size,
                                           gpu_accessor_.common_push_value);
  cudaStreamSynchronize(stream);
}

#ifdef PADDLE_WITH_PSCORE
template class AccessorWrapper<CommonFeatureValueAccessor>;
#endif

#ifdef PADDLE_WITH_PSLIB
template class AccessorWrapper<CommonFeatureValueAccessor>;
#endif

}  // namespace framework
}  // namespace paddle
#endif
