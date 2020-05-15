// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef PADDLE_WITH_BOX_PS
#include <algorithm>
#include <ctime>
#include <memory>
#include <numeric>
#include "paddle/fluid/framework/fleet/box_wrapper.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/platform/gpu_info.h"

namespace paddle {
namespace framework {
#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

__global__ void PullCopy(float** dest, const boxps::FeatureValueGpu* src,
                         const int64_t* len, int hidden, int slot_num,
                         int total_len, uint64_t** keys) {
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
    if (*(keys[x] + y) == 0) {
      *(dest[x] + y * hidden) = 0;
      *(dest[x] + y * hidden + 1) = 0;
      *(dest[x] + y * hidden + 2) = 0;
    } else {
      *(dest[x] + y * hidden) = (src + i)->show;
      *(dest[x] + y * hidden + 1) = (src + i)->clk;
      *(dest[x] + y * hidden + 2) = (src + i)->embed_w;
    }
    if ((src + i)->embedding_size == 0 || *(keys[x] + y) == 0) {
      for (int j = 0; j < 8; j++) {
        *(dest[x] + y * hidden + 3 + j) = 0;
      }
    } else {
      for (int j = 0; j < 8; j++) {
        *(dest[x] + y * hidden + 3 + j) = (src + i)->embedx[1 + j];
      }
    }
  }
}

__global__ void CopyKeysKernel(uint64_t** src_keys, uint64_t* dest_total_keys,
                               const int64_t* len, int slot_num,
                               int total_len) {
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
    dest_total_keys[i] = src_keys[x][y];
  }
}

__global__ void PushCopy(boxps::FeaturePushValueGpu* dest, float** src,
                         int64_t* len, int hidden, int slot_num, int total_len,
                         int bs, int* slot_vector) {
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
    (dest + i)->slot = slot_vector[x];
    (dest + i)->show = *(src[x] + y * hidden);
    (dest + i)->clk = *(src[x] + y * hidden + 1);
    (dest + i)->embed_g = *(src[x] + y * hidden + 2) * -1. * bs;
    for (int j = 0; j < 8; j++) {
      (dest + i)->embedx_g[j] = *(src[x] + y * hidden + 3 + j) * -1. * bs;
    }
  }
}

void BoxWrapper::CopyForPull(const paddle::platform::Place& place,
                             uint64_t** gpu_keys,
                             const std::vector<float*>& values,
                             const boxps::FeatureValueGpu* total_values_gpu,
                             const int64_t* gpu_len, const int slot_num,
                             const int hidden_size,
                             const int64_t total_length) {
  auto stream = dynamic_cast<platform::CUDADeviceContext*>(
                    platform::DeviceContextPool::Instance().Get(
                        BOOST_GET_CONST(platform::CUDAPlace, place)))
                    ->stream();
  auto buf_value = memory::AllocShared(place, values.size() * sizeof(float*));
  float** gpu_values = reinterpret_cast<float**>(buf_value->ptr());
  cudaMemcpy(gpu_values, values.data(), values.size() * sizeof(float*),
             cudaMemcpyHostToDevice);

  PullCopy<<<(total_length + 512 - 1) / 512, 512, 0, stream>>>(
      gpu_values, total_values_gpu, gpu_len, hidden_size, slot_num,
      total_length, gpu_keys);
  cudaStreamSynchronize(stream);
}

void BoxWrapper::CopyKeys(const paddle::platform::Place& place,
                          uint64_t** origin_keys, uint64_t* total_keys,
                          const int64_t* gpu_len, int slot_num, int total_len) {
  auto stream = dynamic_cast<platform::CUDADeviceContext*>(
                    platform::DeviceContextPool::Instance().Get(
                        BOOST_GET_CONST(platform::CUDAPlace, place)))
                    ->stream();
  CopyKeysKernel<<<(total_len + 512 - 1) / 512, 512, 0, stream>>>(
      origin_keys, total_keys, gpu_len, slot_num, total_len);
  cudaStreamSynchronize(stream);
}

void BoxWrapper::CopyForPush(const paddle::platform::Place& place,
                             const std::vector<const float*>& grad_values,
                             boxps::FeaturePushValueGpu* total_grad_values_gpu,
                             const std::vector<int64_t>& slot_lengths,
                             const int hidden_size, const int64_t total_length,
                             const int batch_size) {
  auto stream = dynamic_cast<platform::CUDADeviceContext*>(
                    platform::DeviceContextPool::Instance().Get(
                        BOOST_GET_CONST(platform::CUDAPlace, place)))
                    ->stream();
  auto slot_lengths_lod = slot_lengths;
  for (int i = 1; i < slot_lengths_lod.size(); i++) {
    slot_lengths_lod[i] += slot_lengths_lod[i - 1];
  }
  auto buf_grad_value =
      memory::AllocShared(place, grad_values.size() * sizeof(float*));
  auto buf_length =
      memory::AllocShared(place, slot_lengths.size() * sizeof(int64_t));
  auto buf_slot_vector =
      memory::AllocShared(place, slot_lengths_lod.size() * sizeof(int));

  float** gpu_values = reinterpret_cast<float**>(buf_grad_value->ptr());
  int64_t* gpu_len = reinterpret_cast<int64_t*>(buf_length->ptr());
  int* d_slot_vector = reinterpret_cast<int*>(buf_slot_vector->ptr());

  cudaMemcpy(gpu_values, grad_values.data(),
             grad_values.size() * sizeof(float*), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_len, slot_lengths_lod.data(),
             slot_lengths.size() * sizeof(int64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_slot_vector, slot_vector_.data(),
             slot_lengths_lod.size() * sizeof(int), cudaMemcpyHostToDevice);

  PushCopy<<<(total_length + 512 - 1) / 512, 512, 0, stream>>>(
      total_grad_values_gpu, gpu_values, gpu_len, hidden_size,
      slot_lengths.size(), total_length, batch_size, d_slot_vector);
  cudaStreamSynchronize(stream);
}
}  // end namespace framework
}  // end namespace paddle
#endif
