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

#ifdef PADDLE_WITH_HETERPS
#include "paddle/fluid/framework/fleet/ps_gpu_wrapper.h"
#include <xpu/runtime.h>
#include <algorithm>
#include <ctime>
#include <memory>
#include <numeric>
#include "paddle/fluid/framework/fleet/heter_ps/optimizer_conf.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "xpu/kernel/cluster_header.h"
#include "xpu/kernel/debug.h"
#include "xpu/kernel/math.h"

namespace paddle {
namespace framework {

__global__ void PullCopy(float** dest, const FeatureValue* src,
                         const int64_t* len, int hidden, int slot_num,
                         int total_len, uint64_t** keys) {
  int cid = core_id();
  int ncores = core_num();
  if (cid >= ncores) {
    return;
  }
  int thread_id = ncores * cluster_id() + cid;
  int nthreads = ncores * cluster_num();
  int len_per_loop = 1;

  for (int i = thread_id * len_per_loop; i < total_len;
       i += nthreads * len_per_loop) {
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
      *(dest[x] + y * hidden + 2) = (src + i)->lr;
    }
    if ((src + i)->mf_size == 0 || *(keys[x] + y) == 0) {
      for (int j = 0; j < hidden - 3; j++) {
        *(dest[x] + y * hidden + 3 + j) = 0;
      }
    } else {
      for (int j = 0; j < hidden - 3; j++) {
        *(dest[x] + y * hidden + 3 + j) = (src + i)->mf[1 + j];
      }
    }
  }
}

__global__ void CopyKeysKernel(uint64_t** src_keys, uint64_t* dest_total_keys,
                               const int64_t* len, int slot_num,
                               int total_len) {
  int cid = core_id();
  int ncores = core_num();
  if (cid >= ncores) {
    return;
  }
  int thread_id = ncores * cluster_id() + cid;
  int nthreads = ncores * cluster_num();
  int len_per_loop = 1;

  for (int i = thread_id * len_per_loop; i < total_len;
       i += nthreads * len_per_loop) {
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

__global__ void PushCopy(FeaturePushValue* dest, float** src, int64_t* len,
                         int hidden, int slot_num, int total_len, int bs,
                         int* slot_vector) {
  int cid = core_id();
  int ncores = core_num();
  if (cid >= ncores) {
    return;
  }
  int thread_id = ncores * cluster_id() + cid;
  int nthreads = ncores * cluster_num();
  int len_per_loop = 1;

  for (int i = thread_id * len_per_loop; i < total_len;
       i += nthreads * len_per_loop) {
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
    (dest + i)->lr_g = *(src[x] + y * hidden + 2) * -1. * bs;
    for (int j = 0; j < hidden - 3; j++) {
      (dest + i)->mf_g[j] = *(src[x] + y * hidden + 3 + j) * -1. * bs;
    }
  }
}

void PSGPUWrapper::CopyForPull(const paddle::platform::Place& place,
                               uint64_t** gpu_keys,
                               const std::vector<float*>& values,
                               const FeatureValue* total_values_gpu,
                               const int64_t* gpu_len, const int slot_num,
                               const int hidden_size,
                               const int64_t total_length) {
  XPUStream stream = nullptr;
  auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
  stream = static_cast<platform::XPUDeviceContext*>(dev_ctx)
               ->x_context()
               ->xpu_stream;
  T* buf_value = nullptr;
  xpu_malloc(reinterpret_cast<void**>(&buf_value),
             values.size() * sizeof(float*));
  float** gpu_values = reinterpret_cast<float**>(&buf_value);
  ​ xpu_memcpy(gpu_values, values.data(), ​ values.size() * sizeof(float*),
                 XPU_HOST_TO_DEVICE);

  PullCopy<<<2, 64, stream>>>(gpu_values, total_values_gpu, gpu_len,
                              hidden_size, slot_num, total_length, gpu_keys);

  xpu_wait(stream);
}

void PSGPUWrapper::CopyKeys(const paddle::platform::Place& place,
                            uint64_t** origin_keys, uint64_t* total_keys,
                            const int64_t* gpu_len, int slot_num,
                            int total_len) {
  XPUStream stream = nullptr;
  auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
  stream = static_cast<platform::XPUDeviceContext*>(dev_ctx)
               ->x_context()
               ->xpu_stream;
  CopyKeysKernel<<<2, 64, stream>>>(origin_keys, total_keys, gpu_len, slot_num,
                                    total_len);
  xpu_wait(stream);
}

void PSGPUWrapper::CopyForPush(const paddle::platform::Place& place,
                               const std::vector<const float*>& grad_values,
                               FeaturePushValue* total_grad_values_gpu,
                               const std::vector<int64_t>& slot_lengths,
                               const int hidden_size,
                               const int64_t total_length,
                               const int batch_size) {
  XPUStream stream = nullptr;
  auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
  stream = static_cast<platform::XPUDeviceContext*>(dev_ctx)
               ->x_context()
               ->xpu_stream;
  auto slot_lengths_lod = slot_lengths;
  for (int i = 1; i < slot_lengths_lod.size(); i++) {
    slot_lengths_lod[i] += slot_lengths_lod[i - 1];
  }

  T* buf_grad_value = nullptr;
  T* buf_length = nullptr;
  T* buf_slot_vector = nullptr;

  xpu_malloc(reinterpret_cast<void**>(&buf_grad_value),
             grad_values.size() * sizeof(float*));
  xpu_malloc(reinterpret_cast<void**>(&buf_length),
             slot_lengths.size() * sizeof(int64_t));
  xpu_malloc(reinterpret_cast<void**>(&buf_slot_vector),
             slot_lengths_lod.size() * sizeof(int));

  float** gpu_values = reinterpret_cast<float**>(&buf_grad_value);
  int64_t* gpu_len = reinterpret_cast<int64_t*>(buf_length);
  int* d_slot_vector = reinterpret_cast<int*>(buf_slot_vector);

  ​xpu_memcpy(gpu_values, grad_values.data(),
                grad_values.size() * sizeof(float*), XPU_HOST_TO_DEVICE);
  ​xpu_memcpy(gpu_len, slot_lengths_lod.data(),
                slot_lengths.size() * sizeof(int64_t), XPU_HOST_TO_DEVICE);
  ​xpu_memcpy(d_slot_vector, slot_vector_.data(),
                slot_lengths_lod.size() * sizeof(int), XPU_HOST_TO_DEVICE);

  PushCopy<<<2, 64, stream>>>(total_grad_values_gpu, gpu_values, gpu_len,
                              hidden_size, slot_lengths.size(), total_length,
                              batch_size, d_slot_vector);
  xpu_wait(stream);
}

}  // end namespace framework
}  // end namespace paddle
#endif
