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
#include <algorithm>
#include <ctime>
#include <memory>
#include <numeric>

#include "paddle/fluid/framework/fleet/heter_ps/optimizer_conf.h"
#include "paddle/fluid/framework/fleet/ps_gpu_wrapper.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/core/platform/device/gpu/gpu_info.h"

namespace paddle {
namespace framework {

const int CUDA_NUM_THREADS = phi::PADDLE_CUDA_NUM_THREADS;
#define GET_BLOCK(N) ((N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS)
#define CUDA_BLOCK(N) GET_BLOCK(N), CUDA_NUM_THREADS, 0

__global__ void CopyKeysKernel(uint64_t** src_keys,
                               uint64_t* dest_total_keys,
                               const int64_t* len,
                               int slot_num,
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

__global__ void PushCopy(FeaturePushValue* dest,
                         float** src,
                         int64_t* len,
                         int hidden,
                         int slot_num,
                         int total_len,
                         int bs,
                         int* slot_vector) {
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
    (dest + i)->lr_g = *(src[x] + y * hidden + 2) * -1. * bs;
    for (int j = 0; j < hidden - 3; j++) {
      (dest + i)->mf_g[j] = *(src[x] + y * hidden + 3 + j) * -1. * bs;
    }
  }
}

PSGPUWrapper::~PSGPUWrapper() { delete HeterPs_; }

void PSGPUWrapper::CopyKeys(const phi::Place& place,
                            uint64_t** origin_keys,
                            uint64_t* total_keys,
                            const int64_t* gpu_len,
                            int slot_num,
                            int total_len) {
  int device_id = place.GetDeviceId();
  platform::CUDADeviceGuard guard(device_id);
  auto stream = dynamic_cast<phi::GPUContext*>(
                    phi::DeviceContextPool::Instance().Get(place))
                    ->stream();
  CopyKeysKernel<<<(total_len + 1024 - 1) / 1024, 1024, 0, stream>>>(
      origin_keys, total_keys, gpu_len, slot_num, total_len);
  cudaStreamSynchronize(stream);
}

__global__ void CopyKeysKernel2(const int total_len,
                                uint64_t** src_keys,
                                uint64_t* dest_total_keys,
                                const int slot_num,
                                const int64_t* slot_lens,
                                int* key2slots) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int low = 0;
    int high = slot_num - 1;
    while (low < high) {
      int mid = (low + high) / 2;
      if (i < slot_lens[mid + 1]) {
        high = mid;
      } else {
        low = mid + 1;
      }
    }
    key2slots[i] = low;
    int y = i - slot_lens[low];
    dest_total_keys[i] = src_keys[low][y];
  }
}

void PSGPUWrapper::CopyKeys(const phi::Place& place,
                            uint64_t** origin_keys,
                            uint64_t* total_keys,
                            const int64_t* slot_lens,
                            int slot_num,
                            int total_len,
                            int* key2slot) {
  int device_id = place.GetDeviceId();
  platform::CUDADeviceGuard guard(device_id);
  auto stream = dynamic_cast<phi::GPUContext*>(
                    phi::DeviceContextPool::Instance().Get(place))
                    ->stream();
  CopyKeysKernel2<<<CUDA_BLOCK(total_len), stream>>>(
      total_len, origin_keys, total_keys, slot_num, slot_lens, key2slot);
  cudaStreamSynchronize(stream);
}

void PSGPUWrapper::SetSparseSGD(float nonclk_coeff,
                                float clk_coeff,
                                float min_bound,
                                float max_bound,
                                float learning_rate,
                                float initial_g2sum,
                                float initial_range,
                                float beta1_decay_rate,
                                float beta2_decay_rate,
                                float ada_epsilon) {
  optimizer_config_.set_sparse_sgd(nonclk_coeff,
                                   clk_coeff,
                                   min_bound,
                                   max_bound,
                                   learning_rate,
                                   initial_g2sum,
                                   initial_range,
                                   beta1_decay_rate,
                                   beta2_decay_rate,
                                   ada_epsilon);
}

void PSGPUWrapper::SetEmbedxSGD(float mf_create_thresholds,
                                float mf_learning_rate,
                                float mf_initial_g2sum,
                                float mf_initial_range,
                                float mf_min_bound,
                                float mf_max_bound,
                                float mf_beta1_decay_rate,
                                float mf_beta2_decay_rate,
                                float mf_ada_epsilon,
                                float nodeid_slot,
                                float feature_learning_rate) {
  optimizer_config_.set_embedx_sgd(mf_create_thresholds,
                                   mf_learning_rate,
                                   mf_initial_g2sum,
                                   mf_initial_range,
                                   mf_min_bound,
                                   mf_max_bound,
                                   mf_beta1_decay_rate,
                                   mf_beta2_decay_rate,
                                   mf_ada_epsilon,
                                   nodeid_slot,
                                   feature_learning_rate);
}

}  // namespace framework
}  // namespace paddle
#endif
