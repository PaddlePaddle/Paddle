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
#include "paddle/fluid/platform/device/gpu/gpu_info.h"

namespace paddle {
namespace framework {

__global__ void PullCopy(float** dest,
                         const FeatureValue* src,
                         const int64_t* len,
                         int hidden,
                         int slot_num,
                         int total_len,
                         uint64_t** keys) {
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

__global__ void PullCopy(float** dest,
                         const FeatureValue* src,
                         const int64_t* len,
                         int slot_num,
                         int total_len,
                         uint64_t** keys,
                         uint64_t max_val_size,
                         int* gpu_dim) {
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
    FeatureValue* feature_value_ptr =
        (FeatureValue*)((char*)src + uint64_t(i) * uint64_t(max_val_size));
    int mf_dim = gpu_dim[x] - 3;
    if (*(keys[x] + y) == 0) {
      *(dest[x] + y * (mf_dim + 3)) = 0;
      *(dest[x] + y * (mf_dim + 3) + 1) = 0;
      *(dest[x] + y * (mf_dim + 3) + 2) = 0;
    } else {
      *(dest[x] + y * (mf_dim + 3)) = feature_value_ptr->show;
      *(dest[x] + y * (mf_dim + 3) + 1) = feature_value_ptr->clk;
      *(dest[x] + y * (mf_dim + 3) + 2) = feature_value_ptr->lr;
    }
    if ((feature_value_ptr)->mf_size == 0 || *(keys[x] + y) == 0) {
      for (int j = 0; j < mf_dim; j++) {
        *(dest[x] + y * (mf_dim + 3) + 3 + j) = 0;
      }
    } else {
      for (int j = 0; j < mf_dim; j++) {
        *(dest[x] + y * (mf_dim + 3) + 3 + j) = feature_value_ptr->mf[1 + j];
      }
    }
  }
}

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

__global__ void PushCopyWithPool(FeaturePushValue* dest,
                                 float** src,
                                 int64_t* len,
                                 int slot_num,
                                 uint64_t total_len,
                                 int bs,
                                 int* slot_vector,
                                 int* mf_dim_vector,
                                 size_t grad_value_size) {
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
    FeaturePushValue* cur =
        (FeaturePushValue*)((char*)dest + i * grad_value_size);
    cur->slot = slot_vector[x];
    int mf_dim = mf_dim_vector[x];
    cur->mf_dim = mf_dim;
    cur->show = *(src[x] + y * (mf_dim + 3));
    cur->clk = *(src[x] + y * (mf_dim + 3) + 1);
    cur->lr_g = *(src[x] + y * (mf_dim + 3) + 2) * -1. * bs;
    for (int j = 0; j < cur->mf_dim; j++) {
      cur->mf_g[j] = *(src[x] + y * (mf_dim + 3) + 3 + j) * -1. * bs;
    }
  }
}
PSGPUWrapper::~PSGPUWrapper() { delete HeterPs_; }

void PSGPUWrapper::CopyForPull(const paddle::platform::Place& place,
                               uint64_t** gpu_keys,
                               const std::vector<float*>& values,
                               const FeatureValue* total_values_gpu,
                               const int64_t* gpu_len,
                               const int slot_num,
                               const int hidden_size,
                               const int64_t total_length) {
  auto stream = dynamic_cast<platform::CUDADeviceContext*>(
                    platform::DeviceContextPool::Instance().Get(place))
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
      hidden_size,
      slot_num,
      total_length,
      gpu_keys);
  cudaStreamSynchronize(stream);
}

void PSGPUWrapper::CopyForPull(const paddle::platform::Place& place,
                               uint64_t** gpu_keys,
                               const std::vector<float*>& values,
                               const FeatureValue* total_values_gpu,
                               const int64_t* gpu_len,
                               const int slot_num,
                               const int hidden_size,
                               const int64_t total_length,
                               int* gpu_dim) {
  auto stream = dynamic_cast<platform::CUDADeviceContext*>(
                    platform::DeviceContextPool::Instance().Get(place))
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
      val_type_size_,
      gpu_dim);
  cudaStreamSynchronize(stream);
}

void PSGPUWrapper::CopyKeys(const paddle::platform::Place& place,
                            uint64_t** origin_keys,
                            uint64_t* total_keys,
                            const int64_t* gpu_len,
                            int slot_num,
                            int total_len) {
  auto stream = dynamic_cast<platform::CUDADeviceContext*>(
                    platform::DeviceContextPool::Instance().Get(place))
                    ->stream();
  CopyKeysKernel<<<(total_len + 1024 - 1) / 1024, 1024, 0, stream>>>(
      origin_keys, total_keys, gpu_len, slot_num, total_len);
  cudaStreamSynchronize(stream);
}

void PSGPUWrapper::CopyForPush(const paddle::platform::Place& place,
                               const std::vector<const float*>& grad_values,
                               FeaturePushValue* total_grad_values_gpu,
                               const std::vector<int64_t>& slot_lengths,
                               const int hidden_size,
                               const int64_t total_length,
                               const int batch_size) {
  auto stream = dynamic_cast<platform::CUDADeviceContext*>(
                    platform::DeviceContextPool::Instance().Get(place))
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

  float** gpu_values = reinterpret_cast<float**>(buf_grad_value->ptr());
  int64_t* gpu_len = reinterpret_cast<int64_t*>(buf_length->ptr());
  int* d_slot_vector = reinterpret_cast<int*>(buf_slot_vector->ptr());

  cudaMemcpy(gpu_values,
             grad_values.data(),
             grad_values.size() * sizeof(float*),
             cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_len,
             slot_lengths_lod.data(),
             slot_lengths.size() * sizeof(int64_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_slot_vector,
             slot_vector_.data(),
             slot_lengths_lod.size() * sizeof(int),
             cudaMemcpyHostToDevice);

  PushCopy<<<(total_length + 1024 - 1) / 1024, 1024, 0, stream>>>(
      total_grad_values_gpu,
      gpu_values,
      gpu_len,
      hidden_size,
      slot_lengths.size(),
      total_length,
      batch_size,
      d_slot_vector);
  cudaStreamSynchronize(stream);
}

void PSGPUWrapper::CopyForPush(const paddle::platform::Place& place,
                               const std::vector<const float*>& grad_values,
                               FeaturePushValue* total_grad_values_gpu,
                               const std::vector<int64_t>& slot_lengths,
                               const uint64_t total_length,
                               const int batch_size,
                               size_t grad_value_size) {
  auto stream = dynamic_cast<platform::CUDADeviceContext*>(
                    platform::DeviceContextPool::Instance().Get(place))
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
  cudaMemcpy(gpu_values,
             grad_values.data(),
             grad_values.size() * sizeof(float*),
             cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_len,
             slot_lengths_lod.data(),
             slot_lengths.size() * sizeof(int64_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_slot_vector,
             slot_vector_.data(),
             slot_lengths_lod.size() * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_mf_dim_vector,
             slot_mf_dim_vector_.data(),
             slot_lengths_lod.size() * sizeof(int),
             cudaMemcpyHostToDevice);
  PushCopyWithPool<<<(total_length + 1024 - 1) / 1024, 1024, 0, stream>>>(
      total_grad_values_gpu,
      gpu_values,
      gpu_len,
      slot_lengths.size(),
      total_length,
      batch_size,
      d_slot_vector,
      d_mf_dim_vector,
      grad_value_size);
  cudaStreamSynchronize(stream);
}

void PSGPUWrapper::SetSparseSGD(float nonclk_coeff,
                                float clk_coeff,
                                float min_bound,
                                float max_bound,
                                float learning_rate,
                                float initial_g2sum,
                                float initial_range) {
  OptimizerConfig optimizer_config;
  optimizer_config.set_sparse_sgd(nonclk_coeff,
                                  clk_coeff,
                                  min_bound,
                                  max_bound,
                                  learning_rate,
                                  initial_g2sum,
                                  initial_range);
  HeterPs_->set_sparse_sgd(optimizer_config);
}

void PSGPUWrapper::SetEmbedxSGD(float mf_create_thresholds,
                                float mf_learning_rate,
                                float mf_initial_g2sum,
                                float mf_initial_range,
                                float mf_min_bound,
                                float mf_max_bound) {
  OptimizerConfig optimizer_config;
  optimizer_config.set_embedx_sgd(mf_create_thresholds,
                                  mf_learning_rate,
                                  mf_initial_g2sum,
                                  mf_initial_range,
                                  mf_min_bound,
                                  mf_max_bound);
  HeterPs_->set_embedx_sgd(optimizer_config);
}

}  // end namespace framework
}  // end namespace paddle
#endif
