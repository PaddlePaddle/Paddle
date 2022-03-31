/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/framework/fleet/heter_ps/heter_comm_kernel.h"
#ifdef PADDLE_WITH_HETERPS
namespace paddle {
namespace framework {

#ifdef PADDLE_WITH_CUDA

template <typename T>
__global__ void fill_idx(T* idx, int64 len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    idx[i] = i;
  }
}

// template <typename T>
// void show_tensor(T* input, int64 len, gpuStream_t stream, std::string
// name)
// {
//  T tmp[len];  // NOLINT
//  cudaMemcpyAsync(&tmp, input, sizeof(T) * len, cudaMemcpyDeviceToHost,
//  stream);
//  cudaStreamSynchronize(stream);
//  std::cout << name;
//  for (int i = 0; i < len; ++i) {
//    std::cout << ":" << tmp[i];
//  }
//  std::cout << std::endl;
//}

template <typename T>
__global__ void calc_shard_offset(T* idx, T* left, T* right, int64 len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len - 1) {
    if (idx[i] != idx[i + 1]) {
      right[idx[i]] = i;
      left[idx[i + 1]] = i + 1;
    }
  }
  if (i == 0) {
    left[idx[i]] = i;
  }
  if (i == (len - 1)) {
    right[idx[i]] = i;
  }
}

template <typename KeyType, typename T>
__global__ void calc_shard_index(KeyType* d_keys, int64 len, T* shard_index,
                                 int total_gpu) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    shard_index[i] = d_keys[i] % total_gpu;
  }
}

template <typename KeyType, typename T>
__global__ void fill_shard_key(KeyType* d_shard_keys, KeyType* d_keys, T* idx,
                               int64 len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    d_shard_keys[i] = d_keys[idx[i]];
  }
}

template <typename KeyType, typename GradType, typename T>
__global__ void fill_shard_grads(KeyType* d_shard_keys, KeyType* d_keys,
                                 GradType* d_shard_grads, GradType* d_grads,
                                 T* idx, int64 len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    d_shard_keys[i] = d_keys[idx[i]];
    d_shard_grads[i] = d_grads[idx[i]];
  }
}

template <typename ValType, typename T>
__global__ void fill_dvals(ValType* d_shard_vals, ValType* d_vals, T* idx,
                           int64 len) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    d_vals[idx[i]] = d_shard_vals[i];
  }
}

// cuda implemention of  heter_comm_kernel.h
template <typename T, typename StreamType>
void HeterCommKernel::fill_idx(T* idx, int64 len, StreamType& stream) {
  int grid_size = (len - 1) / block_size_ + 1;
  fill_idx<<<grid_size, block_size_, 0, stream>>>(idx, len);
}

template <typename T, typename StreamType>
void HeterCommKernel::calc_shard_offset(T* idx, T* left, T* right, int64 len,
                                        int total_devs, StreamType& stream) {
  int grid_size = (len - 1) / block_size_ + 1;
  calc_shard_offset<<<grid_size, block_size_, 0, stream>>>(idx, left, right,
                                                           len);
}

template <typename KeyType, typename T, typename StreamType>
void HeterCommKernel::calc_shard_index(KeyType* d_keys, int64 len,
                                       T* shard_index, int total_gpu,
                                       StreamType& stream) {
  int grid_size = (len - 1) / block_size_ + 1;
  calc_shard_index<<<grid_size, block_size_, 0, stream>>>(
      d_keys, len, shard_index, total_gpu);
}

template <typename KeyType, typename T, typename StreamType>
void HeterCommKernel::fill_shard_key(KeyType* d_shard_keys, KeyType* d_keys,
                                     T* idx, int64 len, StreamType& stream) {
  int grid_size = (len - 1) / block_size_ + 1;
  fill_shard_key<<<grid_size, block_size_, 0, stream>>>(d_shard_keys, d_keys,
                                                        idx, len);
}

template <typename KeyType, typename GradType, typename T, typename StreamType>
void HeterCommKernel::fill_shard_grads(KeyType* d_shard_keys, KeyType* d_keys,
                                       GradType* d_shard_grads,
                                       GradType* d_grads, T* idx, int64 len,
                                       StreamType& stream) {
  int grid_size = (len - 1) / block_size_ + 1;
  fill_shard_grads<<<grid_size, block_size_, 0, stream>>>(
      d_shard_keys, d_keys, d_shard_grads, d_grads, idx, len);
}

template <typename ValType, typename T, typename StreamType>
void HeterCommKernel::fill_dvals(ValType* d_shard_vals, ValType* d_vals, T* idx,
                                 int64 len, StreamType& stream) {
  int grid_size = (len - 1) / block_size_ + 1;
  fill_dvals<<<grid_size, block_size_, 0, stream>>>(d_shard_vals, d_vals, idx,
                                                    len);
}
#endif

}  // end namespace framework
}  // end namespace paddle
#endif
