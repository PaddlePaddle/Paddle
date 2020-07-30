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
#include <memory.h>
#include "cub/cub.cuh"
#include "paddle/fluid/operators/math/math_function.h"

#define NORM_POS(idx, row, col) (((idx)*block_cols + (col)) * ins_num + (row))
#define SCALE_MEAN_POS(idx, col) ((idx)*block_cols + (col))
#define INPUT_POS(idx, row, col) \
  (((embed_dim * (idx)) + (col)) * ins_num + (row))

#define INPUT_POS_FF(idx, row, col) \
  (embed_dim * (idx) + (col) + (row)*input_cols)

#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
static inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}
namespace paddle {
namespace operators {

template <typename T>
__global__ void nncross_normforward_multi(int len, int n, int embed_dim,
                                          int ins_num, const T* inputs,
                                          T* norm_output, const T* mean,
                                          const T* scale) {
  CUDA_KERNEL_LOOP(i, len) {
    int norm_cols = n * (embed_dim * 3 + 1);
    int block_cols = embed_dim * 3 + 1;
    int input_cols = embed_dim * 2 * n;

    int row = i / norm_cols;
    int col_global = i % norm_cols;
    int block_idx = col_global / block_cols;
    int col = col_global % block_cols;

    if (col < embed_dim) {
      norm_output[i] =
          (inputs[INPUT_POS_FF(block_idx * 2, row, col)] - mean[col_global]) *
          scale[col_global];
    } else if (col < embed_dim * 2) {
      col -= embed_dim;
      norm_output[i] = (inputs[INPUT_POS_FF(block_idx * 2 + 1, row, col)] -
                        mean[col_global]) *
                       scale[col_global];
    } else if (col < embed_dim * 3) {
      col -= 2 * embed_dim;
      norm_output[i] = (inputs[INPUT_POS_FF(block_idx * 2, row, col)] *
                            inputs[INPUT_POS_FF(block_idx * 2 + 1, row, col)] -
                        mean[col_global]) *
                       scale[col_global];
    }
  }
}

template <typename T>
__global__ void nncross_normforward_multi_sim(int len, int N, int embed_dim,
                                              int ins_num, const T* inputs,
                                              T* norm_output, const T* mean,
                                              const T* scale) {
  CUDA_KERNEL_LOOP(i, len) {
    int block_cols = embed_dim * 3 + 1;
    int row = i / N;
    int col_global = (i % N + 1) * block_cols - 1;
    int block_idx = col_global / block_cols;
    int input_cols = embed_dim * 2 * N;

    T sum = 0;
    for (int j = 0; j < embed_dim; ++j) {
      sum += inputs[INPUT_POS_FF(block_idx * 2, row, j)] *
             inputs[INPUT_POS_FF(block_idx * 2 + 1, row, j)];
    }
    norm_output[row * block_cols * N + col_global] =
        (sum - mean[col_global]) * scale[col_global];
  }
}

template <typename T>
__global__ void nncross_normbackpropagate_multi(int len, int N, int embed_dim,
                                                int ins_num, const T* inputs,
                                                const T* norm_grad, T* grads,
                                                const T* mean, const T* scale) {
  CUDA_KERNEL_LOOP(i, len) {
    int row = i % ins_num;
    int col_global = i / ins_num;
    int a_idx = col_global / embed_dim;
    int col = col_global % embed_dim;
    int block_cols = embed_dim * 3 + 1;

    // grad 0
    grads[i] += norm_grad[NORM_POS(a_idx / 2, row, col)] *
                scale[SCALE_MEAN_POS(a_idx / 2, col)];
    // grad 1
    grads[i] += norm_grad[NORM_POS(a_idx / 2, row, (embed_dim * 2 + col))] *
                scale[SCALE_MEAN_POS(a_idx / 2, (embed_dim * 2 + col))] *
                inputs[INPUT_POS((1 + (a_idx / 2) * 4 - a_idx), row, col)];
    // grad 2
    grads[i] += norm_grad[NORM_POS(a_idx / 2, row, (embed_dim * 3))] *
                scale[SCALE_MEAN_POS(a_idx / 2, (embed_dim * 3))] *
                inputs[INPUT_POS((1 + (a_idx / 2) * 4 - a_idx), row, col)];
  }
}

template <typename T>
__global__ void kernel_mean_scale(int N, const T* summary, T* mean, T* scale) {
  CUDA_KERNEL_LOOP(i, N) {
    mean[i] = summary[i + N] / summary[i];
    scale[i] = sqrt(summary[i] / summary[i + 2 * N]);
  }
}

template <typename T>
__global__ void kernel_normbackwardsummary_x0(int len, int row, T* in_val,
                                              T* sum_grad, const T* means,
                                              const T* scale,
                                              const T squared_sum_epsilon) {
  CUDA_KERNEL_LOOP(i, len) {
    in_val[i] = in_val[i] / scale[i / row] + means[i / row];
  }
}

template <typename T>
__global__ void kernel_normbackwardsummary_plus_mean(
    int len, int row, T* in_val, T* sum_grad, const T* means, const T* scale,
    const T squared_sum_epsilon) {
  CUDA_KERNEL_LOOP(i, len) {
    in_val[i] = (in_val[i] - means[i / row]) * (in_val[i] - means[i / row]);
  }
}

template <typename T>
__global__ void kernel_normbackwardsummary_place_sum(int len, T* buf1, T* buf2,
                                                     T* out_val, int row,
                                                     T squared_sum_epsilon) {
  CUDA_KERNEL_LOOP(i, len) {
    out_val[3 * i] = row;
    out_val[3 * i + 1] = buf1[i];
    out_val[3 * i + 2] = buf2[i] + row * squared_sum_epsilon;
  }
}

template <typename T>
void nncross_norm_ff(int N, int embed_dim, int ins_num, const T* inputs,
                     T* norm_output, const T* summary, T* mean, T* scale,
                     cudaStream_t stream) {
  int norm_cols = N * (embed_dim * 3 + 1);
  kernel_mean_scale<<<GET_BLOCKS(norm_cols), CUDA_NUM_THREADS, 0, stream>>>(
      norm_cols, summary, mean, scale);
  nncross_normforward_multi<<<GET_BLOCKS(norm_cols * ins_num), CUDA_NUM_THREADS,
                              0, stream>>>(norm_cols * ins_num, N, embed_dim,
                                           ins_num, inputs, norm_output, mean,
                                           scale);
  nncross_normforward_multi_sim<<<GET_BLOCKS(N * ins_num), CUDA_NUM_THREADS, 0,
                                  stream>>>(N * ins_num, N, embed_dim, ins_num,
                                            inputs, norm_output, mean, scale);
}

template <typename T>
void nncross_norm_bp(int N, int embed_dim, int ins_num, const T* inputs,
                     T* norm_output, const T* norm_grad, T* grads, T* sum_grad,
                     T* summary, const T* mean, const T* scale,
                     const T squared_sum_epsilon, cudaStream_t stream,
                     T* sum_grad_buf, int* sum_offset, T* sum_grad_buf2,
                     const framework::ExecutionContext& ctx) {
  auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
  int norm_cols = N * (embed_dim * 3 + 1);
  int intput_cols = N * embed_dim;

  kernel_normbackwardsummary_x0<<<GET_BLOCKS(norm_cols * ins_num),
                                  CUDA_NUM_THREADS, 0, stream>>>(
      norm_cols * ins_num, ins_num, norm_output, sum_grad, mean, scale,
      squared_sum_epsilon);

  size_t temp_storage_bytes;
  cub::DeviceSegmentedReduce::Sum(NULL, temp_storage_bytes, norm_output,
                                  sum_grad_buf, norm_cols, sum_offset,
                                  sum_offset + 1);
  auto temp_cub_buf = memory::Alloc(dev_ctx, temp_storage_bytes);
  T* cub_buf = reinterpret_cast<T*>(temp_cub_buf->ptr());

  cub::DeviceSegmentedReduce::Sum(cub_buf, temp_storage_bytes, norm_output,
                                  sum_grad_buf, norm_cols, sum_offset,
                                  sum_offset + 1, stream);

  kernel_normbackwardsummary_plus_mean<<<GET_BLOCKS(norm_cols * ins_num),
                                         CUDA_NUM_THREADS, 0, stream>>>(
      norm_cols * ins_num, ins_num, norm_output, sum_grad, mean, scale,
      squared_sum_epsilon);
  cub::DeviceSegmentedReduce::Sum(cub_buf, temp_storage_bytes, norm_output,
                                  sum_grad_buf2, norm_cols, sum_offset,
                                  sum_offset + 1, stream);
  kernel_normbackwardsummary_place_sum<<<GET_BLOCKS(norm_cols),
                                         CUDA_NUM_THREADS, 0, stream>>>(
      norm_cols, sum_grad_buf, sum_grad_buf2, sum_grad, ins_num,
      squared_sum_epsilon);

  nncross_normbackpropagate_multi<<<GET_BLOCKS(intput_cols * ins_num * 2),
                                    CUDA_NUM_THREADS, 0, stream>>>(
      intput_cols * ins_num * 2, N, embed_dim, ins_num, inputs, norm_grad,
      grads, mean, scale);
}

template <typename T>
__global__ void KernelUpdateParam(int C, const T* d_summary, T* summary,
                                  const float decay_rate) {
  CUDA_KERNEL_LOOP(i, C) {
    summary[i] = summary[i] * decay_rate + d_summary[i];
  }
}

template <typename T>
void update_norm_param(cudaStream_t stream, int C, const T* d_summary,
                       T* summary, const float decay_rate) {
  KernelUpdateParam<<<GET_BLOCKS(C), CUDA_NUM_THREADS, 0, stream>>>(
      C, d_summary, summary, decay_rate);
}

}  // namespace operators
}  // namespace paddle
