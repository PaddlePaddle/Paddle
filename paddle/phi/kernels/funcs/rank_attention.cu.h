// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
#include "paddle/common/dim.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

const int CUDA_NUM_THREADS = 1024;
static inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

template <typename T>
__global__ void expand_input_by_rank_kernel(const T* input,
                                            int input_row,
                                            int input_col,
                                            T* output,
                                            int output_row,
                                            int output_col,
                                            const int* rank_offset,
                                            int rank_offset_row,
                                            int rank_offset_col,
                                            T* ins_rank,
                                            int max_rank) {
  CUDA_KERNEL_LOOP(idx, output_row * output_col) {
    int output_col_idx = idx % output_col;
    int output_row_idx = idx / output_col;
    int k = output_col_idx / input_col;

    int faster = rank_offset[output_row_idx * rank_offset_col + 2 * k + 1] - 1;
    if (output_col_idx == 0) {
      ins_rank[output_row_idx] = rank_offset[output_row_idx * rank_offset_col];
    }

    if (rank_offset[output_row_idx * rank_offset_col] - 1 < 0 || faster < 0) {
      continue;
    }

    int rank_input_col_idx = output_col_idx % input_col;
    int index = rank_offset[output_row_idx * rank_offset_col + 2 * k + 2];
    output[idx] = input[rank_input_col_idx + index * input_col];
  }
}

template <typename T>
void expand_rank_attention_input(gpuStream_t stream,
                                 const T* input,
                                 int input_row,
                                 int input_col,
                                 T* output,
                                 int output_row,
                                 int output_col,
                                 const int* rank_offset,
                                 int rank_offset_row,
                                 int rank_offset_col,
                                 T* ins_rank,
                                 int max_rank) {
  expand_input_by_rank_kernel<<<GET_BLOCKS(output_row * output_col),
                                CUDA_NUM_THREADS,
                                0,
                                stream>>>(input,
                                          input_row,
                                          input_col,
                                          output,
                                          output_row,
                                          output_col,
                                          rank_offset,
                                          rank_offset_row,
                                          rank_offset_col,
                                          ins_rank,
                                          max_rank);
}

template <typename T>
__global__ void expand_rank_attention_param_kernel(const T* input,
                                                   int input_row,
                                                   int input_col,
                                                   const int* rank_offset,
                                                   int rank_offset_row,
                                                   int rank_offset_col,
                                                   const T* param,
                                                   int param_row,
                                                   int param_col,
                                                   T* output_param,
                                                   int output_param_row,
                                                   int output_param_col,
                                                   int max_rank) {
  CUDA_KERNEL_LOOP(idx, output_param_row * output_param_col) {
    int output_col_idx = idx % output_param_col;
    int output_row_idx = idx / output_param_col;

    int block_matrix_row = max_rank * input_col;
    int ins_idx = output_row_idx / block_matrix_row;
    int start_offset = output_row_idx % block_matrix_row;

    int k = start_offset / input_col;
    int k_offset = start_offset % input_col;

    int lower = rank_offset[ins_idx * rank_offset_col] - 1;
    int faster = rank_offset[2 * k + 1 + rank_offset_col * ins_idx] - 1;

    if (lower < 0 || faster < 0) {
      continue;
    }
    int start = lower * max_rank + faster;
    int ori_idx =
        start * param_col * input_col + k_offset * param_col + output_col_idx;
    output_param[idx] = param[ori_idx];
  }
}

template <typename T>
void expand_rank_attention_param(gpuStream_t stream,
                                 const T* input,
                                 int input_row,
                                 int input_col,
                                 const int* rank_offset,
                                 int rank_offset_row,
                                 int rank_offset_col,
                                 const T* param,
                                 int param_row,
                                 int param_col,
                                 T* output_param,
                                 int output_param_row,
                                 int output_param_col,
                                 int max_rank) {
  expand_rank_attention_param_kernel<<<GET_BLOCKS(output_param_row *
                                                  output_param_col),
                                       CUDA_NUM_THREADS,
                                       0,
                                       stream>>>(input,
                                                 input_row,
                                                 input_col,
                                                 rank_offset,
                                                 rank_offset_row,
                                                 rank_offset_col,
                                                 param,
                                                 param_row,
                                                 param_col,
                                                 output_param,
                                                 output_param_row,
                                                 output_param_col,
                                                 max_rank);
}

template <typename T>
__global__ void merge_param_gradient_kernel(T* expanded_grad,
                                            int expanded_grad_row,
                                            int expanded_grad_col,
                                            T* param_grad,
                                            int param_grad_row,
                                            int param_grad_col,
                                            const T* ins_rank,
                                            int ins_num,
                                            int max_rank,
                                            int input_col) {
  CUDA_KERNEL_LOOP(tid, param_grad_row * param_grad_col) {
    int param_col_idx = tid % param_grad_col;
    int param_row_idx = tid / param_grad_col;

    int block_matrix_row = max_rank * input_col;
    int rank_idx = param_row_idx / block_matrix_row;
    int rank_offset = param_row_idx % block_matrix_row;

    T tmp = 0;
    for (int i = 0; i < ins_num; ++i) {
      if (ins_rank[i] == rank_idx + 1) {
        int row = i * block_matrix_row + rank_offset;
        tmp += expanded_grad[row * expanded_grad_col + param_col_idx];
      }
    }
    param_grad[tid] = tmp;
  }
}

template <typename T>
void merge_rank_attention_param_grad(gpuStream_t stream,
                                     T* expanded_grad,
                                     int expanded_grad_row,
                                     int expanded_grad_col,
                                     T* param_grad,
                                     int param_grad_row,
                                     int param_grad_col,
                                     const T* ins_rank,
                                     int ins_num,
                                     int max_rank,
                                     int input_col) {
  merge_param_gradient_kernel<<<GET_BLOCKS(param_grad_row * param_grad_col),
                                CUDA_NUM_THREADS,
                                0,
                                stream>>>(expanded_grad,
                                          expanded_grad_row,
                                          expanded_grad_col,
                                          param_grad,
                                          param_grad_row,
                                          param_grad_col,
                                          ins_rank,
                                          ins_num,
                                          max_rank,
                                          input_col);
}

}  // namespace phi
