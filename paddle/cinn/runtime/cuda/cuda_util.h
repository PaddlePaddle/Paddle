// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include <absl/container/flat_hash_map.h>
#include <cuda_runtime.h>

#include <string>
#include <vector>

#include "paddle/cinn/common/type.h"
#include "paddle/cinn/runtime/cinn_runtime.h"

namespace cinn {
namespace runtime {
namespace cuda {

const int kCUDAMaxCards{8};

void cinn_gpu_cublas_mul(const std::vector<int>& attrs,
                         cinn_buffer_t* input1,
                         cinn_buffer_t* input2,
                         cinn_buffer_t* output,
                         cudaStream_t stream = nullptr);

void cinn_gpu_cublas_gemm(const std::vector<int>& attrs,
                          cinn_buffer_t* lhs,
                          cinn_buffer_t* rhs,
                          cinn_buffer_t* bias,
                          cinn_buffer_t* output,
                          cudaStream_t stream = nullptr);

void cinn_call_gaussian_random(void* v_args,
                               int num_args,
                               float mean,
                               float std,
                               int seed,
                               void* stream = nullptr);

void cinn_call_uniform_random(void* v_args,
                              int num_args,
                              float min,
                              float max,
                              int seed,
                              void* stream = nullptr);

void cinn_call_randint(void* v_args,
                       int num_args,
                       int seed,
                       void* stream = nullptr);

void cinn_call_cholesky_nvgpu(void* v_args,
                              int num_args,
                              int batch_size,
                              int m,
                              bool upper,
                              void* stream = nullptr);

void cinn_call_triangular_solve_nvgpu(void* v_args,
                                      int num_args,
                                      int batch_size,
                                      int m,
                                      int k,
                                      bool left_side,
                                      bool upper,
                                      bool transpose_a,
                                      bool unit_diagonal,
                                      void* stream = nullptr);

void cinn_call_cuda_memset(void* v_args,
                           int num_args,
                           int value,
                           size_t count,
                           void* stream = nullptr);
void cinn_call_cuda_memcpy(void* v_args,
                           int num_args,
                           size_t count,
                           void* stream = nullptr);

int64_t cinn_get_value_in_cuda_kernel_args(void* v_args, int idx);
void* cinn_get_item_in_cuda_kernel_args(void* v_args, int idx);

void infer_shape_set_value(int row, int col, int64_t value, int64_t** v);

/**
 * Call a CUDA compiled kernel.
 *
 * @param kernel_fn the compiled PTX kernel.
 * @param args an array of cinn_pod_value_ts(consists of scalars and buffers).
 */
void cinn_call_cuda_kernel(void* kernel_fn,
                           void* v_args,
                           int num_args,
                           int grid_x,
                           int grid_y,
                           int grid_z,
                           int block_x,
                           int block_y,
                           int block_z,
                           int shared_memory_bytes,
                           void* stream);

void cinn_call_cublas(void* v_args,
                      int num_args,
                      bool trans_a,
                      bool trans_b,
                      bool trans_o,
                      float alpha,
                      float beta,
                      int a1,
                      int a2,
                      int a3,
                      int a4,
                      int b1,
                      int b2,
                      int b3,
                      int b4,
                      void* stream);

void cinn_call_batched_cublas(void* v_args,
                              int num_args,
                              int opside,
                              bool trans_a,
                              bool trans_b,
                              bool trans_o,
                              float alpha,
                              float beta,
                              int a1,
                              int a2,
                              int a3,
                              int a4,
                              int b1,
                              int b2,
                              int b3,
                              int b4,
                              void* stream);

#ifdef CINN_WITH_CUDNN
void cinn_gpu_cudnn_conv2d(
    const absl::flat_hash_map<std::string, int>& attr,
    cinn_buffer_t* x,
    cinn_buffer_t* w,
    cinn_buffer_t* y,
    cudaStream_t stream = nullptr,
    cinn::common::Layout target = cinn::common::Layout::kNCHW);

void cinn_gpu_cudnn_conv2d_backward_data(
    const absl::flat_hash_map<std::string, int>& attr,
    cinn_buffer_t* w,
    cinn_buffer_t* dy,
    cinn_buffer_t* dx,
    cudaStream_t stream = nullptr);

void cinn_gpu_cudnn_conv2d_backward_filter(
    const absl::flat_hash_map<std::string, int>& attr,
    cinn_buffer_t* x,
    cinn_buffer_t* dy,
    cinn_buffer_t* dw,
    cudaStream_t stream = nullptr);

void cinn_gpu_cudnn_pool2d(const std::vector<int>& attrs,
                           const std::vector<std::string>& str_attrs,
                           cinn_buffer_t* input,
                           cinn_buffer_t* output,
                           cudaStream_t stream = nullptr);

void cinn_gpu_cudnn_softmax(const std::vector<int>& attrs,
                            cinn_buffer_t* input,
                            cinn_buffer_t* output,
                            cudaStream_t stream = nullptr);

void cinn_call_cudnn_conv2d_forward(void* v_args,
                                    int num_args,
                                    int format,
                                    float alpha,
                                    float beta,
                                    int input_n,
                                    int input_c,
                                    int input_h,
                                    int input_w,
                                    int filter_n,
                                    int filter_c,
                                    int filter_h,
                                    int filter_w,
                                    int pad_h,
                                    int pad_w,
                                    int stride_h,
                                    int stride_w,
                                    int dilation_h,
                                    int dilation_w,
                                    int groups,
                                    int output_n,
                                    int output_c,
                                    int output_h,
                                    int output_w,
                                    void* stream);

void cinn_call_cudnn_conv2d_backward_data(void* v_args,
                                          int num_args,
                                          int format,
                                          float alpha,
                                          float beta,
                                          int input_n,
                                          int input_c,
                                          int input_h,
                                          int input_w,
                                          int filter_n,
                                          int filter_c,
                                          int filter_h,
                                          int filter_w,
                                          int pad_h,
                                          int pad_w,
                                          int stride_h,
                                          int stride_w,
                                          int dilation_h,
                                          int dilation_w,
                                          int groups,
                                          int output_n,
                                          int output_c,
                                          int output_h,
                                          int output_w,
                                          void* stream);

void cinn_call_cudnn_conv2d_backward_filter(void* v_args,
                                            int num_args,
                                            int format,
                                            float alpha,
                                            float beta,
                                            int input_n,
                                            int input_c,
                                            int input_h,
                                            int input_w,
                                            int filter_n,
                                            int filter_c,
                                            int filter_h,
                                            int filter_w,
                                            int pad_h,
                                            int pad_w,
                                            int stride_h,
                                            int stride_w,
                                            int dilation_h,
                                            int dilation_w,
                                            int groups,
                                            int output_n,
                                            int output_c,
                                            int output_h,
                                            int output_w,
                                            void* stream);

void cinn_call_cudnn_pool2d_forward(void* v_args,
                                    int num_args,
                                    int mode,
                                    int format,
                                    float alpha,
                                    float beta,
                                    int input_n,
                                    int input_c,
                                    int input_h,
                                    int input_w,
                                    int kernel_h,
                                    int kernel_w,
                                    int pad_h,
                                    int pad_w,
                                    int stride_h,
                                    int stride_w,
                                    int output_n,
                                    int output_c,
                                    int output_h,
                                    int output_w,
                                    void* stream);

void cinn_call_cudnn_pool2d_backward(void* v_args,
                                     int num_args,
                                     int mode,
                                     int format,
                                     float alpha,
                                     float beta,
                                     int input_n,
                                     int input_c,
                                     int input_h,
                                     int input_w,
                                     int kernel_h,
                                     int kernel_w,
                                     int pad_h,
                                     int pad_w,
                                     int stride_h,
                                     int stride_w,
                                     int output_n,
                                     int output_c,
                                     int output_h,
                                     int output_w,
                                     void* stream);

void cinn_call_cudnn_softmax_forward(void* v_args,
                                     int num_args,
                                     int mode,
                                     int format,
                                     float alpha,
                                     float beta,
                                     int input_n,
                                     int input_c,
                                     int input_h,
                                     int input_w,
                                     int output_n,
                                     int output_c,
                                     int output_h,
                                     int output_w,
                                     void* stream);

void cinn_call_cudnn_softmax_backward(void* v_args,
                                      int num_args,
                                      int mode,
                                      int format,
                                      float alpha,
                                      float beta,
                                      int input_n,
                                      int input_c,
                                      int input_h,
                                      int input_w,
                                      int output_n,
                                      int output_c,
                                      int output_h,
                                      int output_w,
                                      void* stream);

#endif  // CINN_WITH_CUDNN
}  // namespace cuda
}  // namespace runtime
}  // namespace cinn
