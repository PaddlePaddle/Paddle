// Copyright (c) 2024 CINN Authors. All Rights Reserved.
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

#include <mutex>  // NOLINT
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

#ifdef CINN_WITH_CNNL
#include <cnnl.h>
#include <cnrt.h>
#endif

namespace cinn {
namespace runtime {
namespace sycl {

/**
 * Call a SYCL compiled kernel.
 *
 * @param kernel_fn the func pointer.
 * @param args an array of cinn_pod_value_ts(consists of scalars and buffers).
 */
void cinn_call_sycl_kernel(void* kernel_fn,
                           void* v_args,
                           int num_args,
                           int grid_x,
                           int grid_y,
                           int grid_z,
                           int block_x,
                           int block_y,
                           int block_z);

void cinn_call_sycl_memset(void* v_args, int num_args, int value, size_t count);

void cinn_call_sycl_memcpy(void* v_args, int num_args, size_t count);

#ifdef CINN_WITH_CNNL
#define CNRT_CALL(func)                                  \
  {                                                      \
    auto status = func;                                  \
    if (status != cnrtSuccess) {                         \
      std::stringstream ss;                              \
      ss << "CNRT Error : " << cnrtGetErrorStr(status);  \
      PADDLE_THROW(::common::errors::Fatal(              \
          ("SYCL Error in Paddle CINN: %s", ss.str()))); \
    }                                                    \
  }

#define CNNL_CALL(func)                                    \
  {                                                        \
    auto status = func;                                    \
    if (status != CNNL_STATUS_SUCCESS) {                   \
      std::stringstream ss;                                \
      ss << "CNNL Error : " << cnnlGetErrorString(status); \
      PADDLE_THROW(::common::errors::Fatal(                \
          ("SYCL Error in Paddle CINN: %s", ss.str())));   \
    }                                                      \
  }

void cinn_call_cnnl_gaussian_random(
    void* v_args, int num_args, float mean, float std, int seed);

void cinn_call_cnnl_uniform_random(
    void* v_args, int num_args, float min, float max, int seed);

void cinn_call_cnnl_randint(void* v_args, int num_args, int seed);

void cinn_call_cnnl_matmul(void* v_args,
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
                           int b4);

void cinn_call_cnnl_conv2d_forward(void* v_args,
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
                                   int output_w);

void cinn_call_cnnl_conv2d_backward_data(void* v_args,
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
                                         int output_w);

void cinn_call_cnnl_conv2d_backward_filter(void* v_args,
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
                                           int output_w);

void cinn_call_cnnl_pool2d_forward(void* v_args,
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
                                   int output_w);

void cinn_call_cnnl_pool2d_backward(void* v_args,
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
                                    int output_w);
#endif  // CINN_WITH_CNNL

}  // namespace sycl
}  // namespace runtime
}  // namespace cinn
