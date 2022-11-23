// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/fusion/cutlass/conv2d/conv2d_util.h"
#include <iostream>
#include <vector>

namespace phi {
namespace fusion {

struct logical_struct {
  int n;
  int c;
  int h;
  int w;
};

int nchw(struct logical_struct shape, struct logical_struct index) {
  return index.n * shape.c * shape.h * shape.w + index.c * shape.h * shape.w +
         index.h * shape.w + index.w;
}

int nhwc(struct logical_struct shape, struct logical_struct index) {
  return index.n * shape.h * shape.w * shape.c + index.h * shape.w * shape.c +
         index.w * shape.c + index.c;
}

void naive_conv_cpu(const half *input,
                    const half *weight,
                    const half *bias,
                    float *output,
                    int batch,
                    int ic,
                    int ih,
                    int iw,
                    int kh,
                    int kw,
                    int oc,
                    int pad_h,
                    int pad_w,
                    int stride_h,
                    int stride_w,
                    const half *residual,
                    std::string op_name) {
  int oh = (ih + pad_h * 2 - kh) / stride_h + 1;
  int ow = (iw + pad_w * 2 - kw) / stride_w + 1;
  struct logical_struct input_shape {
    batch, ic, ih, iw
  };
  struct logical_struct output_shape {
    batch, oc, oh, ow
  };
  struct logical_struct weight_shape {
    oc, ic, kh, kw
  };
  for (int bs_i = 0; bs_i < batch; bs_i++) {
    for (int oc_i = 0; oc_i < oc; oc_i++) {
      for (int oh_i = 0; oh_i < oh; oh_i++) {
        for (int ow_i = 0; ow_i < ow; ow_i++) {
          struct logical_struct output_index {
            bs_i, oc_i, oh_i, ow_i
          };
          float *out_ptr = output + nhwc(output_shape, output_index);
          float sum = 0.f;

          for (int kh_i = 0; kh_i < kh; kh_i++) {
            for (int kw_i = 0; kw_i < kw; kw_i++) {
              int ih_i = oh_i * stride_h - pad_h + kh_i;
              int iw_i = ow_i * stride_w - pad_w + kw_i;
              if (ih_i < 0 || ih_i >= ih) continue;
              if (iw_i < 0 || iw_i >= iw) continue;

              for (int ic_i = 0; ic_i < ic; ic_i++) {
                struct logical_struct input_index {
                  bs_i, ic_i, ih_i, iw_i
                };
                struct logical_struct weight_index {
                  oc_i, ic_i, kh_i, kw_i
                };
                const half *in_ptr = input + nhwc(input_shape, input_index);
                const half *weight_ptr =
                    weight + nhwc(weight_shape, weight_index);
                sum += __half2float(*in_ptr) * __half2float(*weight_ptr);
              }
            }
          }
          sum += __half2float(*(bias + oc_i));
          if (op_name == "conv2d_bias") {
            *out_ptr = sum;
          } else if (op_name == "conv2d_bias_relu") {
            *out_ptr = sum > 0 ? sum : 0.f;
          } else if (op_name == "conv2d_bias_add_relu") {
            sum += __half2float(*(residual + nhwc(output_shape, output_index)));
            *out_ptr = sum > 0 ? sum : 0.f;
          }
        }
      }
    }
  }
}
float diff(const half *c, const float *c_baseline, int n) {
  float max_diff = -1.;
  for (int i = 0; i < n; i++) {
    float c_value = __half2float(c[i]);
    if (std::abs(c_baseline[i] - c_value) > max_diff) {
      max_diff = std::abs(c_baseline[i] - c_value);
    }
  }
  return max_diff;
}

float conv2d_diff_cpu(COMMON_CONV_PARAMS,
                      const half *residual,
                      std::string op_name) {
  // debug code
  half *cpu_input, *cpu_weight, *cpu_bias;
  float *cpu_output;
  half *output_from_cutlass;

  int input_size = batch * ic * ih * iw;
  int weight_size = oc * ic * kh * kw;
  cpu_input = reinterpret_cast<half *>(malloc(sizeof(half) * input_size));
  cpu_weight = reinterpret_cast<half *>(malloc(sizeof(half) * weight_size));
  cpu_bias = reinterpret_cast<half *>(malloc(sizeof(half) * oc));
  cudaMemcpy(
      cpu_input, input, input_size * sizeof(half), cudaMemcpyDeviceToHost);
  cudaMemcpy(
      cpu_weight, weight, weight_size * sizeof(half), cudaMemcpyDeviceToHost);
  cudaMemcpy(cpu_bias, bias, oc * sizeof(half), cudaMemcpyDeviceToHost);

  int oh = (ih + pad_h * 2 - kh) / stride_h + 1;
  int ow = (iw + pad_w * 2 - kw) / stride_w + 1;

  int output_size = batch * oc * oh * ow;
  cpu_output = reinterpret_cast<float *>(malloc(sizeof(float) * output_size));
  output_from_cutlass =
      reinterpret_cast<half *>(malloc(sizeof(half) * output_size));
  cudaMemcpy(output_from_cutlass,
             output,
             output_size * sizeof(half),
             cudaMemcpyDeviceToHost);

  naive_conv_cpu(cpu_input,
                 cpu_weight,
                 cpu_bias,
                 cpu_output,
                 batch,
                 ic,
                 ih,
                 iw,
                 kh,
                 kw,
                 oc,
                 pad_h,
                 pad_w,
                 stride_h,
                 stride_w,
                 residual,
                 op_name);
  float max_diff = diff(output_from_cutlass, cpu_output, output_size);
  free(cpu_output);
  free(cpu_input);
  free(cpu_weight);
  free(cpu_bias);
  free(output_from_cutlass);
  return max_diff;
}

__device__ int gpu_nhwc(struct logical_struct shape,
                        struct logical_struct index) {
  return index.n * shape.h * shape.w * shape.c + index.h * shape.w * shape.c +
         index.w * shape.c + index.c;
}

__global__ void naive_conv2d_kernel(const half *input,
                                    const half *weight,
                                    const half *bias,
                                    float *output,
                                    int batch,
                                    int ic,
                                    int ih,
                                    int iw,
                                    int kh,
                                    int kw,
                                    int oc,
                                    int pad_h,
                                    int pad_w,
                                    int stride_h,
                                    int stride_w,
                                    int oh,
                                    int ow,
                                    const half *residual,
                                    int op_type) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= batch * oc * oh * ow) return;
  int batch_i = idx / (oc * oh * ow);
  int remain = idx % (oc * oh * ow);
  int oc_i = remain / (oh * ow);
  remain = idx % (oh * ow);
  int oh_i = remain / ow;
  int ow_i = remain % ow;
  struct logical_struct input_shape {
    batch, ic, ih, iw
  };
  struct logical_struct output_shape = {batch, oc, oh, ow};
  struct logical_struct output_index = {batch_i, oc_i, oh_i, ow_i};
  struct logical_struct weight_shape = {oc, ic, kh, kw};
  float *out_ptr = output + gpu_nhwc(output_shape, output_index);
  float sum = 0.f;

  for (int kh_i = 0; kh_i < kh; kh_i++) {
    for (int kw_i = 0; kw_i < kw; kw_i++) {
      int ih_i = oh_i * stride_h - pad_h + kh_i;
      int iw_i = ow_i * stride_w - pad_w + kw_i;
      if (ih_i < 0 || ih_i >= ih) continue;
      if (iw_i < 0 || iw_i >= iw) continue;

      for (int ic_i = 0; ic_i < ic; ic_i++) {
        struct logical_struct input_index {
          batch_i, ic_i, ih_i, iw_i
        };
        struct logical_struct weight_index {
          oc_i, ic_i, kh_i, kw_i
        };
        const half *in_ptr = input + gpu_nhwc(input_shape, input_index);
        const half *weight_ptr = weight + gpu_nhwc(weight_shape, weight_index);
        sum += __half2float(*in_ptr) * __half2float(*weight_ptr);
      }
    }
  }
  sum += __half2float(*(bias + oc_i));
  float x = sum;
  if (op_type == 0) {
    // conv2d_bias
    *out_ptr = x;
  } else if (op_type == 1) {
    // conv2d_bias_relu
    *out_ptr = x > 0 ? x : 0;
  } else if (op_type == 2) {
    // conv2d_bias_silu
    *out_ptr = x * (1.f / (1 + exp(-x)));
  } else if (op_type == 3) {
    // conv2d_bias_add_relu
    x += __half2float(*(residual + gpu_nhwc(output_shape, output_index)));
    *out_ptr = x > 0 ? x : 0;
  }
}

float conv2d_diff_gpu(COMMON_CONV_PARAMS,
                      const half *residual,
                      std::string op_name) {
  int op_type = 0;
  if (op_name == "conv2d_bias") {
    op_type = 0;
  } else if (op_name == "conv2d_bias_relu") {
    op_type = 1;
  } else if (op_name == "conv2d_bias_silu") {
    op_type = 2;
  } else if (op_name == "conv2d_bias_add_relu") {
    op_type = 3;
  }

  int oh = (ih + pad_h * 2 - kh) / stride_h + 1;
  int ow = (iw + pad_w * 2 - kw) / stride_w + 1;

  int onhwc = batch * oh * ow * oc;
  uint3 grid = {onhwc / 256 + 1, 1, 1};
  uint3 block = {256, 1, 1};

  int output_size = batch * oc * oh * ow;
  half *output_from_cutlass =
      reinterpret_cast<half *>(malloc(sizeof(half) * output_size));
  cudaMemcpy(output_from_cutlass,
             output,
             output_size * sizeof(half),
             cudaMemcpyDeviceToHost);

  float *gpu_output;
  cudaMalloc(&gpu_output, output_size * sizeof(float));
  naive_conv2d_kernel<<<grid, block>>>(input,
                                       weight,
                                       bias,
                                       gpu_output,
                                       batch,
                                       ic,
                                       ih,
                                       iw,
                                       kh,
                                       kw,
                                       oc,
                                       pad_h,
                                       pad_w,
                                       stride_h,
                                       stride_w,
                                       oh,
                                       ow,
                                       residual,
                                       op_type);
  float *output_from_gpu =
      reinterpret_cast<float *>(malloc(sizeof(float) * output_size));
  cudaMemcpy(output_from_gpu,
             gpu_output,
             output_size * sizeof(float),
             cudaMemcpyDeviceToHost);
  float max_diff = diff(output_from_cutlass, output_from_gpu, output_size);

  free(output_from_cutlass);
  free(output_from_gpu);
  cudaFree(gpu_output);
  return max_diff;
}
}  // namespace fusion
}  // namespace phi
