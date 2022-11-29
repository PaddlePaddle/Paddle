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
                    OpType op_type,
                    const half *residual,
                    float alpha) {
  int oh = (ih + pad_h * 2 - kh) / stride_h + 1;
  int ow = (iw + pad_w * 2 - kw) / stride_w + 1;
  struct logical_struct input_shape = {batch, ic, ih, iw};
  struct logical_struct output_shape = {batch, oc, oh, ow};
  struct logical_struct weight_shape = {oc, ic, kh, kw};
  for (int bs_i = 0; bs_i < batch; bs_i++) {
    for (int oc_i = 0; oc_i < oc; oc_i++) {
      for (int oh_i = 0; oh_i < oh; oh_i++) {
        for (int ow_i = 0; ow_i < ow; ow_i++) {
          struct logical_struct output_index = {bs_i, oc_i, oh_i, ow_i};
          float *out_ptr = output + nhwc(output_shape, output_index);
          float sum = 0.f;

          for (int kh_i = 0; kh_i < kh; kh_i++) {
            for (int kw_i = 0; kw_i < kw; kw_i++) {
              int ih_i = oh_i * stride_h - pad_h + kh_i;
              int iw_i = ow_i * stride_w - pad_w + kw_i;
              if (ih_i < 0 || ih_i >= ih) continue;
              if (iw_i < 0 || iw_i >= iw) continue;

              for (int ic_i = 0; ic_i < ic; ic_i++) {
                struct logical_struct input_index = {bs_i, ic_i, ih_i, iw_i};
                struct logical_struct weight_index = {oc_i, ic_i, kh_i, kw_i};
                const half *in_ptr = input + nhwc(input_shape, input_index);
                const half *weight_ptr =
                    weight + nhwc(weight_shape, weight_index);
                sum += __half2float(*in_ptr) * __half2float(*weight_ptr);
              }
            }
          }
          sum += __half2float(*(bias + oc_i));
          float x = sum;
          if (op_type == CONV2D_BIAS) {
            *out_ptr = x;
          } else if (op_type == CONV2D_BIAS_RELU) {
            *out_ptr = x > 0 ? x : 0.f;
          } else if (op_type == CONV2D_BIAS_ADD_RELU) {
            x += __half2float(*(residual + nhwc(output_shape, output_index)));
            *out_ptr = x > 0 ? x : 0.f;
          } else if (op_type == CONV2D_BIAS_LEAKY_RELU) {
            *out_ptr = x > 0 ? x : (x * alpha);
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

float conv2d_diff_cpu(ConvAllParams params, OpType op_type) {
  const half *input = params.input;
  const half *weight = params.weight;
  const half *bias = params.bias;
  half *output = params.output;
  int batch = params.batch;
  int ic = params.ic;
  int ih = params.ih;
  int iw = params.iw;
  int kh = params.kh;
  int kw = params.kw;
  int oc = params.oc;
  int pad_h = params.pad_h;
  int pad_w = params.pad_w;
  int stride_h = params.stride_h;
  int stride_w = params.stride_w;
  const half *residual = params.residual;

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
                 op_type,
                 residual,
                 params.alpha);
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
                                    float alpha,  // for leaky_relu
                                    OpType op_type) {
  int M = batch * oh * ow;
  int N = oc;
  int K = ic * kh * kw;
  int m_i = threadIdx.x + blockIdx.x * blockDim.x;
  int n_i = threadIdx.y + blockIdx.y * blockDim.y;
  if (m_i >= M || n_i >= N) return;

  int batch_i = m_i / (oh * ow);
  int oh_i = (m_i % (oh * ow)) / ow;
  int ow_i = (m_i % (oh * ow)) % ow;
  int oc_i = n_i;

  struct logical_struct weight_shape = {oc, ic, kh, kw};
  struct logical_struct input_shape = {batch, ic, ih, iw};
  int out_offset = m_i * N + n_i;
  float *out_ptr = output + out_offset;
  float sum = 0.f;

  for (int k_i = 0; k_i < K; k_i++) {
    int ic_i = k_i / (kh * kw);
    int kh_i = (k_i % (kh * kw)) / kw;
    int kw_i = (k_i % (kh * kw)) % kw;

    struct logical_struct weight_index = {oc_i, ic_i, kh_i, kw_i};

    int ih_i = oh_i * stride_h - pad_h + kh_i;
    int iw_i = ow_i * stride_w - pad_w + kw_i;

    if (ih_i < 0 || ih_i >= ih) continue;
    if (iw_i < 0 || iw_i >= iw) continue;

    struct logical_struct input_index = {batch_i, ic_i, ih_i, iw_i};
    const half *weight_ptr = weight + gpu_nhwc(weight_shape, weight_index);
    const half *in_ptr = input + gpu_nhwc(input_shape, input_index);
    sum += __half2float(*in_ptr) * __half2float(*weight_ptr);
  }

  sum += __half2float(*(bias + oc_i));
  float x = sum;

  switch (op_type) {
    case CONV2D_BIAS:
      *out_ptr = x;
      break;
    case CONV2D_BIAS_RELU:
      *out_ptr = x > 0 ? x : 0;
      break;
    case CONV2D_BIAS_SILU:
      *out_ptr = x * (1.f / (1 + exp(-x)));
      break;
    case CONV2D_BIAS_ADD_RELU:
      x += __half2float(*(residual + out_offset));
      *out_ptr = x > 0 ? x : 0;
      break;
    case CONV2D_BIAS_LEAKY_RELU:
      *out_ptr = x > 0 ? x : (x * alpha);
      break;
    default:
      break;
  }
}

float conv2d_diff_gpu(ConvAllParams params, OpType op_type) {
  const half *input = params.input;
  const half *weight = params.weight;
  const half *bias = params.bias;
  half *output = params.output;
  int batch = params.batch;
  int ic = params.ic;
  int ih = params.ih;
  int iw = params.iw;
  int kh = params.kh;
  int kw = params.kw;
  int oc = params.oc;
  int pad_h = params.pad_h;
  int pad_w = params.pad_w;
  int stride_h = params.stride_h;
  int stride_w = params.stride_w;
  const half *residual = params.residual;

  int oh = (ih + pad_h * 2 - kh) / stride_h + 1;
  int ow = (iw + pad_w * 2 - kw) / stride_w + 1;
  int M = batch * oh * ow;
  int N = oc;

  constexpr int blockM = 16;
  constexpr int blockN = 16;
  uint3 grid = {(M + blockM - 1) / blockM, (N + blockN - 1) / blockN, 1};
  uint3 block = {blockM, blockN, 1};

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
                                       params.alpha,
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
