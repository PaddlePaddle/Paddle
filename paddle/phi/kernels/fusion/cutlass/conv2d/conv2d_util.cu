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

namespace phi {
namespace fusion {
namespace cutlass_internal {
struct logical_coord {
  int n;
  int c;
  int h;
  int w;
};

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

__device__ int gpu_nhwc(struct logical_coord shape,
                        struct logical_coord index) {
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
                                    int dilation_h,
                                    int dilation_w,
                                    int oh,
                                    int ow,
                                    int groups,
                                    const half *residual,
                                    float alpha,  // for leaky_relu
                                    OpType op_type) {
  int M = batch * oh * ow;
  int N = oc;
  int kc = ic / groups;
  int K = kc * kh * kw;
  int m_i = threadIdx.x + blockIdx.x * blockDim.x;
  int n_i = threadIdx.y + blockIdx.y * blockDim.y;
  if (m_i >= M || n_i >= N) return;

  int batch_i = m_i / (oh * ow);
  int oh_i = (m_i % (oh * ow)) / ow;
  int ow_i = (m_i % (oh * ow)) % ow;
  int oc_i = n_i;
  int groups_i = (oc_i / (oc / groups));

  struct logical_coord weight_shape = {oc, kc, kh, kw};
  struct logical_coord input_shape = {batch, ic, ih, iw};
  int out_offset = m_i * N + n_i;
  float *out_ptr = output + out_offset;
  float sum = 0.f;

  for (int k_i = 0; k_i < K; k_i++) {
    int ic_i = k_i / (kh * kw) + groups_i * kc;
    int kh_i = (k_i % (kh * kw)) / kw;
    int kw_i = (k_i % (kh * kw)) % kw;

    struct logical_coord weight_index = {oc_i, k_i / (kh * kw), kh_i, kw_i};

    int ih_i = oh_i * stride_h - pad_h + kh_i * dilation_h;
    int iw_i = ow_i * stride_w - pad_w + kw_i * dilation_w;

    if (ih_i < 0 || ih_i >= ih) continue;
    if (iw_i < 0 || iw_i >= iw) continue;

    struct logical_coord input_index = {batch_i, ic_i, ih_i, iw_i};
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

float conv2d_diff_gpu(const ConvAllParams &params, OpType op_type) {
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
  int pad_h = params.pad_h0;
  int pad_w = params.pad_w0;
  int stride_h = params.stride_h;
  int stride_w = params.stride_w;
  int dilation_h = params.dilation_h;
  int dilation_w = params.dilation_w;
  const half *residual = params.residual;
  int groups = params.groups;

  int oh = params.oh;
  int ow = params.ow;
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
                                       dilation_h,
                                       dilation_w,
                                       oh,
                                       ow,
                                       groups,
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

std::string OpType2String(OpType op_type) {
  switch (op_type) {
    case CONV2D_BIAS:
      return "conv2d_bias";
      break;
    case CONV2D_BIAS_RELU:
      return "conv2d_bias_relu";
      break;
    case CONV2D_BIAS_SILU:
      return "conv2d_bias_silu";
      break;
    case CONV2D_BIAS_ADD_RELU:
      return "conv2d_bias_add_relu";
      break;
    case CONV2D_BIAS_LEAKY_RELU:
      return "conv2d_bias_leaky_relu";
    default:
      break;
  }
  return "unnamed_op";
}

int ProfileToGetBestConfig(
    const std::vector<std::function<cutlass::Status(ConvAllParams)>> &all_func,
    const ConvAllParams &params,
    OpType op_type) {
  constexpr int WARMUP = 10;
  constexpr int REPEAT = 100;
  float min_time = 100000.f;
  int min_time_index = -1;
  for (int i = 0; i < all_func.size(); i++) {
    cutlass::Status status;
    auto func = all_func[i];
    // When func has large diff, we will make it nullptr.
    if (!func) continue;

    for (int ii = 0; ii < WARMUP; ii++) {
      status = func(params);
    }

    cudaEvent_t beg, end;
    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventCreate(&beg));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventCreate(&end));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventRecord(beg));
    for (int ii = 0; ii < REPEAT; ii++) {
      status = func(params);
    }

    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventRecord(end));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventSynchronize(end));
    float elapsed_time;
    PADDLE_ENFORCE_GPU_SUCCESS(cudaEventElapsedTime(&elapsed_time, beg, end));
    if (elapsed_time < min_time && status == cutlass::Status::kSuccess) {
      min_time = elapsed_time;
      min_time_index = i;
      // debug code
      VLOG(3) << OpType2String(op_type) << ": tactic " << i << " has max diff "
              << conv2d_diff_gpu(params, op_type) << " compared with baseline,"
              << "cost_time: " << elapsed_time << "ms.";
    }
  }

  if (min_time_index < 0) {
    PADDLE_THROW(
        phi::errors::NotFound("Can't find any cutlass config for this %s op.",
                              OpType2String(op_type).c_str()));
  }
  return min_time_index;
}

}  // namespace cutlass_internal
}  // namespace fusion
}  // namespace phi
