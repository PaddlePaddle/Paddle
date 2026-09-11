// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

namespace phi {
namespace funcs {
namespace bilinear_compat {

// Match the separate FP32 operations in torch 2.12's _upsample_linear.
// In particular, a fused multiply-add changes both coordinates and values.
__device__ __forceinline__ float Coordinate(int64_t index, float scale) {
  return fmaxf(
      __fsub_rn(__fmul_rn(__fadd_rn(static_cast<float>(index), 0.5f), scale),
                0.5f),
      0.0f);
}

__device__ __forceinline__ float Weight(int64_t index, float scale) {
  float coordinate = Coordinate(index, scale);
  auto lower = static_cast<int64_t>(coordinate);
  return fminf(__fsub_rn(coordinate, static_cast<float>(lower)), 1.0f);
}

__device__ __forceinline__ float Lerp(float left, float right, float weight) {
  return __fadd_rn(left, __fmul_rn(__fsub_rn(right, left), weight));
}

static __global__ void Forward(const float* input,
                               float* output,
                               int64_t numel,
                               int64_t in_h,
                               int64_t in_w,
                               int64_t out_h,
                               int64_t out_w,
                               float scale_h,
                               float scale_w) {
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < numel;
       index += stride) {
    int64_t x = index % out_w;
    int64_t y = index / out_w % out_h;
    int64_t plane = index / (out_h * out_w);
    float fy = Coordinate(y, scale_h);
    float fx = Coordinate(x, scale_w);
    auto y0 = static_cast<int64_t>(fy);
    auto x0 = static_cast<int64_t>(fx);
    int64_t y1 = y0 + 1 < in_h ? y0 + 1 : in_h - 1;
    int64_t x1 = x0 + 1 < in_w ? x0 + 1 : in_w - 1;
    float wy = fminf(__fsub_rn(fy, static_cast<float>(y0)), 1.0f);
    float wx = fminf(__fsub_rn(fx, static_cast<float>(x0)), 1.0f);
    const float* values = input + plane * in_h * in_w;
    float top = Lerp(values[y0 * in_w + x0], values[y0 * in_w + x1], wx);
    float bottom = Lerp(values[y1 * in_w + x0], values[y1 * in_w + x1], wx);
    output[index] = Lerp(top, bottom, wy);
  }
}

// Each corner's index map is monotone. Its inverse is an interval, so we
// can reproduce stable index sorting without allocating or sorting indices.
__device__ __forceinline__ int64_t LowerBound(int64_t target,
                                              int corner,
                                              int64_t in_size,
                                              int64_t out_size,
                                              float scale) {
  int64_t first = 0;
  int64_t last = out_size;
  while (first < last) {
    int64_t mid = first + (last - first) / 2;
    int64_t source = static_cast<int64_t>(Coordinate(mid, scale)) + corner;
    source = source < in_size ? source : in_size - 1;
    if (source < target) {
      first = mid + 1;
    } else {
      last = mid;
    }
  }
  return first;
}

__device__ __forceinline__ float CornerGrad(float grad,
                                            float wy,
                                            float wx,
                                            int corner) {
  float bottom = __fmul_rn(grad, wy);
  float row = corner >= 2 ? bottom : __fsub_rn(grad, bottom);
  float right = __fmul_rn(row, wx);
  return corner % 2 ? right : __fsub_rn(row, right);
}

template <bool WarpReduce>
__device__ __forceinline__ float GatherGrad(const float* grad,
                                            int64_t y,
                                            int64_t x,
                                            int64_t in_h,
                                            int64_t in_w,
                                            int64_t out_h,
                                            int64_t out_w,
                                            float scale_h,
                                            float scale_w,
                                            int corner) {
  int64_t y_begin = LowerBound(y, corner / 2, in_h, out_h, scale_h);
  int64_t y_end = LowerBound(y + 1, corner / 2, in_h, out_h, scale_h);
  int64_t x_begin = LowerBound(x, corner % 2, in_w, out_w, scale_w);
  int64_t x_end = LowerBound(x + 1, corner % 2, in_w, out_w, scale_w);
  int64_t width = x_end - x_begin;
  int64_t count = (y_end - y_begin) * width;
  float sum = 0.0f;
  int64_t tail = 0;
  if constexpr (WarpReduce) {
    // Torch's deterministic index_put sums full groups with a warp tree,
    // then lane 0 adds the remainder in original output-index order.
    tail = count / 32 * 32;
    for (int64_t i = static_cast<int64_t>(threadIdx.x) % 32; i < tail;
         i += 32) {
      int64_t oy = y_begin + i / width;
      int64_t ox = x_begin + i % width;
      sum = __fadd_rn(sum,
                      CornerGrad(grad[oy * out_w + ox],
                                 Weight(oy, scale_h),
                                 Weight(ox, scale_w),
                                 corner));
    }
    for (int offset = 16; offset > 0; offset /= 2) {
      sum = __fadd_rn(sum, __shfl_down_sync(0xffffffff, sum, offset));
    }
  }
  if (!WarpReduce || threadIdx.x % 32 == 0) {
    for (int64_t i = tail; i < count; ++i) {
      int64_t oy = y_begin + i / width;
      int64_t ox = x_begin + i % width;
      sum = __fadd_rn(sum,
                      CornerGrad(grad[oy * out_w + ox],
                                 Weight(oy, scale_h),
                                 Weight(ox, scale_w),
                                 corner));
    }
  }
  // index_put accumulates into a zero-initialized gradient tensor.
  return __fadd_rn(0.0f, sum);
}

template <bool WarpReduce>
static __global__ void Backward(const float* out_grad,
                                float* in_grad,
                                int64_t numel,
                                int64_t in_h,
                                int64_t in_w,
                                int64_t out_h,
                                int64_t out_w,
                                float scale_h,
                                float scale_w) {
  constexpr int threads_per_input = WarpReduce ? 32 : 1;
  int64_t stride =
      static_cast<int64_t>(blockDim.x) * gridDim.x / threads_per_input;
  for (int64_t index =
           (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) /
           threads_per_input;
       index < numel;
       index += stride) {
    int64_t x = index % in_w;
    int64_t y = index / in_w % in_h;
    int64_t plane = index / (in_h * in_w);
    float result = 0.0f;
    // The four gather nodes are created in 00,01,10,11 order. Autograd
    // accumulates their input gradients in the reverse order, left to right.
    for (int corner = 3; corner >= 0; --corner) {
      float part = GatherGrad<WarpReduce>(out_grad + plane * out_h * out_w,
                                          y,
                                          x,
                                          in_h,
                                          in_w,
                                          out_h,
                                          out_w,
                                          scale_h,
                                          scale_w,
                                          corner);
      result = corner == 3 ? part : __fadd_rn(result, part);
    }
    if (!WarpReduce || threadIdx.x % 32 == 0) {
      in_grad[index] = result;
    }
  }
}

inline void LaunchForward(const float* input,
                          float* output,
                          int64_t numel,
                          int64_t in_h,
                          int64_t in_w,
                          int64_t out_h,
                          int64_t out_w,
                          cudaStream_t stream) {
  if (numel == 0) return;
  int blocks = static_cast<int>(std::min<int64_t>((numel + 255) / 256, 4096));
  Forward<<<blocks, 256, 0, stream>>>(
      input,
      output,
      numel,
      in_h,
      in_w,
      out_h,
      out_w,
      static_cast<float>(static_cast<double>(in_h) / out_h),
      static_cast<float>(static_cast<double>(in_w) / out_w));
}

inline void LaunchBackward(const float* out_grad,
                           float* in_grad,
                           int64_t numel,
                           int64_t in_h,
                           int64_t in_w,
                           int64_t out_h,
                           int64_t out_w,
                           cudaStream_t stream) {
  if (numel == 0) return;
  float scale_h = static_cast<float>(static_cast<double>(in_h) / out_h);
  float scale_w = static_cast<float>(static_cast<double>(in_w) / out_w);
  // At most 2x upsampling has fewer than 32 contributions per corner even
  // at clamped boundaries. A thread can reproduce the warp's remainder sum.
  // Keep the fast-path bound in the range where d + 0.5 is exact in FP32.
  if (out_h <= 2 * in_h && out_w <= 2 * in_w && out_h <= (1 << 23) &&
      out_w <= (1 << 23)) {
    int blocks = static_cast<int>(std::min<int64_t>((numel + 255) / 256, 4096));
    Backward<false><<<blocks, 256, 0, stream>>>(
        out_grad, in_grad, numel, in_h, in_w, out_h, out_w, scale_h, scale_w);
  } else {
    int blocks = static_cast<int>(std::min<int64_t>((numel + 7) / 8, 4096));
    Backward<true><<<blocks, 256, 0, stream>>>(
        out_grad, in_grad, numel, in_h, in_w, out_h, out_w, scale_h, scale_w);
  }
}

}  // namespace bilinear_compat
}  // namespace funcs
}  // namespace phi
