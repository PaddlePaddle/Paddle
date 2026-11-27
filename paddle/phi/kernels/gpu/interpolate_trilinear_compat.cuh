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

#include "paddle/phi/kernels/funcs/interpolate_function.h"
#include "paddle/phi/kernels/gpu/interpolate_bilinear_compat.cuh"

namespace phi {
namespace funcs {
namespace trilinear_compat {

// The 3D counterpart of bilinear_compat: torch 2.12's _upsample_linear
// lerps W, then H, then D, and its backward is eight index_put reductions.
using bilinear_compat::Coordinate;
using bilinear_compat::Lerp;
using bilinear_compat::LowerBound;
using bilinear_compat::Weight;

// Spatial extents and coordinate scales in (d, h, w) order.
struct Geometry {
  int64_t in[3];
  int64_t out[3];
  float scale[3];
  bool align_corners;
};

// Splits a memory-order index of a (d, h, w) tensor into its voxel and
// returns where the same (n, c) volume starts in the other tensor. NDHWC
// keeps channels innermost, like bilinear_compat::Locate.
template <bool NDHWC>
__device__ __forceinline__ int64_t Locate(int64_t index,
                                          const int64_t* size,
                                          const int64_t* other,
                                          int64_t channels,
                                          int64_t* position) {
  int64_t voxel = NDHWC ? index / channels : index;
  position[2] = voxel % size[2];
  position[1] = voxel / size[2] % size[1];
  position[0] = voxel / (size[1] * size[2]) % size[0];
  int64_t volume = voxel / (size[0] * size[1] * size[2]);
  int64_t other_volume = other[0] * other[1] * other[2];
  if constexpr (NDHWC) {
    return volume * other_volume * channels + index % channels;
  } else {
    return volume * other_volume;
  }
}

template <typename T, bool NDHWC>
static __global__ void Forward(const T* input,
                               T* output,
                               int64_t numel,
                               int64_t channels,
                               Geometry geometry) {
  const int64_t step = NDHWC ? channels : 1;
  const int64_t in_h = geometry.in[1];
  const int64_t in_w = geometry.in[2];
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < numel;
       index += stride) {
    int64_t position[3];
    const T* values =
        input +
        Locate<NDHWC>(index, geometry.out, geometry.in, channels, position);
    int64_t source[3][2];
    float weight[3];
    for (int k = 0; k < 3; ++k) {
      float coordinate =
          Coordinate(position[k], geometry.scale[k], geometry.align_corners);
      int64_t lower = static_cast<int64_t>(coordinate);
      source[k][0] = lower;
      source[k][1] =
          lower + 1 < geometry.in[k] ? lower + 1 : geometry.in[k] - 1;
      weight[k] = fminf(__fsub_rn(coordinate, static_cast<float>(lower)), 1.0f);
    }
    auto at = [&](int d, int h, int w) {
      return static_cast<float>(
          values[((source[0][d] * in_h + source[1][h]) * in_w + source[2][w]) *
                 step]);
    };
    float plane[2];
    for (int d = 0; d < 2; ++d) {
      float row[2];
      for (int h = 0; h < 2; ++h) {
        row[h] = Lerp(at(d, h, 0), at(d, h, 1), weight[2]);
      }
      plane[d] = Lerp(row[0], row[1], weight[1]);
    }
    output[index] = static_cast<T>(Lerp(plane[0], plane[1], weight[0]));
  }
}

// Corner bits are (d, h, w) from high to low, the order in which the
// decomposition creates its gathers. Each lerp level hands `grad * w` to the
// upper corner and `grad - grad * w` to the lower one, outermost (D) first.
__device__ __forceinline__ float CornerGrad(float grad,
                                            const float* weight,
                                            int corner) {
  for (int k = 0; k < 3; ++k) {
    float upper = __fmul_rn(grad, weight[k]);
    grad = (corner >> (2 - k)) & 1 ? upper : __fsub_rn(grad, upper);
  }
  return grad;
}

// Reproduces torch's deterministic index_put for one corner: contributions
// in output row-major order, full groups of 32 summed per lane and reduced by
// a shuffle-down tree, then the remainder added in order by lane 0. Without
// WarpReduce one thread replays the lanes, which keeps it exact for any count.
template <typename T, bool WarpReduce>
__device__ __forceinline__ float GatherGrad(const T* grad,
                                            int64_t step,
                                            const int64_t* position,
                                            const Geometry& geometry,
                                            int corner) {
  int64_t begin[3];
  int64_t extent[3];
  for (int k = 0; k < 3; ++k) {
    int bit = (corner >> (2 - k)) & 1;
    begin[k] = LowerBound(position[k],
                          bit,
                          geometry.in[k],
                          geometry.out[k],
                          geometry.scale[k],
                          geometry.align_corners);
    extent[k] = LowerBound(position[k] + 1,
                           bit,
                           geometry.in[k],
                           geometry.out[k],
                           geometry.scale[k],
                           geometry.align_corners) -
                begin[k];
  }
  auto contribution = [&](int64_t i) {
    int64_t o[3] = {begin[0] + i / (extent[1] * extent[2]),
                    begin[1] + i / extent[2] % extent[1],
                    begin[2] + i % extent[2]};
    float weight[3];
    for (int k = 0; k < 3; ++k) {
      weight[k] = Weight(o[k], geometry.scale[k], geometry.align_corners);
    }
    int64_t offset =
        ((o[0] * geometry.out[1] + o[1]) * geometry.out[2] + o[2]) * step;
    return CornerGrad(static_cast<float>(grad[offset]), weight, corner);
  };
  int64_t count = extent[0] * extent[1] * extent[2];
  int64_t full = count / 32 * 32;
  float sum = 0.0f;
  if constexpr (WarpReduce) {
    for (int64_t i = static_cast<int64_t>(threadIdx.x) % 32; i < full;
         i += 32) {
      sum = __fadd_rn(sum, contribution(i));
    }
    for (int offset = 16; offset > 0; offset /= 2) {
      sum = __fadd_rn(sum, __shfl_down_sync(0xffffffff, sum, offset));
    }
  } else if (full > 0) {
    float lanes[32] = {};
    for (int64_t i = 0; i < full; ++i) {
      lanes[i % 32] = __fadd_rn(lanes[i % 32], contribution(i));
    }
    for (int offset = 16; offset > 0; offset /= 2) {
      for (int lane = 0; lane < offset; ++lane) {
        lanes[lane] = __fadd_rn(lanes[lane], lanes[lane + offset]);
      }
    }
    sum = lanes[0];
  }
  if (!WarpReduce || threadIdx.x % 32 == 0) {
    for (int64_t i = full; i < count; ++i) {
      sum = __fadd_rn(sum, contribution(i));
    }
  }
  // index_put accumulates into a zero-initialized gradient tensor.
  return __fadd_rn(0.0f, sum);
}

template <typename T, bool WarpReduce, bool NDHWC>
static __global__ void Backward(const T* out_grad,
                                T* in_grad,
                                int64_t numel,
                                int64_t channels,
                                Geometry geometry) {
  const int64_t step = NDHWC ? channels : 1;
  constexpr int threads_per_input = WarpReduce ? 32 : 1;
  int64_t stride =
      static_cast<int64_t>(blockDim.x) * gridDim.x / threads_per_input;
  for (int64_t index =
           (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) /
           threads_per_input;
       index < numel;
       index += stride) {
    int64_t position[3];
    const T* grad =
        out_grad +
        Locate<NDHWC>(index, geometry.in, geometry.out, channels, position);
    float result = 0.0f;
    // Autograd accumulates the eight gathers' gradients in reverse creation
    // order, left to right.
    for (int corner = 7; corner >= 0; --corner) {
      float part =
          GatherGrad<T, WarpReduce>(grad, step, position, geometry, corner);
      result = corner == 7 ? part : __fadd_rn(result, part);
    }
    if (!WarpReduce || threadIdx.x % 32 == 0) {
      in_grad[index] = static_cast<T>(result);
    }
  }
}

inline Geometry MakeGeometry(int64_t in_d,
                             int64_t in_h,
                             int64_t in_w,
                             int64_t out_d,
                             int64_t out_h,
                             int64_t out_w,
                             bool align_corners,
                             double scale_factor_d,
                             double scale_factor_h,
                             double scale_factor_w) {
  return {
      {in_d, in_h, in_w},
      {out_d, out_h, out_w},
      {AreaPixelComputeScale<float>(in_d, out_d, align_corners, scale_factor_d),
       AreaPixelComputeScale<float>(in_h, out_h, align_corners, scale_factor_h),
       AreaPixelComputeScale<float>(
           in_w, out_w, align_corners, scale_factor_w)},
      align_corners};
}

template <typename T>
inline void LaunchForward(const T* input,
                          T* output,
                          int64_t numel,
                          int64_t in_d,
                          int64_t in_h,
                          int64_t in_w,
                          int64_t out_d,
                          int64_t out_h,
                          int64_t out_w,
                          int64_t channels,
                          bool ndhwc,
                          bool align_corners,
                          double scale_factor_d,
                          double scale_factor_h,
                          double scale_factor_w,
                          cudaStream_t stream) {
  if (numel == 0) return;
  Geometry geometry = MakeGeometry(in_d,
                                   in_h,
                                   in_w,
                                   out_d,
                                   out_h,
                                   out_w,
                                   align_corners,
                                   scale_factor_d,
                                   scale_factor_h,
                                   scale_factor_w);
  int blocks = static_cast<int>(std::min<int64_t>((numel + 255) / 256, 4096));
  auto kernel = ndhwc ? Forward<T, true> : Forward<T, false>;
  kernel<<<blocks, 256, 0, stream>>>(input, output, numel, channels, geometry);
}

template <typename T>
inline void LaunchBackward(const T* out_grad,
                           T* in_grad,
                           int64_t numel,
                           int64_t in_d,
                           int64_t in_h,
                           int64_t in_w,
                           int64_t out_d,
                           int64_t out_h,
                           int64_t out_w,
                           int64_t channels,
                           bool ndhwc,
                           bool align_corners,
                           double scale_factor_d,
                           double scale_factor_h,
                           double scale_factor_w,
                           cudaStream_t stream) {
  if (numel == 0) return;
  Geometry geometry = MakeGeometry(in_d,
                                   in_h,
                                   in_w,
                                   out_d,
                                   out_h,
                                   out_w,
                                   align_corners,
                                   scale_factor_d,
                                   scale_factor_h,
                                   scale_factor_w);
  // Both paths are exact. Up to 2x upsampling each input sees at most 64
  // contributions per corner, cheap enough for one thread; larger factors
  // spread the reduction over a warp as torch does.
  bool small_factor =
      align_corners ||
      (scale_factor_d <= 2. && scale_factor_h <= 2. && scale_factor_w <= 2.);
  if (small_factor && out_d <= 2 * in_d && out_h <= 2 * in_h &&
      out_w <= 2 * in_w) {
    int blocks = static_cast<int>(std::min<int64_t>((numel + 255) / 256, 4096));
    auto kernel = ndhwc ? Backward<T, false, true> : Backward<T, false, false>;
    kernel<<<blocks, 256, 0, stream>>>(
        out_grad, in_grad, numel, channels, geometry);
  } else {
    int blocks = static_cast<int>(std::min<int64_t>((numel + 7) / 8, 4096));
    auto kernel = ndhwc ? Backward<T, true, true> : Backward<T, true, false>;
    kernel<<<blocks, 256, 0, stream>>>(
        out_grad, in_grad, numel, channels, geometry);
  }
}

}  // namespace trilinear_compat
}  // namespace funcs
}  // namespace phi
