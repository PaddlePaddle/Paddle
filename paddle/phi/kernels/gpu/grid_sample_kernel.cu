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

#include "paddle/phi/kernels/grid_sample_kernel.h"

#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/gpu/grid_sample_utils.h"

namespace phi {

template <typename T>
static __forceinline__ __device__ T Unnormalize(T coord,
                                                int size,
                                                bool align_corners) {
  if (align_corners) {
    return ((coord + 1.f) / 2) * (size - 1);
  } else {
    return ((coord + 1.f) * size - 1) / 2;
  }
}

template <typename T>
static __forceinline__ __device__ T ClipIndexes(T in, int max_value) {
  return min(static_cast<T>(max_value), max(in, static_cast<T>(0)));
}

template <typename T>
static __forceinline__ __device__ T ReflectIndexes(T in,
                                                   int twice_low,
                                                   int twice_high) {
  if (twice_low == twice_high) {
    return static_cast<T>(0);
  }
  T min = static_cast<T>(twice_low) / 2;
  T span = static_cast<T>(twice_high - twice_low) / 2;
  in = fabs(in - min);
  T extra = fmod(in, span);
  int flips = static_cast<int>(floor(in / span));
  if (flips % 2 == 0) {
    return extra + min;
  } else {
    return span - extra + min;
  }
}

template <typename T>
static __forceinline__ __device__ T ComputePositions(T coord,
                                                     int size,
                                                     PaddingMode padding_mode,
                                                     bool align_corners) {
  coord = Unnormalize<T>(coord, size, align_corners);
  if (padding_mode == PaddingMode::border) {
    coord = ClipIndexes(coord, size - 1);
  } else if (padding_mode == PaddingMode::reflect) {
    if (align_corners) {
      coord = ReflectIndexes(coord, 0, 2 * (size - 1));
    } else {
      coord = ReflectIndexes(coord, -1, 2 * size - 1);
    }
    coord = ClipIndexes(coord, size - 1);
  }
  return coord;
}

template <typename T>
__global__ void GridSampleCudaKernel(const int nthreads,
                                     int n,
                                     int out_c,
                                     int out_h,
                                     int out_w,
                                     int in_h,
                                     int in_w,
                                     const T* input,
                                     const T* grid,
                                     T* output,
                                     const Mode mode,
                                     const PaddingMode padding_mode,
                                     bool align_corners) {
  int inp_sN = out_c * in_h * in_w;

  int inp_sC = in_h * in_w;
  int inp_sH = in_w;
  int inp_sW = 1;
  int grid_sN = out_h * out_w * 2;
  int grid_sH = out_w * 2;
  int grid_sW = 2;
  int grid_sCoor = 1;
  int out_sN = out_c * out_h * out_w;
  int out_sC = out_h * out_w;
  int out_sH = out_w;
  int out_sW = 1;
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % out_w;
    const int h = (index / out_w) % out_h;
    const int n = index / (out_h * out_w);
    const int grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;

    T ix = grid[grid_offset];
    T iy = grid[grid_offset + grid_sCoor];

    ix = ComputePositions(ix, in_w, padding_mode, align_corners);
    iy = ComputePositions(iy, in_h, padding_mode, align_corners);
    if (mode == Mode::bilinear) {
      int ix_nw = static_cast<int>(floor(ix));
      int iy_nw = static_cast<int>(floor(iy));
      int ix_ne = ix_nw + 1;
      int iy_ne = iy_nw;
      int ix_sw = ix_nw;
      int iy_sw = iy_nw + 1;
      int ix_se = ix_nw + 1;
      int iy_se = iy_nw + 1;

      T nw = (ix_se - ix) * (iy_se - iy);
      T ne = (ix - ix_sw) * (iy_sw - iy);
      T sw = (ix_ne - ix) * (iy - iy_ne);
      T se = (ix - ix_nw) * (iy - iy_nw);

      auto inp_offset_NC = n * inp_sN;

      auto out_ptr_NCHW = output + n * out_sN + h * out_sH + w * out_sW;
      for (int c = 0; c < out_c;
           ++c, inp_offset_NC += inp_sC, out_ptr_NCHW += out_sC) {
        *out_ptr_NCHW = static_cast<T>(0);
        if (InBounds(iy_nw, ix_nw, in_h, in_w)) {
          *out_ptr_NCHW +=
              input[inp_offset_NC + iy_nw * inp_sH + ix_nw * inp_sW] * nw;
        }
        if (InBounds(iy_ne, ix_ne, in_h, in_w)) {
          *out_ptr_NCHW +=
              input[inp_offset_NC + iy_ne * inp_sH + ix_ne * inp_sW] * ne;
        }
        if (InBounds(iy_sw, ix_sw, in_h, in_w)) {
          *out_ptr_NCHW +=
              input[inp_offset_NC + iy_sw * inp_sH + ix_sw * inp_sW] * sw;
        }
        if (InBounds(iy_se, ix_se, in_h, in_w)) {
          *out_ptr_NCHW +=
              input[inp_offset_NC + iy_se * inp_sH + ix_se * inp_sW] * se;
        }
      }
    } else if (mode == Mode::nearest) {
      int ix_nearest = static_cast<int>(std::nearbyint(ix));
      int iy_nearest = static_cast<int>(std::nearbyint(iy));
      auto inp_offset_NC = n * inp_sN;
      auto out_ptr_NCHW = output + n * out_sN + h * out_sH + w * out_sW;
      for (int c = 0; c < out_c;
           ++c, inp_offset_NC += inp_sC, out_ptr_NCHW += out_sC) {
        if (InBounds(iy_nearest, ix_nearest, in_h, in_w)) {
          *out_ptr_NCHW =
              input[inp_offset_NC + iy_nearest * inp_sH + ix_nearest * inp_sW];
        } else {
          *out_ptr_NCHW = static_cast<T>(0);
        }
      }
    }
  }
}

template <typename T, typename Context>
void GridSampleKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& grid,
                      const std::string& mode,
                      const std::string& padding_mode,
                      bool align_corners,
                      DenseTensor* out) {
  PaddingMode enum_padding_mode;
  Mode enum_mode;
  if (padding_mode == "border") {
    enum_padding_mode = PaddingMode::border;
  } else if (padding_mode == "reflection") {
    enum_padding_mode = PaddingMode::reflect;
  } else {
    enum_padding_mode = PaddingMode::zeros;
  }

  if (mode == "nearest") {
    enum_mode = Mode::nearest;
  } else {
    enum_mode = Mode::bilinear;
  }

  const int n = grid.dims()[0];
  const int out_h = grid.dims()[1];
  const int out_w = grid.dims()[2];
  const int c = x.dims()[1];
  const int in_h = x.dims()[2];
  const int in_w = x.dims()[3];
  VLOG(3) << "n: " << n << "; c: " << c << "; out_h: " << out_h
          << "; out_w: " << out_w;

  auto* output_data = dev_ctx.template Alloc<T>(out);
  VLOG(3) << "out dims: " << out->dims()[0] << "; " << out->dims()[1] << "; "
          << out->dims()[2] << "; " << out->dims()[3];

  int count = static_cast<int>(n * out_h * out_w);
  auto cu_stream = dev_ctx.stream();
  backends::gpu::GpuLaunchConfig config =
      backends::gpu::GetGpuLaunchConfig1D(dev_ctx, count);
  GridSampleCudaKernel<
      T><<<config.block_per_grid, config.thread_per_block, 0, cu_stream>>>(
      count,
      n,
      c,
      out_h,
      out_w,
      in_h,
      in_w,
      x.data<T>(),
      grid.data<T>(),
      output_data,
      enum_mode,
      enum_padding_mode,
      align_corners);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    grid_sample, GPU, ALL_LAYOUT, phi::GridSampleKernel, float, double) {}
