// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/funcs/affine_grid_utils.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_device_function.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"

namespace phi {
namespace funcs {

template <typename T>
__global__ void CreateBaseGridKernel_4D_Kernel(
    T* base_grid_data, int64_t n, int64_t h, int64_t w, bool align_corners) {
  int64_t total_elements = n * h * w;
  CUDA_KERNEL_LOOP(idx, total_elements) {
    int64_t w_idx = idx % w;
    int64_t h_idx = (idx / w) % h;
    int64_t n_idx = idx / (h * w);

    int64_t grid_idx = n_idx * h * w + h_idx * w + w_idx;

    T x, y;
    T start_data = static_cast<T>(-1);
    T stop_data = static_cast<T>(1);

    if (w > 1) {
      T step = (stop_data - start_data) / (w - 1);
      int64_t w_half = w / 2;
      if (w_idx < w_half) {
        x = static_cast<T>(start_data + step * w_idx);
      } else {
        x = static_cast<T>(stop_data - step * (w - w_idx - 1));
      }
      if (!align_corners) {
        x = (x * static_cast<T>(w - 1)) *
            (static_cast<T>(1) / static_cast<T>(w));
      }
    } else {
      x = static_cast<T>(0);
    }

    if (h > 1) {
      T step = (stop_data - start_data) / (h - 1);
      int64_t h_half = h / 2;
      if (h_idx < h_half) {
        y = static_cast<T>(start_data + step * h_idx);
      } else {
        y = static_cast<T>(stop_data - step * (h - h_idx - 1));
      }
      if (!align_corners) {
        y = (y * static_cast<T>(h - 1)) *
            (static_cast<T>(1) / static_cast<T>(h));
      }
    } else {
      y = static_cast<T>(0);
    }

    base_grid_data[grid_idx * 3 + 0] = x;
    base_grid_data[grid_idx * 3 + 1] = y;
    base_grid_data[grid_idx * 3 + 2] = static_cast<T>(1);
  }
}

template <typename T>
__global__ void CreateBaseGridKernel_5D_Kernel(T* base_grid_data,
                                               int64_t n,
                                               int64_t d,
                                               int64_t h,
                                               int64_t w,
                                               bool align_corners) {
  int64_t total_elements = n * d * h * w;
  CUDA_KERNEL_LOOP(idx, total_elements) {
    int64_t w_idx = idx % w;
    int64_t h_idx = (idx / w) % h;
    int64_t d_idx = (idx / (w * h)) % d;
    int64_t n_idx = idx / (d * h * w);

    int64_t grid_idx = n_idx * d * h * w + d_idx * h * w + h_idx * w + w_idx;

    T x, y, z;
    T start_data = static_cast<T>(-1);
    T stop_data = static_cast<T>(1);

    // X coordinate (W dimension)
    if (w > 1) {
      T step = (stop_data - start_data) / (w - 1);
      int64_t w_half = w / 2;
      if (w_idx < w_half) {
        x = static_cast<T>(start_data + step * w_idx);
      } else {
        x = static_cast<T>(stop_data - step * (w - w_idx - 1));
      }
      if (!align_corners) {
        x = (x * static_cast<T>(w - 1)) *
            (static_cast<T>(1) / static_cast<T>(w));
      }
    } else {
      x = static_cast<T>(0);
    }

    // Y coordinate (H dimension)
    if (h > 1) {
      T step = (stop_data - start_data) / (h - 1);
      int64_t h_half = h / 2;
      if (h_idx < h_half) {
        y = static_cast<T>(start_data + step * h_idx);
      } else {
        y = static_cast<T>(stop_data - step * (h - h_idx - 1));
      }
      if (!align_corners) {
        y = (y * static_cast<T>(h - 1)) *
            (static_cast<T>(1) / static_cast<T>(h));
      }
    } else {
      y = static_cast<T>(0);
    }

    // Z coordinate (D dimension)
    if (d > 1) {
      T step = (stop_data - start_data) / (d - 1);
      int64_t d_half = d / 2;
      if (d_idx < d_half) {
        z = static_cast<T>(start_data + step * d_idx);
      } else {
        z = static_cast<T>(stop_data - step * (d - d_idx - 1));
      }
      if (!align_corners) {
        z = (z * static_cast<T>(d - 1)) *
            (static_cast<T>(1) / static_cast<T>(d));
      }
    } else {
      z = static_cast<T>(0);
    }

    base_grid_data[grid_idx * 4 + 0] = x;
    base_grid_data[grid_idx * 4 + 1] = y;
    base_grid_data[grid_idx * 4 + 2] = z;
    base_grid_data[grid_idx * 4 + 3] = static_cast<T>(1);
  }
}

template <typename T, typename Context>
void CreateBaseGridKernel_4D(const Context& dev_ctx,
                             T* base_grid_data,
                             int64_t n,
                             int64_t h,
                             int64_t w,
                             bool align_corners) {
  int64_t total_elements = n * h * w;
  auto stream = dev_ctx.stream();
  int64_t block_size = 512;
  int64_t grid_size = (total_elements + block_size - 1) / block_size;
  CreateBaseGridKernel_4D_Kernel<T><<<grid_size, block_size, 0, stream>>>(
      base_grid_data, n, h, w, align_corners);
}

template <typename T, typename Context>
void CreateBaseGridKernel_5D(const Context& dev_ctx,
                             T* base_grid_data,
                             int64_t n,
                             int64_t d,
                             int64_t h,
                             int64_t w,
                             bool align_corners) {
  int64_t total_elements = n * d * h * w;
  auto stream = dev_ctx.stream();
  int64_t block_size = 512;
  int64_t grid_size = (total_elements + block_size - 1) / block_size;
  CreateBaseGridKernel_5D_Kernel<T><<<grid_size, block_size, 0, stream>>>(
      base_grid_data, n, d, h, w, align_corners);
}

template void CreateBaseGridKernel_4D<float, phi::GPUContext>(
    const phi::GPUContext&, float*, int64_t, int64_t, int64_t, bool);
template void CreateBaseGridKernel_4D<double, phi::GPUContext>(
    const phi::GPUContext&, double*, int64_t, int64_t, int64_t, bool);

template void CreateBaseGridKernel_5D<float, phi::GPUContext>(
    const phi::GPUContext&, float*, int64_t, int64_t, int64_t, int64_t, bool);
template void CreateBaseGridKernel_5D<double, phi::GPUContext>(
    const phi::GPUContext&, double*, int64_t, int64_t, int64_t, int64_t, bool);

}  // namespace funcs
}  // namespace phi
