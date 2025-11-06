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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

using Array1 = Eigen::DSizes<int64_t, 1>;
using Array2 = Eigen::DSizes<int64_t, 2>;
using Array3 = Eigen::DSizes<int64_t, 3>;
using Array4 = Eigen::DSizes<int64_t, 4>;
using Array5 = Eigen::DSizes<int64_t, 5>;

template <typename Context, typename T>
struct Linspace {
  void operator()(T start,
                  T end,
                  int count,
                  bool align_corners,
                  DenseTensor* numbers,
                  const Context& dev_ctx);
};

template <typename Context, typename T>
inline void GetIdxMap4D(int n,
                        int h,
                        int w,
                        bool align_corners,
                        DenseTensor* grid,
                        const Context& dev_ctx) {
  auto& place = *dev_ctx.eigen_device();
  grid->Resize(common::make_ddim({n, h, w, 3}));
  dev_ctx.template Alloc<T>(grid);
  auto grid_t = EigenTensor<T, 4>::From(*grid);
  // Get indexes of height with shape [height, width, 1]
  DenseTensor h_idx;
  Linspace<Context, T> linspace;
  linspace((T)-1, (T)1, h, align_corners, &h_idx, dev_ctx);
  auto h_idx_t = EigenTensor<T, 1>::From(h_idx);
  // Get indexes of width with shape [height, width, 1]
  DenseTensor w_idx;
  linspace((T)-1, (T)1, w, align_corners, &w_idx, dev_ctx);
  auto w_idx_t = EigenTensor<T, 1>::From(w_idx);
  // Get constant ones tensor with shape [height, width, 1]
  DenseTensor ones;
  ones.Resize(common::make_ddim({h, w, 1}));
  dev_ctx.template Alloc<T>(&ones);

  phi::funcs::SetConstant<Context, T>()(dev_ctx, &ones, static_cast<T>(1));
  auto ones_t = EigenTensor<T, 3>::From(ones);
  // Get grid tensor with shape [n, h, w, 3] by concatenating h_idx, w_idx and
  // ones
  DenseTensor w_idx_map;
  w_idx_map.Resize(common::make_ddim({h, w, 1}));
  dev_ctx.template Alloc<T>(&w_idx_map);
  auto w_idx_map_t = EigenTensor<T, 3>::From(w_idx_map);

  DenseTensor h_idx_map;
  h_idx_map.Resize(common::make_ddim({h, w, 1}));
  dev_ctx.template Alloc<T>(&h_idx_map);
  auto h_idx_map_t = EigenTensor<T, 3>::From(h_idx_map);

  DenseTensor w_h_idx_map;
  w_h_idx_map.Resize(common::make_ddim({h, w, 2}));
  dev_ctx.template Alloc<T>(&w_h_idx_map);
  auto w_h_idx_map_t = EigenTensor<T, 3>::From(w_h_idx_map);

  DenseTensor w_h_one_idx_map;
  w_h_one_idx_map.Resize(common::make_ddim({h, w, 3}));
  dev_ctx.template Alloc<T>(&w_h_one_idx_map);
  auto w_h_one_idx_map_t = EigenTensor<T, 3>::From(w_h_one_idx_map);

  w_idx_map_t.device(place) = w_idx_t.reshape(Array2(1, w))
                                  .broadcast(Array2(h, 1))
                                  .reshape(Array3(h, w, 1));
  h_idx_map_t.device(place) = h_idx_t.reshape(Array2(1, h))
                                  .broadcast(Array2(w, 1))
                                  .shuffle(Array2(1, 0))
                                  .reshape(Array3(h, w, 1));

  w_h_idx_map_t.device(place) = w_idx_map_t.concatenate(h_idx_map_t, 2);
  w_h_one_idx_map_t.device(place) = w_h_idx_map_t.concatenate(ones_t, 2);
  grid_t.device(place) = w_h_one_idx_map_t.reshape(Array4(1, h, w, 3))
                             .broadcast(Array4(n, 1, 1, 1));
}

template <typename Context, typename T>
inline void GetIdxMap5D(int n,
                        int d,
                        int h,
                        int w,
                        bool align_corners,
                        DenseTensor* grid,
                        const Context& dev_ctx) {
  auto& place = *dev_ctx.eigen_device();
  grid->Resize(common::make_ddim({n, d, h, w, 4}));
  dev_ctx.template Alloc<T>(grid);
  auto grid_t = EigenTensor<T, 5>::From(*grid);
  // Get indexes of height with shape [depth, height, width, 1]
  DenseTensor d_idx;
  Linspace<Context, T> linspace;
  linspace((T)-1, (T)1, d, align_corners, &d_idx, dev_ctx);
  auto d_idx_t = EigenTensor<T, 1>::From(d_idx);
  // Get indexes of height with shape [depth, height, width, 1]
  DenseTensor h_idx;
  linspace((T)-1, (T)1, h, align_corners, &h_idx, dev_ctx);
  auto h_idx_t = EigenTensor<T, 1>::From(h_idx);
  // Get indexes of width with shape [depth, height, width, 1]
  DenseTensor w_idx;
  linspace((T)-1, (T)1, w, align_corners, &w_idx, dev_ctx);
  auto w_idx_t = EigenTensor<T, 1>::From(w_idx);
  // Get constant ones tensor with shape [depth, height, width, 1]
  DenseTensor ones;
  ones.Resize(common::make_ddim({d, h, w, 1}));
  dev_ctx.template Alloc<T>(&ones);

  phi::funcs::SetConstant<Context, T>()(dev_ctx, &ones, static_cast<T>(1));
  auto ones_t = EigenTensor<T, 4>::From(ones);
  // Get grid tensor with shape [n, d, h, w, 4] by concatenating d_idx, h_idx,
  // w_idx and ones
  DenseTensor w_idx_map;
  w_idx_map.Resize(common::make_ddim({d, h, w, 1}));
  dev_ctx.template Alloc<T>(&w_idx_map);
  auto w_idx_map_t = EigenTensor<T, 4>::From(w_idx_map);

  DenseTensor h_idx_map;
  h_idx_map.Resize(common::make_ddim({d, h, w, 1}));
  dev_ctx.template Alloc<T>(&h_idx_map);
  auto h_idx_map_t = EigenTensor<T, 4>::From(h_idx_map);

  DenseTensor d_idx_map;
  d_idx_map.Resize(common::make_ddim({d, h, w, 1}));
  dev_ctx.template Alloc<T>(&d_idx_map);
  auto d_idx_map_t = EigenTensor<T, 4>::From(d_idx_map);

  DenseTensor w_h_idx_map;
  w_h_idx_map.Resize(common::make_ddim({d, h, w, 2}));
  dev_ctx.template Alloc<T>(&w_h_idx_map);
  auto w_h_idx_map_t = EigenTensor<T, 4>::From(w_h_idx_map);

  DenseTensor w_h_d_idx_map;
  w_h_d_idx_map.Resize(common::make_ddim({d, h, w, 3}));
  dev_ctx.template Alloc<T>(&w_h_d_idx_map);
  auto w_h_d_idx_map_t = EigenTensor<T, 4>::From(w_h_d_idx_map);

  DenseTensor w_h_d_one_idx_map;
  w_h_d_one_idx_map.Resize(common::make_ddim({d, h, w, 4}));
  dev_ctx.template Alloc<T>(&w_h_d_one_idx_map);
  auto w_h_d_one_idx_map_t = EigenTensor<T, 4>::From(w_h_d_one_idx_map);

  w_idx_map_t.device(place) = w_idx_t.reshape(Array3(1, 1, w))
                                  .broadcast(Array3(d, h, 1))
                                  .reshape(Array4(d, h, w, 1));
  h_idx_map_t.device(place) = h_idx_t.reshape(Array3(1, h, 1))
                                  .broadcast(Array3(d, 1, w))
                                  .reshape(Array4(d, h, w, 1));
  d_idx_map_t.device(place) = d_idx_t.reshape(Array3(d, 1, 1))
                                  .broadcast(Array3(1, h, w))
                                  .reshape(Array4(d, h, w, 1));

  w_h_idx_map_t.device(place) = w_idx_map_t.concatenate(h_idx_map_t, 3);
  w_h_d_idx_map_t.device(place) = w_h_idx_map_t.concatenate(d_idx_map_t, 3);

  w_h_d_one_idx_map_t.device(place) = w_h_d_idx_map_t.concatenate(ones_t, 3);
  grid_t.device(place) = w_h_d_one_idx_map_t.reshape(Array5(1, d, h, w, 4))
                             .broadcast(Array5(n, 1, 1, 1, 1));
}

namespace funcs {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#if defined(__CUDACC__) || defined(__HIPCC__)
template <typename T>
__global__ void CreateBaseGridKernel_4D(
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
__global__ void CreateBaseGridKernel_5D(T* base_grid_data,
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
#endif  // __CUDACC__ || __HIPCC__
#endif  // PADDLE_WITH_CUDA || PADDLE_WITH_HIP
}  // namespace funcs

}  // namespace phi
