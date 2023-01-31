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
  grid->Resize(phi::make_ddim({n, h, w, 3}));
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
  ones.Resize(phi::make_ddim({h, w, 1}));
  dev_ctx.template Alloc<T>(&ones);

  phi::funcs::SetConstant<Context, T>()(dev_ctx, &ones, static_cast<T>(1));
  auto ones_t = EigenTensor<T, 3>::From(ones);
  // Get grid tensor with shape [n, h, w, 3] by concatenating h_idx, w_idx and
  // ones
  DenseTensor w_idx_map;
  w_idx_map.Resize(phi::make_ddim({h, w, 1}));
  dev_ctx.template Alloc<T>(&w_idx_map);
  auto w_idx_map_t = EigenTensor<T, 3>::From(w_idx_map);

  DenseTensor h_idx_map;
  h_idx_map.Resize(phi::make_ddim({h, w, 1}));
  dev_ctx.template Alloc<T>(&h_idx_map);
  auto h_idx_map_t = EigenTensor<T, 3>::From(h_idx_map);

  DenseTensor w_h_idx_map;
  w_h_idx_map.Resize(phi::make_ddim({h, w, 2}));
  dev_ctx.template Alloc<T>(&w_h_idx_map);
  auto w_h_idx_map_t = EigenTensor<T, 3>::From(w_h_idx_map);

  DenseTensor w_h_one_idx_map;
  w_h_one_idx_map.Resize(phi::make_ddim({h, w, 3}));
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
  grid->Resize(phi::make_ddim({n, d, h, w, 4}));
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
  ones.Resize(phi::make_ddim({d, h, w, 1}));
  dev_ctx.template Alloc<T>(&ones);

  phi::funcs::SetConstant<Context, T>()(dev_ctx, &ones, static_cast<T>(1));
  auto ones_t = EigenTensor<T, 4>::From(ones);
  // Get grid tensor with shape [n, d, h, w, 4] by concatenating d_idx, h_idx,
  // w_idx and ones
  DenseTensor w_idx_map;
  w_idx_map.Resize(phi::make_ddim({d, h, w, 1}));
  dev_ctx.template Alloc<T>(&w_idx_map);
  auto w_idx_map_t = EigenTensor<T, 4>::From(w_idx_map);

  DenseTensor h_idx_map;
  h_idx_map.Resize(phi::make_ddim({d, h, w, 1}));
  dev_ctx.template Alloc<T>(&h_idx_map);
  auto h_idx_map_t = EigenTensor<T, 4>::From(h_idx_map);

  DenseTensor d_idx_map;
  d_idx_map.Resize(phi::make_ddim({d, h, w, 1}));
  dev_ctx.template Alloc<T>(&d_idx_map);
  auto d_idx_map_t = EigenTensor<T, 4>::From(d_idx_map);

  DenseTensor w_h_idx_map;
  w_h_idx_map.Resize(phi::make_ddim({d, h, w, 2}));
  dev_ctx.template Alloc<T>(&w_h_idx_map);
  auto w_h_idx_map_t = EigenTensor<T, 4>::From(w_h_idx_map);

  DenseTensor w_h_d_idx_map;
  w_h_d_idx_map.Resize(phi::make_ddim({d, h, w, 3}));
  dev_ctx.template Alloc<T>(&w_h_d_idx_map);
  auto w_h_d_idx_map_t = EigenTensor<T, 4>::From(w_h_d_idx_map);

  DenseTensor w_h_d_one_idx_map;
  w_h_d_one_idx_map.Resize(phi::make_ddim({d, h, w, 4}));
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

}  // namespace phi
