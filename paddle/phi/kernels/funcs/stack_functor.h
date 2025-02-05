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

#include "paddle/phi/kernels/funcs/for_range.h"

namespace phi {
namespace funcs {

template <typename VecXType, typename T>
struct StackFunctor {
  HOSTDEVICE StackFunctor(const VecXType &x, T *y, int n, int post)
      : x_(x), y_(y), n_(n), post_(post) {}

  HOSTDEVICE void operator()(int idx) {
    int i = idx / (n_ * post_);
    int which_x = idx / post_ - i * n_;
    int x_index = i * post_ + idx % post_;
    y_[idx] = x_[which_x][x_index];
  }

 private:
  VecXType x_;
  T *y_;
  int n_;
  int post_;
};

template <typename VecDxType, typename T>
struct StackGradFunctor {
  HOSTDEVICE StackGradFunctor(const VecDxType &dx, const T *dy, int n, int post)
      : dx_(dx), dy_(dy), n_(n), post_(post) {}

  HOSTDEVICE void operator()(int idx) {
    int i = idx / (n_ * post_);
    int which_x = idx / post_ - i * n_;
    int x_index = i * post_ + idx % post_;
    if (dx_[which_x] != nullptr) dx_[which_x][x_index] = dy_[idx];
  }

 private:
  VecDxType dx_;
  const T *dy_;
  int n_;
  int post_;
};

template <typename DeviceContext, typename VecXType, typename T>
static inline void StackFunctorForRange(const DeviceContext &ctx,
                                        const VecXType &x,
                                        T *y,
                                        int total_num,
                                        int n,
                                        int post) {
  phi::funcs::ForRange<DeviceContext> for_range(ctx, total_num);
  for_range(StackFunctor<VecXType, T>(x, y, n, post));
}

template <typename DeviceContext, typename VecDxType, typename T>
static inline void StackGradFunctorForRange(const DeviceContext &ctx,
                                            const VecDxType &dx,
                                            const T *dy,
                                            int total_num,
                                            int n,
                                            int post) {
  phi::funcs::ForRange<DeviceContext> for_range(ctx, total_num);
  for_range(StackGradFunctor<VecDxType, T>(dx, dy, n, post));
}

}  // namespace funcs
}  // namespace phi
