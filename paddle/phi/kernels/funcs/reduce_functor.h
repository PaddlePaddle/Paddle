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

#include "paddle/common/macros.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
namespace phi {
namespace funcs {

//////// Frobenius Norm Functor ///////
struct FrobeniusNormFunctor {
  template <typename DeviceContext, typename X, typename Y, typename Dim>
  void operator()(const DeviceContext& place, X* x, Y* y, const Dim& dim) {
    y->device(place) = ((x->square()).sum(dim)).sqrt();
  }
};

struct FrobeniusNormGradFunctor {
  template <typename DeviceContext,
            typename X,
            typename Y,
            typename DX,
            typename DY,
            typename Dim>
  void operator()(const DeviceContext& place,
                  X* x,
                  Y* y,
                  DX* dx,
                  DY* dy,
                  const Dim& dim,
                  int size UNUSED) {
    dx->device(place) = y->broadcast(dim);
    dx->device(place) = *dx + dx->constant(1e-12f);
    dx->device(place) = (*x / *dx) * (dy->broadcast(dim));
  }
};

//////// Max Functor ///////
struct MaxFunctor {
  template <typename DeviceContext, typename X, typename Y, typename Dim>
  void operator()(const DeviceContext& place, X* x, Y* y, const Dim& dim) {
    y->device(place) = x->maximum(dim);
  }
};

//////// Mean Functor ///////
struct MeanFunctor {
  template <typename DeviceContext, typename X, typename Y, typename Dim>
  void operator()(const DeviceContext& place, X* x, Y* y, const Dim& dim) {
    y->device(place) = x->mean(dim);
  }
};

//////// Prod Functor ///////
struct ProdFunctor {
  template <typename DeviceContext, typename X, typename Y, typename Dim>
  void operator()(const DeviceContext& place, X* x, Y* y, const Dim& dim) {
    y->device(place) = x->prod(dim);
  }
};

//////// Sum Functor ///////
struct SumFunctor {
  template <typename DeviceContext, typename X, typename Y, typename Dim>
  void operator()(const DeviceContext& place, X* x, Y* y, const Dim& dim) {
    y->device(place) = x->sum(dim);
  }
};

//////// Min Functor ///////
struct MinFunctor {
  template <typename DeviceContext, typename X, typename Y, typename Dim>
  void operator()(const DeviceContext& place, X* x, Y* y, const Dim& dim) {
    y->device(place) = x->minimum(dim);
  }
};

//////// All Functor ///////
struct AllFunctor {
  template <typename DeviceContext, typename X, typename Y, typename Dim>
  void operator()(const DeviceContext& place, X* x, Y* y, const Dim& dim) {
    y->device(place) = x->all(dim);
  }
};

//////// Any Functor ///////
struct AnyFunctor {
  template <typename DeviceContext, typename X, typename Y, typename Dim>
  void operator()(const DeviceContext& place, X* x, Y* y, const Dim& dim) {
    y->device(place) = x->any(dim);
  }
};

struct MeanGradFunctor {
  template <typename DeviceContext,
            typename X,
            typename Y,
            typename DX,
            typename DY,
            typename Dim>
  void operator()(const DeviceContext& place,
                  X* x UNUSED,
                  Y* y UNUSED,
                  DX* dx,
                  DY* dy,
                  const Dim& dim,
                  int size) {
    dx->device(place) = dy->broadcast(dim) / dx->constant(size);
  }
};

struct SumGradFunctor {
  template <typename DeviceContext,
            typename X,
            typename Y,
            typename DX,
            typename DY,
            typename Dim>
  void operator()(const DeviceContext& place,
                  X* x UNUSED,
                  Y* y UNUSED,
                  DX* dx,
                  DY* dy,
                  const Dim& dim,
                  int size UNUSED) {
    dx->device(place) = dy->broadcast(dim);
  }
};

struct ProdGradFunctor {
  template <typename DeviceContext,
            typename X,
            typename Y,
            typename DX,
            typename DY,
            typename Dim>
  void operator()(const DeviceContext& place,
                  X* x,
                  Y* y,
                  DX* dx,
                  DY* dy,
                  const Dim& dim,
                  int size UNUSED) {
    dx->device(place) = dy->broadcast(dim) * y->broadcast(dim) * x->inverse();
  }
};

struct MaxOrMinGradFunctor {
  template <typename DeviceContext,
            typename X,
            typename Y,
            typename DX,
            typename DY,
            typename Dim>
  void operator()(const DeviceContext& place,
                  X* x,
                  Y* y,
                  DX* dx,
                  DY* dy,
                  const Dim& dim,
                  int size UNUSED) {
    auto equals = (*x) == y->broadcast(dim);
    auto ones = dx->constant(1);
    auto zeros = dx->constant(0);
    // If there are multiple minimum or maximum elements, the subgradient of
    // each is the set [0, 1], and we pass gradient to all of them here.
    dx->device(place) = dy->broadcast(dim).reshape(x->dimensions()) *
                        equals.select(ones, zeros);
  }
};

#define HANDLE_AXIS_DIM(BROADCAST_DIM, AXIS_DIM)                 \
  if (broadcast_dim_size == BROADCAST_DIM && rank == AXIS_DIM) { \
    AMaxOrAMinAxisIsListGradFunctor<DeviceContext,               \
                                    X,                           \
                                    Y,                           \
                                    DX,                          \
                                    DY,                          \
                                    Dim,                         \
                                    BROADCAST_DIM,               \
                                    AXIS_DIM>(                   \
        place, x, y, dx, dy, dim, axis_dim);                     \
  }

template <typename DeviceContext,
          typename X,
          typename Y,
          typename DX,
          typename DY,
          typename Dim,
          int R,
          int D>
void AMaxOrAMinAxisIsListGradFunctor(const DeviceContext& place,
                                     X* x,
                                     Y* y,
                                     DX* dx,
                                     DY* dy,
                                     const Dim& dim,
                                     const std::vector<int>& axis_dim) {
  // R is x->dimensions().size();
  // D is axis_dim->dimensions().size();
  auto axis = Eigen::array<int, D>();
  auto reshape_x = Eigen::array<int, R>();
  auto reshape_y = Eigen::array<int, R>();

  for (int i = 0; i < D; i++) axis[i] = axis_dim[i];
  for (int i = 0; i < R; i++) {
    reshape_x[i] = x->dimensions()[i];
    reshape_y[i] = y->dimensions()[i];
  }

  auto equals = (*x) == y->broadcast(dim);
  auto ones = dx->constant(1);
  auto zeros = dx->constant(0);
  auto mask = equals.select(ones, zeros);
  dx->device(place) =
      dy->broadcast(dim) * mask /
      mask.reshape(reshape_x).sum(axis).reshape(reshape_y).broadcast(dim);
}

struct AMaxOrAMinGradFunctor {
  template <typename DeviceContext,
            typename X,
            typename Y,
            typename DX,
            typename DY,
            typename Dim>
  void operator()(const DeviceContext& place,
                  X* x,
                  Y* y,
                  DX* dx,
                  DY* dy,
                  const Dim& dim,
                  int size) {
    auto equals = (*x) == y->broadcast(dim);
    auto ones = dx->constant(1);
    auto zeros = dx->constant(0);
    auto mask = equals.select(ones, zeros);

    // If there are multiple minimum or maximum elements,
    // we evenly distribute gradient between these equal values
    size_t x_numel = 1;
    for (size_t i = 0; i < x->dimensions().size(); i++)
      x_numel *= x->dimensions()[i];
    // reduce_all
    if (size == static_cast<int>(x_numel)) {
      auto equal_number = mask.sum()
                              .reshape(Eigen::array<int, 1>({{1}}))
                              .broadcast(Eigen::array<int, 1>({size}));
      dx->device(place) =
          dy->broadcast(dim).reshape(x->dimensions()) * mask / equal_number;
      return;
    }

    // compute forward reduce axis_dim by dim (which is broadcast_dim)
    std::vector<int> axis_dim;
    int broadcast_dim_size = static_cast<int>(dim.size());
    for (int i = 0; i < broadcast_dim_size; i++) {
      if (dim[i] > 1) {
        axis_dim.push_back(i);
      }
    }

    int rank = static_cast<int>(axis_dim.size());
    // axis is a int element
    if (rank == 1) {
      auto axis = Eigen::array<int, 1>({axis_dim[0]});
      dx->device(place) =
          dy->broadcast(dim) * mask /
          mask.sum(axis).reshape(dy->dimensions()).broadcast(dim);
      return;
    }
    // axis is list, HANDLE_AXIS_DIM(broadcast_dim_size, rank)
    HANDLE_AXIS_DIM(3, 2);
    HANDLE_AXIS_DIM(4, 2);
    HANDLE_AXIS_DIM(4, 3);
    // comments for accelerating compiling temporarily.
    // HANDLE_AXIS_DIM(5, 2);
    // HANDLE_AXIS_DIM(5, 3);
    // HANDLE_AXIS_DIM(5, 4);
    // HANDLE_AXIS_DIM(6, 2);
    // HANDLE_AXIS_DIM(6, 3);
    // HANDLE_AXIS_DIM(6, 4);
    // HANDLE_AXIS_DIM(6, 5);
  }
};

}  // namespace funcs
}  // namespace phi
