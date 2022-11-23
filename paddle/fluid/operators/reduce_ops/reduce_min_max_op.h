// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/reduce_ops/reduce_op.h"

namespace paddle {
namespace operators {

struct MaxFunctor {
  template <typename DeviceContext, typename X, typename Y, typename Dim>
  void operator()(const DeviceContext& place, X* x, Y* y, const Dim& dim) {
    y->device(place) = x->maximum(dim);
  }
};

struct MinFunctor {
  template <typename DeviceContext, typename X, typename Y, typename Dim>
  void operator()(const DeviceContext& place, X* x, Y* y, const Dim& dim) {
    y->device(place) = x->minimum(dim);
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
                  int size) {
    auto equals = (*x) == y->broadcast(dim);
    auto ones = dx->constant(1);
    auto zeros = dx->constant(0);
    // If there are multiple minimum or maximum elements, the subgradient of
    // each is the set [0, 1], and we pass gradient to all of them here.
    dx->device(place) = dy->broadcast(dim) * equals.select(ones, zeros);
  }
};

}  // namespace operators
}  // namespace paddle
