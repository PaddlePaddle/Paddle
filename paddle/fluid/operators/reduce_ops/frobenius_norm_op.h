// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <vector>

#include "paddle/fluid/operators/reduce_ops/reduce_op.h"

namespace paddle {
namespace operators {

// \partial \| X \|_F = \frac{X}{ \| X \|_F }
template <typename DeviceContext, typename T, typename Functor>
class FrobeniusNormGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    // default use Eigen broadcast
    ReduceGradKernel<DeviceContext, T, Functor, false> kernel;
    kernel.Compute(context);
  }
};

struct FrobeniusNormFunctor {
  template <typename DeviceContext, typename X, typename Y, typename Dim>
  void operator()(const DeviceContext& place, X* x, Y* y, const Dim& dim) {
    y->device(place) = ((x->square()).sum(dim)).sqrt();
  }
};

struct FrobeniusNormGradFunctor {
  template <typename DeviceContext, typename X, typename Y, typename DX,
            typename DY, typename Dim>
  void operator()(const DeviceContext& place, X* x, Y* y, DX* dx, DY* dy,
                  const Dim& dim, int size) {
    dx->device(place) = y->broadcast(dim);
    dx->device(place) = *dx + dx->constant(1e-12f);
    dx->device(place) = (*x / *dx) * (dy->broadcast(dim));
  }
};

}  // namespace operators
}  // namespace paddle
