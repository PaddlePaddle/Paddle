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

#include "paddle/fluid/operators/reduce_ops/reduce_op.h"

namespace paddle {
namespace operators {

struct LogsumexpFunctor {
  template <typename DeviceContext, typename X, typename Y, typename Dim>
  void operator()(const DeviceContext& place, X* x, Y* y, const Dim& dim) {
    auto x_dim = x->dimensions();
    auto t_dim = x_dim;
    for (int i = 0; i < static_cast<int>(dim.size()); i++) {
      t_dim[dim[i]] = 1;
    }

    auto r_dim = x_dim;
    for (int i = 0; i < static_cast<int>(r_dim.size()); i++) {
      r_dim[i] = 1;
    }
    for (int i = 0; i < static_cast<int>(dim.size()); i++) {
      r_dim[dim[i]] = x_dim[dim[i]];
    }

    auto y_dim = y->dimensions();
    auto x_max = x->maximum(dim);
    y->device(place) =
        (x_max +
         (*x - x_max.reshape(t_dim).broadcast(r_dim)).exp().sum(dim).log())
            .reshape(y_dim);
  }
};

struct LogsumexpGradFunctor {
  template <typename DeviceContext, typename X, typename Y, typename DX,
            typename DY, typename Dim>
  void operator()(const DeviceContext& place, X* x, Y* y, DX* dx, DY* dy,
                  const Dim& dim, int size) {
    dx->device(place) = dy->broadcast(dim) * (*x - y->broadcast(dim)).exp();
  }
};

}  // namespace operators
}  // namespace paddle
