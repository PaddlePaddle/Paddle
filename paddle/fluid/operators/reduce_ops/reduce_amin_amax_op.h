// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/framework/eigen.h"

namespace paddle {
namespace operators {

struct AmaxOrAminGradFunctor {
  template <typename DeviceContext, typename X, typename Y, typename DX,
            typename DY, typename Dim, size_t D>
  void operator()(const DeviceContext& place, X* x, Y* y, DX* dx, DY* dy,
                  const Dim& dim, int size, const std::array<int, D> dims) {
    auto equals = (*x) == y->broadcast(dim);
    auto ones = dx->constant(1);
    auto zeros = dx->constant(0);

    auto number = equals.template cast<typename DY::Scalar>().sum(dims).reshape(dy->dimensions()).broadcast(dim);
    dx->device(place) = dy->broadcast(dim) * equals.select(ones, zeros) / number;
    // If there are multiple minimum or maximum elements, the subgradient of
    // each is the set [0, 1], and we pass gradient to all of them here.
    // auto x_shape = x->dimensions();
    // auto y_shape = y->dimensions();
    // std::vector<int> dims;
    // for(int i = 0; i < static_cast <int>(x_shape.size()); i++){
    //   if(x_shape[i] != y_shape[i]){
    //     dims.push_back(i);
    //   }
    // }

    // int rank = dims.size();
    // switch (rank) {
        // case 1:
        //   {std::array<int, 1> array_dims;
        //   std::copy(dims.begin(), dims.end(), array_dims.begin());
        //   auto number = equals.template cast<typename DY::Scalar>().sum(array_dims).reshape(dy->dimensions()).broadcast(dim);
        //   dx->device(place) = dy->broadcast(dim) * equals.select(ones, zeros) / number;}
        //   break;
        // case 2:
        //   {std::array<int, 2> array_dims;
        //   std::copy(dims.begin(), dims.end(), array_dims.begin());
        //   auto number = equals.template cast<typename DY::Scalar>().sum(array_dims).reshape(dy->dimensions()).broadcast(dim);
        //   dx->device(place) = dy->broadcast(dim) * equals.select(ones, zeros) / number;}
        //   break;
        // case 3:
        //   {std::array<int, 3> array_dims;
        //   std::copy(dims.begin(), dims.end(), array_dims.begin());
        //   auto number = equals.template cast<typename DY::Scalar>().sum(array_dims).reshape(dy->dimensions()).broadcast(dim);
        //   dx->device(place) = dy->broadcast(dim) * equals.select(ones, zeros) / number;}
        //   break;
        // case 4:
        //   {std::array<int, 4> array_dims;
        //   std::copy(dims.begin(), dims.end(), array_dims.begin());
        //   auto number = equals.template cast<typename DY::Scalar>().sum(array_dims).reshape(dy->dimensions()).broadcast(dim);
        //   dx->device(place) = dy->broadcast(dim) * equals.select(ones, zeros) / number;}
        //   break;
        // case 5:
        //   {std::array<int, 5> array_dims;
        //   std::copy(dims.begin(), dims.end(), array_dims.begin());
        //   auto number = equals.template cast<typename DY::Scalar>().sum(array_dims).reshape(dy->dimensions()).broadcast(dim);
        //   dx->device(place) = dy->broadcast(dim) * equals.select(ones, zeros) / number;}
        //   break;
        // case 6:
        //   {std::array<int, 6> array_dims;
        //   std::copy(dims.begin(), dims.end(), array_dims.begin());
        //   auto number = equals.template cast<typename DY::Scalar>().sum(array_dims).reshape(dy->dimensions()).broadcast(dim);
        //   dx->device(place) = dy->broadcast(dim) * equals.select(ones, zeros) / number;}
        //   break;
        // case 7:
        //   {std::array<int, 7> array_dims;
        //   std::copy(dims.begin(), dims.end(), array_dims.begin());
        //   auto number = equals.template cast<typename DY::Scalar>().sum(array_dims).reshape(dy->dimensions()).broadcast(dim);
        //   dx->device(place) = dy->broadcast(dim) * equals.select(ones, zeros) / number;}
        //   break;
        // case 8:
        //   {std::array<int, 8> array_dims;
        //   std::copy(dims.begin(), dims.end(), array_dims.begin());
        //   auto number = equals.template cast<typename DY::Scalar>().sum(array_dims).reshape(dy->dimensions()).broadcast(dim);
        //   dx->device(place) = dy->broadcast(dim) * equals.select(ones, zeros) / number;}
        //   break;
        // case 9:
        //   {std::array<int, 9> array_dims;
        //   std::copy(dims.begin(), dims.end(), array_dims.begin());
        //   auto number = equals.template cast<typename DY::Scalar>().sum(array_dims).reshape(dy->dimensions()).broadcast(dim);
        //   dx->device(place) = dy->broadcast(dim) * equals.select(ones, zeros) / number;}
        //   break;
        // case 10:
        //   {std::array<int, 10> array_dims;
        //   std::copy(dims.begin(), dims.end(), array_dims.begin());
        //   auto number = equals.template cast<typename DY::Scalar>().sum(array_dims).reshape(dy->dimensions()).broadcast(dim);
        //   dx->device(place) = dy->broadcast(dim) * equals.select(ones, zeros) / number;}
        //   break;
    // }
  }
};

}  // namespace operators
}  // namespace paddle