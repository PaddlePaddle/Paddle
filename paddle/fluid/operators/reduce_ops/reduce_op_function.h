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
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DDim = framework::DDim;
template <typename T, size_t D, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenTensor = framework::EigenTensor<T, D, MajorType, IndexType>;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenScalar = framework::EigenScalar<T, MajorType, IndexType>;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

template <typename DeviceContext, typename T, size_t D, size_t R_D,
          typename Functor>
void ReduceFunctor(const DeviceContext& context, const framework::Tensor& input,
                   framework::Tensor* output, const std::vector<int>& dims,
                   bool keep_dim) {
  auto x = EigenTensor<T, D>::From(input);
  auto x_rank = static_cast<int>(x.dimensions().size());
  auto reduce_dim = Eigen::array<int, R_D>();
  std::vector<int> dims_ref = dims;
  for (size_t i = 0; i < dims_ref.size(); ++i) {
    if (dims_ref[i] < 0) dims_ref[i] = x_rank + dims_ref[i];
    reduce_dim[i] = dims_ref[i];
  }
  // construct the squeezed output tensor
  DDim out_dims = output->dims();
  if (keep_dim && x_rank > 1) {
    const int kDelFlag = -2;
    auto dims_vector = framework::vectorize(out_dims);
    for (size_t i = 0; i < dims_ref.size(); ++i) {
      dims_vector[dims_ref[i]] = kDelFlag;
    }
    dims_vector.erase(remove(dims_vector.begin(), dims_vector.end(), kDelFlag),
                      dims_vector.end());
    out_dims = framework::make_ddim(dims_vector);
  }
  auto& place = *context.eigen_device();
  Functor functor;

  if (D == 1) {
    auto out = EigenScalar<T>::From(*output);
    functor(place, &x, &out, reduce_dim);
  } else {
    auto out = EigenTensor<T, (D - R_D)>::From(*output, out_dims);
    functor(place, &x, &out, reduce_dim);
  }
}

template <typename DeviceContext, typename T, size_t D, typename Functor>
void ReduceGradFunctor(const DeviceContext& context,
                       const framework::Tensor& input0,
                       const framework::Tensor& input1,
                       const framework::Tensor& input2,
                       framework::Tensor* output, Functor functor,
                       const std::vector<int>& dims) {
  auto x = EigenTensor<T, D>::From(input0);
  auto x_grad = EigenTensor<T, D>::From(*output);
  auto x_rank = static_cast<int>(x.dimensions().size());
  auto x_dims = input0.dims();
  auto reduced_dims_v = framework::vectorize(x_dims);
  std::vector<int> dims_ref = dims;
  Eigen::array<int, D> broadcast_dim;
  for (size_t i = 0; i < D; ++i) broadcast_dim[i] = 1;

  int broad_cats_times = 1;
  for (size_t i = 0; i < dims_ref.size(); ++i) {
    if (dims_ref[i] < 0) {
      dims_ref[i] = x_rank + dims_ref[i];
    }
    reduced_dims_v[dims_ref[i]] = 1;
    broadcast_dim[dims_ref[i]] = x_dims[dims_ref[i]];
    broad_cats_times *= x_dims[dims_ref[i]];
  }
  auto reduced_dims = framework::make_ddim(reduced_dims_v);
  auto x_reduce = EigenTensor<T, D>::From(input1, reduced_dims);
  auto x_reduce_grad = EigenTensor<T, D>::From(input2, reduced_dims);

  auto& place = *context.eigen_device();

  functor(place, &x, &x_reduce, &x_grad, &x_reduce_grad, broadcast_dim,
          broad_cats_times);
}

}  // namespace operators
}  // namespace paddle
