/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/framework/tensor.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace paddle {
namespace framework {

// EigenDim converts paddle::platform::DDim into Eigen::DSizes.
template <int D>
struct EigenDim {
  using Type = Eigen::DSizes<Eigen::DenseIndex, D>;

  static Type From(const DDim& dims) {
    Type ret;
    if (arity(dims) == 1 && dims[0] == 1) {
      ret[0] = 0;
    } else {
      PADDLE_ENFORCE(arity(dims) == D, "D must match arity(DDim)");
      for (int d = 0; d < arity(dims); d++) {
        ret[d] = dims[d];
      }
    }
    return ret;
  }
};

// Interpret paddle::platform::Tensor as EigenTensor and EigenConstTensor.
template <typename T, size_t D, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
struct EigenTensor {
  // TODO(qijun) Now, default type in unaligned, and we will make a benchmark on
  // the speed of aligned and unaligned version in future.
  using Type = Eigen::TensorMap<Eigen::Tensor<T, D, MajorType, IndexType>>;

  using ConstType =
      Eigen::TensorMap<Eigen::Tensor<const T, D, MajorType, IndexType>>;

  static Type From(Tensor& tensor, DDim dims) {
    return Type(tensor.data<T>(), EigenDim<D>::From(dims));
  }

  static Type From(Tensor& tensor) { return From(tensor, tensor.dims_); }

  static ConstType From(const Tensor& tensor, DDim dims) {
    return ConstType(tensor.data<T>(), EigenDim<D>::From(dims));
  }

  static ConstType From(const Tensor& tensor) {
    return From(tensor, tensor.dims_);
  }
};

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
struct EigenMatrix : public EigenTensor<T, 2, MajorType, IndexType> {};

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
struct EigenVector : public EigenTensor<T, 1, MajorType, IndexType> {
  // Flatten reshapes a Tensor into an EigenVector.
  static typename EigenVector::Type Flatten(Tensor& tensor) {
    return EigenVector::From(
        tensor, make_ddim({static_cast<int>(product(tensor.dims_))}));
  }

  static typename EigenVector::ConstType Flatten(const Tensor& tensor) {
    return EigenVector::From(
        tensor, make_ddim({static_cast<int>(product(tensor.dims_))}));
  }
};

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
struct EigenScalar : public EigenTensor<T, 0, MajorType, IndexType> {};

}  // namespace framework
}  // namespace paddle
