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
  typedef Eigen::DSizes<Eigen::DenseIndex, D> Type;

  static Type From(const DDim& dims) {
    PADDLE_ENFORCE(arity(dims) == D, "D must match arity(DDim)");
    Type ret;
    for (int d = 0; d < rank; d++) {
      ret[d] = dims[d];
    }
    return ret;
  }
};

// Interpret paddle::platform::Tensor as EigenTensor and EigenConstTensor.
template <typename T, size_t D, typename IndexType = Eigen::DenseIndex>
struct EigenTensor {
  using Type = Eigen::TensorMap<Eigen::Tensor<T, D, Eigen::RowMajor, IndexType>,
                                Eigen::Aligned>;

  using ConstType =
      Eigen::TensorMap<Eigen::Tensor<const T, D, Eigen::RowMajor, IndexType>,
                       Eigen::Aligned>
          ConstTensor;

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

// Interpret paddle::platform::Tensor as EigenVecotr and EigenConstVector.
template <typename T, typename IndexType = Eigen::DenseIndex>
struct EigenVector {
  using EigenVector =
      Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, IndexType>,
                       Eigen::Aligned>;

  using EigenConstVector =
      Eigen::TensorMap<Eigen::Tensor<const T, 1, Eigen::RowMajor, IndexType>,
                       Eigen::Aligned>;

  static Type From(Tensor& tensor) { return EigenTensor<T, 1>::From(tensor); }

  static ConstType From(const Tensor& tensor) {
    return EigenTensor<T, 1>::From(tensor);
  }
};

// Interpret paddle::platform::Tensor as EigenMatrix and EigenConstMatrix.
template <typename T, typename IndexType = Eigen::DenseIndex>
struct EigenMatrix {
  template <typename T, typename IndexType = Eigen::DenseIndex>
  using EigenMatrix =
      Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor, IndexType>,
                       Eigen::Aligned>;

  template <typename T, typename IndexType = Eigen::DenseIndex>
  using EigenConstMatrix =
      Eigen::TensorMap<Eigen::Tensor<const T, 2, Eigen::RowMajor, IndexType>,
                       Eigen::Aligned>;

  static Type From(Tensor& tensor) { return EigenTensor<T, 2>::From(tensor); }

  static ConstType From(const Tensor& tensor) {
    return EigenTensor<T, 2>::From(tensor);
  }
};

}  // namespace framework
}  // namespace paddle
