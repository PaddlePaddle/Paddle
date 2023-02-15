/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include <stdint.h>

#include "paddle/fluid/framework/tensor.h"
#include "paddle/phi/kernels/funcs/eigen/extensions.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace paddle {
namespace framework {

// EigenDim converts paddle::platform::DDim into Eigen::DSizes.
template <int D>
struct EigenDim {
  using Type = Eigen::DSizes<Eigen::DenseIndex, D>;

  static Type From(const DDim& dims) {
    PADDLE_ENFORCE_EQ(arity(dims),
                      D,
                      platform::errors::InvalidArgument(
                          "Input dimension size should be equal to %d, but "
                          "received dimension size is %d.",
                          arity(dims),
                          D));
    Type ret;
    for (int64_t d = 0; d < arity(dims); d++) {
      ret[d] = dims[d];
    }
    return ret;
  }
};

// Interpret paddle::platform::Tensor as EigenTensor and EigenConstTensor.
template <typename T,
          size_t D,
          int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
struct EigenTensor {
  // TODO(qijun) Now, default type in unaligned, and we will make a benchmark on
  // the speed of aligned and unaligned version in future.
  using Type = Eigen::TensorMap<Eigen::Tensor<T, D, MajorType, IndexType>>;

  using ConstType =
      Eigen::TensorMap<Eigen::Tensor<const T, D, MajorType, IndexType>>;

  static Type From(phi::DenseTensor& tensor, DDim dims) {  // NOLINT
    return Type(tensor.data<T>(), EigenDim<D>::From(dims));
  }

  static Type From(phi::DenseTensor& tensor) {  // NOLINT
    return From(tensor, tensor.dims());
  }  // NOLINT

  static ConstType From(const phi::DenseTensor& tensor, DDim dims) {
    return ConstType(tensor.data<T>(), EigenDim<D>::From(dims));
  }

  static ConstType From(const phi::DenseTensor& tensor) {
    return From(tensor, tensor.dims());
  }
};

template <typename T,
          int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
struct EigenMatrix : public EigenTensor<T, 2, MajorType, IndexType> {
  static typename EigenMatrix::Type Reshape(phi::DenseTensor& tensor,  // NOLINT
                                            int num_col_dims) {
    int rank = tensor.dims().size();
    PADDLE_ENFORCE_EQ((num_col_dims > 0 && num_col_dims < rank),
                      true,
                      platform::errors::InvalidArgument(
                          "Input dimension number(num_col_dims) must be "
                          "between 0 and %d, but received number is %d.",
                          rank,
                          num_col_dims));
    return EigenMatrix::From(tensor,
                             phi::flatten_to_2d(tensor.dims(), num_col_dims));
  }

  static typename EigenMatrix::ConstType Reshape(const phi::DenseTensor& tensor,
                                                 int num_col_dims) {
    int rank = tensor.dims().size();
    PADDLE_ENFORCE_EQ((num_col_dims > 0 && num_col_dims < rank),
                      true,
                      platform::errors::InvalidArgument(
                          "Input dimension number(num_col_dims) must be "
                          "between 0 and %d, but received number is %d.",
                          rank,
                          num_col_dims));
    return EigenMatrix::From(tensor,
                             phi::flatten_to_2d(tensor.dims(), num_col_dims));
  }
};

template <typename T,
          int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
struct EigenVector : public EigenTensor<T, 1, MajorType, IndexType> {
  // Flatten reshapes a phi::DenseTensor into an EigenVector.
  static typename EigenVector::Type Flatten(
      phi::DenseTensor& tensor) {  // NOLINT
    return EigenVector::From(tensor, {product(tensor.dims())});
  }

  static typename EigenVector::ConstType Flatten(
      const phi::DenseTensor& tensor) {  // NOLINT
    return EigenVector::From(tensor, {product(tensor.dims())});
  }
};

template <typename T,
          int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
struct EigenScalar {
  // Scalar tensor (implemented as a rank-0 tensor) of scalar type T.
  using Type = Eigen::TensorMap<
      Eigen::TensorFixedSize<T, Eigen::Sizes<>, MajorType, IndexType>>;
  using ConstType = Eigen::TensorMap<
      Eigen::TensorFixedSize<const T, Eigen::Sizes<>, MajorType, IndexType>>;

  static Type From(phi::DenseTensor& tensor) {  // NOLINT
    return Type(tensor.data<T>());
  }

  static ConstType From(const phi::DenseTensor& tensor) {
    return ConstType(tensor.data<T>());
  }
};

// Define phi::DenseTensor with 32-bit index.
template <typename T, int D, int MajorType = Eigen::RowMajor>
using Tensor32BitIndex =
    Eigen::TensorMap<Eigen::Tensor<T, D, MajorType, int>, Eigen::Aligned>;

template <typename DSizes>
Eigen::DSizes<int, DSizes::count> To32BitDims(const DSizes& in) {
  Eigen::DSizes<int, DSizes::count> out;
  for (int i = 0; i < DSizes::count; ++i) {
    out[i] = in[i];
  }
  return out;
}

template <typename EigenTensor>
Tensor32BitIndex<typename EigenTensor::Scalar, EigenTensor::NumIndices>
To32BitIndex(EigenTensor in) {
  using RetType =
      Tensor32BitIndex<typename EigenTensor::Scalar, EigenTensor::NumIndices>;
  return RetType(in.data(), To32BitDims(in.dimensions()));
}

}  // namespace framework
}  // namespace paddle
