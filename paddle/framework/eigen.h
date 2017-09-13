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

#include "paddle/framework/ddim.h"
#include "paddle/framework/tensor.h"
#include "paddle/platform/variant.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace paddle {
namespace framework {

template <int arity>
using EigenDim = Eigen::DSizes<Eigen::DenseIndex, arity>;

using EigenDDim = boost::variant<EigenDim<1>, EigenDim<2>, EigenDim<3>,
                                 EigenDim<4>, EigenDim<5>, EigenDim<6>,
                                 EigenDim<7>, EigenDim<8>, EigenDim<9>>;

struct EigenDDimConvertVisitor : public boost::static_visitor<EigenDDim> {
  template <typename DimType>
  EigenDDim operator()(const DimType& dims) const {
    constexpr int arity = DimType::dimensions;
    Eigen::DSizes<Eigen::DenseIndex, arity> ret;
    for (int64_t d = 0; d < arity; ++d) {
      ret[d] = dims[d];
    }
    return ret;
  }
};

inline EigenDDim DDimToEigenDDim(const DDim& dims) {
  return boost::apply_visitor(EigenDDimConvertVisitor(), dims);
}

template <typename Visitor>
inline auto VisitEigenDDim(Visitor visitor, const EigenDDim& ddim) ->
    typename Visitor::result_type {
  return boost::apply_visitor(visitor, ddim);
}

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
    return Type(tensor.data<T>(),
                boost::get<EigenDim<D>>(DDimToEigenDDim(dims)));
  }

  static Type From(Tensor& tensor) { return From(tensor, tensor.dims_); }

  static ConstType From(const Tensor& tensor, DDim dims) {
    return ConstType(tensor.data<T>(),
                     boost::get<EigenDim<D>>(DDimToEigenDDim(dims)));
  }

  static ConstType From(const Tensor& tensor) {
    return From(tensor, tensor.dims_);
  }
};

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
struct EigenMatrix : public EigenTensor<T, 2, MajorType, IndexType> {
  static typename EigenMatrix::Type Reshape(Tensor& tensor, int num_col_dims) {
    int rank = tensor.dims_.size();
    PADDLE_ENFORCE(num_col_dims > 0 && num_col_dims < rank,
                   "`num_col_dims` must be between (0, rank_of_tensor).");
    return EigenMatrix::From(tensor,
                             flatten_to_2d(tensor.dims(), num_col_dims));
  }

  static typename EigenMatrix::ConstType Reshape(const Tensor& tensor,
                                                 int num_col_dims) {
    int rank = tensor.dims_.size();
    PADDLE_ENFORCE(num_col_dims > 0 && num_col_dims < rank,
                   "`num_col_dims` must be between (0, rank_of_tensor).");
    return EigenMatrix::From(tensor,
                             flatten_to_2d(tensor.dims(), num_col_dims));
  }
};

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
struct EigenVector : public EigenTensor<T, 1, MajorType, IndexType> {
  // Flatten reshapes a Tensor into an EigenVector.
  static typename EigenVector::Type Flatten(Tensor& tensor) {
    return EigenVector::From(tensor, {product(tensor.dims_)});
  }

  static typename EigenVector::ConstType Flatten(const Tensor& tensor) {
    return EigenVector::From(tensor, {product(tensor.dims_)});
  }
};

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
struct EigenScalar {
  // Scalar tensor (implemented as a rank-0 tensor) of scalar type T.
  using Type = Eigen::TensorMap<
      Eigen::TensorFixedSize<T, Eigen::Sizes<>, MajorType, IndexType>>;
  using ConstType = Eigen::TensorMap<
      Eigen::TensorFixedSize<const T, Eigen::Sizes<>, MajorType, IndexType>>;

  static Type From(Tensor& tensor) { return Type(tensor.data<T>()); }

  static ConstType From(const Tensor& tensor) {
    return ConstType(tensor.data<T>());
  }
};

}  // namespace framework
}  // namespace paddle
