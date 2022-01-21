/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include "paddle/pten/kernels/funcs/eigen/eigen_function.h"

namespace pten {
namespace funcs {

template <typename T>
struct EigenL1Norm<Eigen::DefaultDevice, T> {
  using InType = Eigen::TensorMap<
      Eigen::Tensor<const T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  using OutType = Eigen::TensorMap<Eigen::TensorFixedSize<T,
                                                          Eigen::Sizes<>,
                                                          Eigen::RowMajor,
                                                          Eigen::DenseIndex>>;
  static void Eval(const Eigen::DefaultDevice& dev,
                   OutType out,
                   const InType& in) {
    out.device(dev) = in.abs().sum();
  }
};

template <typename T>
struct EigenL1NormGrad<Eigen::DefaultDevice, T> {
  using Array = Eigen::DSizes<Eigen::DenseIndex, 1>;
  using InType = Eigen::TensorMap<
      Eigen::Tensor<const T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  using OutType =
      Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  static void Eval(const Eigen::DefaultDevice& dev,
                   OutType din,
                   const InType& dout,
                   const InType& in,
                   const Array& bcast) {
    din.device(dev) = dout.broadcast(bcast) * in.sign();
  }
};

template struct EigenL1Norm<Eigen::DefaultDevice, float>;
template struct EigenL1NormGrad<Eigen::DefaultDevice, float>;

}  // namespace funcs
}  // namespace pten
