/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/funcs/eigen/extensions.h"

namespace phi {
namespace funcs {

template <typename T>
struct EigenGenericI0e<Eigen::DefaultDevice, T> {
  using InType = Eigen::TensorMap<
      Eigen::Tensor<const T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  using OutType =
      Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  static void Eval(const Eigen::DefaultDevice& dev,
                   OutType out,
                   const InType& in) {
    out.device(dev) = in.bessel_i0e();
  }
};

template <typename T>
struct EigenGenericI0eGrad<Eigen::DefaultDevice, T> {
  using InType = Eigen::TensorMap<
      Eigen::Tensor<const T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  using OutType =
      Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  static void Eval(const Eigen::DefaultDevice& dev,
                   OutType din,
                   const InType& in,
                   const InType& dout) {

  }
};

template <typename T>
struct EigenGenericI0<Eigen::DefaultDevice, T> {
  using InType = Eigen::TensorMap<
      Eigen::Tensor<const T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  using OutType =
      Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  static void Eval(const Eigen::DefaultDevice& dev,
                   OutType out,
                   const InType& in) {
          out.device(dev) = in.bessel_i0();
  }
};

template <typename T>
struct EigenGenericI0Grad<Eigen::DefaultDevice, T> {
  using InType = Eigen::TensorMap<
      Eigen::Tensor<const T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  using OutType =
      Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  static void Eval(const Eigen::DefaultDevice& dev,
                   OutType din,
                   const InType& in,
                   const InType& dout) {

  }
};

#define INSTANTIATION(FUNCTOR)                           \
  template struct FUNCTOR<Eigen::DefaultDevice, float>;  \
  template struct FUNCTOR<Eigen::DefaultDevice, double>
INSTANTIATION(EigenGenericI0e);
INSTANTIATION(EigenGenericI0eGrad);
INSTANTIATION(EigenGenericI0);
INSTANTIATION(EigenGenericI0Grad);
#undef INSTANTIATION

}  // namespace funcs
}  // namespace phi
