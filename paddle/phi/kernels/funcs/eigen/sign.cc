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
#include "paddle/phi/common/complex.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"

namespace phi::funcs {

template <typename T>
struct EigenSign<Eigen::DefaultDevice, T> {
  using InType = Eigen::TensorMap<
      Eigen::Tensor<const T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  using OutType =
      Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  static void Eval(const Eigen::DefaultDevice& dev,
                   OutType out,
                   const InType& in) {
    out.device(dev) = in.sign();
  }
};

template <typename T>
struct EigenSign<Eigen::DefaultDevice, phi::dtype::complex<T>> {
  using InType = Eigen::TensorMap<Eigen::Tensor<const phi::dtype::complex<T>,
                                                1,
                                                Eigen::RowMajor,
                                                Eigen::DenseIndex>>;
  using OutType = Eigen::TensorMap<Eigen::Tensor<phi::dtype::complex<T>,
                                                 1,
                                                 Eigen::RowMajor,
                                                 Eigen::DenseIndex>>;
  static void Eval(const Eigen::DefaultDevice& dev,
                   OutType out,
                   const InType& in) {
    out.device(dev) = in.unaryExpr(
        [](const phi::dtype::complex<T>& z) -> phi::dtype::complex<T> {
          T abs_val = abs(z);
          if (abs_val == 0) {
            return phi::dtype::complex<T>(0, 0);
          } else {
            return phi::dtype::complex<T>(z.real / abs_val, z.imag / abs_val);
          }
        });
  }
};

template struct EigenSign<Eigen::DefaultDevice, uint8_t>;
template struct EigenSign<Eigen::DefaultDevice, int8_t>;
template struct EigenSign<Eigen::DefaultDevice, int16_t>;
template struct EigenSign<Eigen::DefaultDevice, int32_t>;
template struct EigenSign<Eigen::DefaultDevice, int64_t>;
template struct EigenSign<Eigen::DefaultDevice, float>;
template struct EigenSign<Eigen::DefaultDevice, double>;
template struct EigenSign<Eigen::DefaultDevice, phi::dtype::complex<float>>;
template struct EigenSign<Eigen::DefaultDevice, phi::dtype::complex<double>>;

}  // namespace phi::funcs
