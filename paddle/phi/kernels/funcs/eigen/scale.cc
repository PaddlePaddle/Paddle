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
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"

namespace phi {
namespace funcs {

template <typename T>
struct EigenScale<Eigen::DefaultDevice, T> {
  using InType = Eigen::TensorMap<
      Eigen::Tensor<const T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  using OutType =
      Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  static void Eval(const Eigen::DefaultDevice& dev,
                   OutType out,
                   const InType& in,
                   const T scale,
                   const T bias,
                   const bool bias_after_scale) {
    if (bias_after_scale) {
      out.device(dev) = scale * in + bias;
    } else {
      out.device(dev) = scale * (in + bias);
    }
  }
};

template struct EigenScale<Eigen::DefaultDevice, float>;
template struct EigenScale<Eigen::DefaultDevice, double>;
template struct EigenScale<Eigen::DefaultDevice, dtype::bfloat16>;
template struct EigenScale<Eigen::DefaultDevice, dtype::float16>;
template struct EigenScale<Eigen::DefaultDevice, uint8_t>;
template struct EigenScale<Eigen::DefaultDevice, int8_t>;
template struct EigenScale<Eigen::DefaultDevice, int16_t>;
template struct EigenScale<Eigen::DefaultDevice, int>;
template struct EigenScale<Eigen::DefaultDevice, int64_t>;
template struct EigenScale<Eigen::DefaultDevice, dtype::complex<float>>;
template struct EigenScale<Eigen::DefaultDevice, dtype::complex<double>>;

}  // namespace funcs
}  // namespace phi
