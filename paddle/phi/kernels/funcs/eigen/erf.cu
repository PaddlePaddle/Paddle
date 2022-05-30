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
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include "paddle/phi/common/float16.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/funcs/eigen/extensions.h"

namespace phi {
namespace funcs {

template <typename T>
struct EigenErf<Eigen::GpuDevice, T> {
  using InType = Eigen::TensorMap<
      Eigen::Tensor<const T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  using OutType =
      Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  static void Eval(const Eigen::GpuDevice& dev, OutType out, const InType& in) {
    out.device(dev) = in.erf();
  }
};

template <typename T>
struct EigenErfGrad<Eigen::GpuDevice, T> {
  using InType = Eigen::TensorMap<
      Eigen::Tensor<const T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  using OutType =
      Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  static void Eval(const Eigen::GpuDevice& dev,
                   OutType din,
                   const InType& in,
                   const InType& dout) {
    din.device(dev) =
        dout * static_cast<T>(M_2_SQRTPI) * (-(in.square())).exp();
  }
};

#define INSTANTIATION(FUNCTOR)                       \
  template struct FUNCTOR<Eigen::GpuDevice, float>;  \
  template struct FUNCTOR<Eigen::GpuDevice, double>; \
  template struct FUNCTOR<Eigen::GpuDevice, dtype::float16>
INSTANTIATION(EigenErf);
INSTANTIATION(EigenErfGrad);
#undef INSTANTIATION

}  // namespace funcs
}  // namespace phi
