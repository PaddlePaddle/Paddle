/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include "paddle/fluid/operators/eigen/eigen_function.h"
#include "paddle/fluid/platform/complex128.h"
#include "paddle/fluid/platform/complex64.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

template <typename T, int Rank>
struct EigenSlice<Eigen::DefaultDevice, T, Rank> {
  using Array = Eigen::DSizes<Eigen::DenseIndex, Rank>;
  using InType = Eigen::TensorMap<
      Eigen::Tensor<const T, Rank, Eigen::RowMajor, Eigen::DenseIndex>>;
  using OutType = Eigen::TensorMap<
      Eigen::Tensor<T, Rank, Eigen::RowMajor, Eigen::DenseIndex>>;
  static void Eval(Eigen::DefaultDevice dev, OutType out, InType in,
                   const Array& offsets, const Array& extents) {
    out.device(dev) = in.slice(offsets, extents);
  }
};

template <typename T, int Rank>
struct EigenPad<Eigen::DefaultDevice, T, Rank> {
  using Array =
      std::array<std::pair<Eigen::DenseIndex, Eigen::DenseIndex>, Rank>;
  using InType = Eigen::TensorMap<
      Eigen::Tensor<const T, Rank, Eigen::RowMajor, Eigen::DenseIndex>>;
  using OutType = Eigen::TensorMap<
      Eigen::Tensor<T, Rank, Eigen::RowMajor, Eigen::DenseIndex>>;
  static void Eval(Eigen::DefaultDevice dev, OutType out, InType in,
                   const Array& padding, T padding_value) {
    out.device(dev) = in.pad(padding, padding_value);
  }
};

#define INSTANTIATION(FUNCTOR, T)                      \
  template struct FUNCTOR<Eigen::DefaultDevice, T, 1>; \
  template struct FUNCTOR<Eigen::DefaultDevice, T, 2>; \
  template struct FUNCTOR<Eigen::DefaultDevice, T, 3>; \
  template struct FUNCTOR<Eigen::DefaultDevice, T, 4>; \
  template struct FUNCTOR<Eigen::DefaultDevice, T, 5>; \
  template struct FUNCTOR<Eigen::DefaultDevice, T, 6>
INSTANTIATION(EigenSlice, float);
INSTANTIATION(EigenSlice, platform::float16);
INSTANTIATION(EigenSlice, double);
INSTANTIATION(EigenSlice, int);
INSTANTIATION(EigenSlice, int64_t);
INSTANTIATION(EigenSlice, platform::complex64);
INSTANTIATION(EigenSlice, platform::complex128);
INSTANTIATION(EigenPad, float);
INSTANTIATION(EigenPad, platform::float16);
INSTANTIATION(EigenPad, double);
INSTANTIATION(EigenPad, int);
INSTANTIATION(EigenPad, int64_t);
INSTANTIATION(EigenPad, platform::complex64);
INSTANTIATION(EigenPad, platform::complex128);
#undef INSTANTIATION

}  // namespace operators
}  // namespace paddle
