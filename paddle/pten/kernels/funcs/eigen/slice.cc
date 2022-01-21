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
#include "paddle/pten/common/bfloat16.h"
#include "paddle/pten/common/complex.h"
#include "paddle/pten/common/float16.h"
#include "paddle/pten/kernels/funcs/eigen/eigen_function.h"

namespace pten {
namespace funcs {

template <typename T, int Rank>
struct EigenSlice<Eigen::DefaultDevice, T, Rank> {
  using Array = Eigen::DSizes<Eigen::DenseIndex, Rank>;
  using Array32Bit = Eigen::DSizes<int, Rank>;
  using InType = Eigen::TensorMap<
      Eigen::Tensor<const T, Rank, Eigen::RowMajor, Eigen::DenseIndex>>;
  using InType32BitIndex =
      Eigen::TensorMap<Eigen::Tensor<const T, Rank, Eigen::RowMajor, int>,
                       Eigen::Aligned>;
  using OutType = Eigen::TensorMap<
      Eigen::Tensor<T, Rank, Eigen::RowMajor, Eigen::DenseIndex>>;
  using OutType32BitIndex =
      Eigen::TensorMap<Eigen::Tensor<T, Rank, Eigen::RowMajor, int>,
                       Eigen::Aligned>;

  static void Eval(const Eigen::DefaultDevice& dev,
                   OutType out,
                   const InType& in,
                   const Array& offsets,
                   const Array& extents) {
    out.device(dev) = in.slice(offsets, extents);
  }

  static void Eval(const Eigen::DefaultDevice& dev,
                   OutType32BitIndex out,
                   const InType32BitIndex& in,
                   const Array32Bit& offsets,
                   const Array32Bit& extents) {
    out.device(dev) = in.slice(offsets, extents);
  }
};

#define INSTANTIATION(FUNCTOR, TYPE)                      \
  template struct FUNCTOR<Eigen::DefaultDevice, TYPE, 1>; \
  template struct FUNCTOR<Eigen::DefaultDevice, TYPE, 2>; \
  template struct FUNCTOR<Eigen::DefaultDevice, TYPE, 3>; \
  template struct FUNCTOR<Eigen::DefaultDevice, TYPE, 4>; \
  template struct FUNCTOR<Eigen::DefaultDevice, TYPE, 5>; \
  template struct FUNCTOR<Eigen::DefaultDevice, TYPE, 6>; \
  template struct FUNCTOR<Eigen::DefaultDevice, TYPE, 7>; \
  template struct FUNCTOR<Eigen::DefaultDevice, TYPE, 8>; \
  template struct FUNCTOR<Eigen::DefaultDevice, TYPE, 9>
INSTANTIATION(EigenSlice, bool);
INSTANTIATION(EigenSlice, int);
INSTANTIATION(EigenSlice, int8_t);
INSTANTIATION(EigenSlice, uint8_t);
INSTANTIATION(EigenSlice, int16_t);
INSTANTIATION(EigenSlice, int64_t);
INSTANTIATION(EigenSlice, float);
INSTANTIATION(EigenSlice, double);
INSTANTIATION(EigenSlice, dtype::float16);
INSTANTIATION(EigenSlice, dtype::bfloat16);
INSTANTIATION(EigenSlice, dtype::complex<float>);
INSTANTIATION(EigenSlice, dtype::complex<double>);
#undef INSTANTIATION

}  // namespace funcs
}  // namespace pten
