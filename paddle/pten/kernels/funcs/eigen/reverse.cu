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

template <typename T, int Rank>
struct EigenReverse<Eigen::GpuDevice, T, Rank> {
  using Array = Eigen::DSizes<bool, Rank>;
  using InType = Eigen::TensorMap<
      Eigen::Tensor<const T, Rank, Eigen::RowMajor, Eigen::DenseIndex>>;
  using OutType = Eigen::TensorMap<
      Eigen::Tensor<T, Rank, Eigen::RowMajor, Eigen::DenseIndex>>;
  static void Eval(const Eigen::GpuDevice& dev,
                   OutType out,
                   const InType& in,
                   const Array& reverse) {
    out.device(dev) = in.reverse(reverse);
  }
};

#define INSTANTIATION(FUNCTOR, TYPE)                  \
  template struct FUNCTOR<Eigen::GpuDevice, TYPE, 1>; \
  template struct FUNCTOR<Eigen::GpuDevice, TYPE, 2>; \
  template struct FUNCTOR<Eigen::GpuDevice, TYPE, 3>; \
  template struct FUNCTOR<Eigen::GpuDevice, TYPE, 4>; \
  template struct FUNCTOR<Eigen::GpuDevice, TYPE, 5>; \
  template struct FUNCTOR<Eigen::GpuDevice, TYPE, 6>
INSTANTIATION(EigenReverse, int);
INSTANTIATION(EigenReverse, uint8_t);
INSTANTIATION(EigenReverse, int64_t);
INSTANTIATION(EigenReverse, bool);
INSTANTIATION(EigenReverse, float);
INSTANTIATION(EigenReverse, double);
#undef INSTANTIATION

}  // namespace funcs
}  // namespace pten
