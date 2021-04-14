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
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

template <typename T, int Rank>
struct EigenBroadcast<Eigen::DefaultDevice, T, Rank> {
  using Array = Eigen::DSizes<Eigen::DenseIndex, Rank>;
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

  static void Eval(const Eigen::DefaultDevice& dev, OutType out, InType in,
                   const Array& bcast) {
    out.device(dev) = in.broadcast(bcast);
  }

  static void Eval(const Eigen::DefaultDevice& dev, OutType32BitIndex out,
                   InType32BitIndex in, const Array& bcast) {
    out.device(dev) = in.broadcast(bcast);
  }
};

template <typename T, int Rank>
struct EigenBroadcastGrad<Eigen::DefaultDevice, T, Rank> {
  using Array = Eigen::DSizes<Eigen::DenseIndex, Rank>;
  using Array2 = Eigen::DSizes<Eigen::DenseIndex, Rank * 2>;
  using InType = Eigen::TensorMap<
      Eigen::Tensor<const T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  using OutType =
      Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  static void Eval(const Eigen::DefaultDevice& dev, OutType out, InType in,
                   const Array& reduce_dims, const Array2& reshape_dims) {
    out.device(dev) =
        in.reshape(reshape_dims).sum(reduce_dims).reshape(out.dimensions());
  }
};

#define INSTANTIATION(FUNCTOR, T)                      \
  template struct FUNCTOR<Eigen::DefaultDevice, T, 1>; \
  template struct FUNCTOR<Eigen::DefaultDevice, T, 2>; \
  template struct FUNCTOR<Eigen::DefaultDevice, T, 3>; \
  template struct FUNCTOR<Eigen::DefaultDevice, T, 4>; \
  template struct FUNCTOR<Eigen::DefaultDevice, T, 5>; \
  template struct FUNCTOR<Eigen::DefaultDevice, T, 6>
INSTANTIATION(EigenBroadcast, bool);
INSTANTIATION(EigenBroadcast, platform::float16);
INSTANTIATION(EigenBroadcast, float);
INSTANTIATION(EigenBroadcast, double);
INSTANTIATION(EigenBroadcast, int);
INSTANTIATION(EigenBroadcast, int64_t);
INSTANTIATION(EigenBroadcastGrad, bool);
INSTANTIATION(EigenBroadcastGrad, float);
INSTANTIATION(EigenBroadcastGrad, platform::float16);
INSTANTIATION(EigenBroadcastGrad, double);
INSTANTIATION(EigenBroadcastGrad, int);
INSTANTIATION(EigenBroadcastGrad, int64_t);
template struct EigenBroadcastGrad<Eigen::DefaultDevice, float, 0>;
template struct EigenBroadcastGrad<Eigen::DefaultDevice, double, 0>;
template struct EigenBroadcastGrad<Eigen::DefaultDevice, int, 0>;
template struct EigenBroadcastGrad<Eigen::DefaultDevice, int64_t, 0>;
#undef INSTANTIATION

}  // namespace operators
}  // namespace paddle
