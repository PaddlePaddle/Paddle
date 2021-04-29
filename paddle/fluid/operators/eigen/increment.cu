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

namespace paddle {
namespace operators {

template <typename T>
struct EigenIncrement<Eigen::GpuDevice, T> {
  using InType = Eigen::TensorMap<Eigen::TensorFixedSize<
      const T, Eigen::Sizes<>, Eigen::RowMajor, Eigen::DenseIndex>>;
  using OutType = Eigen::TensorMap<Eigen::TensorFixedSize<
      T, Eigen::Sizes<>, Eigen::RowMajor, Eigen::DenseIndex>>;
  static void Eval(const Eigen::GpuDevice& dev, OutType out, const InType& in,
                   const T value) {
    out.device(dev) = in + value;
  }
};

template struct EigenIncrement<Eigen::GpuDevice, float>;
template struct EigenIncrement<Eigen::GpuDevice, double>;
template struct EigenIncrement<Eigen::GpuDevice, int>;
template struct EigenIncrement<Eigen::GpuDevice, int64_t>;

}  // namespace operators
}  // namespace paddle
