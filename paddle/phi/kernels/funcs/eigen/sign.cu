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
#include "paddle/phi/common/float16.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/funcs/eigen/extensions.h"

namespace phi {
namespace funcs {

template <typename T>
struct EigenSign<Eigen::GpuDevice, T> {
  using InType = Eigen::TensorMap<
      Eigen::Tensor<const T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  using OutType =
      Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  static void Eval(const Eigen::GpuDevice& dev, OutType out, const InType& in) {
    out.device(dev) = in.sign();
  }
};

template struct EigenSign<Eigen::GpuDevice, float>;
template struct EigenSign<Eigen::GpuDevice, double>;
template struct EigenSign<Eigen::GpuDevice, dtype::float16>;

}  // namespace funcs
}  // namespace phi
