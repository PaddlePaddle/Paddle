// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <string>
#include <unordered_map>

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {

template <typename T>
using EigenArrayMap =
    Eigen::Map<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>>;
template <typename T>
using ConstEigenArrayMap =
    Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>>;
template <typename T>
using EigenVectorArrayMap = Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>>;
template <typename T>
using ConstEigenVectorArrayMap =
    Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>>;

template <typename T, typename Context>
void AffineChannelKernel(const Context& dev_ctx,
                         const DenseTensor& x_in,
                         const DenseTensor& scale_in,
                         const DenseTensor& bias_in,
                         const std::string& data_layout,
                         DenseTensor* out) {
  auto* x = &x_in;
  auto* scale = &scale_in;
  auto* bias = &bias_in;

  auto* y = out;
  dev_ctx.template Alloc<T>(y);

  const phi::DataLayout layout = common::StringToDataLayout(data_layout);

  auto dims = x->dims();
  int N = static_cast<int>(dims[0]);
  int C = static_cast<int>(
      layout == phi::DataLayout::kNCHW ? dims[1] : dims[dims.size() - 1]);
  int HxW = static_cast<int>(x->numel() / N / C);

  auto* scale_d = scale->data<T>();
  auto* bias_d = bias->data<T>();
  ConstEigenVectorArrayMap<T> a_e(scale_d, C);
  ConstEigenVectorArrayMap<T> b_e(bias_d, C);

  auto* x_d = x->data<T>();
  auto* y_d = y->data<T>();
  if (layout == phi::DataLayout::kNCHW) {
    int stride = C * HxW;
    for (int i = 0; i < N; i++) {
      ConstEigenArrayMap<T> x_e(x_d, HxW, C);
      EigenArrayMap<T> y_e(y_d, HxW, C);
      y_e = (x_e.rowwise() * a_e.transpose()).rowwise() + b_e.transpose();
      x_d += stride;
      y_d += stride;
    }
  } else {
    int num = N * HxW;
    ConstEigenArrayMap<T> x_e(x_d, C, num);
    EigenArrayMap<T> y_e(y_d, C, num);
    y_e = (x_e.colwise() * a_e).colwise() + b_e;
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(
    affine_channel, CPU, ALL_LAYOUT, phi::AffineChannelKernel, float, double) {}
