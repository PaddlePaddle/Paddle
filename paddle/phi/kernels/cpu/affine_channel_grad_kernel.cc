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
void AffineChannelGradKernel(const Context& dev_ctx,
                             const DenseTensor& x_in,
                             const DenseTensor& scale_in,
                             const DenseTensor& bias_in,
                             const DenseTensor& out_grad,
                             const std::string& data_layout,
                             DenseTensor* x_grad,
                             DenseTensor* scale_grad,
                             DenseTensor* bias_grad) {
  auto* x = &x_in;
  auto* scale = &scale_in;
  auto* dy = &out_grad;

  auto* dx = x_grad;
  auto* dscale = scale_grad;
  auto* dbias = bias_grad;

  const phi::DataLayout layout = common::StringToDataLayout(data_layout);

  auto dims = x->dims();
  int N = static_cast<int>(dims[0]);
  int C = static_cast<int>(
      layout == phi::DataLayout::kNCHW ? dims[1] : dims[dims.size() - 1]);
  int HxW = static_cast<int>(x->numel() / N / C);

  auto* dy_d = dy->data<T>();
  auto* scale_d = scale->data<T>();
  ConstEigenVectorArrayMap<T> scale_e(scale_d, C);

  T* dx_d = dx ? dev_ctx.template Alloc<T>(dx) : nullptr;
  T* dscale_d = dscale ? dev_ctx.template Alloc<T>(dscale) : nullptr;
  T* dbias_d = dbias ? dev_ctx.template Alloc<T>(dbias) : nullptr;
  EigenVectorArrayMap<T> dscale_e(dscale_d, C);
  EigenVectorArrayMap<T> dbias_e(dbias_d, C);

  if (layout == phi::DataLayout::kNCHW) {
    // compute dscale and dbias
    int stride = C * HxW;
    auto* original_dy_d = dy_d;
    if (dscale && dbias) {
      auto* x_d = x->data<T>();
      for (int i = 0; i < N; i++) {
        ConstEigenArrayMap<T> x_e(x_d, HxW, C);
        ConstEigenArrayMap<T> dy_e(dy_d, HxW, C);
        if (i == 0) {
          dscale_e = (x_e * dy_e).colwise().sum();
        } else {
          dscale_e += (x_e * dy_e).colwise().sum();
        }
        if (i == 0) {
          dbias_e = dy_e.colwise().sum();
        } else {
          dbias_e += dy_e.colwise().sum();
        }
        x_d += stride;
        dy_d += stride;
      }
    }

    // compute dx
    if (dx) {
      dy_d = original_dy_d;
      for (int i = 0; i < N; i++) {
        ConstEigenArrayMap<T> dy_e(dy_d, HxW, C);
        EigenArrayMap<T> dx_e(dx_d, HxW, C);
        dx_e = dy_e.rowwise() * scale_e.transpose();
        dy_d += stride;
        dx_d += stride;
      }
    }
  } else {
    int num = N * HxW;
    ConstEigenArrayMap<T> dy_e(dy_d, C, num);
    // compute dscale and dbias
    if (dscale && dbias) {
      auto* x_d = x->data<T>();
      ConstEigenArrayMap<T> x_e(x_d, C, num);
      dscale_e = (x_e * dy_e).rowwise().sum();
      dbias_e = dy_e.rowwise().sum();
    }

    // compute dx
    if (dx) {
      EigenArrayMap<T> dx_e(dx_d, C, num);
      dx_e = dy_e.colwise() * scale_e;
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(affine_channel_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::AffineChannelGradKernel,
                   float,
                   double) {}
