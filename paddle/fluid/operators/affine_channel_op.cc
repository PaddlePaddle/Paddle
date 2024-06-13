/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
Indicesou may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <string>
#include <unordered_map>

#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace paddle {
namespace operators {

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

template <typename T, typename DeviceContext>
class AffineChannelKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* scale = ctx.Input<phi::DenseTensor>("Scale");
    auto* bias = ctx.Input<phi::DenseTensor>("Bias");

    auto* y = ctx.Output<phi::DenseTensor>("Out");
    y->mutable_data<T>(ctx.GetPlace());

    const phi::DataLayout layout =
        common::StringToDataLayout(ctx.Attr<std::string>("data_layout"));

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
};

template <typename T, typename DeviceContext>
class AffineChannelGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* scale = ctx.Input<phi::DenseTensor>("Scale");
    auto* dy = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));

    auto* dx = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto* dscale =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("Scale"));
    auto* dbias = ctx.Output<phi::DenseTensor>(framework::GradVarName("Bias"));

    const phi::DataLayout layout =
        common::StringToDataLayout(ctx.Attr<std::string>("data_layout"));

    auto dims = x->dims();
    int N = static_cast<int>(dims[0]);
    int C = static_cast<int>(
        layout == phi::DataLayout::kNCHW ? dims[1] : dims[dims.size() - 1]);
    int HxW = static_cast<int>(x->numel() / N / C);

    auto* dy_d = dy->data<T>();
    auto* scale_d = scale->data<T>();
    ConstEigenVectorArrayMap<T> scale_e(scale_d, C);

    T* dx_d = dx ? dx->mutable_data<T>(ctx.GetPlace()) : nullptr;
    T* dscale_d = dscale ? dscale->mutable_data<T>(ctx.GetPlace()) : nullptr;
    T* dbias_d = dbias ? dbias->mutable_data<T>(ctx.GetPlace()) : nullptr;
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
};

}  // namespace operators
}  // namespace paddle

PD_REGISTER_STRUCT_KERNEL(
    affine_channel, CPU, ALL_LAYOUT, ops::AffineChannelKernel, float, double) {}

PD_REGISTER_STRUCT_KERNEL(affine_channel_grad,
                          CPU,
                          ALL_LAYOUT,
                          ops::AffineChannelGradKernel,
                          float,
                          double) {}
