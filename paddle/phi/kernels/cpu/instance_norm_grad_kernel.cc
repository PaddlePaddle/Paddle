// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/instance_norm_grad_kernel.h"

#include <memory>
#include <string>
#include <unordered_map>

#include "paddle/common/layout.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/extensions.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/norm_utils.h"

namespace phi {

template <typename T>
using ConstEigenArrayMap =
    Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>>;
template <typename T>
using ConstEigenVectorArrayMap =
    Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>>;
template <typename T>
using EigenArrayMap =
    Eigen::Map<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>>;
template <typename T>
using EigenVectorArrayMap = Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>>;

template <typename T, typename Context>
void InstanceNormGradKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const paddle::optional<DenseTensor>& scale,
                            const DenseTensor& saved_mean,
                            const DenseTensor& saved_variance,
                            const DenseTensor& d_y,
                            float epsilon UNUSED,
                            DenseTensor* d_x,
                            DenseTensor* d_scale,
                            DenseTensor* d_bias) {
  const auto* scale_ptr = scale.get_ptr();

  const auto& x_dims = x.dims();

  const int N = static_cast<int>(x_dims[0]);
  const int C = static_cast<int>(x_dims[1]);
  const int NxC = N * C;
  const int sample_size = static_cast<int>(x.numel() / N / C);

  dev_ctx.template Alloc<T>(d_x);
  auto* place = dev_ctx.eigen_device();

  Eigen::DSizes<int, 2> rshape(NxC, sample_size);
  Eigen::DSizes<int, 2> param_shape(N, C);
  Eigen::DSizes<int, 2> shape(NxC, sample_size);
#ifndef EIGEN_HAS_INDEX_LIST
  Eigen::DSizes<int, 1> rdims(0);
  Eigen::DSizes<int, 1> mean_rdims(1);
  Eigen::DSizes<int, 2> bcast(1, sample_size);
  Eigen::DSizes<int, 2> C_shape(C, 1);
  Eigen::DSizes<int, 2> NxC_shape(NxC, 1);
#else
  Eigen::IndexList<Eigen::type2index<0>> rdims;
  Eigen::IndexList<Eigen::type2index<1>> mean_rdims;
  Eigen::IndexList<Eigen::type2index<1>, int> bcast;
  bcast.set(1, sample_size);
  Eigen::IndexList<int, Eigen::type2index<1>> C_shape;
  C_shape.set(0, C);
  Eigen::IndexList<int, Eigen::type2index<1>> NxC_shape;
  NxC_shape.set(0, NxC);
#endif

  phi::funcs::SetConstant<CPUContext, T> set_constant;

  DenseTensor scale_data;
  if (!scale_ptr) {
    scale_data.Resize({C});
    dev_ctx.template Alloc<T>(&scale_data);
    set_constant(dev_ctx, &scale_data, static_cast<T>(1));
  }

  auto scale_e =
      scale_ptr ? EigenVector<T>::Flatten(*scale_ptr)
                : EigenVector<T>::Flatten(
                      const_cast<const DenseTensor&>(scale_data));  // NOLINT
  auto mean_e = EigenVector<T>::Flatten(saved_mean);
  auto inv_var_e = EigenVector<T>::Flatten(saved_variance);
  auto dy_e = EigenVector<T>::Flatten(d_y);
  auto x_e = EigenVector<T>::Flatten(x);

  auto scale_arr = scale_e.reshape(C_shape);
  auto mean_arr = mean_e.reshape(NxC_shape);
  auto inv_var_arr = inv_var_e.reshape(NxC_shape);
  auto dy_arr = dy_e.reshape(shape);
  auto x_arr = x_e.reshape(shape);

  auto tmp = (x_arr - mean_arr.eval().broadcast(bcast)) *
             inv_var_arr.eval().broadcast(bcast);

  // math: d_bias = np.sum(d_y, axis=(n,h,w))
  // math: d_scale = np.sum((X-mean) / inv_std * dy, axis=(n, h,w))
  if (d_scale && d_bias) {
    dev_ctx.template Alloc<T>(d_scale);
    dev_ctx.template Alloc<T>(d_bias);
    set_constant(dev_ctx, d_scale, static_cast<T>(0));
    set_constant(dev_ctx, d_bias, static_cast<T>(0));

    auto d_scale_e = EigenVector<T>::Flatten(*d_scale);
    auto d_scale_data = d_scale_e.reshape(C_shape);
    auto d_bias_e = EigenVector<T>::Flatten(*d_bias);
    auto d_bias_data = d_bias_e.reshape(C_shape);
    d_bias_data.device(*place) =
        dy_arr.sum(mean_rdims).reshape(param_shape).sum(rdims);
    d_scale_data.device(*place) =
        (tmp * dy_arr).sum(mean_rdims).reshape(param_shape).sum(rdims);
  }

  auto dy_mean =
      dy_arr.mean(mean_rdims).reshape(NxC_shape).eval().broadcast(bcast);

  Eigen::DSizes<int, 2> bcast_param(N, sample_size);
  set_constant(dev_ctx, d_x, static_cast<T>(0));
  // math: d_x = scale * inv_var * d_y - scale * inv_var * np.sum(d_y,
  // axis=(h,w))
  //             - scale * (X - mean) * inv_var.pow(3) * np.sum(d_y * (X -
  //             mean),
  //             axis=(h,w))
  auto dx_e = EigenVector<T>::Flatten(*d_x);
  auto dx_arr = dx_e.reshape(shape);
  dx_arr.device(*place) = scale_arr.broadcast(bcast_param) *
                          inv_var_arr.broadcast(bcast) *
                          (dy_arr - dy_mean -
                           tmp * (dy_arr * tmp)
                                     .mean(mean_rdims)
                                     .reshape(NxC_shape)
                                     .eval()
                                     .broadcast(bcast));
}

template <typename T, typename Context>
void InstanceNormDoubleGradKernel(const Context& dev_ctx,
                                  const DenseTensor& x,
                                  const paddle::optional<DenseTensor>& scale,
                                  const DenseTensor& saved_mean,
                                  const DenseTensor& saved_variance,
                                  const DenseTensor& dy,
                                  const paddle::optional<DenseTensor>& ddx,
                                  const paddle::optional<DenseTensor>& ddscale,
                                  const paddle::optional<DenseTensor>& ddbias,
                                  float epsilon UNUSED,
                                  DenseTensor* dx,
                                  DenseTensor* dscale,
                                  DenseTensor* ddy) {
  const auto* Scale = scale.get_ptr();
  const auto* ddScale = ddscale.get_ptr();
  const auto* ddX = ddx.get_ptr();
  const auto* ddBias = ddbias.get_ptr();
  phi::funcs::SetConstant<CPUContext, T> set_constant;
  const auto& x_dims = x.dims();
  int N = 0, C = 0, H = 0, W = 0, D = 0;
  funcs::ExtractNCWHD(x_dims, DataLayout::kNCHW, &N, &C, &H, &W, &D);
  const int sample_size = static_cast<int>(x.numel() / N / C);
  const int NxC = N * C;

  const T* mean_data = saved_mean.data<T>();
  const T* inv_var_data = saved_variance.data<T>();
  DenseTensor mean_tensor;
  DenseTensor inv_var_tensor;
  ConstEigenArrayMap<T> x_arr(x.data<T>(), sample_size, NxC);
  ConstEigenVectorArrayMap<T> mean_arr(mean_data, NxC);
  ConstEigenVectorArrayMap<T> inv_var_arr(inv_var_data, NxC);

  DenseTensor mean_tile;
  mean_tile.Resize({sample_size, NxC});
  dev_ctx.template Alloc<T>(&mean_tile);
  EigenArrayMap<T> mean_tile_data(mean_tile.data<T>(), sample_size, NxC);
  DenseTensor inv_var_tile;
  inv_var_tile.Resize({sample_size, NxC});
  dev_ctx.template Alloc<T>(&inv_var_tile);
  EigenArrayMap<T> inv_var_tile_data(inv_var_tile.data<T>(), sample_size, NxC);

  mean_tile_data = mean_arr.transpose().replicate(sample_size, 1);
  inv_var_tile_data = inv_var_arr.transpose().replicate(sample_size, 1);

  DenseTensor Scale_data;
  if (!Scale) {
    Scale_data.Resize({C});
    dev_ctx.template Alloc<T>(&Scale_data);
    set_constant(dev_ctx, &Scale_data, static_cast<T>(1));
  }
  ConstEigenVectorArrayMap<T> scale_arr(
      Scale ? Scale->data<T>() : Scale_data.data<T>(), C);

  DenseTensor scale_tile;
  scale_tile.Resize({sample_size, NxC});
  dev_ctx.template Alloc<T>(&scale_tile);
  EigenArrayMap<T> scale_tile_data(scale_tile.data<T>(), sample_size, NxC);
  scale_tile_data = scale_arr.transpose().replicate(sample_size, N);
  ConstEigenArrayMap<T> dy_arr(dy.data<T>(), sample_size, NxC);
  ConstEigenArrayMap<T> ddx_arr(ddX->data<T>(), sample_size, NxC);
  // math: dx = scale * ((x - mean) * inv_var / HxW * (np.mean(ddx,
  //          axis=(h,w)) * np.sum(dy, axis=(h,w)) -
  //          np.sum(dy * ddx, axis=(h,w)) + 3 * np.mean(dy * (x - mean),
  //          axis=(h,w)) * inv_var.pow(2) *
  //          np.sum(ddx * (x - mean), axis=(h,w))) + inv_var.pow(3) / HxW *
  //          np.sum(ddx * (x - mean)) *
  //          (np.mean(dy, axis=(h,w)) - dy) + inv_var.pow(3) / HxW *
  //          np.sum(dy, axis=(h,w)) * (x - mean) *
  //          (np.mean(ddx, axis=(h,w)) - ddx)) + ddr * (dy * inv_var -
  //          inv_var * np.mean(dy, axis=(h,w)) - inv_var.pow(3) *
  //          (x - mean) * np.mean(dy * (x - mean),  axis=(h,w)))

  DenseTensor x_sub_mean_mul_invstd;
  x_sub_mean_mul_invstd.Resize({sample_size, NxC});
  dev_ctx.template Alloc<T>(&x_sub_mean_mul_invstd);
  EigenArrayMap<T> x_sub_mean_mul_invstd_arr(
      x_sub_mean_mul_invstd.data<T>(), sample_size, NxC);
  x_sub_mean_mul_invstd_arr = (x_arr - mean_tile_data) * inv_var_tile_data;

  if (dx) {
    dev_ctx.template Alloc<T>(dx);
    set_constant(dev_ctx, dx, static_cast<T>(0));
    EigenArrayMap<T> dx_arr(dx->data<T>(), sample_size, NxC);
    if (ddX) {
      dx_arr +=
          x_sub_mean_mul_invstd_arr * inv_var_tile_data * inv_var_tile_data /
          sample_size *
          (ddx_arr.colwise().sum() * dy_arr.colwise().sum() / sample_size -
           (dy_arr * ddx_arr).colwise().sum() +
           3. * (dy_arr * x_sub_mean_mul_invstd_arr).colwise().sum() *
               (ddx_arr * x_sub_mean_mul_invstd_arr).colwise().sum() /
               sample_size);
      dx_arr += (ddx_arr * x_sub_mean_mul_invstd_arr).colwise().sum() /
                sample_size * inv_var_tile_data * inv_var_tile_data *
                (dy_arr.colwise().sum() / sample_size - dy_arr);
      dx_arr += (dy_arr * x_sub_mean_mul_invstd_arr).colwise().sum() /
                sample_size * inv_var_tile_data * inv_var_tile_data *
                (ddx_arr.colwise().sum() / sample_size - ddx_arr);
      dx_arr = scale_tile_data * dx_arr;
    }
    if (ddScale) {
      ConstEigenVectorArrayMap<T> ddscale_arr(ddScale->data<T>(), C);
      DenseTensor ddscale_tile;
      ddscale_tile.Resize({sample_size, NxC});
      dev_ctx.template Alloc<T>(&ddscale_tile);
      EigenArrayMap<T> ddscale_tile_data(
          ddscale_tile.data<T>(), sample_size, NxC);
      ddscale_tile_data = ddscale_arr.transpose().replicate(sample_size, N);
      dx_arr += (dy_arr * inv_var_tile_data -
                 dy_arr.colwise().sum() / sample_size * inv_var_tile_data -
                 x_sub_mean_mul_invstd_arr * inv_var_tile_data *
                     (dy_arr * x_sub_mean_mul_invstd_arr).colwise().sum() /
                     sample_size) *
                ddscale_tile_data;
    }
  }
  if (dscale) {
    // math: dscale = inv_var * (dy - np.mean(dy, axis=(h,w) - (x-mean) *
    //            inv_var.pow(2) * np.mean(dy * (x-mean), axis=(h,w)))) * ddx
    dev_ctx.template Alloc<T>(dscale);
    set_constant(dev_ctx, dscale, static_cast<T>(0));
    EigenVectorArrayMap<T> dscale_arr(dscale->data<T>(), C);
    if (ddX) {
      DenseTensor first_grad;
      first_grad.Resize({sample_size, NxC});
      dev_ctx.template Alloc<T>(&first_grad);
      set_constant(dev_ctx, &first_grad, static_cast<T>(0));
      EigenArrayMap<T> first_grad_arr(first_grad.data<T>(), sample_size, NxC);
      first_grad_arr +=
          inv_var_tile_data *
          (dy_arr -
           dy_arr.colwise().sum().replicate(sample_size, 1) / sample_size -
           x_sub_mean_mul_invstd_arr *
               (dy_arr * x_sub_mean_mul_invstd_arr)
                   .colwise()
                   .sum()
                   .replicate(sample_size, 1) /
               sample_size);
      first_grad_arr = first_grad_arr * ddx_arr;
      for (int nc = 0; nc < NxC; ++nc) {
        int c = nc % C;
        dscale_arr(c) += first_grad_arr.colwise().sum()(nc);
      }
    }
  }
  if (ddy) {
    // math: ddy = (x - mean) * inv_var * ddscale + ddbias +
    //           scale * inv_var * (ddx - (x - mean) * inv_var.pow(2) *
    //           np.mean(ddx * (x - mean), axis=(h,w)))
    dev_ctx.template Alloc<T>(ddy);
    set_constant(dev_ctx, ddy, static_cast<T>(0));
    EigenArrayMap<T> ddy_arr(ddy->data<T>(), sample_size, NxC);
    if (ddX) {
      ddy_arr += scale_tile_data * inv_var_tile_data *
                 (ddx_arr - ddx_arr.colwise().sum() / sample_size -
                  x_sub_mean_mul_invstd_arr *
                      (ddx_arr * x_sub_mean_mul_invstd_arr).colwise().sum() /
                      sample_size);
    }
    if (ddScale && ddBias) {
      ConstEigenVectorArrayMap<T> ddscale_arr(ddScale->data<T>(), C);
      DenseTensor ddscale_tile;
      ddscale_tile.Resize({sample_size, NxC});
      dev_ctx.template Alloc<T>(&ddscale_tile);
      EigenArrayMap<T> ddscale_tile_data(
          ddscale_tile.data<T>(), sample_size, NxC);
      ddscale_tile_data = ddscale_arr.transpose().replicate(sample_size, N);

      ConstEigenVectorArrayMap<T> ddbias_arr(ddBias->data<T>(), C);
      DenseTensor ddbias_tile;
      ddbias_tile.Resize({sample_size, NxC});
      dev_ctx.template Alloc<T>(&ddbias_tile);
      EigenArrayMap<T> ddbias_tile_data(
          ddbias_tile.data<T>(), sample_size, NxC);
      ddbias_tile_data = ddbias_arr.transpose().replicate(sample_size, N);

      ddy_arr += x_sub_mean_mul_invstd_arr * ddscale_tile_data;
      ddy_arr += ddbias_tile_data;
    }
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(instance_norm_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::InstanceNormGradKernel,
                   float,
                   double) {}
PD_REGISTER_KERNEL(instance_norm_double_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::InstanceNormDoubleGradKernel,
                   float,
                   double) {}
