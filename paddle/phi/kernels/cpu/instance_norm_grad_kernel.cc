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
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/layout.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/extensions.h"
#include "paddle/phi/kernels/funcs/math_function.h"
namespace phi {

template <typename T, typename Context>
void InstanceNormGradKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const DenseTensor& d_y,
                            const paddle::optional<DenseTensor>& scale,
                            const DenseTensor& saved_mean,
                            const DenseTensor& saved_variance,
                            float epsilon,
                            DenseTensor* d_x,
                            DenseTensor* d_scale,
                            DenseTensor* d_bias) {
  const auto* scale_ptr = scale.get_ptr();

  const auto& x_dims = x.dims();

  const int N = x_dims[0];
  const int C = x_dims[1];
  const int NxC = N * C;
  const int sample_size = x.numel() / N / C;

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
      scale_ptr
          ? EigenVector<T>::Flatten(*scale_ptr)
          : EigenVector<T>::Flatten(const_cast<const DenseTensor&>(scale_data));
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
                           tmp *
                               (dy_arr * tmp)
                                   .mean(mean_rdims)
                                   .reshape(NxC_shape)
                                   .eval()
                                   .broadcast(bcast));
}

}  // namespace phi

PD_REGISTER_KERNEL(instance_norm_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::InstanceNormGradKernel,
                   float,
                   double) {}
