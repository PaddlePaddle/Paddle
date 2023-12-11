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

#include "paddle/phi/kernels/instance_norm_kernel.h"

#include <memory>
#include <string>
#include <unordered_map>

#include "paddle/common/layout.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/funcs/eigen/extensions.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T, typename Context>
void InstanceNormKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const paddle::optional<DenseTensor>& scale,
                        const paddle::optional<DenseTensor>& bias,
                        float epsilon_f,
                        DenseTensor* y,
                        DenseTensor* saved_mean,
                        DenseTensor* saved_variance) {
  const auto& x_dims = x.dims();
  T epsilon = static_cast<T>(epsilon_f);
  const int N = static_cast<int>(x_dims[0]);
  const int C = static_cast<int>(x_dims[1]);
  const int NxC = N * C;
  const int sample_size = static_cast<int>(x.numel() / N / C);
  auto* place = dev_ctx.eigen_device();

  Eigen::DSizes<int, 2> shape(NxC, sample_size);
// Once eigen on Windows is updated, the if branch can be removed.
#ifndef EIGEN_HAS_INDEX_LIST
  Eigen::DSizes<int, 2> bcast(1, sample_size);
  Eigen::DSizes<int, 2> C_shape(C, 1);
  Eigen::DSizes<int, 2> NxC_shape(NxC, 1);
  Eigen::DSizes<int, 1> rdims(1);
#else
  Eigen::IndexList<Eigen::type2index<1>, int> bcast;
  bcast.set(1, sample_size);
  Eigen::IndexList<int, Eigen::type2index<1>> C_shape;
  C_shape.set(0, C);
  Eigen::IndexList<int, Eigen::type2index<1>> NxC_shape;
  NxC_shape.set(0, NxC);
  Eigen::IndexList<Eigen::type2index<1>> rdims;
#endif

  phi::funcs::SetConstant<CPUContext, T> set_constant;
  DenseTensor saved_mean_tmp, saved_variance_tmp;
  if (saved_mean) {
    dev_ctx.template Alloc<T>(saved_mean);
    set_constant(dev_ctx, saved_mean, static_cast<T>(0));
  } else {
    saved_mean_tmp = phi::Full<T>(dev_ctx, {NxC}, 0);
  }
  if (saved_variance) {
    dev_ctx.template Alloc<T>(saved_variance);
    set_constant(dev_ctx, saved_variance, static_cast<T>(0));
  } else {
    saved_variance_tmp = phi::Full<T>(dev_ctx, {NxC}, 0);
  }

  auto saved_mean_a =
      EigenVector<T>::Flatten(saved_mean ? *saved_mean : saved_mean_tmp);
  auto saved_mean_e = saved_mean_a.reshape(NxC_shape);
  auto saved_variance_a = EigenVector<T>::Flatten(
      saved_variance ? *saved_variance : saved_variance_tmp);
  auto saved_variance_e = saved_variance_a.reshape(NxC_shape);

  auto x_e = EigenVector<T>::Flatten(x);
  auto x_arr = x_e.reshape(shape);

  saved_mean_e.device(*place) = x_arr.mean(rdims);
  auto saved_variance_arr =
      (x_arr - saved_mean_e.broadcast(bcast)).square().mean(rdims) + epsilon;

  saved_variance_e.device(*place) = saved_variance_arr.sqrt().inverse();

  const auto scale_ptr = scale.get_ptr();
  const auto bias_ptr = bias.get_ptr();

  DenseTensor scale_data;
  DenseTensor bias_data;
  if (!scale_ptr) {
    scale_data.Resize({C});
    dev_ctx.template Alloc<T>(&scale_data);
    set_constant(dev_ctx, &scale_data, static_cast<T>(1));
  }

  if (!bias_ptr) {
    bias_data.Resize({C});
    dev_ctx.template Alloc<T>(&bias_data);
    set_constant(dev_ctx, &bias_data, static_cast<T>(0));
  }
  auto scale_e =
      scale_ptr ? EigenVector<T>::Flatten(*scale_ptr)
                : EigenVector<T>::Flatten(
                      const_cast<const DenseTensor&>(scale_data));  // NOLINT
  auto scale_arr = scale_e.reshape(C_shape);
  auto bias_e = bias_ptr
                    ? EigenVector<T>::Flatten(*bias_ptr)
                    : EigenVector<T>::Flatten(
                          const_cast<const DenseTensor&>(bias_data));  // NOLINT
  auto bias_arr = bias_e.reshape(C_shape);

  dev_ctx.template Alloc<T>(y);
  auto y_e = EigenVector<T>::Flatten(*y);
  auto y_arr = y_e.reshape(shape);

  // (x - mean) * inv_std * scale + bias
  Eigen::DSizes<int, 2> bcast_param(N, sample_size);
  y_arr.device(*place) = (x_arr - saved_mean_e.broadcast(bcast)) *
                             saved_variance_e.broadcast(bcast) *
                             scale_arr.broadcast(bcast_param) +
                         bias_arr.broadcast(bcast_param);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    instance_norm, CPU, ALL_LAYOUT, phi::InstanceNormKernel, float, double) {}
