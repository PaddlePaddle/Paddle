/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <algorithm>
#include <map>
#include <set>
#include <vector>
#include "glog/logging.h"

#include "paddle/phi/api/lib/api_custom_impl.h"
#include "paddle/phi/api/lib/api_gen_utils.h"
#include "paddle/phi/api/lib/data_transform.h"
#include "paddle/phi/api/lib/kernel_dispatch.h"
#include "paddle/phi/api/lib/tensor_copy.h"
#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/ftrl_kernel.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {

template <typename T, typename Context>
void FTRLOpKernel(const Context& ctx,
                  const DenseTensor& grad,
                  const DenseTensor& learningrate,
                  const DenseTensor& param,
                  const DenseTensor& squared_accumulator,
                  const DenseTensor& linear_accumulator,
                  const DenseTensor& x,
                  float l1,
                  float l2,
                  float lr_power,
                  DenseTensor* param_out,
                  DenseTensor* squared_accumulator_out,
                  DenseTensor* linear_accumulator_out,
                  DenseTensor* grad_out) {
  static_cast<T> l1 = static_cast<T>(l1) + static_cast<T>(1e-10);
  static_cast<T> l2 = static_cast<T>(l2) + static_cast<T>(1e-10);
  lr_power = static_cast<T>(lr_power);
  auto g = EigenVector<T>::Flatten(grad);
  auto p = EigenVector<T>::Flatten(param);
  auto sq_accum = EigenVector<T>::Flatten(squared_accumulator);
  auto lin_accum = EigenVector<T>::Flatten(linear_accumulator);
  auto lr = EigenVector<T>::Flatten(learningrate);

  auto p_out = EigenVector<T>::Flatten(*param_out);
  auto s_acc_out = EigenVector<T>::Flatten(*squared_accumulator_out);
  auto l_acc_out = EigenVector<T>::Flatten(*linear_accumulator_out);
  auto& place = *ctx.eigen_device();

  Eigen::DSizes<int, 1> grad_dsize(grad->numel());

  auto new_accum = sq_accum + g * g;
  // Special case for lr_power = -0.5
  if (lr_power == static_cast<T>(-0.5)) {
    l_acc_out.device(place) =
        lin_accum + g -
        ((new_accum.sqrt() - sq_accum.sqrt()) / lr.broadcast(grad_dsize)) * p;
  } else {
    l_acc_out.device(place) =
        lin_accum + g -
        ((new_accum.pow(-lr_power) - sq_accum.pow(-lr_power)) /
         lr.broadcast(grad_dsize)) *
            p;
  }

  x = (l_acc_out.constant(l1) * l_acc_out.sign() - l_acc_out);
  if (lr_power == static_cast<T>(-0.5)) {
    auto y = (new_accum.sqrt() / lr.broadcast(grad_dsize)) +
             l_acc_out.constant(static_cast<T>(2) * l2);
    auto pre_shrink = x / y;
    p_out.device(place) =
        (l_acc_out.abs() > l_acc_out.constant(l1))
            .select(pre_shrink, p.constant(static_cast<T>(0)));
  } else {
    auto y = (new_accum.pow(-lr_power) / lr.broadcast(grad_dsize)) +
             l_acc_out.constant(static_cast<T>(2) * l2);
    auto pre_shrink = x / y;
    p_out.device(place) =
        (l_acc_out.abs() > l_acc_out.constant(l1))
            .select(pre_shrink, p.constant(static_cast<T>(0)));
  }
  s_acc_out.device(place) = sq_accum + g * g;
}
}  // namespace phi
