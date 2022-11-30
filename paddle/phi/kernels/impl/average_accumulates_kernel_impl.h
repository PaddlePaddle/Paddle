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

#pragma once
#include "paddle/phi/kernels/average_accumulates_kernel.h"

#include <algorithm>

#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T, typename Context>
void AverageAccumulatesKernel(const Context& dev_ctx,
                              const DenseTensor& param,
                              const DenseTensor& in_sum_1,
                              const DenseTensor& in_sum_2,
                              const DenseTensor& in_sum_3,
                              const DenseTensor& in_num_accumulates,
                              const DenseTensor& in_old_num_accumulates,
                              const DenseTensor& in_num_updates,
                              float average_window,
                              int64_t max_average_window,
                              int64_t min_average_window,
                              DenseTensor* out_sum_1,
                              DenseTensor* out_sum_2,
                              DenseTensor* out_sum_3,
                              DenseTensor* out_num_accumulates,
                              DenseTensor* out_old_num_accumulates,
                              DenseTensor* out_num_updates) {
  // It is used to avoid loss of precision
  static const int64_t kMaxNumAccumulates = 16384;
  // Get accumulators from input
  // int64_t num_updates = 0;
  // int64_t num_accumulates = 0;
  // int64_t old_num_accumulates = 0;

  auto num_updates_cpu =
      paddle::memory::Alloc(phi::CPUPlace(), sizeof(int64_t));
  int64_t* num_updates_cpu_ptr =
      reinterpret_cast<int64_t*>(num_updates_cpu->ptr());

  auto num_accumulates_cpu =
      paddle::memory::Alloc(phi::CPUPlace(), sizeof(int64_t));
  int64_t* num_accumulates_cpu_ptr =
      reinterpret_cast<int64_t*>(num_accumulates_cpu->ptr());

  auto old_num_accumulates_cpu =
      paddle::memory::Alloc(phi::CPUPlace(), sizeof(int64_t));
  int64_t* old_num_accumulates_cpu_ptr =
      reinterpret_cast<int64_t*>(old_num_accumulates_cpu->ptr());

  GetAccumulators<Context>(dev_ctx,
                           in_num_accumulates,
                           in_old_num_accumulates,
                           in_num_updates,
                           num_updates_cpu_ptr,
                           num_accumulates_cpu_ptr,
                           old_num_accumulates_cpu_ptr);
  // Get attrs
  // float average_window = ctx.Attr<float>("average_window");
  // int64_t max_average_window = ctx.Attr<int64_t>("max_average_window");
  // int64_t min_average_window = ctx.Attr<int64_t>("min_average_window");
  PADDLE_ENFORCE_LE(
      min_average_window,
      max_average_window,
      errors::InvalidArgument(
          "The min_average_window > "
          "max_average_window is not right, min_average_window is %ld, "
          "max_average_window is %ld.",
          min_average_window,
          max_average_window));

  // Get inputs
  // auto* param = ctx.Input<phi::DenseTensor>("param");
  // auto* in_sum_1 = ctx.Input<phi::DenseTensor>("in_sum_1");
  // auto* in_sum_2 = ctx.Input<phi::DenseTensor>("in_sum_2");
  // auto* in_sum_3 = ctx.Input<phi::DenseTensor>("in_sum_3");
  auto param_tensor = EigenVector<T>::Flatten(param);
  auto in_sum_1_tensor = EigenVector<T>::Flatten(in_sum_1);
  auto in_sum_2_tensor = EigenVector<T>::Flatten(in_sum_2);
  auto in_sum_3_tensor = EigenVector<T>::Flatten(in_sum_3);

  // Get outputs
  // auto* out_sum_1 = ctx.Output<phi::DenseTensor>("out_sum_1");
  // auto* out_sum_2 = ctx.Output<phi::DenseTensor>("out_sum_2");
  // auto* out_sum_3 = ctx.Output<phi::DenseTensor>("out_sum_3");
  dev_ctx.template Alloc<T>(out_sum_1);
  dev_ctx.template Alloc<T>(out_sum_2);
  dev_ctx.template Alloc<T>(out_sum_3);

  auto out_sum_1_tensor = EigenVector<T>::Flatten(*out_sum_1);
  auto out_sum_2_tensor = EigenVector<T>::Flatten(*out_sum_2);
  auto out_sum_3_tensor = EigenVector<T>::Flatten(*out_sum_3);

  // Compute
  // auto& place = *ctx.template device_context<DeviceContext>().eigen_device();
  auto& place = *dev_ctx.eigen_device();

  funcs::SetConstant<Context, T> constant_functor;
  ++(*num_updates_cpu_ptr);
  ++(*num_accumulates_cpu_ptr);
  out_sum_1_tensor.device(place) = in_sum_1_tensor + param_tensor;
  out_sum_2_tensor.device(place) = in_sum_2_tensor;
  out_sum_3_tensor.device(place) = in_sum_3_tensor;
  if ((*num_updates_cpu_ptr) % kMaxNumAccumulates == 0) {
    // Move the sum to a different buffer to avoid loss of precision due to
    // too many sums.
    out_sum_2_tensor.device(place) = in_sum_2_tensor + in_sum_1_tensor;
    constant_functor(dev_ctx, out_sum_1, static_cast<T>(0));
  }
  if ((*num_accumulates_cpu_ptr) >= min_average_window &&
      (*num_accumulates_cpu_ptr) >=
          std::min<int64_t>(max_average_window,
                            (*num_updates_cpu_ptr) * average_window)) {
    //  Now the average window is too long, discard the old sum.
    out_sum_3_tensor.device(place) = in_sum_1_tensor + in_sum_2_tensor;
    constant_functor(dev_ctx, out_sum_1, static_cast<T>(0));
    constant_functor(dev_ctx, out_sum_2, static_cast<T>(0));
    (*old_num_accumulates_cpu_ptr) = (*num_accumulates_cpu_ptr);
    (*num_accumulates_cpu_ptr) = 0;
  }

  // Set accumulators to output
  SetAccumulators<Context>(dev_ctx,
                           *num_updates_cpu_ptr,
                           *num_accumulates_cpu_ptr,
                           *old_num_accumulates_cpu_ptr,
                           out_num_accumulates,
                           out_old_num_accumulates,
                           out_num_updates);
}

}  // namespace phi
