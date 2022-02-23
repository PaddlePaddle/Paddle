/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext>
void GetAccumulators(const framework::ExecutionContext& ctx,
                     int64_t* num_updates, int64_t* num_accumulates,
                     int64_t* old_num_accumulates);

template <typename DeviceContext>
void SetAccumulators(const framework::ExecutionContext& ctx,
                     int64_t num_updates, int64_t num_accumulates,
                     int64_t old_num_accumulates);

template <typename DeviceContext, typename T>
class AverageAccumulatesKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    // It is used to avoid loss of precision
    static const int64_t kMaxNumAccumulates = 16384;
    // Get accumulators from input
    int64_t num_updates = 0;
    int64_t num_accumulates = 0;
    int64_t old_num_accumulates = 0;
    GetAccumulators<DeviceContext>(ctx, &num_updates, &num_accumulates,
                                   &old_num_accumulates);

    // Get attrs
    float average_window = ctx.Attr<float>("average_window");
    int64_t max_average_window = ctx.Attr<int64_t>("max_average_window");
    int64_t min_average_window = ctx.Attr<int64_t>("min_average_window");
    PADDLE_ENFORCE_LE(
        min_average_window, max_average_window,
        platform::errors::InvalidArgument(
            "The min_average_window > "
            "max_average_window is not right, min_average_window is %ld, "
            "max_average_window is %ld.",
            min_average_window, max_average_window));

    // Get inputs
    auto* param = ctx.Input<Tensor>("param");
    auto* in_sum_1 = ctx.Input<Tensor>("in_sum_1");
    auto* in_sum_2 = ctx.Input<Tensor>("in_sum_2");
    auto* in_sum_3 = ctx.Input<Tensor>("in_sum_3");
    auto param_tensor = framework::EigenVector<T>::Flatten(*param);
    auto in_sum_1_tensor = framework::EigenVector<T>::Flatten(*in_sum_1);
    auto in_sum_2_tensor = framework::EigenVector<T>::Flatten(*in_sum_2);
    auto in_sum_3_tensor = framework::EigenVector<T>::Flatten(*in_sum_3);

    // Get outputs
    auto* out_sum_1 = ctx.Output<Tensor>("out_sum_1");
    auto* out_sum_2 = ctx.Output<Tensor>("out_sum_2");
    auto* out_sum_3 = ctx.Output<Tensor>("out_sum_3");
    auto out_sum_1_tensor = framework::EigenVector<T>::Flatten(*out_sum_1);
    auto out_sum_2_tensor = framework::EigenVector<T>::Flatten(*out_sum_2);
    auto out_sum_3_tensor = framework::EigenVector<T>::Flatten(*out_sum_3);

    // Compute
    auto& place = *ctx.template device_context<DeviceContext>().eigen_device();
    phi::funcs::SetConstant<DeviceContext, T> constant_functor;
    ++num_updates;
    ++num_accumulates;
    out_sum_1_tensor.device(place) = in_sum_1_tensor + param_tensor;
    out_sum_2_tensor.device(place) = in_sum_2_tensor;
    out_sum_3_tensor.device(place) = in_sum_3_tensor;
    if (num_updates % kMaxNumAccumulates == 0) {
      // Move the sum to a different buffer to avoid loss of precision due to
      // too many sums.
      out_sum_2_tensor.device(place) = in_sum_2_tensor + in_sum_1_tensor;
      constant_functor(ctx.template device_context<DeviceContext>(), out_sum_1,
                       0.0);
    }
    if (num_accumulates >= min_average_window &&
        num_accumulates >= std::min<int64_t>(max_average_window,
                                             num_updates * average_window)) {
      //  Now the average window is too long, discard the old sum.
      out_sum_3_tensor.device(place) = in_sum_1_tensor + in_sum_2_tensor;
      constant_functor(ctx.template device_context<DeviceContext>(), out_sum_1,
                       0.0);
      constant_functor(ctx.template device_context<DeviceContext>(), out_sum_2,
                       0.0);
      old_num_accumulates = num_accumulates;
      num_accumulates = 0;
    }

    // Set accumulators to output
    SetAccumulators<DeviceContext>(ctx, num_updates, num_accumulates,
                                   old_num_accumulates);
  }
};

}  // namespace operators
}  // namespace paddle
