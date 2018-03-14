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
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

template <typename DeviceContext>
void getAccumulators(const framework::ExecutionContext& ctx,
                     int64_t& num_updates_, int64_t& num_accumulates_,
                     int64_t& old_num_accumulates_);

template <typename DeviceContext>
void setAccumulators(const framework::ExecutionContext& ctx,
                     int64_t num_updates_, int64_t num_accumulates_,
                     int64_t old_num_accumulates_);

template <typename DeviceContext, typename T>
class AverageAccumulatesKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    static const int64_t kMaxNumAccumulates = 16384;
    // accumulators
    int64_t num_updates_ = 0;
    int64_t num_accumulates_ = 0;
    int64_t old_num_accumulates_ = 0;
    // attrs
    int64_t min_average_window_;
    int64_t max_average_window_;
    float average_window_;

    auto* param = ctx.Input<Tensor>("Param");
    auto* in_sum_1 = ctx.Input<Tensor>("sum_1");
    auto* in_sum_2 = ctx.Input<Tensor>("sum_2");
    auto* in_sum_3 = ctx.Input<Tensor>("sum_3");

    auto* out_sum_1 = ctx.Output<Tensor>("sum_1");
    auto* out_sum_2 = ctx.Output<Tensor>("sum_2");
    auto* out_sum_3 = ctx.Output<Tensor>("sum_3");

    getAccumulators<DeviceContext>(ctx, num_updates_, num_accumulates_,
                                   old_num_accumulates_);
    average_window_ = ctx.Attr<float>("average_window");
    max_average_window_ =
        ctx.Attr<int64_t>("max_average_window");  // default bach number
    min_average_window_ =
        ctx.Attr<int64_t>("min_average_window");  // default 10000L
    min_average_window_ =
        std::min<int64_t>(min_average_window_, max_average_window_);

    auto param_tensor = EigenVector<T>::Flatten(*param);
    auto in_sum_1_tensor = EigenVector<T>::Flatten(*in_sum_1);
    auto in_sum_2_tensor = EigenVector<T>::Flatten(*in_sum_2);
    auto in_sum_3_tensor = EigenVector<T>::Flatten(*in_sum_3);
    auto out_sum_1_tensor = EigenVector<T>::Flatten(*out_sum_1);
    auto out_sum_2_tensor = EigenVector<T>::Flatten(*out_sum_2);
    auto out_sum_3_tensor = EigenVector<T>::Flatten(*out_sum_3);

    auto& place = *ctx.template device_context<DeviceContext>().eigen_device();
    math::SetConstant<DeviceContext, T> constant_functor;
    // start batch
    ++num_updates_;
    ++num_accumulates_;

    // update
    out_sum_1_tensor.device(place) = in_sum_1_tensor + param_tensor;

    out_sum_2_tensor.device(place) = in_sum_2_tensor;
    out_sum_3_tensor.device(place) = in_sum_3_tensor;
    // needSpecialTraversal
    if (num_updates_ % kMaxNumAccumulates == 0) {
      out_sum_2_tensor.device(place) = in_sum_2_tensor + in_sum_1_tensor;
      constant_functor(ctx.template device_context<DeviceContext>(), out_sum_1,
                       0.0);
    }

    if (num_accumulates_ >= min_average_window_ &&
        num_accumulates_ >= std::min<int64_t>(max_average_window_,
                                              num_updates_ * average_window_)) {
      out_sum_3_tensor.device(place) = in_sum_1_tensor + in_sum_2_tensor;
      constant_functor(ctx.template device_context<DeviceContext>(), out_sum_1,
                       0.0);
      constant_functor(ctx.template device_context<DeviceContext>(), out_sum_2,
                       0.0);

      // finishBatch
      old_num_accumulates_ = num_accumulates_;
      num_accumulates_ = 0;
    }
    setAccumulators<DeviceContext>(ctx, num_updates_, num_accumulates_,
                                   old_num_accumulates_);
  }
};

}  // namespace operators
}  // namespace paddle
