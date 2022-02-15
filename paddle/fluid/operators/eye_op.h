/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/for_range.h"
#include "paddle/pten/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

template <typename T>
struct EyeFunctor {
  EyeFunctor(int64_t num_columns, T* output)
      : num_columns_(num_columns), output_(output) {}

  HOSTDEVICE void operator()(size_t idx) const {
    output_[idx * num_columns_ + idx] = static_cast<T>(1);
  }

  int64_t num_columns_;
  T* output_;
};

template <typename DeviceContext, typename T>
class EyeKernel : public framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    auto num_rows = ctx.Attr<int64_t>("num_rows");
    auto num_columns = ctx.Attr<int64_t>("num_columns");
    if (num_columns == -1) num_columns = num_rows;

    auto* out_tensor = ctx.Output<framework::Tensor>("Out");
    T* out_data = out_tensor->mutable_data<T>(ctx.GetPlace());

    pten::funcs::SetConstant<DeviceContext, T> set_zero;
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    set_zero(dev_ctx, out_tensor, static_cast<T>(0));

    int64_t num_eyes = (std::min)(num_rows, num_columns);
    platform::ForRange<DeviceContext> for_range(dev_ctx, num_eyes);
    EyeFunctor<T> functor(num_columns, out_data);
    for_range(functor);
  }
};
}  // namespace operators
}  // namespace paddle
