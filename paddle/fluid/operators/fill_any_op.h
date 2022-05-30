/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class FillAnyKernel : public framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext &ctx) const override {
    auto *out = ctx.Output<framework::Tensor>("Out");
    auto floatvar = ctx.template Attr<float>("value_float");
    auto intvar = ctx.template Attr<int>("value_int");
    auto isfloat = ((typeid(float) == typeid(T)) ||
                    (typeid(double) == typeid(T) ||
                     typeid(paddle::platform::float16) == typeid(T)));

    T fill_var = static_cast<T>(floatvar);
    if (!isfloat) {
      fill_var = static_cast<T>(intvar);
    }

    PADDLE_ENFORCE_EQ(
        std::isnan(static_cast<double>(fill_var)), false,
        platform::errors::InvalidArgument("fill value should not be NaN,"
                                          " but received NaN"));

    out->mutable_data<T>(ctx.GetPlace());
    auto &dev_ctx = ctx.template device_context<DeviceContext>();
    phi::funcs::SetConstant<DeviceContext, T> functor;
    functor(reinterpret_cast<const DeviceContext &>(dev_ctx), out,
            static_cast<T>(fill_var));
  }
};

template <typename DeviceContext, typename T>
class FillAnyGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext &ctx) const override {
    auto *dx = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    if (dx) {
      dx->mutable_data<T>(ctx.GetPlace());
      auto &dev_ctx = ctx.template device_context<DeviceContext>();
      phi::funcs::SetConstant<DeviceContext, T> functor;
      functor(reinterpret_cast<const DeviceContext &>(dev_ctx), dx, T(0));
    }
  }
};

}  // namespace operators
}  // namespace paddle
