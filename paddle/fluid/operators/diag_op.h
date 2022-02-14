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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/for_range.h"
#include "paddle/pten/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

template <typename T>
struct DiagFunctor {
  DiagFunctor(const T* diagonal, int64_t numel, T* output)
      : diagonal_(diagonal), numel_(numel), output_(output) {}

  HOSTDEVICE void operator()(size_t idx) const {
    output_[idx * numel_ + idx] = diagonal_[idx];
  }

  const T* diagonal_;
  int64_t numel_;
  T* output_;
};

template <typename DeviceContext, typename T>
class DiagKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* diagonal = context.Input<framework::Tensor>("Diagonal");
    auto* diag_data = diagonal->data<T>();
    auto numel = diagonal->numel();
    auto* out = context.Output<framework::Tensor>("Out");
    T* out_data = out->mutable_data<T>(context.GetPlace());

    pten::funcs::SetConstant<DeviceContext, T> set_zero;
    auto& dev_ctx = context.template device_context<DeviceContext>();
    set_zero(dev_ctx, out, static_cast<T>(0));

    platform::ForRange<DeviceContext> for_range(dev_ctx, numel);
    DiagFunctor<T> functor(diag_data, numel, out_data);
    for_range(functor);
  }
};

}  // namespace operators
}  // namespace paddle
