/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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

namespace paddle {
namespace operators {

template <typename T>
class TrilTriuCompute {
 public:
  HOSTDEVICE TrilTriuCompute(const T* in, const int diagonal, const bool lower,
                             const int64_t H, const int64_t W, T* out)
      : in_(in), diagonal_(diagonal), lower_(lower), H_(H), W_(W), out_(out) {}

  HOSTDEVICE void operator()(int64_t idx) {
    const int64_t row = (idx / W_) % H_;
    const int64_t col = idx % W_;
    const bool mask =
        lower_ ? (col - row > diagonal_) : (col - row < diagonal_);
    out_[idx] = mask ? static_cast<T>(0) : in_[idx];
  }

 private:
  const T* in_;
  const int diagonal_;
  const bool lower_;
  const int64_t H_;
  const int64_t W_;
  T* out_;
};

template <typename DeviceContext, typename T>
class TrilTriuOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const auto* x = context.Input<framework::Tensor>("X");
    const auto* x_data = x->data<T>();
    auto* out = context.Output<framework::Tensor>("Out");
    auto* out_data = out->mutable_data<T>(context.GetPlace());

    const int diagonal = context.Attr<int>("diagonal");
    const bool lower = context.Attr<bool>("lower");

    const auto& dims = x->dims();
    const auto H = dims[dims.size() - 2];
    const auto W = dims[dims.size() - 1];

    platform::ForRange<DeviceContext> for_range(
        context.template device_context<DeviceContext>(),
        static_cast<size_t>(x->numel()));

    paddle::operators::TrilTriuCompute<T> tril_triu_computer(
        x_data, diagonal, lower, H, W, out_data);
    for_range(tril_triu_computer);
  }
};

template <typename DeviceContext, typename T>
class TrilTriuGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const auto* d_out =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));
    const auto* dout_data = d_out->data<T>();
    auto* d_x = context.Output<framework::Tensor>(framework::GradVarName("X"));
    auto* dx_data = d_x->mutable_data<T>(context.GetPlace());

    const int diagonal = context.Attr<int>("diagonal");
    const bool lower = context.Attr<bool>("lower");

    const auto& dims = d_out->dims();
    const auto H = dims[dims.size() - 2];
    const auto W = dims[dims.size() - 1];

    platform::ForRange<DeviceContext> for_range(
        context.template device_context<DeviceContext>(),
        static_cast<size_t>(d_out->numel()));

    paddle::operators::TrilTriuCompute<T> tril_triu_grad_computer(
        dout_data, diagonal, lower, H, W, dx_data);
    for_range(tril_triu_grad_computer);
  }
};

}  // namespace operators
}  // namespace paddle
