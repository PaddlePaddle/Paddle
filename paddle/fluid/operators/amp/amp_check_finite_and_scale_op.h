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

#include <string>
#include <vector>
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/operators/isfinite_op.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class AmpCheckFiniteAndScaleKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    const auto xs = ctx.MultiInput<framework::Tensor>("X");
    const auto* scale = ctx.Input<framework::Tensor>("Scale");
    auto outs = ctx.MultiOutput<framework::Tensor>("Out");
    auto* found_inf = ctx.Output<framework::Tensor>("FoundInfinite");

    const T* scale_data = scale->data<T>();
    bool* found_inf_data = found_inf->mutable_data<bool>(dev_ctx.GetPlace());

    *found_inf_data = false;
    framework::Tensor is_finite =
        ctx.AllocateTmpTensor<bool, DeviceContext>({1}, dev_ctx);
    bool* is_finite_data = is_finite.template data<bool>();

    auto& dev = *ctx.template device_context<DeviceContext>().eigen_device();
    for (size_t i = 0; i < xs.size(); ++i) {
      const auto* x = xs[i];
      auto* out = outs[i];
      out->mutable_data<T>(dev_ctx.GetPlace());
      if (!(*found_inf_data)) {
        framework::TensorIsfinite(*x, &is_finite);
        if (*is_finite_data) {
          auto eigen_out = framework::EigenVector<T>::Flatten(*out);
          auto eigen_in = framework::EigenVector<T>::Flatten(*x);
          eigen_out.device(dev) = (*scale_data) * eigen_in;
        } else {
          *found_inf_data = true;
          break;
        }
      }
    }
    return;
  }
};

}  // namespace operators
}  // namespace paddle
