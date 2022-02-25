/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/sequence_padding.h"
#include "paddle/fluid/operators/math/sequence_scale.h"
#include "paddle/fluid/platform/dynload/warpctc.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

template <typename DeviceContext, typename T>
class WarpCTCKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {

  }
};

template <typename DeviceContext, typename T>
class WarpCTCGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* warpctc_grad = ctx.Input<LoDTensor>("WarpCTCGrad");
    auto* logits_grad = ctx.Output<LoDTensor>(framework::GradVarName("Logits"));
    const Tensor* loss_grad = ctx.Input<Tensor>(framework::GradVarName("Loss"));

    logits_grad->mutable_data<T>(ctx.GetPlace());
    bool norm_by_times = ctx.Attr<bool>("norm_by_times");

    if (ctx.HasInput("LogitsLength")) {
      int max_seq_length = warpctc_grad->dims()[0];  // Tmax
      int num_sequences = warpctc_grad->dims()[1];   // B
      int seq_width = warpctc_grad->dims()[2];       // D

      auto* logits_length = ctx.Input<framework::Tensor>("LogitsLength");
      // B
      auto logits_len_e =
          framework::EigenTensor<int64_t, 1>::From(*logits_length);
      // (B, 1)
      auto loss_grad_e = framework::EigenTensor<T, 2>::From(*loss_grad);
      // (T, B, D)
      auto warpctc_grad_e = framework::EigenTensor<T, 3>::From(*warpctc_grad);

      auto logits_grad_e = framework::EigenTensor<T, 3>::From(*logits_grad);

      Eigen::DSizes<int, 3> grad_shape(1, num_sequences, 1);
      Eigen::DSizes<int, 3> bcast(max_seq_length, 1, seq_width);
      auto logits_g = warpctc_grad_e *
                      loss_grad_e.reshape(grad_shape).broadcast(bcast).eval();

      auto* place = ctx.template device_context<DeviceContext>().eigen_device();
      if (norm_by_times) {
        auto scales = logits_len_e.cast<T>()
                          .inverse()
                          .reshape(grad_shape)
                          .broadcast(bcast)
                          .eval();
        logits_grad_e.device(*place) = logits_g * scales;
      } else {
        logits_grad_e.device(*place) = logits_g;
      }
    } else {
      math::UnpaddingLoDTensorFunctor<DeviceContext, T>()(
          ctx.template device_context<DeviceContext>(), *warpctc_grad,
          logits_grad, -1, 0, norm_by_times, math::kLengthBatchWidth);

      const T* loss_grad_data = loss_grad->data<T>();
      math::ScaleLoDTensorFunctor<DeviceContext, T>()(
          ctx.template device_context<DeviceContext>(), loss_grad_data,
          logits_grad);
    }
  }
};

}  // namespace operators
}  // namespace paddle
