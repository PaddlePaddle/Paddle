/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/pten/core/hostdevice.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using Array1 = Eigen::DSizes<int64_t, 1>;

template <typename T>
struct KLDivLossForward {
  HOSTDEVICE KLDivLossForward() {}

  HOSTDEVICE T operator()(const T& target, const T& input) const {
    if (target <= 0) {
      return 0;
    } else {
      return target * (std::log(target) - input);
    }
  }
};

template <typename T>
struct KLDivLossBackward {
  HOSTDEVICE KLDivLossBackward() {}

  HOSTDEVICE T operator()(const T& target, const T& grad) const {
    if (target <= 0) {
      return 0;
    } else {
      return static_cast<T>(-1.) * grad;
    }
  }
};

template <typename DeviceContext, typename T>
class KLDivLossKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& place = *ctx.template device_context<DeviceContext>().eigen_device();
    auto* input = ctx.Input<Tensor>("X");
    auto* target = ctx.Input<Tensor>("Target");
    auto* loss = ctx.Output<Tensor>("Loss");
    auto reduction = ctx.Attr<std::string>("reduction");

    const int n = input->dims()[0];

    loss->mutable_data<T>(ctx.GetPlace());
    auto input_t = framework::EigenVector<T>::Flatten(*input);
    auto target_t = framework::EigenVector<T>::Flatten(*target);
    auto loss_t = framework::EigenVector<T>::Flatten(*loss);
    auto output = target_t.binaryExpr(input_t, KLDivLossForward<T>());
    if ("none" == reduction) {
      loss_t.device(place) = output;
    } else if ("batchmean" == reduction) {
      auto output_sum = output.sum();
      if (n > 0) {
        loss_t.device(place) = output_sum / output_sum.constant(n);
      } else {
        loss_t.device(place) = output_sum;
      }
    } else if ("mean" == reduction) {
      loss_t.device(place) = output.mean();
    } else if ("sum" == reduction) {
      loss_t.device(place) = output.sum();
    }
  }
};

template <typename DeviceContext, typename T>
class KLDivLossGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& place = *ctx.template device_context<DeviceContext>().eigen_device();
    auto* target = ctx.Input<Tensor>("Target");
    auto reduction = ctx.Attr<std::string>("reduction");
    auto* input_grad = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* loss_grad = ctx.Input<Tensor>(framework::GradVarName("Loss"));

    const int n = input_grad->dims()[0];
    const int numel = input_grad->numel();
    const int expand = numel / loss_grad->numel();

    input_grad->mutable_data<T>(ctx.GetPlace());

    auto target_t = framework::EigenVector<T>::Flatten(*target);

    auto input_grad_t = framework::EigenVector<T>::Flatten(*input_grad);
    auto loss_grad_t = framework::EigenVector<T>::Flatten(*loss_grad);

    auto loss_grad_expand = loss_grad_t.broadcast(Array1(expand));
    auto grad_t = target_t * loss_grad_expand;
    input_grad_t.device(place) =
        target_t.binaryExpr(grad_t, KLDivLossBackward<T>());

    if ("mean" == reduction) {
      input_grad_t.device(place) = input_grad_t / static_cast<T>(numel);
    } else if ("batchmean" == reduction) {
      input_grad_t.device(place) = input_grad_t / static_cast<T>(n);
    }
  }
};

}  // namespace operators
}  // namespace paddle
