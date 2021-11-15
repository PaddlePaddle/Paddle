// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class LerpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto x = ctx.Input<framework::Tensor>("X");
    auto y = ctx.Input<framework::Tensor>("Y");
    auto w = ctx.Input<framework::Tensor>("Weight");
    auto out = ctx.Output<framework::Tensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());

    auto eigen_x = framework::EigenTensor<T, 1>::From(*x);
    auto eigen_y = framework::EigenTensor<T, 1>::From(*y);
    auto eigen_w = framework::EigenTensor<T, 1>::From(*w);
    auto eigen_out = framework::EigenTensor<T, 1>::From(*out);

    auto& place = *ctx.template device_context<DeviceContext>().eigen_device();
    eigen_out.device(place) = eigen_x + eigen_w * (eigen_y - eigen_x);
  }
};

template <typename DeviceContext, typename T>
class LerpGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    VLOG(1) << "LerpGradKernel Computing...";
  }
};

}  // namespace operators
}  // namespace paddle
