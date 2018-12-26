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
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/operators/elementwise/elementwise_add_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/operators/k_select/k_select.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class DGCOpKernel : public framework::OpKernel<T> {
 public:
  // FIXME(gongwb): add gradient clipping.
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto u = ctx.Input<framework::Tensor>("U");
    auto v = ctx.Input<framework::Tensor>("V");
    auto g = ctx.Input<framework::Tensor>("Grad");
    auto local_g = ctx.Input<framework::Tensor>("GradLocal");
    auto m = static_cast<T>(ctx.Attr<float>("m"));

    auto u_out = ctx.Output<framework::Tensor>("U");
    auto v_out = ctx.Output<framework::Tensor>("V");
    auto g_out = ctx.Output<framework::Tensor>("Grad");

    // local_g = local_g + g
    elementwise_add<DeviceContext, T>(ctx, local_g, g, g_out);

    // FIXME(gognwb): use cublas.
    // u = m * u + g
    auto u_out_e = framework::EigenVector<T>::Flatten(*u_out);
    auto u_e = framework::EigenVector<T>::Flatten(*u);
    auto g_e = framework::EigenVector<T>::Flatten(*g);
    auto& dev = *ctx.template device_context<DeviceContext>().eigen_device();
    u_out_e.device(dev) = m * u_e + g_e;

    // v = u + e
    elementwise_add<DeviceContext, T>(ctx, u, v, v_out);

    // k_select();
  }
};
}  // namespace operators
}  // namespace paddle
