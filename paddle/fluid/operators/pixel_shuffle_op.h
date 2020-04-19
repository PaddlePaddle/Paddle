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
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class PixelShuffleOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<framework::Tensor>("X");
    auto* out = ctx.Output<framework::Tensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());

    int factor = ctx.Attr<int>("upscale_factor");

    auto in_dims = in->dims();
    auto o_dims = out->dims();

    framework::Tensor t;
    t.ShareDataWith(*in);
    t.Resize({in_dims[0], o_dims[1], factor, factor, in_dims[2], in_dims[3]});

    std::vector<int> axis = {0, 1, 4, 2, 5, 3};

    framework::Tensor o;
    o.ShareDataWith(*out);
    o.Resize({in_dims[0], o_dims[1], in_dims[2], factor, in_dims[3], factor});

    math::Transpose<DeviceContext, T, 6> trans;
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    trans(dev_ctx, t, &o, axis);
    out->Resize(o_dims);
  }
};

template <typename DeviceContext, typename T>
class PixelShuffleGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* dout = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    dx->mutable_data<T>(ctx.GetPlace());

    int factor = ctx.Attr<int>("upscale_factor");

    auto do_dims = dout->dims();
    auto dx_dims = dx->dims();

    framework::Tensor t;
    t.ShareDataWith(*dout);
    t.Resize({do_dims[0], do_dims[1], dx_dims[2], factor, dx_dims[3], factor});

    std::vector<int> axis = {0, 1, 3, 5, 2, 4};

    framework::Tensor o;
    o.ShareDataWith(*dx);
    o.Resize({do_dims[0], do_dims[1], factor, factor, dx_dims[2], dx_dims[3]});

    math::Transpose<DeviceContext, T, 6> trans;
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    trans(dev_ctx, t, &o, axis);
    dx->Resize(dx_dims);
  }
};

}  // namespace operators
}  // namespace paddle
