/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;
using LoD = framework::LoD;

// @TODO clean code
template <typename DeviceContext, typename T>
class SequencePadOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x_ptr = ctx.Input<LoDTensor>("X");
    auto* out_ptr = ctx.Output<LoDTensor>("Out");

    out_ptr->mutable_data<T>(ctx.GetPlace());

    T pad_value = static_cast<T>(ctx.Attr<float>("pad_value"));

    math::SetConstant<DeviceContext, T> set_func;
    set_func(ctx.template device_context<DeviceContext>(), out_ptr, pad_value);

    auto& x_lod = x_ptr->lod();
    auto& x_last_level_lod = x_lod[x_lod.size() - 1];
    auto seq_num = x_last_level_lod.size() - 1;
    auto max_len = out_ptr->dims()[0] / seq_num;

    PADDLE_ENFORCE_EQ(max_len * seq_num, out_ptr->dims()[0],
                      "First dimension of `Out` should be equal to "
                      "maximum length mulplied by sequence number.");

    for (size_t i = 1; i < x_last_level_lod.size(); ++i) {
      auto x_start = x_last_level_lod[i - 1];
      auto x_end = x_last_level_lod[i];
      auto out_start = (i - 1) * max_len;
      auto out_end = out_start + (x_end - x_start);
      auto x_sub_tensor = x_ptr->Slice(x_start, x_end);
      auto out_sub_tensor = out_ptr->Slice(out_start, out_end);
      framework::TensorCopy(x_sub_tensor, ctx.GetPlace(), &out_sub_tensor);
    }
  }
};

template <typename DeviceContext, typename T>
class SequencePadGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x_ptr = ctx.Input<LoDTensor>("X");
    auto* g_out_ptr = ctx.Input<LoDTensor>(framework::GradVarName("Out"));
    auto* g_x_ptr = ctx.Output<LoDTensor>(framework::GradVarName("X"));

    math::SetConstant<DeviceContext, T> set_func;
    set_func(ctx.template device_context<DeviceContext>(), g_x_ptr,
             static_cast<T>(0));

    auto& x_lod = x_ptr->lod();
    auto& x_last_level_lod = x_lod[x_lod.size() - 1];
    auto seq_num = x_last_level_lod.size() - 1;
    int64_t max_len = g_out_ptr->dims()[0] / seq_num;

    PADDLE_ENFORCE_EQ(max_len * seq_num, g_out_ptr->dims()[0],
                      "First dimension of `Out` should be equal to "
                      "maximum length mulplied by sequence number.");

    for (size_t i = 1; i < x_last_level_lod.size(); ++i) {
      auto x_start = x_last_level_lod[i - 1];
      auto x_end = x_last_level_lod[i];
      auto out_start = (i - 1) * max_len;
      auto out_end = out_start + (x_end - x_start);

      auto g_out_sub = g_out_ptr->Slice(out_start, out_end);
      auto g_x_sub = g_x_ptr->Slice(x_start, x_end);
      framework::TensorCopy(g_x_sub, ctx.GetPlace(), &g_out_sub);
    }
  }
};

}  // namespace operators
}  // namespace paddle
