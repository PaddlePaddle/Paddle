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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;

template <typename DeviceContext, typename T>
class SequenceExpandKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<LoDTensor>("X");
    auto* y = context.Input<LoDTensor>("Y");
    auto* out = context.Output<LoDTensor>("Out");
    int ref_level = context.Attr<int>("ref_level");

    out->mutable_data<T>(context.GetPlace());
    auto& x_lod = x->lod();
    auto& y_lod = y->lod();

    PADDLE_ENFORCE_GT(y_lod.size(), 0,
                      "Level number of `Y`'s lod should be greater than 0.");

    PADDLE_ENFORCE(
        ref_level == -1 || (ref_level >= 0 && ref_level < y_lod.size()),
        "Invlid `ref_level`, which should be either equal to -1 "
        "or in [0, %d)",
        y_lod.size());

    if (ref_level == -1) ref_level = y_lod.size() - 1;

    if (y_lod[ref_level].size() <= 1) {
      framework::TensorCopy(*x, context.GetPlace(), out);
      return;
    }

    auto& out_lod = *out->mutable_lod();
    if (x_lod.size() == 1) {
      out_lod.resize(1);
      out_lod[0] = {0};
    }

    int out_offset = 0;
    for (size_t i = 1; i < y_lod[ref_level].size(); ++i) {
      int repeat_num = y_lod[ref_level][i] - y_lod[ref_level][i - 1];
      int x_start = i - 1;
      int x_end = i;
      if (x_lod.size() == 1) {
        x_start = x_lod[0][i - 1];
        x_end = x_lod[0][i];
      }
      int x_seq_len = x_end - x_start;
      auto x_sub_tensor = x->Slice(x_start, x_end);
      for (size_t j = 0; j < repeat_num; ++j) {
        int out_start = out_offset;
        if (x_lod.size() == 1) {
          out_start = out_lod[0][out_offset];
          out_lod[0].push_back(x_seq_len);
        }
        auto out_sub_tensor = out->Slice(out_start, out_start + x_seq_len);
        framework::TensorCopy(x_sub_tensor, context.GetPlace(),
                              &out_sub_tensor);
        out_offset++;
      }
    }
  }
};

/*
 *Given Grad(Out)
 *
 *    Grad(Out).lod = [[0,                            2],
 *                     [0,              3,            6]]
 *    Grad(Out).data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
 * Then
 *    Grad(X).data = [(0.1 + 0.2 + 0.3), (0.4 + 0.5 + 0.6)]
 *                 = [0.6, 1.5]
 *    Grad(X).lod = Input(X).lod
 *
 * */
template <typename DeviceContext, typename T>
class SequenceExpandGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* g_out = context.Input<LoDTensor>(framework::GradVarName("Out"));
    auto* x = context.Input<LoDTensor>("X");
    auto* y = context.Input<LoDTensor>("Y");
    auto* g_x = context.Output<LoDTensor>(framework::GradVarName("X"));
    int ref_level = context.Attr<int>("ref_level");

    g_x->mutable_data<T>(context.GetPlace());
    g_x->set_lod(x->lod());

    auto& x_lod = x->lod();
    auto& y_lod = y->lod();

    if (ref_level == -1) ref_level = y_lod.size() - 1;

    // just copy the gradient
    if (y_lod[ref_level].size() <= 1) {
      framework::TensorCopy(*g_out, context.GetPlace(), g_x);
      return;
    }

    auto& dev_ctx = context.template device_context<DeviceContext>();

    int g_out_offset = 0;
    for (size_t i = 1; i < y_lod[ref_level].size(); ++i) {
      int repeat_num = y_lod[ref_level][i] - y_lod[ref_level][i - 1];
      if (repeat_num > 0) {
        int x_start = i - 1;
        int x_end = i;
        if (x_lod.size() == 1) {
          x_start = x_lod[0][i - 1];
          x_end = x_lod[0][i];
        }
        int x_seq_len = x_end - x_start;
        auto column = x_seq_len * x->dims()[1];
        auto g_x_sub = g_x->Slice(x_start, x_end);
        g_x_sub = framework::ReshapeToMatrix(g_x_sub, column);
        int g_out_end = g_out_offset + repeat_num * x_seq_len;
        auto g_out_sub = g_out->Slice(g_out_offset, g_out_end);
        g_out_sub = framework::ReshapeToMatrix(g_out_sub, column);
        math::ColwiseSum<DeviceContext, T> col_sum;
        col_sum(dev_ctx, g_out_sub, &g_x_sub);
        g_out_offset += repeat_num * x_seq_len;
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
