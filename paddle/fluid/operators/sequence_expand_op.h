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
#include <numeric>  // std::itoa

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename DeviceContext, typename T>
struct SequenceExpandFunctor {
  void operator()(const DeviceContext& ctx, const LoDTensor& x, LoDTensor* out);
};

template <typename DeviceContext, typename T>
struct SequenceExpandGradFunctor {
  void operator()(const DeviceContext& ctx, const LoDTensor& x,
                  const LoDTensor& out, const LoDTensor& dout, LoDTensor* dx);
};

template <typename T>
struct SequenceExpandFunctor<platform::CPUDeviceContext, T> {
  void operator()(const platform::CPUDeviceContext& context, const LoDTensor& x,
                  LoDTensor* out) {
    auto& out_lod = out->lod()[0];
    framework::Vector<size_t> x_lod;
    if (x.lod() == 1) {
      x_lod = x.lod()[0];
    } else {
      x_lod.reserve(out_lod.size());
      std::itoa(x_lod.begin(), x_lod.end(), 0);  // fill 0 ~ out_lod.size()-1
    }
    int out_offset = 0;
    auto& eigen_place = *context.eigen_device();
    for (size_t i = 1; i < out_lod.size(); ++i) {
      int repeat_num = y_lod[ref_level][i] - y_lod[ref_level][i - 1];
      int x_start = x_lod[i - 1];
      int x_end = x_lod[i];
      int x_seq_len = x_end - x_start;
      if (repeat_num > 0) {
        auto x_sub_tensor = x->Slice(x_start, x_end);
        x_sub_tensor.Resize({1, x_sub_tensor.numel()});
        int out_start = out_offset;
        if (x_lod.size() == 1) {
          out_start = out_lod[0][out_offset];
        }
        auto out_sub_tensor =
            out->Slice(out_start, out_start + x_seq_len * repeat_num);
        out_sub_tensor.Resize({repeat_num, x_sub_tensor.dims()[1]});
        EigenMatrix<T>::From(out_sub_tensor).device(eigen_place) =
            EigenMatrix<T>::From(x_sub_tensor)
                .broadcast(Eigen::array<int, 2>({{repeat_num, 1}}));
      }
    }
  }
};

template <typename DeviceContext, typename T>
class SequenceExpandKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<LoDTensor>("X");
    auto* y = context.Input<LoDTensor>("Y");
    auto* out = context.Output<LoDTensor>("Out");

    int ref_level = context.Attr<int>("ref_level");
    auto& x_lod = x->lod();
    auto& y_lod = y->lod();

    if (ref_level == -1) ref_level = y_lod.size() - 1;

    out->mutable_data<T>(context.GetPlace());

    if (y_lod[ref_level].size() <= 1) {
      framework::TensorCopy(*x, context.GetPlace(), out);
      return;
    }

    auto& out_lod = *out->mutable_lod();
    // x lod level is at most 1.
    if (x_lod.size() == 0) {
      out_lod = y_lod[ref_level];
    } else if (x_lod.size() == 1) {
      out_lod.resize(1);
      out_lod[0] = {0};
      int out_offset = 0;
      for (size_t i = 1; i < y_lod[ref_level].size(); ++i) {
        int repeat_num = y_lod[ref_level][i] - y_lod[ref_level][i - 1];
        int x_start = x_lod[0][i - 1];
        int x_end = x_lod[0][i];
        int x_seq_len = x_end - x_start;
        for (int j = 0; j < repeat_num; ++j) {
          out_lod[0].push_back(out_lod[0].back() + x_seq_len);
          out_offset++;
        }
      }
    }

    SequenceExpandFunctor<DeviceContext, T> functor;
    functor(context.template device_context<DeviceContext>(), *x, out);
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
template <typename T>
struct SequenceExpandGradFunctor<platform::CPUDeviceContext, T> {
  void operator()(const platform::CPUDeviceContext& context, const LoDTensor& x,
                  const LoDTensor& out, const LoDTensor& dout, LoDTensor* dx) {
    auto& dev_ctx = context.template device_context<DeviceContext>();

    math::SetConstant<DeviceContext, T> set_zero;
    set_zero(dev_ctx, g_x, static_cast<T>(0));

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
        auto g_x_sub = g_x->Slice(x_start, x_end);
        g_x_sub.Resize(flatten_to_1d(g_x_sub.dims()));
        int g_out_end = g_out_offset + repeat_num * x_seq_len;
        auto g_out_sub = g_out->Slice(g_out_offset, g_out_end);
        g_out_sub.Resize({repeat_num, g_x_sub.dims()[0]});
        math::ColwiseSum<DeviceContext, T> col_sum;
        col_sum(dev_ctx, g_out_sub, &g_x_sub);
        g_out_offset += repeat_num * x_seq_len;
      }
    }
  }
};

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

    SequenceExpandGradFunctor<DeviceContext, T> functor;
    functor(context.template device_context<DeviceContext>(), *x, *y, *g_out,
            g_x);
  }
};

}  // namespace operators
}  // namespace paddle
