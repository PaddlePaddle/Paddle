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
#include <numeric>  // std::iota

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
  void operator()(
      const DeviceContext& ctx, const LoDTensor& x,
      const framework::Vector<size_t>& x_lod,   /*expand source lod*/
      const framework::Vector<size_t>& ref_lod, /*expand referenced lod*/
      LoDTensor* out);
};

template <typename DeviceContext, typename T>
struct SequenceExpandGradFunctor {
  void operator()(
      const DeviceContext& ctx, const LoDTensor& dout,
      const framework::Vector<size_t>& x_lod,   /*expand source lod*/
      const framework::Vector<size_t>& ref_lod, /*expand referenced lod*/
      LoDTensor* dx);
};

template <typename T>
struct SequenceExpandFunctor<platform::CPUDeviceContext, T> {
  void operator()(
      const platform::CPUDeviceContext& context, const LoDTensor& x,
      const framework::Vector<size_t>& x_lod,   /*expand source lod*/
      const framework::Vector<size_t>& ref_lod, /*expand referenced lod*/
      LoDTensor* out) {
    int out_offset = 0;
    auto& eigen_place = *context.eigen_device();
    for (size_t i = 1; i < ref_lod.size(); ++i) {
      int repeat_num = ref_lod[i] - ref_lod[i - 1];
      int x_start = x_lod[i - 1];
      int x_end = x_lod[i];
      int x_seq_len = x_end - x_start;
      if (repeat_num > 0) {
        auto x_sub_tensor = x.Slice(x_start, x_end);
        x_sub_tensor.Resize({1, x_sub_tensor.numel()});
        int out_start = out_offset;
        if (out->lod().size() == 1) {
          out_start = out->lod()[0][out_offset];
        }
        auto out_sub_tensor =
            out->Slice(out_start, out_start + x_seq_len * repeat_num);
        out_sub_tensor.Resize({repeat_num, x_sub_tensor.dims()[1]});
        EigenMatrix<T>::From(out_sub_tensor).device(eigen_place) =
            EigenMatrix<T>::From(x_sub_tensor)
                .broadcast(Eigen::array<int, 2>({{repeat_num, 1}}));
      }
      out_offset += repeat_num;
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

    // x lod level is at most 1.
    framework::Vector<size_t> out_lod;
    if (x_lod.size() == 1) {
      out_lod.push_back(0);
      int out_offset = 0;
      for (size_t i = 1; i < y_lod[ref_level].size(); ++i) {
        int repeat_num = y_lod[ref_level][i] - y_lod[ref_level][i - 1];
        int x_start = x_lod[0][i - 1];
        int x_end = x_lod[0][i];
        int x_seq_len = x_end - x_start;
        for (int j = 0; j < repeat_num; ++j) {
          out_lod.push_back(out_lod.back() + x_seq_len);
          out_offset++;
        }
      }
      // write lod to out if x has lod
      auto& ref_lod = *out->mutable_lod();
      ref_lod[0] = out_lod;
    }
    framework::Vector<size_t> ref_x_lod;
    if (x->lod().size() == 1) {
      ref_x_lod = x->lod()[0];
    } else {
      // x_lod doesn't has lod, use fake x lod, level = 0
      ref_x_lod.resize(x->dims()[0] + 1);
      std::iota(ref_x_lod.begin(), ref_x_lod.end(), 0);
    }
    SequenceExpandFunctor<DeviceContext, T> functor;
    functor(context.template device_context<DeviceContext>(), *x, ref_x_lod,
            y_lod[ref_level], out);
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
  void operator()(
      const platform::CPUDeviceContext& context, const LoDTensor& dout,
      const framework::Vector<size_t>& x_lod,   /*expand source lod*/
      const framework::Vector<size_t>& ref_lod, /*expand referenced lod*/
      LoDTensor* dx) {
    math::SetConstant<platform::CPUDeviceContext, T> set_zero;
    set_zero(context, dx, static_cast<T>(0));

    int dout_offset = 0;
    for (size_t i = 1; i < ref_lod.size(); ++i) {
      int repeat_num = ref_lod[i] - ref_lod[i - 1];
      if (repeat_num > 0) {
        int x_start = x_lod[i - 1];
        int x_end = x_lod[i];
        int x_seq_len = x_end - x_start;
        auto dx_sub = dx->Slice(x_start, x_end);
        dx_sub.Resize(flatten_to_1d(dx_sub.dims()));
        int dout_end = dout_offset + repeat_num * x_seq_len;
        auto dout_sub = dout.Slice(dout_offset, dout_end);
        dout_sub.Resize({repeat_num, dx_sub.dims()[0]});
        math::ColwiseSum<platform::CPUDeviceContext, T> col_sum;
        col_sum(context, dout_sub, &dx_sub);
        dout_offset += repeat_num * x_seq_len;
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

    auto& y_lod = y->lod();
    if (ref_level == -1) ref_level = y_lod.size() - 1;
    // just copy the gradient
    if (y_lod[ref_level].size() <= 1) {
      framework::TensorCopy(*g_out, context.GetPlace(), g_x);
      return;
    }

    framework::Vector<size_t> ref_x_lod;
    framework::Vector<size_t> ref_lod = y_lod[ref_level];
    if (x->lod().size() == 1) {
      ref_x_lod = x->lod()[0];
    } else {
      // x_lod doesn't has lod, use fake x lod, level = 0
      ref_x_lod.resize(x->dims()[0] + 1);
      std::iota(ref_x_lod.begin(), ref_x_lod.end(), 0);
    }
    SequenceExpandGradFunctor<DeviceContext, T> functor;
    functor(context.template device_context<DeviceContext>(), *g_out, ref_x_lod,
            ref_lod, g_x);
  }
};

}  // namespace operators
}  // namespace paddle
