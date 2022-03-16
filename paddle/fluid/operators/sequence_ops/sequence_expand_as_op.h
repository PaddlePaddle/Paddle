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

#include <numeric>  // std::iota
#include <sstream>
#include <vector>
#include "glog/logging.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
struct SequenceExpandAsFunctor {
  void operator()(
      const DeviceContext &ctx, const framework::LoDTensor &x,
      const framework::Vector<size_t> &ref_lod, /*expand referenced lod*/
      framework::LoDTensor *out);
};

template <typename DeviceContext, typename T>
struct SequenceExpandAsGradFunctor {
  void operator()(
      const DeviceContext &ctx, const framework::LoDTensor &dout,
      const framework::Vector<size_t> &ref_lod, /*expand referenced lod*/
      framework::LoDTensor *dx);
};

template <typename T>
struct SequenceExpandAsFunctor<platform::CPUDeviceContext, T> {
  void operator()(
      const platform::CPUDeviceContext &context, const framework::LoDTensor &x,
      const framework::Vector<size_t> &ref_lod, /*expand referenced lod*/
      framework::LoDTensor *out) {
    int64_t height = x.dims()[0];
    int64_t width = phi::product(x.dims()) / height;

    const T *in_data = x.data<T>();
    T *out_data = out->mutable_data<T>(context.GetPlace());

    for (int h_id = 0; h_id < height; ++h_id) {
      size_t span = ref_lod[h_id + 1] - ref_lod[h_id];
      if (span == 0) continue;
      const T *src = in_data + h_id * width;
      for (int64_t w_id = 0; w_id < width; ++w_id) {
        T ele = src[w_id];
        size_t offset = ref_lod[h_id] * width;
        for (size_t k = 0; k < span; ++k) {
          out_data[offset + k * width + w_id] = ele;
        }
      }
    }
  }
};

template <typename DeviceContext, typename T>
class SequenceExpandAsKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *x = context.Input<framework::LoDTensor>("X");
    auto *y = context.Input<framework::LoDTensor>("Y");
    auto *out = context.Output<framework::LoDTensor>("Out");

    PADDLE_ENFORCE_EQ(
        y->lod().empty(), false,
        platform::errors::InvalidArgument(
            "Input(Y) of SequenceExpandAsOp has wrong LoD information. "
            "Expected Y's lod is not empty, but received empty lod."));

    auto &y_lod = y->lod();
    PADDLE_ENFORCE_EQ(y_lod.size(), 1,
                      platform::errors::InvalidArgument(
                          "Input(Y) of SequenceExpandAsOp has wrong LoD "
                          "information. Expected Y's lod level = 1, but "
                          "received  lod level = %d.",
                          y_lod.size()));
    PADDLE_ENFORCE_GT(y_lod[0].size(), 1,
                      platform::errors::InvalidArgument(
                          "Input(Y) of SequenceExpandAsOp has wrong LoD "
                          "information. Expected the size of Y's lod[0] > 1, "
                          "but received lod[0].size = %d.",
                          y_lod[0].size()));

    out->mutable_data<T>(context.GetPlace());

    auto &dev_ctx = context.template device_context<DeviceContext>();
    SequenceExpandAsFunctor<DeviceContext, T> seq_espand_functor;
    seq_espand_functor(dev_ctx, *x, y_lod[0], out);
  }
};

/*
 *Given Grad(Out)
 *
 *    Grad(Out).lod = [[0,              3,            6]]
 *    Grad(Out).data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
 * Then
 *    Grad(X).data = [(0.1 + 0.2 + 0.3), (0.4 + 0.5 + 0.6)]
 *                 = [0.6, 1.5]
 *    Grad(X).lod = Input(X).lod
 *
 * */
template <typename T>
struct SequenceExpandAsGradFunctor<platform::CPUDeviceContext, T> {
  void operator()(
      const platform::CPUDeviceContext &context,
      const framework::LoDTensor &dout,
      const framework::Vector<size_t> &ref_lod, /*expand referenced lod*/
      framework::LoDTensor *dx) {
    int64_t height = dx->dims()[0];
    int64_t width = phi::product(dx->dims()) / height;

    const T *dout_data = dout.data<T>();
    T *dx_data = dx->mutable_data<T>(context.GetPlace());

    for (int64_t h_id = 0; h_id < height; ++h_id) {
      T *dst = dx_data + h_id * width;
      size_t span = ref_lod[h_id + 1] - ref_lod[h_id];
      for (int64_t w_id = 0; w_id < width; ++w_id) {
        T result = 0;
        for (size_t k = 0; k < span; ++k) {
          size_t offset = (ref_lod[h_id] + k) * width;
          result += dout_data[offset + w_id];
        }
        dst[w_id] = result;
      }
    }
  }
};

template <typename DeviceContext, typename T>
class SequenceExpandAsGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *g_out =
        context.Input<framework::LoDTensor>(framework::GradVarName("Out"));
    auto *y = context.Input<framework::LoDTensor>("Y");
    auto *g_x =
        context.Output<framework::LoDTensor>(framework::GradVarName("X"));

    g_x->mutable_data<T>(context.GetPlace());

    SequenceExpandAsGradFunctor<DeviceContext, T> functor;
    functor(context.template device_context<DeviceContext>(), *g_out,
            y->lod()[0], g_x);
  }
};

}  // namespace operators
}  // namespace paddle
