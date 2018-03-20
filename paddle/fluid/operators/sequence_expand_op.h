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
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;

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
    auto x_dims = x.dims();
    size_t element_len = framework::product(x_dims) / x_dims[0];
    const T* x_data = x.data<T>();
    T* out_data = out->mutable_data<T>(context.GetPlace());
    auto out_starts = out->lod().back();

    for (size_t i = 0; i < out_starts.size() - 1; i++) {
      int scale = out_starts[i + 1] - out_starts[i];
      Eigen::TensorMap<
          Eigen::Tensor<const T, 2, Eigen::RowMajor, Eigen::DenseIndex>>
          x_t(x_data, 1, element_len);
      Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor, Eigen::DenseIndex>>
          out_t(out_data, scale, element_len);
      Eigen::array<int, 2> cast({{scale, 1}});
      out_t.device(*context.eigen_device()) = x_t.broadcast(cast);
      x_data += element_len;
      out_data += element_len * scale;
    }
  }
};

template <typename DeviceContext, typename T>
class SequenceExpandKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<LoDTensor>("X");
    auto* out = context.Output<LoDTensor>("Out");
    auto x_dims = x->dims();
    auto* y = context.Input<LoDTensor>("Y");
    PADDLE_ENFORCE(!y->lod().empty(), "y should have lod");
    PADDLE_ENFORCE_EQ(static_cast<size_t>(x_dims[0]),
                      y->lod().back().size() - 1,
                      "The size of last lod level in Input(Y)"
                      "must be equal to dims[0] of Input(X).");
    out->set_lod(y->lod());
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
    auto out_last_level = out.lod().back();
    const T* d_out_data = dout.data<T>();
    T* d_x_data = dx->mutable_data<T>(context.GetPlace());
    size_t element_len = dout.numel() / dout.dims()[0];
    for (size_t i = 0; i < out_last_level.size() - 1; ++i) {
      size_t repeat = out_last_level[i + 1] - out_last_level[i];
      Eigen::TensorMap<
          Eigen::Tensor<const T, 2, Eigen::RowMajor, Eigen::DenseIndex>>
      d_out_t(d_out_data, static_cast<int>(repeat), element_len);
      Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>>
      d_x_t(d_x_data, static_cast<int>(element_len));
      d_x_t.device(*context.eigen_device()) =
          d_out_t.sum(Eigen::array<int, 1>({{0}}));
      d_out_data += (repeat * element_len);
      d_x_data += element_len;
    }
  }
};

template <typename DeviceContext, typename T>
class SequenceExpandGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<LoDTensor>("X");
    auto* out = context.Input<LoDTensor>("Out");
    auto* d_out = context.Input<LoDTensor>(framework::GradVarName("Out"));

    auto* d_x = context.Output<LoDTensor>(framework::GradVarName("X"));
    d_x->set_lod(x->lod());
    SequenceExpandGradFunctor<DeviceContext, T> functor;
    functor(context.template device_context<DeviceContext>(), *x, *out, *d_out,
            d_x);
  }
};

}  // namespace operators
}  // namespace paddle
