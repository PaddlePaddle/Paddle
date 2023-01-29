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

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
struct SequenceSoftmaxFunctor {
  void operator()(
      const DeviceContext &ctx,
      const phi::DenseTensor &x,
      const framework::Vector<size_t> &ref_lod, /*expand referenced lod*/
      phi::DenseTensor *out);
};

template <typename DeviceContext, typename T>
struct SequenceSoftmaxGradFunctor {
  void operator()(const DeviceContext &ctx,
                  const phi::DenseTensor &dout,
                  const phi::DenseTensor &out,
                  const framework::Vector<size_t> &ref_lod, /*referenced lod*/
                  phi::DenseTensor *dx);
};

template <typename T>
struct SequenceSoftmaxFunctor<phi::CPUContext, T> {
  void operator()(const phi::CPUContext &ctx,
                  const phi::DenseTensor &x,
                  const framework::Vector<size_t> &ref_lod, /*referenced lod*/
                  phi::DenseTensor *out) {
    size_t height = ref_lod.size() - 1;
    const T *in_data = x.data<T>();
    T *out_data = out->mutable_data<T>(ctx.GetPlace());
    for (size_t i = 0; i < height; ++i) {
      size_t span = ref_lod[i + 1] - ref_lod[i];
      T result = 0;
      for (size_t j = 0; j < span; ++j) {
        result += exp(in_data[ref_lod[i] + j]);
      }
      for (size_t j = 0; j < span; ++j) {
        out_data[ref_lod[i] + j] = exp(in_data[ref_lod[i] + j]) / result;
      }
    }
  }
};

template <typename T>
struct SequenceSoftmaxGradFunctor<phi::CPUContext, T> {
  void operator()(const phi::CPUContext &ctx,
                  const phi::DenseTensor &dout,
                  const phi::DenseTensor &out,
                  const framework::Vector<size_t> &ref_lod, /*referenced lod*/
                  phi::DenseTensor *dx) {
    size_t height = ref_lod.size() - 1;

    const T *softmax_grad_data = dout.data<T>();
    const T *softmax = out.data<T>();
    T *dx_data = dx->mutable_data<T>(ctx.GetPlace());

    for (size_t i = 0; i < height; ++i) {
      size_t span = ref_lod[i + 1] - ref_lod[i];
      T result = 0;
      for (size_t j = 0; j < span; ++j) {
        result += softmax_grad_data[ref_lod[i] + j] * softmax[ref_lod[i] + j];
      }

      for (size_t j = 0; j < span; ++j) {
        dx_data[ref_lod[i] + j] = (softmax_grad_data[ref_lod[i] + j] - result) *
                                  softmax[ref_lod[i] + j];
      }
    }
  }
};

template <typename DeviceContext, typename T>
class SequenceSoftmaxKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *x = ctx.Input<phi::DenseTensor>("X");
    auto *out = ctx.Output<phi::DenseTensor>("Out");

    auto lod = x->lod();
    auto dims = x->dims();
    PADDLE_ENFORCE_EQ(lod.empty(),
                      false,
                      platform::errors::InvalidArgument(
                          "Input(X) phi::DenseTensor of SequenceSoftmax "
                          "operator does not contain "
                          "LoD information."));

    const size_t level = lod.size() - 1;
    PADDLE_ENFORCE_EQ(
        dims[0],
        static_cast<int64_t>(lod[level].back()),
        platform::errors::InvalidArgument(
            "The first dimension of Input(X) should be equal to the sum of all "
            "sequences' lengths. But the first dimension of Input(X) is %d, "
            "the sum of all sequences' lengths is %d.",
            dims[0],
            static_cast<int64_t>(lod[level].back())));
    PADDLE_ENFORCE_EQ(
        dims[0],
        x->numel(),
        platform::errors::InvalidArgument(
            "The width of each timestep in Input(X) of SequenceSoftmax "
            "operator should be 1. But the first dimension of Input(X) is %d, "
            "the number of elements is %d.",
            dims[0],
            x->numel()));

    out->mutable_data<T>(ctx.GetPlace());

    SequenceSoftmaxFunctor<DeviceContext, T> seq_softmax_functor;
    seq_softmax_functor(
        ctx.template device_context<DeviceContext>(), *x, lod[level], out);
  }
};

template <typename DeviceContext, typename T>
class SequenceSoftmaxGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *out = ctx.Input<phi::DenseTensor>("Out");
    auto *out_grad = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto *x = ctx.Input<phi::DenseTensor>("X");
    auto *x_grad = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    if (!x_grad) {
      return;
    }

    x_grad->set_lod(x->lod());
    auto lod = x->lod();
    const size_t level = lod.size() - 1;
    x_grad->mutable_data<T>(ctx.GetPlace());

    SequenceSoftmaxGradFunctor<DeviceContext, T> seq_softmax_grad_functor;
    seq_softmax_grad_functor(ctx.template device_context<DeviceContext>(),
                             *out_grad,
                             *out,
                             lod[level],
                             x_grad);
  }
};

}  // namespace operators
}  // namespace paddle
