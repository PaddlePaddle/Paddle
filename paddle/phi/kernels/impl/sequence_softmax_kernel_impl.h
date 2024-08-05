// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/mixed_vector.h"

namespace phi {

template <typename Context, typename T>
struct SequenceSoftmaxFunctor {
  void operator()(const Context &ctx,
                  const phi::DenseTensor &x,
                  const phi::Vector<size_t> &ref_lod, /*expand referenced lod*/
                  phi::DenseTensor *out);
};

template <typename Context, typename T>
struct SequenceSoftmaxGradFunctor {
  void operator()(const Context &ctx,
                  const phi::DenseTensor &dout,
                  const phi::DenseTensor &out,
                  const phi::Vector<size_t> &ref_lod, /*referenced lod*/
                  phi::DenseTensor *dx);
};

template <typename T>
struct SequenceSoftmaxFunctor<phi::CPUContext, T> {
  void operator()(const phi::CPUContext &ctx,
                  const phi::DenseTensor &x,
                  const phi::Vector<size_t> &ref_lod, /*referenced lod*/
                  phi::DenseTensor *out) {
    size_t height = ref_lod.size() - 1;
    const T *in_data = x.data<T>();
    T *out_data = ctx.Alloc<T>(out);
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
                  const phi::Vector<size_t> &ref_lod, /*referenced lod*/
                  phi::DenseTensor *dx) {
    size_t height = ref_lod.size() - 1;

    const T *softmax_grad_data = dout.data<T>();
    const T *softmax = out.data<T>();
    T *dx_data = ctx.Alloc<T>(dx);

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

template <typename T, typename Context>
void SequenceSoftmaxKernel(const Context &dev_ctx,
                           const DenseTensor &x_in,
                           DenseTensor *out) {
  auto *x = &x_in;

  auto lod = x->lod();
  auto dims = x->dims();
  PADDLE_ENFORCE_EQ(lod.empty(),
                    false,
                    common::errors::InvalidArgument(
                        "Input(X) phi::DenseTensor of SequenceSoftmax "
                        "operator does not contain "
                        "LoD information."));

  const size_t level = lod.size() - 1;
  PADDLE_ENFORCE_EQ(
      dims[0],
      static_cast<int64_t>(lod[level].back()),
      common::errors::InvalidArgument(
          "The first dimension of Input(X) should be equal to the sum of all "
          "sequences' lengths. But the first dimension of Input(X) is %d, "
          "the sum of all sequences' lengths is %d.",
          dims[0],
          static_cast<int64_t>(lod[level].back())));
  PADDLE_ENFORCE_EQ(
      dims[0],
      x->numel(),
      common::errors::InvalidArgument(
          "The width of each timestep in Input(X) of SequenceSoftmax "
          "operator should be 1. But the first dimension of Input(X) is %d, "
          "the number of elements is %d.",
          dims[0],
          x->numel()));

  dev_ctx.template Alloc<T>(out);

  SequenceSoftmaxFunctor<Context, T> seq_softmax_functor;
  seq_softmax_functor(dev_ctx, *x, lod[level], out);
}

template <typename T, typename Context>
void SequenceSoftmaxGradKernel(const Context &dev_ctx,
                               const DenseTensor &x_in,
                               const DenseTensor &out_in,
                               const DenseTensor &out_grad_in,
                               DenseTensor *x_grad) {
  auto *out = &out_in;
  auto *out_grad = &out_grad_in;
  auto *x = &x_in;
  if (!x_grad) {
    return;
  }

  x_grad->set_lod(x->lod());
  auto lod = x->lod();
  const size_t level = lod.size() - 1;
  dev_ctx.template Alloc<T>(x_grad);

  SequenceSoftmaxGradFunctor<Context, T> seq_softmax_grad_functor;
  seq_softmax_grad_functor(dev_ctx, *out_grad, *out, lod[level], x_grad);
}
}  // namespace phi
