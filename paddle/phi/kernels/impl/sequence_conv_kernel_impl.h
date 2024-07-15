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
#include <algorithm>

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math/context_project.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/utils/optional.h"

namespace phi {

template <typename T, typename Context>
void SequenceConvKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const paddle::optional<DenseTensor>& padding_data,
                        const DenseTensor& filter,
                        int context_length,
                        bool padding_trainable,
                        int context_start,
                        int context_stride,
                        DenseTensor* out) {
  auto* in = &x;
  dev_ctx.template Alloc<T>(out);
  PADDLE_ENFORCE_EQ(in->lod().empty(),
                    false,
                    phi::errors::InvalidArgument(
                        "Input(X) phi::DenseTensor of SequenceConvOp "
                        "does not contain LoD information."));
  PADDLE_ENFORCE_EQ(
      in->lod().size(),
      1UL,
      phi::errors::InvalidArgument(
          "Only support input sequence with lod level equal to 1 at "
          "present. But received: lod level %u.",
          in->lod().size()));

  const phi::DenseTensor* padding_data_p = nullptr;
  if (padding_trainable) {
    padding_data_p = padding_data.get_ptr();
  }

  int up_pad = std::max(0, -context_start);
  int down_pad = std::max(0, context_start + context_length - 1);
  auto sequence_width = static_cast<int64_t>(in->dims()[1]);

  phi::DDim col_shape = {in->dims()[0], context_length * sequence_width};
  phi::DenseTensor col;
  col.Resize(col_shape);
  dev_ctx.template Alloc<T>(&col);
  // Because if padding_trainable is false, padding data should be zeros.
  phi::funcs::SetConstant<Context, T> set_zero;
  auto blas = phi::funcs::GetBlas<Context, T>(dev_ctx);
  set_zero(dev_ctx, &col, static_cast<T>(0));
  phi::math::ContextProjectFunctor<Context, T> seq_project_functor;

  seq_project_functor(dev_ctx,
                      *in,
                      padding_data_p,
                      padding_trainable,
                      context_start,
                      context_length,
                      context_stride,
                      up_pad,
                      down_pad,
                      &col);

  blas.MatMul(col, filter, out);
}

template <typename T, typename Context>
void SequenceConvGradKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const paddle::optional<DenseTensor>& padding_data,
                            const DenseTensor& filter,
                            const DenseTensor& out_grad,
                            int context_length,
                            bool padding_trainable,
                            int context_start,
                            int context_stride,
                            DenseTensor* x_grad,
                            DenseTensor* padding_data_grad,
                            DenseTensor* filter_grad) {
  auto* in_g = x_grad;
  auto* out_g = &out_grad;
  auto* filter_g = filter_grad;
  auto* padding_data_g = padding_data_grad;
  auto* in = &x;

  PADDLE_ENFORCE_EQ(
      in->lod().size(),
      1UL,
      phi::errors::InvalidArgument(
          "Only support input sequence with lod level equal to 1 at "
          "present. But received: lod level %u.",
          in->lod().size()));
  auto lod_g_level_0 = in->lod()[0];

  int up_pad = std::max(0, -context_start);
  int down_pad = std::max(0, context_start + context_length - 1);
  auto sequence_width = static_cast<int64_t>(in->dims()[1]);

  phi::funcs::SetConstant<Context, T> set_zero;
  auto blas = phi::funcs::GetBlas<Context, T>(dev_ctx);
  // use col_shape in the im2col calculation
  phi::DDim col_shape = {in->dims()[0], sequence_width * context_length};
  phi::DenseTensor col;

  if (in_g || filter_g || (padding_trainable && padding_data_g)) {
    col.Resize(col_shape);
    dev_ctx.template Alloc<T>(&col);
    // Because if padding_trainable is false, padding data should be zeros.
    set_zero(dev_ctx, &col, static_cast<T>(0));
    blas.MatMul(*out_g, false, filter, true, &col);
  }
  phi::math::ContextProjectFunctor<Context, T> seq_project_functor;
  phi::math::ContextProjectGradFunctor<Context, T> seq_project_grad_functor;

  if (in_g != nullptr) {
    dev_ctx.template Alloc<T>(in_g);
    in_g->set_lod(in->lod());
    set_zero(dev_ctx, in_g, static_cast<T>(0));

    seq_project_grad_functor(dev_ctx,
                             *in_g,
                             padding_trainable,
                             context_start,
                             context_length,
                             context_stride,
                             up_pad,
                             down_pad,
                             false,
                             true,
                             padding_data_g,
                             &col);
  }

  if (padding_trainable && padding_data_g != nullptr) {
    dev_ctx.template Alloc<T>(padding_data_g);
    set_zero(dev_ctx, padding_data_g, static_cast<T>(0));

    phi::DenseTensor* input = const_cast<phi::DenseTensor*>(in);
    seq_project_grad_functor(dev_ctx,
                             *input,
                             padding_trainable,
                             context_start,
                             context_length,
                             context_stride,
                             up_pad,
                             down_pad,
                             true,
                             false,
                             padding_data_g,
                             &col);
  }

  if (filter_g != nullptr) {
    dev_ctx.template Alloc<T>(filter_g);
    set_zero(dev_ctx, filter_g, static_cast<T>(0));

    phi::DenseTensor out_grad = *out_g;

    const phi::DenseTensor* padding_data_p = nullptr;
    if (padding_trainable) {
      padding_data_p = padding_data.get_ptr();
    }

    seq_project_functor(dev_ctx,
                        *in,
                        padding_data_p,
                        padding_trainable,
                        context_start,
                        context_length,
                        context_stride,
                        up_pad,
                        down_pad,
                        &col);

    blas.MatMul(col, true, out_grad, false, filter_g);
  }
}

}  // namespace phi
