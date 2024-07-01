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
#include <numeric>  // std::iota

#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/mixed_vector.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename DeviceContext, typename T>
struct SequenceExpandFunctor {
  void operator()(const DeviceContext& ctx,
                  const phi::DenseTensor& x,
                  const phi::Vector<size_t>& x_lod,   /*expand source lod*/
                  const phi::Vector<size_t>& ref_lod, /*expand referenced lod*/
                  phi::DenseTensor* out);
};

template <typename DeviceContext, typename T>
struct SequenceExpandGradFunctor {
  void operator()(const DeviceContext& ctx,
                  const phi::DenseTensor& dout,
                  const phi::Vector<size_t>& x_lod,   /*expand source lod*/
                  const phi::Vector<size_t>& ref_lod, /*expand referenced lod*/
                  phi::DenseTensor* dx);
};

template <typename T>
struct SequenceExpandFunctor<phi::CPUContext, T> {
  void operator()(const phi::CPUContext& context UNUSED,
                  const phi::DenseTensor& x,
                  const phi::Vector<size_t>& x_lod,   /*expand source lod*/
                  const phi::Vector<size_t>& ref_lod, /*expand referenced lod*/
                  phi::DenseTensor* out) {
    int out_offset = 0;
    int x_item_length = x.numel() / x.dims()[0];
    auto out_data = out->data<T>();
    auto x_data = x.data<T>();
    for (size_t i = 1; i < ref_lod.size(); ++i) {
      int repeat_num = ref_lod[i] - ref_lod[i - 1];
      int x_start = x_lod[i - 1];
      int x_end = x_lod[i];
      int x_seq_len = x_end - x_start;
      if (repeat_num > 0) {
        int out_start = out_offset;
        if (out->lod().size() == 1) {
          out_start = out->lod()[0][out_offset];
        }
        for (int j = 0; j < repeat_num; j++) {
          for (int k = 0; k < x_seq_len; k++) {
            for (int l = 0; l < x_item_length; l++) {
              out_data[(out_start + j * x_seq_len + k) * x_item_length + l] =
                  x_data[(x_start + k) * x_item_length + l];
            }
          }
        }
      }
      out_offset += repeat_num;
    }
  }
};

template <typename T, typename Context>
void SequenceExpandKernel(const Context& dev_ctx,
                          const DenseTensor& x_in,
                          const DenseTensor& y_in,
                          int ref_level,
                          DenseTensor* out) {
  // From InferShape
  auto x_dims = x_in.dims();
  auto out_dims = x_dims;
  auto& x_lod = x_in.lod();
  auto& y_lod = y_in.lod();

  PADDLE_ENFORCE_LE(x_lod.size(),
                    1UL,
                    phi::errors::InvalidArgument(
                        "Level of Input(X)'s lod should not be "
                        "greater than 1. But received: lod level %u.",
                        x_lod.size()));
  PADDLE_ENFORCE_GT(y_lod.size(),
                    0UL,
                    phi::errors::InvalidArgument(
                        "Level of Input(Y)'s lod should be greater than 0. But "
                        "received: lod level %u.",
                        y_lod.size()));
  PADDLE_ENFORCE_EQ(
      ref_level == -1 ||
          (ref_level >= 0 && ref_level < static_cast<int>(y_lod.size())),
      true,
      phi::errors::InvalidArgument(
          "Invalid `ref_level`, which should be either equal to -1 "
          "or in [0, %d), but received `ref_level` = %u.",
          y_lod.size(),
          ref_level));

  if (ref_level == -1) ref_level = static_cast<int>(y_lod.size() - 1);

  if (!x_lod.empty()) {
    PADDLE_ENFORCE_EQ(
        x_lod[0].size(),
        y_lod[ref_level].size(),
        phi::errors::InvalidArgument(
            "Level number of Input(X)'s lod could be 0. Otherwise "
            "size of Input(X)'s first level lod should be equal to "
            "size of Input(Y)'s referred level lod. But received: "
            "Input(X).lod[0].size() = %u, Input(Y).lod[%d].size() = "
            "%u",
            x_lod[0].size(),
            ref_level,
            y_lod[ref_level].size()));
  } else {
    PADDLE_ENFORCE_EQ(x_dims[0],
                      static_cast<int64_t>(y_lod[ref_level].size()) - 1,
                      phi::errors::InvalidArgument(
                          "When Input(X)'s lod is null, the dims[0] of "
                          "Input(X) should match the "
                          "size of Input(Y)'s referred level lod. But received "
                          "Input(X): input rank %u, input shape [%s]; received "
                          "Input(Y).lod[%d].size() - 1 = %d.",
                          x_dims.size(),
                          x_dims,
                          ref_level,
                          static_cast<int64_t>(y_lod[ref_level].size()) - 1));
  }

  int64_t out_first_dim = 0;
  if (y_lod[ref_level].size() <= 1) {
    out_first_dim = x_dims[0];
  } else {
    for (size_t i = 1; i < y_lod[ref_level].size(); ++i) {
      int x_seq_len = 1;
      if (x_lod.size() == 1) {
        x_seq_len = static_cast<int>(x_lod[0][i] - x_lod[0][i - 1]);
      }
      out_first_dim += static_cast<int64_t>(
          (y_lod[ref_level][i] - y_lod[ref_level][i - 1]) * x_seq_len);
    }
  }
  out_dims[0] = out_first_dim;
  out->Resize(out_dims);

  auto* x = &x_in;
  PADDLE_ENFORCE_EQ(
      y_lod.empty(),
      false,
      phi::errors::InvalidArgument(
          "Input(Y) phi::DenseTensor of SequenceExpandOp does not contain "
          "LoD information."));

  if (ref_level == -1) ref_level = y_lod.size() - 1;

  dev_ctx.template Alloc<T>(out);

  if (y_lod[ref_level].size() <= 1) {
    phi::Copy(dev_ctx, *x, dev_ctx.GetPlace(), false, out);
    return;
  }

  // x lod level is at most 1.
  phi::Vector<size_t> out_lod;
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
  phi::Vector<size_t> ref_x_lod;
  if (x->lod().size() == 1) {
    ref_x_lod = x->lod()[0];
  } else {
    // x_lod doesn't has lod, use fake x lod, level = 0
    ref_x_lod.resize(x->dims()[0] + 1);
    std::iota(ref_x_lod.begin(), ref_x_lod.end(), 0);
  }
  SequenceExpandFunctor<Context, T> functor;
  functor(dev_ctx, *x, ref_x_lod, y_lod[ref_level], out);
}

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
struct SequenceExpandGradFunctor<phi::CPUContext, T> {
  void operator()(const phi::CPUContext& context,
                  const phi::DenseTensor& dout,
                  const phi::Vector<size_t>& x_lod,   /*expand source lod*/
                  const phi::Vector<size_t>& ref_lod, /*expand referenced lod*/
                  phi::DenseTensor* dx) {
    int dout_offset = 0;
    for (size_t i = 1; i < ref_lod.size(); ++i) {
      int repeat_num = ref_lod[i] - ref_lod[i - 1];
      if (repeat_num > 0) {
        int x_start = x_lod[i - 1];
        int x_end = x_lod[i];
        int x_seq_len = x_end - x_start;
        if (x_seq_len == 0) continue;
        auto dx_sub = dx->Slice(x_start, x_end);
        dx_sub.Resize(common::flatten_to_1d(dx_sub.dims()));
        int dout_end = dout_offset + repeat_num * x_seq_len;
        auto dout_sub = dout.Slice(dout_offset, dout_end);
        dout_sub.Resize({repeat_num, dx_sub.dims()[0]});
        phi::funcs::ColwiseSum<phi::CPUContext, T> col_sum;
        col_sum(context, dout_sub, &dx_sub);
        dout_offset += repeat_num * x_seq_len;
      }
    }
  }
};

template <typename T, typename Context>
void SequenceExpandGradKernel(const Context& dev_ctx,
                              const DenseTensor& x_in,
                              const DenseTensor& y_in,
                              const DenseTensor& out_grad,
                              int ref_level,
                              DenseTensor* x_grad) {
  auto* g_out = &out_grad;
  auto* x = &x_in;
  auto* y = &y_in;
  auto* g_x = x_grad;

  dev_ctx.template Alloc<T>(g_x);
  g_x->set_lod(x->lod());

  phi::funcs::SetConstant<Context, T> set_zero;
  set_zero(dev_ctx, g_x, static_cast<T>(0));

  auto& y_lod = y->lod();
  if (ref_level == -1) ref_level = y_lod.size() - 1;
  // just copy the gradient
  if (y_lod[ref_level].size() <= 1) {
    phi::Copy(dev_ctx, *g_out, dev_ctx.GetPlace(), false, g_x);
    return;
  }

  phi::Vector<size_t> ref_x_lod;
  phi::Vector<size_t> ref_lod = y_lod[ref_level];
  if (x->lod().size() == 1) {
    ref_x_lod = x->lod()[0];
  } else {
    // x_lod doesn't has lod, use fake x lod, level = 0
    ref_x_lod.resize(x->dims()[0] + 1);
    std::iota(ref_x_lod.begin(), ref_x_lod.end(), 0);
  }
  SequenceExpandGradFunctor<Context, T> functor;
  functor(dev_ctx, *g_out, ref_x_lod, ref_lod, g_x);
}
}  // namespace phi
