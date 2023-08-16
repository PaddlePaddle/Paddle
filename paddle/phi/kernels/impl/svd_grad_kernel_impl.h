// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/activation_kernel.h"
#include "paddle/phi/kernels/diag_kernel.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/elementwise_multiply_kernel.h"
#include "paddle/phi/kernels/elementwise_subtract_kernel.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/matmul_kernel.h"
#include "paddle/phi/kernels/slice_kernel.h"

namespace phi {

template <class T, class Context>
static DenseTensor Fill(const Context& ctx,
                        std::vector<int> shape,
                        float fill_value) {
  DenseTensor ret;
  ret.Resize(make_ddim(shape));
  ctx.template Alloc<T>(&ret);
  funcs::SetConstant<Context, T>()(ctx, &ret, T(fill_value));
  return ret;
}

template <class T, class Context>
static DenseTensor Eye(const Context& dev_ctx, int n) {
  auto output = Fill<T, Context>(dev_ctx, {n}, 1);
  auto ret = Diag<T, Context>(dev_ctx, output, 0, 0);
  return ret;
}

template <class T, class Context>
static DenseTensor Infinits(const Context& ctx, std::vector<int> shape) {
  auto value = static_cast<T>(std::numeric_limits<double>::infinity());
  return Fill<T, Context>(ctx, shape, value);
}

static DenseTensor Unsqueeze(const DenseTensor& x, int axis = 0) {
  // don't copy data, only change the dims
  DenseTensor out;
  out.ShareDataWith(x);
  std::vector<int> out_shape = phi::vectorize<int>(x.dims());
  if (axis >= 0) {
    auto index = (out_shape.begin() + axis);
    out_shape.insert(index, 1);
  } else if (axis < 0) {
    auto index = (out_shape.end() + axis + 1);
    out_shape.insert(index, 1);
  }
  out.Resize(phi::make_ddim(out_shape));
  return out;
}

template <typename T, typename Context>
void SvdGradKernel(const Context& dev_ctx,
                   const DenseTensor& x UNUSED,
                   const DenseTensor& u,
                   const DenseTensor& vh,
                   const DenseTensor& s,
                   const paddle::optional<DenseTensor>& u_grad,
                   const paddle::optional<DenseTensor>& vh_grad,
                   const paddle::optional<DenseTensor>& s_grad,
                   bool full_matrices,
                   DenseTensor* x_grad) {
  const auto& dX = *x_grad;
  int m = dX.dims()[dX.dims().size() - 2];
  int n = dX.dims()[dX.dims().size() - 1];
  int k = s.dims()[s.dims().size() - 1];
  DenseTensor U, VH, dU, dV, dVH;
  if (full_matrices) {
    // if full_matrices is set, slice the U and VT to k columns
    U = Slice<T, Context>(dev_ctx, u, {u.dims().size() - 1}, {0}, {k});
    // If m < n for input matrices A, we partition A = [X|Y] and R = [U|V]

    VH = Slice<T, Context>(dev_ctx, vh, {vh.dims().size() - 2}, {0}, {k});
    if (u_grad.get_ptr() != nullptr) {
      dU = Slice<T, Context>(
          dev_ctx, *(u_grad.get_ptr()), {u.dims().size() - 1}, {0}, {k});
    }
    if (vh_grad.get_ptr() != nullptr) {
      dVH = Slice<T, Context>(
          dev_ctx, *(vh_grad.get_ptr()), {vh.dims().size() - 2}, {0}, {k});
    }
  } else {
    U = u;
    VH = vh;
    if (u_grad.get_ptr() != nullptr) {
      dU = *(u_grad.get_ptr());
    }
    if (vh_grad.get_ptr() != nullptr) {
      dVH = *(vh_grad.get_ptr());
    }
  }
  auto s_inverse = Pow<T, Context>(dev_ctx, s, -1);
  auto s_square = Pow<T, Context>(dev_ctx, s, 2);
  auto F = Subtract<T, Context>(
      dev_ctx, Unsqueeze(s_square, -2), Unsqueeze(s_square, -1));
  F = Add<T, Context>(
      dev_ctx,
      F,
      Diag<T, Context>(dev_ctx, Infinits<T, Context>(dev_ctx, {k}), 0, 0));
  F = Pow<T, Context>(dev_ctx, F, -1);
  DenseTensor sigma_term = Fill<T, Context>(dev_ctx, {1}, 0.0);
  DenseTensor u_term = Fill<T, Context>(dev_ctx, {1}, 0.0);
  DenseTensor v_term = Fill<T, Context>(dev_ctx, {1}, 0.0);

  if (s_grad.get_ptr() != nullptr) {
    const DenseTensor& gS = *(s_grad.get_ptr());
    sigma_term = Multiply<T, Context>(dev_ctx, Unsqueeze(gS, -2), U);
    sigma_term = Matmul<T, Context>(dev_ctx, sigma_term, VH);
  }

  if (u_grad.get_ptr() != nullptr) {
    auto UTG = Matmul<T, Context>(dev_ctx, U, dU, true, false);
    auto GTU = Matmul<T, Context>(dev_ctx, dU, U, true, false);
    u_term = Multiply<T, Context>(
        dev_ctx,
        Multiply<T, Context>(
            dev_ctx, Subtract<T, Context>(dev_ctx, UTG, GTU), F),
        Unsqueeze(s, -2));
    u_term = Matmul<T, Context>(dev_ctx, U, u_term);
    if (m > k) {
      auto project =
          Subtract<T, Context>(dev_ctx,
                               Eye<T, Context>(dev_ctx, m),
                               Matmul<T, Context>(dev_ctx, U, U, false, true));
      u_term = Add<T, Context>(
          dev_ctx,
          u_term,
          Multiply<T, Context>(dev_ctx,
                               Matmul<T, Context>(dev_ctx, project, dU),
                               Unsqueeze(s_inverse, -2)));
    }
    u_term = Matmul<T, Context>(dev_ctx, u_term, VH);
  }
  if (vh_grad.get_ptr() != nullptr) {
    auto UTG = Matmul<T, Context>(dev_ctx, VH, dVH, false, true);
    auto GTU = Matmul<T, Context>(dev_ctx, dVH, VH, false, true);
    v_term = Multiply<T, Context>(
        dev_ctx,
        Matmul<T, Context>(
            dev_ctx,
            Multiply<T, Context>(
                dev_ctx, Subtract<T, Context>(dev_ctx, UTG, GTU), F),
            VH),
        Unsqueeze(s, -1));
    if (n > k) {
      auto project = Subtract<T, Context>(
          dev_ctx,
          Eye<T, Context>(dev_ctx, n),
          Matmul<T, Context>(dev_ctx, VH, VH, true, false));
      v_term = Add<T, Context>(
          dev_ctx,
          v_term,
          Multiply<T, Context>(dev_ctx,
                               Matmul<T, Context>(dev_ctx, dVH, project),
                               Unsqueeze(s_inverse, -1)));
    }
    v_term = Matmul<T, Context>(dev_ctx, U, v_term);
  }

  *x_grad = Add<T, Context>(
      dev_ctx, Add<T, Context>(dev_ctx, u_term, sigma_term), v_term);
}

}  // namespace phi
