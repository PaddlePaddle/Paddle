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
#include "paddle/phi/core/selected_rows.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/matrix_bit_code.h"

namespace phi {

template <typename T, typename Context>
void HSigmoidLossGradKernelImpl(const Context& ctx,
                                const DenseTensor& x,
                                const DenseTensor& w,
                                const DenseTensor& label,
                                const paddle::optional<DenseTensor>& path,
                                const paddle::optional<DenseTensor>& code,
                                const paddle::optional<DenseTensor>& bias
                                    UNUSED,
                                const DenseTensor& pre_out,
                                const DenseTensor& out_grad,
                                int num_classes,
                                bool is_sparse,
                                DenseTensor* x_grad,
                                DenseTensor* w_grad,
                                DenseTensor* bias_grad,
                                SelectedRows* w_grad_sr = nullptr) {
  funcs::SetConstant<Context, T> zero;
  DenseTensor pre_out_grad;

  pre_out_grad.Resize(pre_out.dims());
  ctx.template Alloc<T>(&pre_out_grad);
  ctx.template Alloc<T>(x_grad);
  zero(ctx, x_grad, static_cast<T>(0.0));

  bool is_custom = false;
  if (path.get_ptr()) {
    is_custom = true;
  }

  std::unique_ptr<phi::funcs::MatrixBitCodeFunctor<T>> bit_code;
  if (!is_custom) {
    bit_code.reset(new phi::funcs::MatrixBitCodeFunctor<T>(
        num_classes, label.template data<int64_t>()));
  } else {
    bit_code.reset(new phi::funcs::MatrixBitCodeFunctor<T>(
        *(path.get_ptr()), *(code.get_ptr()), label.template data<int64_t>()));
  }

  // softrelu derivative

  auto blas = funcs::GetBlas<Context, T>(ctx);

  auto* pre_out_grad_data = pre_out_grad.data<T>();
  auto* pre_out_data = pre_out.template data<T>();
  auto n = pre_out.numel();
  blas.VEXP(n, pre_out_data, pre_out_grad_data);
  blas.VINV(n, pre_out_grad_data, pre_out_grad_data);
  for (int64_t i = 0; i < n; ++i) {
    pre_out_grad_data[i] = 1.0 - pre_out_grad_data[i];
  }
  bit_code->Sub(&pre_out_grad);  // the gradient of clip(w * x + b)
  auto* out_grad_data = out_grad.template data<T>();

  int64_t dim0 = pre_out_grad.dims()[0];
  int64_t dim1 = pre_out_grad.dims()[1];
  for (int64_t i = 0; i < dim0; ++i) {
    T tmp = out_grad_data[i];
    blas.SCAL(dim1, tmp, pre_out_grad_data + i * dim1);
  }
  // TODO(guosheng): multiply pre_out_grad with subgradient of clipping to
  // be consistent with the clipping in forward.
  if (bias_grad) {
    ctx.template Alloc<T>(bias_grad);
    zero(ctx, bias_grad, static_cast<T>(0.0));
    bit_code->AddGrad(pre_out_grad, bias_grad);
  }
  ctx.template Alloc<T>(w_grad);
  zero(ctx, w_grad, static_cast<T>(0.0));
  if (!is_sparse) {
    bit_code->MulGradWeight(pre_out_grad, w_grad, x);
  } else {
    bit_code->MulGradWeight(pre_out_grad, w_grad_sr, x);
  }
  bit_code->MulGradError(pre_out_grad, w, x_grad);
}

}  // namespace phi
