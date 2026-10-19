// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/addcmul_grad_kernel.h"
#include "paddle/phi/kernels/addcmul_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/addcmul_functor.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"

namespace phi {

// Sums grad_b, which has the broadcast output shape, into grad.
template <typename T, typename Context>
void AddcmulReduceGrad(const Context& dev_ctx,
                       const DenseTensor& grad_b,
                       DenseTensor* grad) {
  const DDim grad_dims = grad->dims();
  std::vector<int> reduce_dims =
      funcs::GetReduceDim(grad_dims, grad_b.dims(), -1);
  SumKernel<T, Context>(
      dev_ctx, grad_b, IntArray(reduce_dims), grad_b.dtype(), false, grad);
  grad->Resize(grad_dims);
}

// Computes the gradient of one multiplicand: out_grad * conj(value * other),
// then reduces it to the shape of grad. The broadcast intermediate result is
// only allocated when grad really needs a reduction.
template <typename T, typename Context>
void AddcmulMultiplicandGrad(const Context& dev_ctx,
                             const DenseTensor& out_grad,
                             const DenseTensor& other,
                             const funcs::AddcmulGradFunctor<T>& functor,
                             DenseTensor* grad) {
  if (grad->numel() == out_grad.numel()) {
    const DDim grad_dims = grad->dims();
    grad->Resize(out_grad.dims());
    funcs::ElementwiseCompute<funcs::AddcmulGradFunctor<T>, T>(
        dev_ctx, out_grad, other, functor, grad);
    grad->Resize(grad_dims);
    return;
  }
  DenseTensor grad_b;
  grad_b.Resize(out_grad.dims());
  funcs::ElementwiseCompute<funcs::AddcmulGradFunctor<T>, T>(
      dev_ctx, out_grad, other, functor, &grad_b);
  AddcmulReduceGrad<T, Context>(dev_ctx, grad_b, grad);
}

template <typename T, typename Context>
void AddcmulGradKernel(const Context& dev_ctx,
                       const DenseTensor& input,
                       const DenseTensor& tensor1,
                       const DenseTensor& tensor2,
                       const DenseTensor& out_grad,
                       const Scalar& value,
                       DenseTensor* input_grad,
                       DenseTensor* tensor1_grad,
                       DenseTensor* tensor2_grad) {
  if (out_grad.numel() == 0) {
    // An input can be non-empty even if it is broadcast to an empty output.
    for (DenseTensor* grad : {input_grad, tensor1_grad, tensor2_grad}) {
      if (grad) {
        Full<T, Context>(dev_ctx, grad->dims(), 0, grad);
      }
    }
    return;
  }

  if (input_grad) {
    if (input_grad->numel() == out_grad.numel()) {
      const DDim input_grad_dims = input_grad->dims();
      Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, input_grad);
      input_grad->Resize(input_grad_dims);
    } else {
      AddcmulReduceGrad<T, Context>(dev_ctx, out_grad, input_grad);
    }
  }

  funcs::AddcmulGradFunctor<T> functor(funcs::GetAddcmulValue<T>(value));
  if (tensor1_grad) {
    AddcmulMultiplicandGrad<T, Context>(
        dev_ctx, out_grad, tensor2, functor, tensor1_grad);
  }
  if (tensor2_grad) {
    AddcmulMultiplicandGrad<T, Context>(
        dev_ctx, out_grad, tensor1, functor, tensor2_grad);
  }
}

template <typename T, typename Context>
void AddcmulDoubleGradKernel(
    const Context& dev_ctx,
    const DenseTensor& tensor1,
    const DenseTensor& tensor2,
    const DenseTensor& grad_out,
    const paddle::optional<DenseTensor>& grad_input_grad,
    const paddle::optional<DenseTensor>& grad_tensor1_grad,
    const paddle::optional<DenseTensor>& grad_tensor2_grad,
    const Scalar& value,
    DenseTensor* tensor1_grad,
    DenseTensor* tensor2_grad,
    DenseTensor* grad_out_grad) {
  funcs::AddcmulGradFunctor<T> functor(funcs::GetAddcmulValue<T>(value));
  // The grad of one multiplicand is grad_out * conj(value * other), so its
  // grad w.r.t. the other one has the same form with grad_other_grad.
  auto multiplicand_grad =
      [&](const paddle::optional<DenseTensor>& grad_other_grad,
          DenseTensor* grad) {
        if (!grad) {
          return;
        }
        if (!grad_other_grad || grad_out.numel() == 0) {
          Full<T, Context>(dev_ctx, grad->dims(), 0, grad);
          return;
        }
        AddcmulMultiplicandGrad<T, Context>(
            dev_ctx, grad_out, *grad_other_grad, functor, grad);
      };
  multiplicand_grad(grad_tensor2_grad, tensor1_grad);
  multiplicand_grad(grad_tensor1_grad, tensor2_grad);

  if (grad_out_grad) {
    // grad_out_grad = grad_input_grad + value * (grad_tensor1_grad * tensor2 +
    // grad_tensor2_grad * tensor1), broadcast to the shape of grad_out. The
    // missing grads are skipped instead of being treated as zeros, since
    // value * 0 is nan when value is inf or nan.
    Full<T, Context>(dev_ctx, grad_out_grad->dims(), 0, grad_out_grad);
    if (grad_input_grad) {
      funcs::ElementwiseCompute<funcs::AddFunctor<T>, T>(dev_ctx,
                                                         *grad_out_grad,
                                                         *grad_input_grad,
                                                         funcs::AddFunctor<T>(),
                                                         grad_out_grad);
    }
    if (grad_tensor1_grad) {
      AddcmulKernel<T, Context>(dev_ctx,
                                *grad_out_grad,
                                *grad_tensor1_grad,
                                tensor2,
                                value,
                                grad_out_grad);
    }
    if (grad_tensor2_grad) {
      AddcmulKernel<T, Context>(dev_ctx,
                                *grad_out_grad,
                                *grad_tensor2_grad,
                                tensor1,
                                value,
                                grad_out_grad);
    }
  }
}

}  // namespace phi
