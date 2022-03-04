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

#include "paddle/phi/kernels/sgd_kernel.h"
#include "paddle/fluid/operators/jit/kernels.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {

template <typename T>
void sgd_dense_param_dense_grad_impl(const DenseTensor& param,
                                     const DenseTensor& learning_rate,
                                     const DenseTensor& grad,
                                     DenseTensor* param_out) {
  const auto sz = param_out->numel();
  paddle::operators::jit::sgd_attr_t attr(1, sz, 1, sz, 1);
  const T* lr = learning_rate.data<T>();
  const T* param_data = param.data<T>();
  const T* grad_data = grad.data<T>();
  int64_t rows_idx = 0;
  T* out_data = param_out->data<T>();

  auto sgd =
      paddle::operators::jit::KernelFuncs<paddle::operators::jit::SgdTuple<T>,
                                          phi::CPUPlace>::Cache()
          .At(attr);
  sgd(lr, param_data, grad_data, &rows_idx, out_data, &attr);
}

template <>
void sgd_dense_param_dense_grad_impl<phi::dtype::bfloat16>(
    const DenseTensor& param,
    const DenseTensor& learning_rate,
    const DenseTensor& grad,
    DenseTensor* param_out) {
  auto p = EigenVector<phi::dtype::bfloat16>::Flatten(param);
  auto g = EigenVector<phi::dtype::bfloat16>::Flatten(grad);
  auto o = EigenVector<phi::dtype::bfloat16>::Flatten(*param_out);
  const auto* lr = learning_rate.data<phi::dtype::bfloat16>();

  o = p - lr[0] * g;
}

template <typename T>
void sgd_dense_param_sparse_grad_impl(const DenseTensor& param,
                                      const DenseTensor& learning_rate,
                                      const SelectedRows& grad,
                                      DenseTensor* param_out) {
  const auto& grad_value = grad.value();
  const auto& grad_rows = grad.rows();
  const T* param_data = param.data<T>();
  const T* grad_data = grad_value.data<T>();
  const T* lr = learning_rate.data<T>();
  const int64_t* rows_data = grad_rows.data();
  T* out_data = param_out->data<T>();

  paddle::operators::jit::sgd_attr_t attr;
  attr.param_height = param_out->dims()[0];
  attr.param_width = param_out->numel() / attr.param_height;
  attr.grad_height = grad_rows.size();  // note: it is not grad->height()
  attr.grad_width = grad_value.numel() / attr.grad_height;
  attr.selected_rows_size = grad_rows.size();

  auto sgd =
      paddle::operators::jit::KernelFuncs<paddle::operators::jit::SgdTuple<T>,
                                          phi::CPUPlace>::Cache()
          .At(attr);
  sgd(lr, param_data, grad_data, rows_data, out_data, &attr);
}

template <>
void sgd_dense_param_sparse_grad_impl<phi::dtype::bfloat16>(
    const DenseTensor& param,
    const DenseTensor& learning_rate,
    const SelectedRows& grad,
    DenseTensor* param_out) {
  const auto& grad_value = grad.value();
  const auto& grad_rows = grad.rows();
  const auto grad_height = grad.height();
  const int64_t grad_val_height = static_cast<int64_t>(grad_rows.size());
  const auto grad_width = grad_value.numel() / grad_val_height;

  const auto* grad_data = grad_value.data<phi::dtype::bfloat16>();
  auto* out_data = param_out->data<phi::dtype::bfloat16>();
  const auto* lr = learning_rate.data<phi::dtype::bfloat16>();

  for (size_t i = 0; i < grad_rows.size(); ++i) {
    PADDLE_ENFORCE_LT(
        grad_rows[i],
        grad_height,
        phi::errors::OutOfRange(
            "Grad rows index value should be less than grad height."
            "Got [%s], but expected less than [%s]",
            grad_rows[i],
            grad_height));
    const int64_t row = grad_rows[i];
    for (int64_t j = 0; j < grad_width; ++j) {
      out_data[row * grad_width + j] -= lr[0] * grad_data[i * grad_width + j];
    }
  }
}

template <typename T, typename Context>
void SGDDenseKernel(const Context& dev_ctx,
                    const DenseTensor& param,
                    const DenseTensor& learning_rate,
                    const DenseTensor& grad,
                    paddle::optional<const DenseTensor&> master_param,
                    bool multi_precision,
                    DenseTensor* param_out,
                    DenseTensor* master_param_out) {
  dev_ctx.template Alloc<T>(param_out);
  sgd_dense_param_dense_grad_impl<T>(param, learning_rate, grad, param_out);
}

template <typename T, typename Context>
void SGDDenseParamSparseGradKernel(
    const Context& dev_ctx,
    const DenseTensor& param,
    const DenseTensor& learning_rate,
    const SelectedRows& grad,
    paddle::optional<const DenseTensor&> master_param,
    bool multi_precision,
    DenseTensor* param_out,
    DenseTensor* master_param_out) {
  dev_ctx.template Alloc<T>(param_out);
  sgd_dense_param_sparse_grad_impl<T>(param, learning_rate, grad, param_out);
}

template <typename T, typename Context>
void SGDSparseParamSparseGradKernel(
    const Context& dev_ctx,
    const SelectedRows& param,
    const DenseTensor& learning_rate,
    const SelectedRows& grad,
    paddle::optional<const SelectedRows&> master_param,
    bool multi_precision,
    SelectedRows* param_out,
    SelectedRows* master_param_out) {
  // for distributed training, a sparse var may be empty,
  // just skip updating.
  if (grad.rows().size() == 0) {
    return;
  }

  auto param_row_width = param.value().dims()[1];
  auto grad_row_width = grad.value().dims()[1];
  PADDLE_ENFORCE_EQ(
      param_row_width,
      grad_row_width,
      phi::errors::InvalidArgument(
          "The param_row in SgdOP should have the same size with grad_row. "
          "But received param_row's width is [%s], and grad_row's width is "
          "[%s]",
          param_row_width,
          grad_row_width));

  const auto* lr = learning_rate.data<T>();
  const auto* grad_data = grad.value().data<T>();
  auto* out_data = param_out->mutable_value()->data<T>();
  for (size_t i = 0; i < grad.rows().size(); i++) {
    int64_t id_index = param_out->AutoGrownIndex(grad.rows()[i], false);
    PADDLE_ENFORCE_GE(
        id_index,
        static_cast<int64_t>(0),
        phi::errors::InvalidArgument(
            "The id in SgdOp should be >= 0. But recevied id_index is [%s]",
            id_index));
    for (int64_t j = 0; j < grad_row_width; j++) {
      out_data[id_index * grad_row_width + j] -=
          lr[0] * grad_data[i * grad_row_width + j];
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(sgd,
                   CPU,
                   ALL_LAYOUT,
                   phi::SGDDenseKernel,
                   phi::dtype::bfloat16,
                   float,
                   double) {}

PD_REGISTER_KERNEL(sgd_dense_param_sparse_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::SGDDenseParamSparseGradKernel,
                   phi::dtype::bfloat16,
                   float,
                   double) {}

PD_REGISTER_KERNEL(sgd_sparse_param_sparse_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::SGDSparseParamSparseGradKernel,
                   phi::dtype::bfloat16,
                   float,
                   double) {}
