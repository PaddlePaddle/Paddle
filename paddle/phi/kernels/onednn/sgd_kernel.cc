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

#include "paddle/phi/backends/onednn/axpy_handler.h"
#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void SGDDenseKernel(const Context& dev_ctx,
                    const DenseTensor& param,
                    const DenseTensor& learning_rate,
                    const DenseTensor& grad,
                    const paddle::optional<DenseTensor>& master_param,
                    bool multi_precision,
                    DenseTensor* param_out,
                    DenseTensor* master_param_out) {
  auto* out_data = dev_ctx.template Alloc<T>(param_out);
  const T* param_data = param.data<T>();
  const auto* grad_data = grad.data<T>();
  const auto* lr = learning_rate.data<T>();
  // Since denese SGD is not in place operation, first copy params to output
  // tensor and then update it.
  std::memcpy(out_data, param_data, param.memory_size());
  funcs::OneDNNAXPYHandler<T>(param_out->numel(), -lr[0], dev_ctx.GetEngine())(
      grad_data, out_data);
}

template <typename T, typename Context>
void SGDDenseParamSparseGradKernel(
    const Context& dev_ctx,
    const DenseTensor& param,
    const DenseTensor& learning_rate,
    const SelectedRows& grad,
    const paddle::optional<DenseTensor>& master_param,
    bool multi_precision,
    DenseTensor* param_out,
    DenseTensor* master_param_out) {
  const auto& grad_value = grad.value();
  const auto& grad_rows = grad.rows();
  const auto grad_height = grad.height();
  const int64_t grad_val_height = static_cast<int64_t>(grad_rows.size());
  const auto grad_width = grad_value.numel() / grad_val_height;

  const auto* grad_data = grad_value.data<T>();
  auto* out_data = param_out->data<T>();
  const auto* lr = learning_rate.data<T>();

  funcs::OneDNNAXPYHandler<T> axpy_handler(
      grad_width, -lr[0], dev_ctx.GetEngine());

  for (size_t i = 0; i < grad_rows.size(); ++i) {
    PADDLE_ENFORCE_LT(
        grad_rows[i],
        grad_height,
        errors::OutOfRange(
            "Grad rows index value should be less than grad height."
            "Got [%s], but expected less than [%s]",
            grad_rows[i],
            grad_height));
    const int64_t row = grad_rows[i];
    const auto* src = grad_data + i * grad_width;
    auto* dst = out_data + row * grad_width;
    axpy_handler(src, dst);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    sgd, OneDNN, ALL_LAYOUT, phi::SGDDenseKernel, float, phi::dtype::bfloat16) {
}

PD_REGISTER_KERNEL(sgd_dense_param_sparse_grad,
                   OneDNN,
                   ALL_LAYOUT,
                   phi::SGDDenseParamSparseGradKernel,
                   float,
                   phi::dtype::bfloat16) {}
