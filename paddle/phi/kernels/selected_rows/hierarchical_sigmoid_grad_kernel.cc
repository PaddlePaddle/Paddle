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

#include "paddle/phi/kernels/selected_rows/hierarchical_sigmoid_grad_kernel.h"

#include "paddle/fluid/framework/mixed_vector.h"
#include "paddle/fluid/operators/math/matrix_bit_code.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {
namespace sr {

namespace math = paddle::operators::math;

static std::vector<int64_t> PathToRows(const DenseTensor& path) {
  std::set<int64_t> rows;
  const int64_t* paths = path.data<int64_t>();
  for (int64_t i = 0; i < path.numel(); ++i) {
    int64_t row = paths[i];
    if (row < 0) {
      continue;
    }
    rows.emplace(row);
  }
  return std::vector<int64_t>(rows.begin(), rows.end());
}

template <typename T, typename Context>
void HierarchicalSigmoidGradKernel(const Context& ctx,
                                   const DenseTensor& x,
                                   const DenseTensor& w,
                                   const DenseTensor& label,
                                   const DenseTensor& pre_out,
                                   const DenseTensor& out_grad,
                                   paddle::optional<const DenseTensor&> path,
                                   paddle::optional<const DenseTensor&> code,
                                   paddle::optional<const DenseTensor&> bias,
                                   int num_classes,
                                   bool remote_prefetch,
                                   int trainer_id,
                                   const std::vector<int64_t>& height_sections,
                                   const std::vector<std::string>& epmap,
                                   const std::vector<std::string>& table_names,
                                   bool is_sparse,
                                   DenseTensor* x_grad,
                                   SelectedRows* w_grad,
                                   DenseTensor* bias_grad) {
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

  std::unique_ptr<math::MatrixBitCodeFunctor<T>> bit_code;
  if (!is_custom) {
    bit_code.reset(new math::MatrixBitCodeFunctor<T>(
        num_classes, label.template data<int64_t>()));
  } else {
    bit_code.reset(new math::MatrixBitCodeFunctor<T>(
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
  PADDLE_ENFORCE_NOT_NULL(
      path.get_ptr(),
      errors::NotFound("Custom tree must be set for sparse mode!"));
  paddle::framework::Vector<int64_t> real_rows = PathToRows(*path);
  w_grad->set_rows(real_rows);
  // Build a map of id -> row_index to speed up finding the index of one id
  w_grad->set_height(w.dims()[0]);
  auto* w_grad_value = w_grad->mutable_value();
  phi::DDim temp_dim(w.dims());
  temp_dim[0] = real_rows.size();
  w_grad_value->Resize(temp_dim);
  ctx.template Alloc<T>(w_grad_value);
  zero(ctx, w_grad_value, static_cast<T>(0.0));
  bit_code->MulGradWeight(pre_out_grad, w_grad, x);
  bit_code->MulGradError(pre_out_grad, w, x_grad);
}

}  // namespace sr
}  // namespace phi

PD_REGISTER_KERNEL(hierarchical_sigmoid_grad_sr,
                   CPU,
                   ALL_LAYOUT,
                   phi::sr::HierarchicalSigmoidGradKernel,
                   float,
                   double) {}
